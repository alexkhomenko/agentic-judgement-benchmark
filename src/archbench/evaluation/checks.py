"""Human checks: a reviewer reads every response next to its judgements and approves or flags it.

A check is made while seeing the judges' assessments and the recommendation extraction (never the
candidate's identity). It names the exact judge evaluation and extraction records it saw, so a check goes
stale if those change. Approving lets the judge mean stand, with each critical-failure candidate the judges
raised confirmed or rejected by the reviewer. Flagging a judge or the answer holds the response for a full,
blind human review (`review-export` / `review-import`); flagging only the extraction is reported but does
not block scoring. A changed decision is a new record that supersedes the reviewer's earlier one.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ..domain.common import canonical_json, new_id
from ..domain.evaluation import HumanCheckSubmission
from ..domain.records import Record, RecordKind
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from .records import EVALUATION_CODE_REVISION, effective, read_packet, run_records, selected_attempt
from .review import judge_raw_scores, selection_record_id

if TYPE_CHECKING:
    from .runner import RunContext


class CheckError(ValueError):
    pass


# ----- index of one run's records --------------------------------------------------------------------


@dataclass
class RunIndex:
    ctx: RunContext
    packets: list[Record]  # review order: round, then packet id (random, so it hides configuration order)
    judges: dict[str, dict[str, list[Record]]]  # packet_id -> judge_id -> attempts
    extractions: dict[str, list[Record]]  # packet_id -> attempts
    validations: dict[str, Record]  # response_id -> validation
    reviews: dict[str, list[Record]]  # response_id -> effective full human reviews
    checks: dict[str, list[Record]]  # packet_id -> effective checks (all reviewers)
    selection: Record | None
    reasons: dict[str, list[str]] = field(default_factory=dict)  # packet_id -> selection reasons

    @classmethod
    def build(cls, db: Database, ctx: RunContext) -> RunIndex:
        exp, run_id = ctx.experiment_id, ctx.run.record_id
        packets = sorted(run_records(db, RecordKind.BLINDED_PACKET, exp, run_id),
                         key=lambda p: (int(p.payload["round"]), p.record_id))
        judges: dict[str, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
        for r in run_records(db, RecordKind.JUDGE_EVALUATION, exp, run_id):
            judges[r.payload["packet_id"]][r.payload["judge_id"]].append(r)
        extractions: dict[str, list[Record]] = defaultdict(list)
        for r in run_records(db, RecordKind.RECOMMENDATION_EXTRACTION, exp, run_id):
            extractions[r.payload["packet_id"]].append(r)
        reviews: dict[str, list[Record]] = defaultdict(list)
        for r in effective(run_records(db, RecordKind.HUMAN_REVIEW, exp, run_id)):
            reviews[str(r.subject_id)].append(r)
        checks: dict[str, list[Record]] = defaultdict(list)
        for r in effective(run_records(db, RecordKind.HUMAN_CHECK, exp, run_id)):
            checks[r.payload["packet_id"]].append(r)
        selection = db.get_record(selection_record_id(run_id))
        reasons = {i["packet_id"]: i["reasons"] for i in selection.payload["items"]} if selection else {}
        return cls(
            ctx=ctx, packets=packets, judges=judges, extractions=extractions,
            validations={str(r.subject_id): r for r in run_records(db, RecordKind.VALIDATION, exp, run_id)},
            reviews=reviews, checks=checks, selection=selection, reasons=reasons,
        )

    def packet(self, packet_id: str) -> Record:
        found = next((p for p in self.packets if p.record_id == packet_id), None)
        if found is None:
            raise CheckError(f"packet {packet_id} does not belong to evaluation run {self.ctx.run.record_id}")
        return found

    def chosen_judges(self, packet_id: str) -> dict[str, Record]:
        return {
            j: a for j in self.ctx.judges
            if (a := selected_attempt(self.judges[packet_id].get(j, []))) is not None
        }

    def chosen_extraction(self, packet_id: str) -> Record | None:
        return selected_attempt(self.extractions.get(packet_id, []))

    def seen(self, packet_id: str) -> dict[str, Any]:
        """The exact records a check of this packet is a check of."""
        extraction = self.chosen_extraction(packet_id)
        return {
            "judge_evaluation_ids": {j: r.record_id for j, r in sorted(self.chosen_judges(packet_id).items())},
            "extraction_id": extraction.record_id if extraction else None,
        }

    def current_checks(self, packet_id: str, reviewer_id: str | None = None) -> list[Record]:
        """Effective checks made against the records the packet has now. Others are stale."""
        seen = self.seen(packet_id)
        return [
            c for c in self.checks.get(packet_id, [])
            if (reviewer_id is None or c.payload["reviewer_id"] == reviewer_id)
            and {k: c.payload[k] for k in seen} == seen
        ]

    def state(self, packet_id: str, reviewer_id: str | None = None) -> str:
        checks = self.current_checks(packet_id, reviewer_id)
        if not checks:
            stale = any(reviewer_id is None or c.payload["reviewer_id"] == reviewer_id
                        for c in self.checks.get(packet_id, []))
            return "stale" if stale else "unchecked"
        return "flagged" if any(c.payload["flags"] for c in checks) else "approved"

    def blocking(self, packet_id: str) -> bool:
        return any(c.payload["blocking"] for c in self.current_checks(packet_id))


# ----- recording a check -----------------------------------------------------------------------------


def candidate_failures(judges: dict[str, Record]) -> list[str]:
    return sorted({c["failure"] for r in judges.values() for c in r.payload["output"]["critical_failure_candidates"]})


def record_check(
    db: Database, store: ArtifactStore, index: RunIndex, data: dict[str, Any] | bytes
) -> tuple[Record, bool]:
    """Validate and append a check. Returns (record, created); an unchanged resubmission is not re-recorded."""
    try:
        submission = (HumanCheckSubmission.model_validate_json(data) if isinstance(data, bytes)
                      else HumanCheckSubmission.model_validate(data))
    except ValidationError as exc:
        raise CheckError("; ".join(e["msg"] for e in exc.errors())) from exc
    ctx, rubric = index.ctx, index.ctx.rubric
    if index.selection is None:
        raise CheckError("judge evaluation has not finished for this run; resume the evaluation first")
    packet = index.packet(submission.packet_id)
    judges = index.chosen_judges(packet.record_id)
    extraction = index.chosen_extraction(packet.record_id)

    for flag in submission.flags:
        if flag.startswith("judge:") and flag[6:] not in ctx.judges:
            raise CheckError(f"no judge {flag[6:]!r} in this run")
        if flag == "extraction" and extraction is None:
            raise CheckError("this response has no recommendation extraction to flag")
    if not submission.blocking and not judges:
        raise CheckError("no judge produced a valid evaluation of this response, so there is nothing to approve; "
                         "flag it for a full review")

    candidates = candidate_failures(judges)
    decided: dict[str, bool] = {}
    for d in submission.critical_failures:
        resolved = rubric.critical_failure(d.failure)
        if resolved not in candidates or resolved in decided:
            raise CheckError(f"not a critical-failure candidate raised by the judges, or decided twice: {d.failure!r}")
        decided[resolved] = d.confirmed
    if not submission.blocking and set(decided) != set(candidates):
        raise CheckError("confirm or reject every critical-failure candidate the judges raised before approving")

    seen = index.seen(packet.record_id)
    prior = [c for c in index.checks.get(packet.record_id, []) if c.payload["reviewer_id"] == submission.reviewer_id]
    body = submission.model_dump(mode="json")
    for p in prior:
        if p.payload["submission"] == body and {k: p.payload[k] for k in seen} == seen:
            return p, False

    experiment_id = ctx.experiment_id
    check_id = new_id("hck")
    sink = store.sink(f"{experiment_id}/checks/{check_id}")
    ref = sink.write("submission.json", canonical_json(body))
    inputs = {packet.record_id: packet.payload["packet_sha256"], **{r.record_id: r.payload_sha256 for r in judges.values()}}
    if extraction is not None:
        inputs[extraction.record_id] = extraction.payload_sha256
    record = Record.create(
        kind=RecordKind.HUMAN_CHECK, record_id=check_id, experiment_id=experiment_id, subject_id=packet.subject_id,
        inputs=inputs, policy_revision=f"{ctx.run.policy_revision}/human-check",
        code_revision=EVALUATION_CODE_REVISION, supersedes=prior[0].record_id if prior else None,
        payload={
            "evaluation_run_id": ctx.run.record_id,
            "packet_id": packet.record_id,
            "round": int(packet.payload["round"]),
            "reviewer_id": submission.reviewer_id,
            "submission": body,
            "submission_sha256": ref.sha256,
            "flags": list(submission.flags),
            "blocking": submission.blocking,
            "confirmed_critical_failures": sorted(f for f, c in decided.items() if c),
            **seen,
        },
    )
    with db.transaction():
        db.insert_record(record)
        db.register_artifacts(check_id, sink.refs)
    index.checks[packet.record_id] = [c for c in index.checks.get(packet.record_id, []) if c not in prior] + [record]
    return record, True


# ----- what the reviewer sees ------------------------------------------------------------------------

_QUOTES = "\"'“”‘’«»"
_SEP = r"[\s*`|]+"
_BLOCK_PREFIX = re.compile(r"[ \t]*(?:#{1,6}[ \t]+|[-*+][ \t]+|\d+[.)][ \t]+|>[ \t]*|\|[ \t]*)*")
OPEN, MID, CLOSE = "", "", ""


def locate(quote: str, answer: str) -> tuple[int, int] | None:
    """Where a cited quote sits in the answer, ignoring whitespace, case, and emphasis/code/table markup."""
    tokens = [t for t in re.split(_SEP, quote.strip().strip(_QUOTES + " ")) if t]
    if not tokens or sum(len(t) for t in tokens) < 3:
        return None
    match = re.search(_SEP.join(re.escape(t) for t in tokens), answer, re.IGNORECASE)
    if match is None:
        return None
    start, end = match.span()
    line_start = answer.rfind("\n", 0, start) + 1
    if not answer[line_start:start].strip():  # keep markdown block syntax outside the highlight
        start = max(start, _BLOCK_PREFIX.match(answer, line_start).end())  # type: ignore[union-attr]
    return (start, end) if start < end else None


def mark_answer(answer: str, spans: list[tuple[int, int, int]]) -> str:
    """Wrap cited spans in private-use sentinels (OPEN ids MID text CLOSE) so the page can render markdown and
    still highlight evidence. A highlight never crosses a line break or covers a line's markdown block syntax
    (heading, list, quote, or table markers). `answer` must already be free of sentinels."""
    cuts = {0, len(answer)} | {s for s, _, _ in spans} | {e for _, e, _ in spans}
    syntax: list[tuple[int, int]] = []
    for line_start in [0, *(i + 1 for i, ch in enumerate(answer) if ch == "\n")]:
        end = _BLOCK_PREFIX.match(answer, line_start).end()  # type: ignore[union-attr]
        syntax.append((line_start, end))
        cuts |= {line_start, end}
    cuts |= {i for i, ch in enumerate(answer) if ch == "\n"}
    out = []
    for a, b in pairwise(sorted(cuts)):
        seg = answer[a:b]
        ids = [str(ev) for s, e, ev in spans if s <= a and b <= e]
        if ids and seg.strip() and "\n" not in seg and not any(s <= a and b <= e for s, e in syntax):
            out.append(f"{OPEN}{','.join(ids)}{MID}{seg}{CLOSE}")
        else:
            out.append(seg)
    return "".join(out)


class _Evidence:
    def __init__(self, answer: str) -> None:
        self.answer = re.sub(f"[{OPEN}{MID}{CLOSE}]", "", answer)
        self.spans: list[tuple[int, int, int]] = []
        self.labels: dict[int, str] = {}

    def cite(self, quote: str, label: str) -> dict[str, Any]:
        found = locate(quote, self.answer)
        if found is None:
            return {"quote": quote, "ev": None}
        ev = len(self.labels)
        self.spans.append((*found, ev))
        self.labels[ev] = label
        return {"quote": quote, "ev": ev}


def _attempt_failure(attempts: list[Record]) -> str | None:
    """Why there is no valid output. Only the failure kind and parse problems: raw errors can name models."""
    if not attempts:
        return "not evaluated yet"
    if selected_attempt(attempts) is not None:
        return None
    last = sorted(attempts, key=lambda r: r.payload["attempt_no"])[-1].payload
    problems = "; ".join(last.get("problems") or [])
    kind = last.get("failure_kind") or last.get("status")
    return f"{len(attempts)} attempt(s) failed ({kind}){': ' + problems if problems else ''}"[:400]


def _attention(index: RunIndex, packet: Record, judges: list[dict[str, Any]], extraction: dict[str, Any] | None,
               validation: dict[str, Any]) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    reasons = index.reasons.get(packet.record_id, [])
    scores = {j["judge_id"]: j["raw_score"] for j in judges if j["available"]}
    if "score_disagreement" in reasons and scores:
        spread = max(scores.values()) - min(scores.values())
        notes.append({"level": "warn", "text": f"Judges' overall scores differ by {spread:.1f} points"})
    if "dimension_disagreement" in reasons:
        notes.append({"level": "warn", "text": "Judges are 2+ rating points apart on at least one dimension"})
    if "critical_failure_candidate" in reasons:
        notes.append({"level": "warn", "text": "A judge raised a critical-failure candidate: confirm or reject it"})
    missing = [j["judge_id"] for j in judges if not j["available"]]
    if missing:
        notes.append({"level": "warn", "text": f"No valid evaluation from {', '.join(missing)}"})
    unverified = sum(1 for j in judges for d in j["ratings"].values() for q in d["evidence"] if q["ev"] is None)
    unverified += sum(1 for j in judges for c in j["candidates"] for q in c["evidence"] if q["ev"] is None)
    if unverified:
        notes.append({"level": "info", "text": f"{unverified} quote(s) a judge cited could not be found in the answer"})
    if not validation["valid"]:
        notes.append({"level": "warn", "text": "Mechanical validation failed: "
                      + ", ".join(c["name"] for c in validation["failed"])})
    if extraction and extraction["available"] and any(extraction["catalog_issues"].values()):
        notes.append({"level": "info", "text": "Extraction disagrees with the catalog cross-check"})
    if extraction and not extraction["available"]:
        notes.append({"level": "warn", "text": "No valid recommendation extraction"})
    return notes


def item_view(index: RunIndex, store: ArtifactStore, db: Database, packet_id: str, reviewer_id: str) -> dict[str, Any]:
    ctx, rubric = index.ctx, index.ctx.rubric
    packet_record = index.packet(packet_id)
    packet = read_packet(db, store, packet_id)
    round_number = int(packet["round"])
    dims = rubric.applicable_dimensions(round_number)
    titles = {d.id: d.title for d in dims}
    ev = _Evidence(str(packet["candidate_answer"]))

    chosen = index.chosen_judges(packet_id)
    judges: list[dict[str, Any]] = []
    for judge_id in ctx.judges:
        record = chosen.get(judge_id)
        if record is None:
            judges.append({"judge_id": judge_id, "available": False, "raw_score": None, "ratings": {},
                           "candidates": [],
                           "failure": _attempt_failure(index.judges[packet_id].get(judge_id, []))})
            continue
        output = record.payload["output"]
        judges.append({
            "judge_id": judge_id, "available": True, "failure": None,
            "raw_score": judge_raw_scores(rubric, round_number, {judge_id: output})[judge_id],
            "ratings": {
                dim: {"rating": r["rating"], "rationale": r.get("rationale", ""),
                      "evidence": [ev.cite(q, f"{judge_id} · {titles.get(dim, dim)} ({r['rating']})")
                                   for q in r["evidence"]]}
                for dim, r in output["ratings"].items()
            },
            "candidates": [
                {"failure": c["failure"], "rationale": c.get("rationale", ""),
                 "evidence": [ev.cite(q, f"{judge_id} · critical-failure candidate") for q in c["evidence"]]}
                for c in output["critical_failure_candidates"]
            ],
        })

    extraction: dict[str, Any] | None = None
    if round_number in ctx.extractor_rounds:
        chosen_ext = index.chosen_extraction(packet_id)
        if chosen_ext is None:
            extraction = {"available": False, "failure": _attempt_failure(index.extractions.get(packet_id, [])),
                          "items": [], "flags": {}, "catalog_issues": {}}
        else:
            out = chosen_ext.payload["output"]
            catalog = chosen_ext.payload.get("catalog_check", {})
            extraction = {
                "available": True, "failure": None,
                "items": [{**{k: v for k, v in item.items() if k != "evidence"},
                           "evidence": ev.cite(item["evidence"], f"extraction · {item.get('name_as_written', '')}")}
                          for item in sorted(out["items"], key=lambda i: (i.get("position") is None, i.get("position") or 0))],
                "flags": {name: {"value": f["value"],
                                 "evidence": [ev.cite(q, f"extraction · {name}") for q in f.get("evidence", [])]}
                          for name, f in out.get("flags", {}).items()},
                "catalog_issues": {k: catalog.get(k, []) for k in ("unlisted", "not_extracted", "name_conflicts")},
            }

    validation_record = index.validations.get(str(packet_record.subject_id))
    vp: dict[str, Any] = validation_record.payload if validation_record else {}
    validation = {"valid": bool(vp.get("mechanically_valid", True)),
                  "failed": [c for c in vp.get("checks", []) if not c["passed"]]}

    mine = index.current_checks(packet_id, reviewer_id)
    earlier = [c for c in index.checks.get(packet_id, []) if c.payload["reviewer_id"] == reviewer_id]
    latest = mine or earlier
    decision = latest[-1] if latest else None
    others = [c for c in index.current_checks(packet_id) if c.payload["reviewer_id"] != reviewer_id]
    return {
        "packet_id": packet_id,
        "round": round_number,
        "task": packet["task_conversation"],
        "answer": mark_answer(ev.answer, ev.spans),
        "evidence_labels": ev.labels,
        "rubric": {
            "dimensions": [{"id": d.id, "title": d.title, "guidance": d.guidance, "weight": d.weight} for d in dims],
            "rating_scale": {str(k): v for k, v in sorted(rubric.rating_scale.items())},
        },
        "judges": judges,
        "candidate_failures": candidate_failures(chosen),
        "extraction": extraction,
        "validation": validation,
        "attention": _attention(index, packet_record, judges, extraction, validation),
        "state": index.state(packet_id, reviewer_id),
        "decision": {
            "submission": decision.payload["submission"], "created_at": decision.created_at,
            "stale": not mine,
        } if decision else None,
        "other_reviewers": [{"reviewer_id": c.payload["reviewer_id"], "flags": c.payload["flags"]} for c in others],
    }


def queue_view(index: RunIndex, reviewer_id: str) -> dict[str, Any]:
    items = []
    for n, p in enumerate(index.packets, start=1):
        reasons = index.reasons.get(p.record_id, [])
        items.append({
            "packet_id": p.record_id, "number": n, "round": int(p.payload["round"]),
            "state": index.state(p.record_id, reviewer_id),
            "attention": any(r != "stratified_sample" for r in reasons),
        })
    counts = {s: sum(1 for i in items if i["state"] == s) for s in ("approved", "flagged", "unchecked", "stale")}
    return {
        "evaluation_run_id": index.ctx.run.record_id,
        "policy": str(index.ctx.run.policy_revision),
        "reviewer_id": reviewer_id,
        "judging_settled": index.selection is not None,
        "items": items,
        "counts": counts,
        "total": len(items),
        "done": counts["approved"] + counts["flagged"],
    }

