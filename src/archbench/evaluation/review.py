"""Full human review: attention signals, blinded export, and validated import.

Every response is human-checked (see `checks.py`). A full review is the escalation for a check that flags a
judge or the answer: the reviewer re-scores the response from the blinded packet.

The review selection marks responses that deserve close attention during the check: critical-failure
candidates, judge disagreements, responses lacking a valid evaluation from every judge, high/low examples per
round, and a reproducible stratified sample. It no longer decides which responses get human attention.

Exports contain only the blinded packet and an empty structured template. They never reveal judge
scores, selection reasons, or response identity.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..domain.common import new_id
from ..domain.evaluation import EvaluationPolicy, HumanReviewSubmission, Rubric
from ..domain.records import Record, RecordKind
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from .records import EVALUATION_CODE_REVISION, effective, read_packet, run_records, selected_attempt
from .scoring import compute_score


class ReviewImportError(ValueError):
    pass


def selection_record_id(run_id: str) -> str:
    return f"sel-{run_id}"


def judge_raw_scores(rubric: Rubric, round_number: int, assessments: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        judge_id: compute_score(
            rubric, round_number, {k: v["rating"] for k, v in a["ratings"].items()}, critical_failure_confirmed=False
        ).raw_score
        for judge_id, a in sorted(assessments.items())
    }


def stratified_sample(strata: dict[str, list[str]], *, fraction: float, seed: int) -> list[str]:
    chosen: list[str] = []
    for name in sorted(strata):
        members = sorted(strata[name])
        k = math.ceil(len(members) * fraction)
        chosen += random.Random(f"{seed}:{name}").sample(members, k)
    return sorted(chosen)


def build_selection(
    *,
    policy: EvaluationPolicy,
    rubric: Rubric,
    responses: list[dict[str, Any]],  # response_id, packet_id, round, track
    judge_records: dict[str, list[Record]],  # response_id -> judge evaluation records
) -> dict[str, Any]:
    judge_ids = [j.judge_id for j in policy.judges]
    disagreement = policy.review_selection.disagreement
    reasons: dict[str, list[str]] = defaultdict(list)
    scores: dict[str, dict[str, float]] = {}
    for item in responses:
        rid = item["response_id"]
        by_judge: dict[str, dict[str, Any]] = {}
        for judge_id in judge_ids:
            chosen = selected_attempt([r for r in judge_records.get(rid, []) if r.payload["judge_id"] == judge_id])
            if chosen is not None:
                by_judge[judge_id] = chosen.payload["output"]
        if len(by_judge) < len(judge_ids):
            reasons[rid].append("insufficient_judge_evaluations")
        if any(a["critical_failure_candidates"] for a in by_judge.values()):
            reasons[rid].append("critical_failure_candidate")
        if not by_judge:
            continue
        scores[rid] = judge_raw_scores(rubric, item["round"], by_judge)
        if len(by_judge) > 1:
            values = list(scores[rid].values())
            if max(values) - min(values) > disagreement.score_distance:
                reasons[rid].append("score_disagreement")
            dims = next(iter(by_judge.values()))["ratings"].keys()
            if any(
                max(a["ratings"][d]["rating"] for a in by_judge.values())
                - min(a["ratings"][d]["rating"] for a in by_judge.values()) >= disagreement.dimension_distance
                for d in dims
            ):
                reasons[rid].append("dimension_disagreement")

    k = policy.review_selection.extremes_per_round
    by_round: dict[int, list[str]] = defaultdict(list)
    for item in responses:
        if item["response_id"] in scores and len(scores[item["response_id"]]) == len(judge_ids):
            by_round[item["round"]].append(item["response_id"])
    for ids in by_round.values():
        ranked = sorted(ids, key=lambda r: (sum(scores[r].values()) / len(scores[r]), r))
        for rid in ranked[-k:] if k else []:
            reasons[rid].append("high_example")
        for rid in ranked[:k] if k else []:
            reasons[rid].append("low_example")

    strata: dict[str, list[str]] = defaultdict(list)
    for item in responses:
        if item["response_id"] not in reasons:
            strata[f"{item['track']}/round-{item['round']}"].append(item["response_id"])
    sample = stratified_sample(
        strata, fraction=policy.review_selection.fraction, seed=policy.review_selection.seed
    )
    for rid in sample:
        reasons[rid].append("stratified_sample")

    packets = {i["response_id"]: i for i in responses}
    return {
        "items": [
            {"response_id": rid, "packet_id": packets[rid]["packet_id"], "round": packets[rid]["round"],
             "reasons": sorted(set(reasons[rid]))}
            for rid in sorted(reasons)
        ],
        "strata": {name: len(ids) for name, ids in sorted(strata.items())},
        "sampled": len(sample),
        "judge_raw_scores": {rid: s for rid, s in sorted(scores.items())},
    }


def publish_selection(
    db: Database, run: Record, selection: dict[str, Any], judge_inputs: list[Record]
) -> Record:
    record = Record.create(
        kind=RecordKind.REVIEW_SELECTION, record_id=selection_record_id(run.record_id),
        experiment_id=run.experiment_id, subject_id=None,
        inputs={run.record_id: run.payload_sha256, **{r.record_id: r.payload_sha256 for r in judge_inputs}},
        policy_revision=run.policy_revision, code_revision=EVALUATION_CODE_REVISION,
        payload={"evaluation_run_id": run.record_id, **selection},
    )
    db.insert_record(record)
    return record


def review_template(packet: dict[str, Any]) -> dict[str, Any]:
    rubric = packet["rubric"]
    return {
        "packet_id": packet["packet_id"],
        "reviewer_id": "",
        "ratings": {d["id"]: {"rating": None, "evidence": [], "rationale": ""} for d in rubric["dimensions"]},
        "critical_failures": [
            {"failure": cf, "confirmed": None, "evidence": [], "rationale": ""} for cf in rubric["critical_failures"]
        ],
        "notes": "",
    }


REVIEW_README = """# Full review of flagged responses

These are the responses a human check flagged (a judge or the answer looked wrong). Re-score each one.

Each `<packet>.packet.json` holds the task, the conversation, a candidate answer, the private rubric,
and reference properties. Provider, model, cost, timing, and judge assessments are deliberately absent.

For each packet, fill `<packet>.review.json`:
- `reviewer_id`: your reviewer identifier.
- `ratings`: an integer 0-4 per dimension, with quoted evidence from the answer and a rationale.
- `critical_failures`: decide every listed failure (`confirmed: true/false`); confirmed ones need evidence.

Import with `bench review-import <evaluation-run> <file>...`.
"""


def flagged_for_review(db: Database, run: Record) -> list[dict[str, Any]]:
    """Responses whose current check flags a judge or the answer and that have no full review yet."""
    from .checks import RunIndex
    from .runner import load_run

    index = RunIndex.build(db, load_run(db, run.record_id))
    return [
        {"response_id": str(p.subject_id), "packet_id": p.record_id, "round": int(p.payload["round"])}
        for p in index.packets
        if index.blocking(p.record_id) and str(p.subject_id) not in index.reviews
    ]


def export_reviews(db: Database, store: ArtifactStore, run: Record, out_dir: Path) -> list[Path]:
    selection = db.get_record(selection_record_id(run.record_id))
    if selection is None:
        raise RuntimeError("review selection does not exist yet; judge evaluation must finish first")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for item in flagged_for_review(db, run):
        packet = read_packet(db, store, item["packet_id"])
        for path, value in (
            (out_dir / f"{item['packet_id']}.packet.json", packet),
            (out_dir / f"{item['packet_id']}.review.json", review_template(packet)),
        ):
            with open(path, "x", encoding="utf-8") as fh:  # never overwrite a reviewer's file
                json.dump(value, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            written.append(path)
    readme = out_dir / "README.md"
    if not readme.exists():
        readme.write_text(REVIEW_README)
    return written


def import_review(
    db: Database, store: ArtifactStore, run: Record, rubric: Rubric, path: Path, *, supersede: bool = False
) -> Record:
    data = path.read_bytes()
    try:
        submission = HumanReviewSubmission.model_validate_json(data)
    except ValidationError as exc:
        raise ReviewImportError(f"{path}: {exc}") from exc
    selection = db.get_record(selection_record_id(run.record_id))
    if selection is None:
        raise ReviewImportError("review selection does not exist yet")
    packet_record = db.get_record(submission.packet_id)
    if (packet_record is None or packet_record.kind is not RecordKind.BLINDED_PACKET
            or packet_record.payload.get("evaluation_run_id") != run.record_id):
        raise ReviewImportError(f"packet {submission.packet_id} does not belong to evaluation run {run.record_id}")
    response_id, round_number = str(packet_record.subject_id), int(packet_record.payload["round"])

    expected = {d.id for d in rubric.applicable_dimensions(round_number)}
    if set(submission.ratings) != expected:
        raise ReviewImportError(f"ratings must cover exactly {sorted(expected)}")
    decided: dict[str, bool] = {}
    for decision in submission.critical_failures:
        resolved = rubric.critical_failure(decision.failure)
        if resolved is None or resolved in decided:
            raise ReviewImportError(f"unknown or duplicate critical failure: {decision.failure!r}")
        decided[resolved] = decision.confirmed
    if set(decided) != set(rubric.critical_failures):
        raise ReviewImportError("every rubric critical failure needs an explicit decision")

    experiment_id = str(run.experiment_id)
    prior = [
        r for r in effective(run_records(db, RecordKind.HUMAN_REVIEW, experiment_id, run.record_id))
        if r.payload["packet_id"] == submission.packet_id and r.payload["reviewer_id"] == submission.reviewer_id
    ]
    if prior and not supersede:
        raise ReviewImportError(
            f"{submission.reviewer_id} already reviewed {submission.packet_id} ({prior[0].record_id}); "
            "pass --supersede to record a correction"
        )
    confirmed = sorted(f for f, c in decided.items() if c)
    score = compute_score(
        rubric, round_number, {k: v.rating for k, v in submission.ratings.items()},
        critical_failure_confirmed=bool(confirmed),
    )
    review_id = new_id("hrv")
    sink = store.sink(f"{experiment_id}/reviews/{review_id}")
    ref = sink.write("submission.json", data)
    packet = db.artifacts_for(submission.packet_id)["packet.json"]
    record = Record.create(
        kind=RecordKind.HUMAN_REVIEW, record_id=review_id, experiment_id=experiment_id,
        subject_id=response_id,
        inputs={submission.packet_id: packet.sha256, selection.record_id: selection.payload_sha256},
        policy_revision=f"{run.policy_revision}/rubric@{rubric.revision}", code_revision=EVALUATION_CODE_REVISION,
        supersedes=prior[0].record_id if prior else None,
        payload={
            "evaluation_run_id": run.record_id,
            "packet_id": submission.packet_id,
            "reviewer_id": submission.reviewer_id,
            "submission_sha256": ref.sha256,
            "submission": submission.model_dump(mode="json"),
            "confirmed_critical_failures": confirmed,
            "score": score.model_dump(mode="json"),
        },
    )
    with db.transaction():
        db.insert_record(record)
        db.register_artifacts(review_id, sink.refs)
    return record
