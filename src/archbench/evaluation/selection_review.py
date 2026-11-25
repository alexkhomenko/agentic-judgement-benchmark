"""Explicit human validation of every v2 answer, against exact automated evidence."""
from __future__ import annotations

import secrets
from collections.abc import Callable
from importlib import resources
from typing import Any, Literal

from pydantic import Field, ValidationError

from ..domain.common import Frozen, content_hash
from ..domain.records import Record, RecordKind
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from .checks import CheckError
from .invocations import next_attempt
from .selection_runner import STAGES, _load, _records, publish_selection_report

HUMAN_REVIEW_POLICY = "selection-human-review@2"


class ReviewSubmission(Frozen):
    packet_id: str
    seen: str
    decision: Literal["proceed", "proceed_without_audit", "flag"]
    note: str = Field(default="", max_length=20000)


def evidence_hash(db: Database, run: Record, response_id: str) -> str:
    entry = next(e for e in run.payload["responses"] if e["response_id"] == response_id)
    attempts = {r.record_id: r.payload_sha256 for role in STAGES for r in _records(db, run, role, response_id)}
    return content_hash({"run": run.payload_sha256, "answer": entry["final_answer_sha256"], "attempts": attempts})


def current_reviews(db: Database, run: Record, response_id: str) -> tuple[list[Record], list[Record]]:
    latest: dict[str, Record] = {}
    for record in db.list_records(kind=RecordKind.SELECTION_HUMAN_REVIEW,
                                  experiment_id=run.experiment_id, subject_id=response_id):
        if record.payload.get("evaluation_run_id") == run.record_id:
            latest[record.payload["reviewer_id"]] = record
    seen = evidence_hash(db, run, response_id)
    return ([r for r in latest.values() if r.payload["seen"] == seen],
            [r for r in latest.values() if r.payload["seen"] != seen])


def review_state(db: Database, run: Record, response_id: str) -> str:
    current, stale = current_reviews(db, run, response_id)
    if any(r.payload["decision"] == "flag" for r in current):
        return "flagged"
    if current:
        return "approved"
    return "stale" if stale else "unchecked"


def review_summary(db: Database, run: Record) -> dict[str, Any]:
    states = [review_state(db, run, e["response_id"]) for e in run.payload["responses"]]
    approved, flagged, stale = (states.count(s) for s in ("approved", "flagged", "stale"))
    unavailable = sum(any(r.payload["decision"] == "proceed_without_audit" and r.policy_revision == HUMAN_REVIEW_POLICY
                          for r in current_reviews(db, run, e["response_id"])[0]) for e in run.payload["responses"])
    return {"required": len(states), "approved": approved, "flagged": flagged, "stale": stale,
            "audit_unavailable": unavailable,
            "remaining": len(states)-approved, "complete": bool(states) and approved == len(states)}


def audit_approval(db: Database, run: Record, row: dict[str, Any]) -> dict[str, Any]:
    """A human may resolve an exhausted quality audit only when the core labels are valid."""
    selected = row["selected_records"]
    core_valid = bool(selected["extractor"] and selected["classifier"])
    required = bool(row["audit_reasons"])
    attempts = _records(db, run, "auditor", row["response_id"])
    eligible = bool(core_valid and required and not selected["auditor"] and attempts and
                    next_attempt(attempts, run.payload["policy"]["max_attempts"]) is None)
    current, _ = current_reviews(db, run, row["response_id"])
    resolutions = [r for r in current if eligible and r.policy_revision == HUMAN_REVIEW_POLICY and
                   r.payload["decision"] == "proceed_without_audit" and r.payload["note"].strip()]
    status = ("not_required" if not required else "available" if selected["auditor"] else
              "unavailable" if resolutions else "missing")
    return {"can_proceed": core_valid and (not required or bool(selected["auditor"])),
            "can_proceed_without_audit": eligible,
            "audit_resolution": {"status": status, "human_review_ids": [r.record_id for r in resolutions],
                                 "notes": [{"reviewer_id": r.payload["reviewer_id"], "note": r.payload["note"]}
                                           for r in resolutions]},
            "failed_audit_record_ids": [r.record_id for r in attempts] if eligible else []}


class SelectionReviewApp:
    def __init__(self, db: Database, store: ArtifactStore, run_id: str, reviewer_id: str) -> None:
        if not reviewer_id.strip():
            raise CheckError("a reviewer identifier is required")
        self.db, self.store, self.reviewer_id = db, store, reviewer_id.strip()
        self.run, _, _ = _load(db, run_id)
        self.token = secrets.token_urlsafe(24)
        page = resources.files(__package__).joinpath("selection_review.html").read_text(encoding="utf-8")
        self.page = page.replace("__REVIEW_TOKEN__", self.token).encode("utf-8")

    def packet_id(self, response_id: str) -> str:
        return "packet-" + content_hash([self.run.record_id, response_id])[:24]

    def entry(self, packet_id: str) -> dict[str, Any]:
        for entry in self.run.payload["responses"]:
            if self.packet_id(entry["response_id"]) == packet_id:
                return dict(entry)
        raise CheckError("unknown response packet")

    def queue(self) -> dict[str, Any]:
        report = publish_selection_report(self.db, self.store, self.run.record_id)
        flags = {r["response_id"]: r["flags"] for r in report.payload["responses"]}
        entries = sorted(self.run.payload["responses"], key=lambda e: self.packet_id(e["response_id"]))
        items = [{"packet_id": self.packet_id(e["response_id"]), "number": n,
                  "state": review_state(self.db, self.run, e["response_id"]),
                  "attention": bool(flags[e["response_id"]])} for n, e in enumerate(entries, 1)]
        summary = review_summary(self.db, self.run)
        return {"evaluation_run_id": self.run.record_id, "reviewer_id": self.reviewer_id,
                "total": len(items), "done": summary["approved"], "counts": summary, "items": items}

    def item(self, packet_id: str) -> dict[str, Any]:
        entry = self.entry(packet_id)
        report = publish_selection_report(self.db, self.store, self.run.record_id)
        row = next(r for r in report.payload["responses"] if r["response_id"] == entry["response_id"])
        current, stale = current_reviews(self.db, self.run, entry["response_id"])
        approval = audit_approval(self.db, self.run, row)
        # Deliberately omit configuration, provider, track and ownership comparisons from the review UI.
        return {"packet_id": packet_id, "seen": evidence_hash(self.db, self.run, entry["response_id"]),
                "state": review_state(self.db, self.run, entry["response_id"]),
                "task": entry["packet"]["task_conversation"], "answer": entry["packet"]["candidate_answer"],
                "target": self.run.payload["target"], "requested_count": self.run.payload["requested_count"],
                "flags": [*row["flags"], *(["quality_audit_unavailable"]
                          if approval["audit_resolution"]["status"] == "unavailable" else [])],
                "automated_flags": row["flags"],
                "audit_reasons": row["audit_reasons"],
                "extraction": row["extraction"], "labels": row["labels"], "audit": row["audit"],
                **approval,
                "reviews": [{"reviewer_id": r.payload["reviewer_id"], "decision": r.payload["decision"],
                             "note": r.payload["note"], "stale": r in stale} for r in current + stale]}

    def submit(self, body: dict[str, Any], *, validate: Callable[[], None] | None = None) -> dict[str, Any]:
        try:
            submission = ReviewSubmission.model_validate(body)
        except ValidationError as exc:
            raise CheckError(str(exc)) from None
        with self.store.lock(str(self.run.experiment_id)), self.db.transaction():
            if validate:
                validate()
            item = self.item(submission.packet_id)
            if submission.seen != item["seen"]:
                raise CheckError("evaluation evidence changed; reload and review this answer again")
            if submission.decision == "proceed" and not item["can_proceed"]:
                raise CheckError("finish the required automated evaluations before proceeding")
            if submission.decision == "proceed_without_audit":
                if not item["can_proceed_without_audit"]:
                    raise CheckError("valid preference/compliance labels and an exhausted quality audit are required")
                if not submission.note.strip():
                    raise CheckError("add a note explaining your resolution of the unavailable quality audit")
            if submission.decision == "flag" and not submission.note.strip():
                raise CheckError("add a note explaining what needs attention")
            entry = self.entry(submission.packet_id)
            current, stale = current_reviews(self.db, self.run, entry["response_id"])
            prior = next((r for r in current + stale if r.payload["reviewer_id"] == self.reviewer_id), None)
            payload = {**submission.model_dump(mode="json"), "note": submission.note.strip(),
                       "evaluation_run_id": self.run.record_id, "reviewer_id": self.reviewer_id,
                       "automated_flags_seen": item["automated_flags"]}
            if submission.decision == "proceed_without_audit":
                payload["audit_resolution"] = {"status": "unavailable",
                    "failed_audit_record_ids": item["failed_audit_record_ids"]}
            if prior and prior.payload == payload:
                record, created = prior, False
            else:
                record = Record.create(kind=RecordKind.SELECTION_HUMAN_REVIEW, prefix="shrv",
                    experiment_id=self.run.experiment_id, subject_id=entry["response_id"],
                    policy_revision=HUMAN_REVIEW_POLICY,
                    supersedes=prior.record_id if prior else None,
                    inputs={self.run.record_id: self.run.payload_sha256,
                            entry["response_id"]: entry["final_answer_sha256"], "review-evidence": item["seen"]},
                    payload=payload)
                sink = self.store.sink(f"{self.run.experiment_id}/evaluations/{self.run.record_id}/human/{record.record_id}")
                sink.write_json("human-review.json", record.model_dump(mode="json"))
                with self.db.transaction():
                    self.db.insert_record(record)
                    self.db.register_artifacts(record.record_id, sink.refs)
                created = True
            return {"check_id": record.record_id, "created": created, "queue": self.queue()}
