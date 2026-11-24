"""Live v2 human-review inbox over existing immutable plans, answers and evaluations."""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from importlib import resources
from typing import Any
from urllib.parse import urlencode

from ..domain.common import content_hash, sha256_text
from ..domain.execution import CandidateResponse
from ..domain.plan import ExperimentPlan
from ..domain.records import Record, RecordKind
from ..planning.compiler import load_plan
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from .checks import CheckError
from .invocations import next_attempt
from .selection import scenario_selection
from .selection_review import (
    ReviewSubmission, SelectionReviewApp, current_reviews, evidence_hash, review_state,
)
from .selection_runner import STAGES, _records
from .selection_completion import publish_selection_report, selection_status


@dataclass(frozen=True)
class InboxEntry:
    response: CandidateResponse
    plan: ExperimentPlan
    versions: tuple[Record, ...]

    @property
    def packet_id(self) -> str:
        # Stable across re-evaluation, report publication and queue reordering.
        return "inbox-" + content_hash(self.response.response_id)[:24]

    @property
    def active(self) -> Record | None:
        return self.versions[-1] if self.versions else None


class SelectionInboxApp:
    def __init__(self, db: Database, store: ArtifactStore, reviewer_id: str) -> None:
        if not reviewer_id.strip():
            raise CheckError("a reviewer identifier is required")
        self.db, self.store, self.reviewer_id = db, store, reviewer_id.strip()
        self.token = secrets.token_urlsafe(24)
        page = resources.files(__package__).joinpath("selection_review.html").read_text(encoding="utf-8")
        self.page = page.replace("__REVIEW_TOKEN__", self.token).encode("utf-8")

    def entries(self) -> list[InboxEntry]:
        plans: dict[str, ExperimentPlan] = {}
        for record in self.db.list_records(kind=RecordKind.EXPERIMENT_PLAN):
            scenario_id = record.payload.get("scenario", {}).get("scenario_id", "")
            try:
                scenario_selection(scenario_id)
            except ValueError:
                continue
            plan = load_plan(self.db, str(record.experiment_id))
            plans[plan.experiment_id] = plan
        versions: dict[str, list[Record]] = {}
        for run in self.db.list_records(kind=RecordKind.SELECTION_EVALUATION_RUN):
            if run.experiment_id in plans:
                for response in run.payload["responses"]:
                    versions.setdefault(response["response_id"], []).append(run)
        return [InboxEntry(r, plan, tuple(versions.get(r.response_id, [])))
                for plan in plans.values() for r in self.db.list_responses(plan.experiment_id) if r.round == 1]

    def entry(self, packet_id: str) -> InboxEntry:
        for entry in self.entries():
            if entry.packet_id == packet_id:
                return entry
        raise CheckError("unknown v2 response packet")

    def _run_view(self, run: Record) -> dict[str, Any]:
        try:
            report = publish_selection_report(self.db, self.store, run.record_id)
            status = selection_status(self.db, run.record_id)
            return {"rows": {r["response_id"]: r for r in report.payload["responses"]}, "status": status,
                    "error": None}
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            return {"rows": {}, "status": None, "error": str(exc)}

    def _summary(self, entry: InboxEntry, view: dict[str, Any] | None) -> dict[str, Any]:
        run, response = entry.active, entry.response
        target, count = scenario_selection(entry.plan.scenario.scenario_id)
        state = review_state(self.db, run, response.response_id) if run else "unchecked"
        if state == "unchecked" and any(current_reviews(self.db, old, response.response_id)[0] or
                                         current_reviews(self.db, old, response.response_id)[1]
                                         for old in entry.versions[:-1]):
            state = "stale"
        row = view["rows"].get(response.response_id, {}) if view else {}
        status = view["status"] if view else None
        if view and view["error"]:
            automatic = "blocked"
            blocked_reason = view["error"]
        elif status and run:
            required = ["extractor", "classifier"] + (["auditor"] if row.get("audit_reasons") else [])
            missing = [role for role in required if not row.get("selected_records", {}).get(role)]
            failed = [role for role in missing if next_attempt(_records(self.db, run, role, response.response_id),
                                                               run.payload["policy"]["max_attempts"]) is None]
            if failed:
                automatic = "failed"
                blocked_reason = "Required evaluation failed after its allowed attempts: " + ", ".join(failed)
            elif missing:
                automatic, blocked_reason = "pending", "Evaluation is pending: " + ", ".join(missing)
            else:
                automatic, blocked_reason = "ready", None
        else:
            automatic = "pending"
            blocked_reason = "Evaluation is pending." if run else "No evaluation has been started for this answer."
        dates = [response.created_at]
        if run:
            dates.append(run.created_at)
            dates.extend(r.created_at for role in STAGES for r in _records(self.db, run, role, response.response_id))
            current, _ = current_reviews(self.db, run, response.response_id)
            dates.extend(r.created_at for r in current if r.payload["decision"] == "flag")
        policy = run.payload["policy"] if run else None
        flags = row.get("flags", [])
        resolution = row.get("audit_resolution", {"status": "missing", "notes": [], "human_review_ids": []})
        resolved_unavailable = resolution["status"] == "unavailable"
        return {"packet_id": entry.packet_id, "label": "Response " + entry.packet_id[6:14],
                "state": state, "automatic": automatic, "blocked": automatic != "ready" and not resolved_unavailable,
                "audit_resolution": resolution,
                "blocked_reason": blocked_reason, "attention": bool(flags) or automatic != "ready" or state == "flagged",
                "flags": flags, "scenario_id": entry.plan.scenario.scenario_id,
                "question": entry.plan.scenario.scenario_id.removeprefix("agentic-design-bias-v2-"),
                "target": target, "requested_count": count, "repetition": response.repetition,
                "policy": f"{policy['policy_id']}@{policy['revision']}" if policy else "Awaiting evaluation",
                "evaluation_run_id": run.record_id if run else None, "version_count": len(entry.versions),
                "updated_at": max(dates), "created_at": response.created_at,
                "seen": evidence_hash(self.db, run, response.response_id) if run else response.final_answer_sha256}

    def queue(self) -> dict[str, Any]:
        entries = self.entries()
        runs = {e.active.record_id: e.active for e in entries if e.active}
        views = {run_id: self._run_view(run) for run_id, run in runs.items()}
        items = [self._summary(e, views[e.active.record_id] if e.active else None) for e in entries]
        # Stable tie break, then newest evidence first, then unresolved before resolved.
        items.sort(key=lambda i: i["packet_id"])
        items.sort(key=lambda i: i["updated_at"], reverse=True)
        items.sort(key=lambda i: i["state"] == "approved" and not i["blocked"])
        counts = {state: sum(i["state"] == state for i in items)
                  for state in ("approved", "flagged", "stale", "unchecked")}
        counts.update(blocked=sum(i["blocked"] for i in items),
                      audit_unavailable=sum(i["audit_resolution"]["status"] == "unavailable" for i in items),
                      needs_review=sum(i["state"] != "approved" or i["blocked"] for i in items))
        return {"scope": "v2", "reviewer_id": self.reviewer_id, "total": len(items),
                "done": counts["approved"], "counts": counts, "items": items}

    def _history(self, entry: InboxEntry) -> list[dict[str, Any]]:
        history = []
        for run in reversed(entry.versions):
            current, stale = current_reviews(self.db, run, entry.response.response_id)
            policy = run.payload["policy"]
            history.append({"evaluation_run_id": run.record_id, "created_at": run.created_at,
                            "policy": f"{policy['policy_id']}@{policy['revision']}",
                            "active": run == entry.active,
                            "reviews": [{"reviewer_id": r.payload["reviewer_id"], "decision": r.payload["decision"],
                                         "note": r.payload["note"], "created_at": r.created_at,
                                         "stale": r in stale} for r in current + stale],
                            "attempts": self._attempts(run, entry.response.response_id)})
        return history

    def _attempts(self, run: Record, response_id: str) -> list[dict[str, Any]]:
        attempts = []
        for role in STAGES:
            for record in _records(self.db, run, role, response_id):
                ref = self.db.artifacts_for(record.record_id).get("model-output.txt")
                attempts.append({"stage": role, "attempt_no": record.payload["attempt_no"],
                                 "status": record.payload["status"], "created_at": record.created_at,
                                 "problems": record.payload.get("problems", []),
                                 "error": record.payload.get("error"),
                                 "raw_output": self.store.read_verified(ref).decode("utf-8") if ref else None})
        return attempts

    def item(self, packet_id: str) -> dict[str, Any]:
        entry = self.entry(packet_id)
        run = entry.active
        view = self._run_view(run) if run else None
        summary = self._summary(entry, view)
        if run and view and not view["error"]:
            app = SelectionReviewApp(self.db, self.store, run.record_id, self.reviewer_id)
            item = app.item(app.packet_id(entry.response.response_id))
        else:
            refs = self.db.artifacts_for(entry.response.attempt_id)
            answer = self.store.read_verified(refs["final-answer.txt"]).decode("utf-8")
            if sha256_text(answer) != entry.response.final_answer_sha256:
                raise CheckError("candidate answer does not match its recorded hash")
            messages = self.store.read_json(refs["canonical-messages.json"])
            item = {"answer": answer, "task": messages, "extraction": None, "labels": None, "audit": None,
                    "audit_reasons": [], "reviews": [], "can_proceed": False}
        return {**item, **summary, "can_proceed": item["can_proceed"], "can_flag": run is not None and not (view and view["error"]),
                "history": self._history(entry),
                "report_url": "/report?" + urlencode({"packet": packet_id, "token": self.token}) if run else None}

    def recovery(self, packet_id: str) -> dict[str, Any]:
        entry = self.entry(packet_id)
        if not entry.active:
            return {"message": "Start automatic evaluation for this answer before proceeding.", "command": None}
        summary = self._summary(entry, self._run_view(entry.active))
        if summary["audit_resolution"]["status"] == "unavailable":
            return {"message": "The quality audit was explicitly resolved as unavailable by a human. "
                    "Its failures and missing scores remain recorded; review completion is no longer blocked by this audit.",
                    "command": None}
        if summary["automatic"] == "pending":
            return {"message": "Resume the existing evaluation. Completed calls are retained. Missing calls may incur API charges.",
                    "command": f"bench evaluate-v2 {entry.plan.experiment_id} --resume {entry.active.record_id}"}
        if summary["automatic"] == "ready":
            return {"message": "Automatic evaluation is complete. No recovery is needed.", "command": None}
        return {"message": str(summary["blocked_reason"]) +
                ". Inspect the attempt history. Exhausted retries require an explicitly chosen new evaluation run or policy; "
                "resuming this run will not reset its retry allowance.", "command": None}

    def submit(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            submission = ReviewSubmission.model_validate(body)
        except ValueError as exc:
            raise CheckError(str(exc)) from None
        entry = self.entry(submission.packet_id)
        if entry.active is None:
            raise CheckError("start evaluation before reviewing this answer")
        app = SelectionReviewApp(self.db, self.store, entry.active.record_id, self.reviewer_id)

        def validate_active() -> None:
            # Called inside the experiment lock and DB write transaction, before saving the decision.
            if self.entry(submission.packet_id).active != entry.active:
                raise CheckError("the active evaluation changed; reload and review this answer again")

        result = app.submit({**submission.model_dump(mode="json"),
                             "packet_id": app.packet_id(entry.response.response_id)}, validate=validate_active)
        return {**result, "queue": self.queue()}

    def report(self, packet_id: str) -> bytes:
        entry = self.entry(packet_id)
        if entry.active is None:
            raise CheckError("no evaluation report is available yet")
        from ..reporting.selection_report import publish
        try:
            record = publish(self.db, self.store, entry.plan.experiment_id, entry.active.record_id)
            return self.store.read_verified(self.db.artifacts_for(record.record_id)["report.html"])
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            raise CheckError(f"report unavailable: {exc}") from None
