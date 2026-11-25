"""A human can accept missing quality auditing without fabricating scores or bypassing core labels."""
from __future__ import annotations

import csv
import io
from typing import Any

import pytest
import yaml

from archbench.domain.records import Record, RecordKind
from archbench.evaluation.checks import CheckError
from archbench.evaluation.selection_completion import publish_selection_report, selection_status
from archbench.evaluation.selection_inbox import SelectionInboxApp
from archbench.evaluation.selection_review import HUMAN_REVIEW_POLICY, SelectionReviewApp
from archbench.evaluation.selection_runner import resume_selection_evaluation, start_selection_evaluation
from archbench.execution.limits import CallBudget, CheckpointStop
from archbench.reporting.selection_report import collect, render_html, render_markdown

from .conftest import Env
from .test_selection_evaluation import setup


def failed_audit(env: Env, *, failed_role: str = "auditor") -> tuple[SelectionReviewApp, Any, Any]:
    plan, path, registry, llm, jev = setup(env, repetitions=1, fraction=1)
    config_path = env.tmp/"configs"/f"{failed_role}.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["adapter_options"]["malformed_attempts"] = 9
    config_path.write_text(yaml.safe_dump(config))
    run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    resume_selection_evaluation(env.db, env.store, registry, run.record_id)
    return SelectionReviewApp(env.db, env.store, run.record_id, "human"), llm, jev


def body(app: Any, *, note: str = "Audit outputs failed validation; I reviewed the answer and its preference/compliance labels.") -> dict[str, Any]:
    packet = app.queue()["items"][0]["packet_id"]
    item = app.item(packet)
    return {"packet_id": packet, "seen": item["seen"], "decision": "proceed_without_audit", "note": note}


def test_explicit_note_resolution_preserves_failures_missing_scores_and_cost(env: Env) -> None:
    app, llm, jev = failed_audit(env)
    initial = body(app)
    item = app.item(initial["packet_id"])
    assert item["can_proceed_without_audit"] and not item["can_proceed"]
    before = env.db.list_records(kind=RecordKind.SELECTION_AUDIT, experiment_id=app.run.experiment_id)
    before_calls = len(llm.calls), len(jev.calls)
    with pytest.raises(CheckError, match="note"):
        app.submit({**initial, "note": " \n "})
    with pytest.raises(CheckError, match="automated"):
        app.submit({**initial, "decision": "proceed"})
    saved = app.submit(initial)
    assert saved["queue"]["done"] == 1 and not app.submit(initial)["created"]
    record = env.db.get_record(saved["check_id"])
    assert record and record.policy_revision == HUMAN_REVIEW_POLICY
    assert record.payload["decision"] == "proceed_without_audit" and record.payload["note"] == initial["note"]
    assert record.payload["audit_resolution"]["failed_audit_record_ids"] == [r.record_id for r in before]
    status = selection_status(env.db, app.run.record_id)
    assert status["complete"] and not status["automatic_complete"] and not status["all_required_outputs_valid"]
    assert status["all_required_work_resolved"] and status["audit_resolution"]["human_unavailable"] == 1
    assert status["stages"]["auditor"]["valid"] == 0 and status["stages"]["auditor"]["exhausted"] == 1
    report = publish_selection_report(env.db, env.store, app.run.record_id)
    row = report.payload["responses"][0]
    assert row["quality"] is None and row["audit"] is None and row["audit_resolution"]["status"] == "unavailable"
    assert {"audit_failed", "quality_audit_unavailable"}.issubset(row["flags"])
    assert saved["check_id"] in report.inputs and report.payload["complete"]
    refs = env.db.artifacts_for(report.record_id)
    exported = list(csv.DictReader(io.StringIO(env.store.read_verified(refs["selection-labels.csv"]).decode())))[0]
    assert exported["quality"] == "null" and exported["audit_resolution"] == "unavailable"
    assert "Audit outputs failed" in exported["audit_resolution_notes"]
    assert env.db.list_records(kind=RecordKind.SELECTION_AUDIT, experiment_id=app.run.experiment_id) == before
    data = collect(env.db, env.store, str(app.run.experiment_id), app.run.record_id)
    page = render_html(data, "test-report")
    assert "COMPLETE · AUDIT UNAVAILABLE" in page and "human resolved" in page
    assert "Quality score: <strong>Unavailable" in page and initial["note"] in page
    assert "proceed_without_audit" in render_markdown(data, "test-report")
    assert (len(llm.calls), len(jev.calls)) == before_calls
    assert publish_selection_report(env.db, env.store, app.run.record_id).record_id == report.record_id


@pytest.mark.parametrize("role", ["extractor", "classifier"])
def test_missing_preference_or_compliance_outputs_cannot_be_bypassed(env: Env, role: str) -> None:
    app, _, _ = failed_audit(env, failed_role=role)
    request = body(app)
    assert not app.item(request["packet_id"])["can_proceed_without_audit"]
    with pytest.raises(CheckError, match="preference/compliance"):
        app.submit(request)
    assert not selection_status(env.db, app.run.record_id)["complete"]


@pytest.mark.parametrize("state", ["not_required", "pending", "retry_remaining", "available"])
def test_audit_resolution_requires_an_exhausted_failure(env: Env, state: str) -> None:
    plan, path, registry, _, _ = setup(env, repetitions=1, fraction=0 if state == "not_required" else 1)
    if state == "retry_remaining":
        p = env.tmp/"configs/auditor.yaml"
        config = yaml.safe_load(p.read_text())
        config["adapter_options"]["malformed_attempts"] = 9
        p.write_text(yaml.safe_dump(config))
    run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    if state in ("pending", "retry_remaining"):
        budget = CallBudget(2 if state == "pending" else 3)
        with pytest.raises(CheckpointStop):
            resume_selection_evaluation(env.db, env.store, registry, run.record_id, before_call=budget.before_call)
    else:
        resume_selection_evaluation(env.db, env.store, registry, run.record_id)
    app = SelectionReviewApp(env.db, env.store, run.record_id, "human")
    request = body(app)
    assert not app.item(request["packet_id"])["can_proceed_without_audit"]
    with pytest.raises(CheckError, match="exhausted quality audit"):
        app.submit(request)


def test_resolution_becomes_stale_when_evaluation_evidence_changes(env: Env) -> None:
    app, _, _ = failed_audit(env)
    request = body(app)
    saved = app.submit(request)["check_id"]
    response_id = app.run.payload["responses"][0]["response_id"]
    env.db.insert_record(Record.create(kind=RecordKind.SELECTION_AUDIT,
        experiment_id=app.run.experiment_id, subject_id=response_id,
        payload={"evaluation_run_id": app.run.record_id, "attempt_no": 3, "status": "invalid_output",
                 "problems": ["new evidence"], "cost_incurred": "no"}))
    status = selection_status(env.db, app.run.record_id)
    assert not status["complete"] and status["human_review"]["stale"] == 1
    assert status["audit_resolution"]["human_unavailable"] == 0
    with pytest.raises(CheckError, match="changed"):
        app.submit(request)
    assert env.db.get_record(saved)


def test_inbox_resolves_workflow_block_without_marking_automatic_audit_valid(env: Env) -> None:
    app, _, _ = failed_audit(env)
    inbox = SelectionInboxApp(env.db, env.store, "human")
    request = body(inbox)
    result = inbox.submit(request)
    item = inbox.item(request["packet_id"])
    assert item["state"] == "approved" and item["automatic"] == "failed" and not item["blocked"]
    assert item["audit"] is None and item["audit_resolution"]["status"] == "unavailable"
    assert result["queue"]["counts"]["blocked"] == 0 and result["queue"]["counts"]["audit_unavailable"] == 1
    assert result["queue"]["counts"]["needs_review"] == 0
    assert "explicitly resolved" in inbox.recovery(request["packet_id"])["message"]
    assert b"AUDIT UNAVAILABLE" in inbox.report(request["packet_id"])


def test_another_reviewers_flag_still_blocks_completion(env: Env) -> None:
    app, _, _ = failed_audit(env)
    app.submit(body(app))
    other = SelectionReviewApp(env.db, env.store, app.run.record_id, "second-human")
    other.submit({**body(other), "decision": "flag", "note": "I disagree with the classification."})
    status = selection_status(env.db, app.run.record_id)
    assert status["all_required_work_resolved"] and not status["complete"]
    assert status["human_review"]["flagged"] == 1
