"""V2 human validation is explicit, durable and tied to the evidence actually shown."""
from __future__ import annotations

import json

import pytest

from archbench.domain.records import Record, RecordKind
from archbench.evaluation.checks import CheckError
from archbench.evaluation.selection_review import SelectionReviewApp
from archbench.evaluation.selection_runner import (
    publish_selection_report, resume_selection_evaluation, selection_status, start_selection_evaluation,
)

from .conftest import Env
from .test_selection_evaluation import setup


def app_for(env: Env, *, probability: float = .99, malformed: int = 0) -> SelectionReviewApp:
    plan, path, registry, _, _ = setup(env, repetitions=1, probability=probability, malformed=malformed)
    run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    resume_selection_evaluation(env.db, env.store, registry, run.record_id)
    return SelectionReviewApp(env.db, env.store, run.record_id, "human")


def decision(app: SelectionReviewApp, action: str = "proceed", note: str = "") -> dict[str, str]:
    item = app.item(app.queue()["items"][0]["packet_id"])
    return {"packet_id": item["packet_id"], "seen": item["seen"], "decision": action, "note": note}


def test_every_answer_requires_explicit_human_action_and_publishes_final_state(env: Env) -> None:
    app = app_for(env)
    initial = app.queue()
    assert initial["done"] == 0 and initial["total"] == 1
    assert not selection_status(env.db, app.run.record_id)["complete"]
    item = app.item(initial["items"][0]["packet_id"])
    assert item["can_proceed"]
    assert "hidden-author-identity" not in json.dumps(item)
    assert "configuration_key" not in item and "provider" not in item and "track" not in item
    assert b'Proceed' in app.page and b'__REVIEW_TOKEN__' not in app.page
    assert not env.db.list_records(kind=RecordKind.SELECTION_HUMAN_REVIEW)
    body = decision(app)
    saved = app.submit(body)
    assert saved["created"] and saved["queue"]["done"] == 1
    assert selection_status(env.db, app.run.record_id)["complete"]
    assert not app.submit(body)["created"]
    report = publish_selection_report(env.db, env.store, app.run.record_id)
    assert report.payload["complete"]
    assert report.payload["responses"][0]["human_review_state"] == "approved"
    assert saved["check_id"] in report.inputs
    assert "human-review.json" in env.db.artifacts_for(saved["check_id"])


def test_flags_are_visible_and_human_flags_hold_completion_until_resolved(env: Env) -> None:
    app = app_for(env, probability=.6)
    item = app.item(app.queue()["items"][0]["packet_id"])
    assert "uncertain_rank" in item["flags"] and item["audit"]
    with pytest.raises(CheckError, match="note"):
        app.submit(decision(app, "flag"))
    flagged = app.submit(decision(app, "flag", "Check the provider label."))
    assert flagged["queue"]["counts"]["flagged"] == 1
    assert not selection_status(env.db, app.run.record_id)["complete"]
    approved = app.submit(decision(app))
    record = env.db.get_record(approved["check_id"])
    assert record and record.supersedes == flagged["check_id"]
    assert "uncertain_rank" in record.payload["automated_flags_seen"]
    assert approved["queue"]["done"] == 1


def test_changed_evidence_makes_validation_stale_and_rejects_old_screen(env: Env) -> None:
    app = app_for(env)
    body = decision(app)
    app.submit(body)
    entry = app.run.payload["responses"][0]
    env.db.insert_record(Record.create(kind=RecordKind.SELECTION_AUDIT,
        experiment_id=app.run.experiment_id, subject_id=entry["response_id"],
        payload={"evaluation_run_id": app.run.record_id, "attempt_no": 9, "status": "failed",
                 "error": "additional evidence", "cost_incurred": "no"}))
    assert app.queue()["items"][0]["state"] == "stale"
    assert not selection_status(env.db, app.run.record_id)["complete"]
    with pytest.raises(CheckError, match="changed"):
        app.submit(body)


def test_exhausted_automatic_evaluation_cannot_be_implicitly_approved(env: Env) -> None:
    app = app_for(env, malformed=9)
    item = app.item(app.queue()["items"][0]["packet_id"])
    assert "extraction_failed" in item["flags"] and not item["can_proceed"]
    with pytest.raises(CheckError, match="automated"):
        app.submit(decision(app))


def test_other_reviewers_flag_remains_blocking_and_untrusted_submission_is_rejected(env: Env) -> None:
    app = app_for(env)
    second = SelectionReviewApp(env.db, env.store, app.run.record_id, "second-human")
    app.submit(decision(app, "flag", "The trade-off needs checking."))
    second.submit(decision(second))
    assert not selection_status(env.db, app.run.record_id)["complete"]
    with pytest.raises(CheckError):
        app.submit({**decision(app), "reviewer_id": "spoofed"})
    with pytest.raises(CheckError):
        app.item("outside-this-run")
