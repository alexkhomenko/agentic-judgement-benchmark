"""A live, blinded v2 inbox preserves review provenance across answers and evaluation versions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from archbench.cli import build_parser
from archbench.domain.records import RecordKind
from archbench.evaluation.checks import CheckError
from archbench.evaluation.selection_inbox import SelectionInboxApp
from archbench.evaluation.selection_review import SelectionReviewApp
from archbench.evaluation.selection_runner import resume_selection_evaluation, start_selection_evaluation
from archbench.execution.controller import Controller
from archbench.execution.limits import CallBudget, CheckpointStop

from .conftest import Env
from .test_selection_evaluation import setup
from .test_selection_review import decision


def submit(inbox: SelectionInboxApp, packet: str, action: str = "proceed", note: str = "") -> dict[str, Any]:
    item = inbox.item(packet)
    return inbox.submit({"packet_id": packet, "seen": item["seen"], "decision": action, "note": note})


def new_answer(env: Env, name: str = "another-hidden-model") -> Any:
    config = env.config(name, provider="hidden-author-identity",
        adapter_options={"style": "recommendations", "recommendation_pool": ["Acme"], "recommendation_count": 1})
    plan = env.plan(env.experiment(name, [config], scenario=str(env.tmp/"scenario"), repetitions=1))
    Controller(env.db, env.store, env.registry).run(plan)
    return plan


def test_live_discovery_unresolved_first_and_existing_approval_is_retained(env: Env) -> None:
    first, path, registry, llm, jev = setup(env, repetitions=1)
    run = start_selection_evaluation(env.db, env.store, registry, first.experiment_id, path, base=env.base)
    resume_selection_evaluation(env.db, env.store, registry, run.record_id)
    old = SelectionReviewApp(env.db, env.store, run.record_id, "human")
    approval = old.submit(decision(old))["check_id"]
    inbox = SelectionInboxApp(env.db, env.store, "human")
    q = inbox.queue()
    approved_packet = q["items"][0]["packet_id"]
    assert q["done"] == 1 and q["total"] == 1
    new_answer(env)
    q = inbox.queue()
    assert q["total"] == 2 and q["items"][1]["packet_id"] == approved_packet
    assert q["items"][0]["automatic"] == "pending" and q["items"][0]["blocked"]
    item = inbox.item(q["items"][0]["packet_id"])
    assert item["answer"] and not item["can_proceed"] and not item["can_flag"]
    assert "hidden-author-identity" not in json.dumps(item)
    assert "another-hidden-model" not in json.dumps(item)
    assert env.db.get_record(approval) and old.queue()["done"] == 1
    assert [(i["packet_id"], i["updated_at"]) for i in inbox.queue()["items"]] == [
        (i["packet_id"], i["updated_at"]) for i in q["items"]]
    assert len(llm.calls) == 2 and len(jev.calls) == 1  # inbox operations dispatched no evaluator calls


def test_newest_pending_answer_first_and_non_v2_answers_excluded(env: Env) -> None:
    setup(env, repetitions=1)
    inbox = SelectionInboxApp(env.db, env.store, "human")
    old_packet = inbox.queue()["items"][0]["packet_id"]
    new_answer(env)
    v1 = env.plan(env.experiment("v1-outside-inbox", [env.config("v1-model")], repetitions=1))
    Controller(env.db, env.store, env.registry).run(v1)
    q = inbox.queue()
    assert q["total"] == 2 and q["items"][1]["packet_id"] == old_packet
    assert all(i["state"] == "unchecked" for i in q["items"])


def test_local_answers_recorded_before_the_v2_exclusion_are_hidden(env: Env, monkeypatch: Any) -> None:
    from archbench.planning import compiler

    setup(env, repetitions=1)
    with monkeypatch.context() as m:
        m.setattr(compiler, "V2_SCENARIO_PREFIX", "planned-before-the-exclusion")
        config = env.config("old-local-model", provider="hidden-author-identity", track="local",
            adapter_options={"style": "recommendations", "recommendation_pool": ["Acme"], "recommendation_count": 1})
        local = env.plan(env.experiment("old-local", [config], scenario=str(env.tmp/"scenario"), repetitions=1),
                         force=True)
    Controller(env.db, env.store, env.registry).run(local)
    assert env.db.list_responses(local.experiment_id)
    entries = SelectionInboxApp(env.db, env.store, "alexander").entries()
    assert len(entries) == 1 and all(e.plan.experiment_id != local.experiment_id for e in entries)


def test_new_evaluation_is_one_versioned_item_and_old_approval_does_not_transfer(env: Env) -> None:
    plan, path, registry, _, _ = setup(env, repetitions=1)
    run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    resume_selection_evaluation(env.db, env.store, registry, run.record_id)
    inbox = SelectionInboxApp(env.db, env.store, "human")
    packet = inbox.queue()["items"][0]["packet_id"]
    original = submit(inbox, packet)["check_id"]
    seen = inbox.item(packet)["seen"]
    new_run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    q = inbox.queue()
    assert q["total"] == 1 and q["items"][0]["packet_id"] == packet
    assert q["items"][0]["state"] == "stale" and q["items"][0]["version_count"] == 2
    with pytest.raises(CheckError, match="changed"):
        inbox.submit({"packet_id": packet, "seen": seen, "decision": "proceed"})
    resume_selection_evaluation(env.db, env.store, registry, new_run.record_id)
    item = inbox.item(packet)
    assert len(item["history"]) == 2 and item["history"][1]["reviews"][0]["decision"] == "proceed"
    result = submit(inbox, packet)
    assert result["queue"]["done"] == 1
    saved = env.db.get_record(result["check_id"])
    assert saved and saved.payload["evaluation_run_id"] == new_run.record_id
    assert env.db.get_record(original)


def test_blocked_answer_is_visible_and_can_be_flagged_without_fabricating_audit(env: Env) -> None:
    plan, path, registry, _, _ = setup(env, repetitions=1, malformed=9)
    run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    resume_selection_evaluation(env.db, env.store, registry, run.record_id)
    inbox = SelectionInboxApp(env.db, env.store, "human")
    packet = inbox.queue()["items"][0]["packet_id"]
    item = inbox.item(packet)
    assert item["automatic"] == "failed" and item["can_flag"] and not item["can_proceed"]
    assert "extraction_failed" in item["flags"]
    assert any(a["status"] == "invalid_output" and a["problems"] for a in item["history"][0]["attempts"])
    recovery = inbox.recovery(packet)
    assert recovery["command"] is None and "retry allowance" in recovery["message"]
    result = submit(inbox, packet, "flag", "The extractor needs recovery.")
    assert result["queue"]["counts"]["flagged"] == 1 and result["queue"]["counts"]["blocked"] == 1
    with pytest.raises(CheckError, match="automated"):
        submit(inbox, packet)


def test_ready_answer_can_be_reviewed_while_other_answers_in_its_run_are_pending(env: Env) -> None:
    plan, path, registry, _, _ = setup(env, repetitions=2, fraction=0)
    run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    budget = CallBudget(2)
    with pytest.raises(CheckpointStop):
        resume_selection_evaluation(env.db, env.store, registry, run.record_id, before_call=budget.before_call)
    inbox = SelectionInboxApp(env.db, env.store, "human")
    q = inbox.queue()
    ready = next(i for i in q["items"] if i["automatic"] == "ready")
    pending = next(i for i in q["items"] if i["automatic"] == "pending")
    assert submit(inbox, ready["packet_id"])["queue"]["done"] == 1
    assert not inbox.item(pending["packet_id"])["can_proceed"]
    assert "--resume" in inbox.recovery(pending["packet_id"])["command"]


def test_incompatible_frozen_code_is_a_visible_blocker_not_a_disappearing_answer(env: Env, monkeypatch: Any) -> None:
    from archbench.evaluation import selection_runner
    plan, path, registry, _, _ = setup(env, repetitions=1)
    run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    resume_selection_evaluation(env.db, env.store, registry, run.record_id)
    monkeypatch.setattr(selection_runner, "_code_hashes", lambda: {"changed": "implementation"})
    inbox = SelectionInboxApp(env.db, env.store, "human")
    q = inbox.queue()
    assert q["total"] == 1 and q["items"][0]["automatic"] == "blocked"
    item = inbox.item(q["items"][0]["packet_id"])
    assert item["answer"] and "code changed" in item["blocked_reason"] and not item["can_flag"]


def test_policy_filter_metadata_and_inbox_cli_defaults(env: Env) -> None:
    plan, path, registry, _, _ = setup(env, repetitions=1)
    policy = yaml.safe_load(path.read_text())
    policy["revision"] = 2
    path.write_text(yaml.safe_dump(policy))
    run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    inbox = SelectionInboxApp(env.db, env.store, "human")
    item = inbox.queue()["items"][0]
    assert item["policy"] == "test-selection@2" and item["evaluation_run_id"] == run.record_id
    args = build_parser().parse_args(["review-v2", "--reviewer", "human"])
    assert args.port == 8766 and args.reviewer == "human"
    with pytest.raises(CheckError):
        inbox.item("outside-this-inbox")
    with pytest.raises(CheckError):
        inbox.submit({"packet_id": item["packet_id"], "seen": item["seen"], "decision": "proceed", "reviewer_id": "spoofed"})


def test_inbox_http_report_and_recovery_keep_localhost_and_token_checks(env: Env) -> None:
    import threading
    import urllib.error
    import urllib.request
    from archbench.evaluation.review_app import serve

    plan, path, registry, _, _ = setup(env, repetitions=1)
    run = start_selection_evaluation(env.db, env.store, registry, plan.experiment_id, path, base=env.base)
    resume_selection_evaluation(env.db, env.store, registry, run.record_id)
    inbox = SelectionInboxApp(env.db, env.store, "human")
    server, url = serve(inbox)
    server.timeout = .05
    results: dict[str, Any] = {}

    def call(path: str, *, token: str | None = inbox.token, host: str | None = None) -> tuple[int, Any]:
        headers = {"X-Review-Token": token} if token else {}
        if host:
            headers["Host"] = host
        try:
            with urllib.request.urlopen(urllib.request.Request(url.rstrip("/")+path, headers=headers)) as response:
                body = response.read().decode()
                return response.status, json.loads(body) if path.startswith("/api") else body
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def client() -> None:
        try:
            assert call("/api/queue", token=None)[0] == 403
            assert call("/api/queue", host="outside.example")[0] == 403
            status, queue = call("/api/queue")
            assert status == 200 and queue["total"] == 1
            packet = queue["items"][0]["packet_id"]
            status, item = call("/api/item?packet="+packet)
            assert status == 200
            assert call("/report?packet="+packet)[0] == 403
            assert call(item["report_url"], host="outside.example")[0] == 403
            status, page = call(item["report_url"])
            assert status == 200 and "V2 experiment report" in page
            assert call("/api/recovery?packet="+packet, token=None)[0] == 403
            assert call("/api/recovery?packet="+packet)[0] == 200
            results["ok"] = True
        except BaseException as exc:
            results["error"] = exc

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            server.handle_request()
        assert results.get("ok"), results.get("error")
    finally:
        server.server_close()
        thread.join(timeout=2)
