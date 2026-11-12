"""Controller invariants: branch isolation, independent repetitions, native state, retries, resume."""

from __future__ import annotations

from typing import Any

import pytest

from archbench.adapters.base import ContinuationSnapshot, InvocationContext, PreparedInvocation
from archbench.adapters.fake import FakeAdapter
from archbench.adapters.registry import AdapterRegistry
from archbench.domain.configuration import ModelConfiguration
from archbench.domain.execution import AttemptStatus, CostIncurred, FailureKind, TaskStatus
from archbench.domain.scenario import Message, Role
from archbench.execution.controller import Controller, RunSummary, task_id_for

from .conftest import Env


class SpyAdapter(FakeAdapter):
    """Records what every invocation receives; optionally mutates or interrupts."""

    def __init__(self, *, mutate_parent: bool = False, interrupt_on: tuple[int, int] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.mutate_parent = mutate_parent
        self.interrupt_on = interrupt_on  # (repetition-agnostic round, attempt_no)

    def prepare(
        self,
        config: ModelConfiguration,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        ctx: InvocationContext,
    ) -> PreparedInvocation:
        prepared = super().prepare(config, messages, parent, ctx)
        if ctx.purpose == "scored":
            self.calls.append({
                "task_id": ctx.task_id,
                "round": ctx.round,
                "messages": messages,
                "parent_response_id": parent.native_state.get("response_id") if parent else None,
            })
        if parent is not None and self.mutate_parent:
            parent.native_state["state_token"] = "tampered-by-sibling"
            parent.native_state["leak"] = f"from {ctx.task_id}"
        return prepared

    def execute(self, config, prepared, ctx, sink):  # type: ignore[no-untyped-def]
        if self.interrupt_on == (ctx.round, ctx.attempt_no) and ctx.purpose == "scored":
            raise KeyboardInterrupt("simulated controller crash during dispatch")
        return super().execute(config, prepared, ctx, sink)


def run(env: Env, plan: Any, adapter: FakeAdapter | None = None) -> RunSummary:
    registry = AdapterRegistry({"fake": adapter or FakeAdapter()})
    return Controller(env.db, env.store, registry).run(plan)


def answer(env: Env, task_id: str) -> str:
    response = env.db.response_for_task(task_id)
    assert response is not None
    return env.store.read_verified(env.db.artifacts_for(response.attempt_id)["final-answer.txt"]).decode()


def native(env: Env, task_id: str, name: str) -> Any:
    response = env.db.response_for_task(task_id)
    assert response is not None
    return env.store.read_json(env.db.artifacts_for(response.attempt_id)[name])


def test_full_graph_completes_with_branching_structure(env: Env) -> None:
    plan = env.plan(env.experiment("exp-full", [env.config("alpha"), env.config("beta")]))
    summary = run(env, plan)
    assert summary.task_status == {"succeeded": 24}
    assert summary.responses == 24
    for task in env.db.list_tasks(plan.experiment_id):
        if task.round == 1:
            assert task.parent_task_id is None
        else:
            assert task.parent_task_id == task_id_for(plan.experiment_id, task.configuration_key, task.repetition, 1)


def test_branches_see_only_their_own_baseline_and_follow_up(env: Env) -> None:
    spy = SpyAdapter()
    plan = env.plan(env.experiment("exp-iso", [env.config("alpha")]))
    run(env, plan, spy)
    scenario = plan.scenario
    key = plan.configurations[0].configuration.key
    for rep in range(1, 4):
        base_id = task_id_for(plan.experiment_id, key, rep, 1)
        baseline_answer = answer(env, base_id)
        baseline_native_id = native(env, base_id, "native-response.json")["id"]
        for spec in scenario.follow_ups:
            call = next(c for c in spy.calls if c["task_id"] == task_id_for(plan.experiment_id, key, rep, spec.round))
            assert call["messages"] == (
                Message(role=Role.SYSTEM, content=scenario.system_prompt),
                Message(role=Role.USER, content=scenario.baseline.prompt),
                Message(role=Role.ASSISTANT, content=baseline_answer),
                Message(role=Role.USER, content=spec.prompt),
            )
            # Native continuation state comes from this repetition's baseline, never another repetition's.
            assert call["parent_response_id"] == baseline_native_id
            request = native(env, call["task_id"], "native-request.json")
            assert request["body"]["continuation_token"] == native(env, base_id, "continuation-state.json")[
                "native_state"]["state_token"]
            sibling_prompts = {s.prompt for s in scenario.follow_ups if s.round != spec.round}
            assert not sibling_prompts & {m.content for m in call["messages"]}


def test_repetitions_start_fresh_conversations(env: Env) -> None:
    spy = SpyAdapter()
    plan = env.plan(env.experiment("exp-reps", [env.config("alpha")]))
    run(env, plan, spy)
    baselines = [c for c in spy.calls if c["round"] == 1]
    assert len(baselines) == 3
    for call in baselines:
        assert call["parent_response_id"] is None
        assert [m.role for m in call["messages"]] == [Role.SYSTEM, Role.USER]
    parents = {c["parent_response_id"] for c in spy.calls if c["round"] != 1}
    assert len(parents) == 3  # each repetition's branches fork from a different baseline


def test_sibling_cannot_corrupt_shared_baseline_state(env: Env) -> None:
    spy = SpyAdapter(mutate_parent=True)
    plan = env.plan(env.experiment("exp-mutate", [env.config("alpha")]))
    summary = run(env, plan, spy)
    # Every sibling still received a valid, untampered snapshot loaded from verified evidence.
    assert summary.task_status == {"succeeded": 12}
    key = plan.configurations[0].configuration.key
    stored = native(env, task_id_for(plan.experiment_id, key, 1, 1), "continuation-state.json")
    assert "leak" not in stored["native_state"]


def test_tampered_continuation_state_is_rejected() -> None:
    adapter = FakeAdapter()
    snapshot = ContinuationSnapshot(
        adapter="fake",
        configuration_key="x@1",
        messages=(Message(role=Role.SYSTEM, content="s"), Message(role=Role.USER, content="u"),
                  Message(role=Role.ASSISTANT, content="a")),
        native_state={"format": "fake-transcript-v1", "state_token": "forged"},
    )
    from archbench.adapters.base import AdapterError

    config = ModelConfiguration(configuration_id="x", revision=1, track="fake", adapter="fake", provider="h",
                                model="m")
    ctx = InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=2)
    with pytest.raises(AdapterError) as err:
        adapter.prepare(config, snapshot.messages + (Message(role=Role.USER, content="f"),), snapshot, ctx)
    assert err.value.kind is FailureKind.INVALID_STATE


def test_transient_failure_retried_as_new_attempt_and_first_success_selected(env: Env) -> None:
    cfg = env.config("beta", adapter_options={"transient_failures": {"1": 1}})
    plan = env.plan(env.experiment("exp-retry", [cfg]))
    summary = run(env, plan)
    assert summary.task_status == {"succeeded": 12}
    assert summary.attempts == 15  # three baselines needed a second attempt
    key = plan.configurations[0].configuration.key
    for rep in range(1, 4):
        tid = task_id_for(plan.experiment_id, key, rep, 1)
        attempts = env.db.list_attempts(task_id=tid)
        assert [a.status for a in attempts] == [AttemptStatus.FAILED, AttemptStatus.SUCCEEDED]
        assert attempts[0].failure_kind is FailureKind.TRANSIENT_TRANSPORT
        response = env.db.response_for_task(tid)
        assert response is not None
        assert response.attempt_id == attempts[1].attempt_id
        assert response.selection_policy == "first_success"
        # The failed attempt keeps its own evidence.
        assert "error.json" in env.db.artifacts_for(attempts[0].attempt_id)


def test_exhausted_baseline_blocks_follow_ups_and_rerun_adds_no_attempts(env: Env) -> None:
    cfg = env.config("flaky", adapter_options={"transient_failures": {"1": 9}})
    plan = env.plan(env.experiment("exp-exhaust", [cfg], repetitions=1))
    summary = run(env, plan)
    assert summary.task_status == {"blocked": 3, "failed": 1}
    assert summary.attempts == 3
    blocked = [t for t in env.db.list_tasks(plan.experiment_id) if t.status is TaskStatus.BLOCKED]
    assert all("produced no response" in (t.status_reason or "") for t in blocked)
    again = run(env, plan)
    assert again.attempts == 3


def test_permanent_failure_is_not_retried(env: Env) -> None:
    cfg = env.config("perm", adapter_options={"permanent_failure_rounds": [3]})
    plan = env.plan(env.experiment("exp-perm", [cfg], repetitions=1))
    summary = run(env, plan)
    assert summary.task_status == {"failed": 1, "succeeded": 3}
    assert summary.attempts == 4


def test_truncated_answer_is_kept_not_regenerated(env: Env) -> None:
    cfg = env.config("trunc", adapter_options={"truncate_rounds": [2]})
    plan = env.plan(env.experiment("exp-trunc", [cfg], repetitions=1))
    summary = run(env, plan)
    assert summary.attempts == 4
    key = plan.configurations[0].configuration.key
    response = env.db.response_for_task(task_id_for(plan.experiment_id, key, 1, 2))
    assert response is not None and response.truncated and response.finish_reason == "max_output_tokens"


def test_context_overflow_rejected_before_dispatch(env: Env) -> None:
    cfg = env.config("tiny", context_limit=2000)
    plan = env.plan(env.experiment("exp-ctx", [cfg], repetitions=1), force=True)
    summary = run(env, plan)
    assert summary.task_status == {"blocked": 3, "failed": 1}
    (attempt,) = env.db.list_attempts(experiment_id=plan.experiment_id)
    assert attempt.failure_kind is FailureKind.CONTEXT_EXCEEDED
    assert attempt.cost_incurred is CostIncurred.NO
    metrics = env.store.read_json(env.db.artifacts_for(attempt.attempt_id)["metrics.json"])
    assert metrics["client_wall_ms"]["value"] is None
    assert metrics["client_wall_ms"]["missing_reason"] == "request was not dispatched"


def test_crash_then_resume_records_interruption_with_unknown_cost(env: Env) -> None:
    plan = env.plan(env.experiment("exp-crash", [env.config("alpha")], repetitions=1))
    with pytest.raises(KeyboardInterrupt):
        run(env, plan, SpyAdapter(interrupt_on=(3, 1)))
    running = env.db.list_attempts(experiment_id=plan.experiment_id, status=AttemptStatus.RUNNING)
    assert len(running) == 1
    crashed = running[0].attempt_id

    summary = run(env, plan)
    assert summary.reconcile.interrupted_attempts == [crashed]
    assert summary.reconcile.integrity_ok
    assert summary.task_status == {"succeeded": 4}
    interrupted = env.db.get_attempt(crashed)
    assert interrupted.status is AttemptStatus.INTERRUPTED
    assert interrupted.cost_incurred is CostIncurred.UNKNOWN
    # Evidence written before the crash was adopted, not discarded.
    assert {"canonical-messages.json", "native-request.json"} <= set(env.db.artifacts_for(crashed))
    # Completed tasks were not re-run.
    assert summary.attempts == 5


def test_controller_lock_prevents_concurrent_runs(env: Env) -> None:
    plan = env.plan(env.experiment("exp-lock", [env.config("alpha")], repetitions=1))
    with env.store.lock(plan.experiment_id), pytest.raises(RuntimeError, match="holds the lock"):
        run(env, plan)
