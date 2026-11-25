"""Deterministic experiment controller.

Each repetition runs: fresh conversation -> R1 baseline -> immutable snapshot -> independent R2/R3/R4
branches. A child receives the original system/user messages, its own repetition's baseline assistant
turn and native state, and its own follow-up. It never sees a sibling. The controller is the only
database writer; evidence is durable on disk before an attempt is marked complete.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable

from pydantic import Field

from ..adapters.base import ContinuationSnapshot, InferenceAdapter, InvocationContext
from ..adapters.registry import AdapterRegistry
from ..domain.common import Frozen, new_id, sha256_text, utc_now
from ..domain.execution import (
    AttemptStatus,
    FailureKind,
    CandidateResponse,
    InvocationAttempt,
    ResponseTask,
    TaskStatus,
)
from ..domain.plan import ConfigBinding, ExperimentPlan, RetryPolicy
from ..domain.scenario import Message, Role, Scenario
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from ..storage.reconcile import ReconcileReport, reconcile
from .invoke import invoke


class RunSummary(Frozen):
    experiment_id: str
    reconcile: ReconcileReport
    task_status: dict[str, int] = Field(default_factory=dict)
    responses: int = 0
    attempts: int = 0


def task_id_for(experiment_id: str, configuration_key: str, repetition: int, round_number: int) -> str:
    return f"{experiment_id}:{configuration_key}:rep{repetition}:r{round_number}"


def planned_tasks(plan: ExperimentPlan) -> list[ResponseTask]:
    tasks: list[ResponseTask] = []
    base = plan.scenario.baseline.round
    for binding in plan.configurations:
        key = binding.configuration.key
        for rep in range(1, plan.repetitions + 1):
            parent = task_id_for(plan.experiment_id, key, rep, base)
            tasks.append(ResponseTask(task_id=parent, experiment_id=plan.experiment_id, configuration_key=key,
                                      repetition=rep, round=base, parent_task_id=None))
            for spec in plan.scenario.follow_ups:
                tasks.append(ResponseTask(
                    task_id=task_id_for(plan.experiment_id, key, rep, spec.round),
                    experiment_id=plan.experiment_id, configuration_key=key, repetition=rep,
                    round=spec.round, parent_task_id=parent,
                ))
    return tasks


def build_conversation(scenario: Scenario, round_number: int, parent: ContinuationSnapshot | None) -> tuple[Message, ...]:
    spec = scenario.round_spec(round_number)
    root = (
        Message(role=Role.SYSTEM, content=scenario.system_prompt),
        Message(role=Role.USER, content=scenario.baseline.prompt),
    )
    if spec.parent_round is None:
        return root
    if parent is None:
        raise ValueError(f"round {round_number} requires a baseline snapshot")
    if parent.messages[: len(root)] != root or len(parent.messages) != len(root) + 1:
        raise ValueError("baseline snapshot does not match the scenario's baseline conversation")
    return parent.messages + (Message(role=Role.USER, content=spec.prompt),)


def _rate_limited(policy: RetryPolicy, attempt: InvocationAttempt) -> bool:
    return bool(policy.rate_limit_backoff_s) and attempt.failure_kind is FailureKind.RATE_LIMITED


def counted_attempts(policy: RetryPolicy, attempts: list[InvocationAttempt]) -> int:
    """Attempts that consume max_attempts: all of them, except rate-limited ones when backoff is configured."""
    return sum(not _rate_limited(policy, a) for a in attempts)


def may_retry(policy: RetryPolicy, last: InvocationAttempt, attempts_used: int) -> bool:
    if last.status is AttemptStatus.SUCCEEDED or last.failure_kind not in policy.retry_on:
        return False
    return _rate_limited(policy, last) or attempts_used < policy.max_attempts


class Controller:
    def __init__(self, db: Database, store: ArtifactStore, adapters: AdapterRegistry,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.db = db
        self.store = store
        self.adapters = adapters
        self.sleep = sleep

    def run(self, plan: ExperimentPlan) -> RunSummary:
        with self.store.lock(plan.experiment_id):
            report = reconcile(self.db, self.store, plan.experiment_id)
            self.db.ensure_tasks(planned_tasks(plan))
            for key in plan.scheduling.resolved_order:
                binding = plan.binding(key)
                adapter = self.adapters.get(binding.configuration.adapter)
                for rep in range(1, plan.repetitions + 1):
                    self._run_repetition(plan, binding, adapter, rep)
            return self.summary(plan.experiment_id, report)

    def summary(self, experiment_id: str, report: ReconcileReport) -> RunSummary:
        counts = Counter(t.status.value for t in self.db.list_tasks(experiment_id))
        return RunSummary(
            experiment_id=experiment_id,
            reconcile=report,
            task_status=dict(sorted(counts.items())),
            responses=len(self.db.list_responses(experiment_id)),
            attempts=len(self.db.list_attempts(experiment_id=experiment_id)),
        )

    # ----- graph -----------------------------------------------------------------------------------

    def _run_repetition(self, plan: ExperimentPlan, binding: ConfigBinding, adapter: InferenceAdapter, rep: int) -> None:
        key = binding.configuration.key
        base_task = self.db.get_task(task_id_for(plan.experiment_id, key, rep, plan.scenario.baseline.round))
        baseline_ok = self._run_task(plan, binding, adapter, base_task, parent=None)
        for spec in plan.scenario.follow_ups:
            task = self.db.get_task(task_id_for(plan.experiment_id, key, rep, spec.round))
            if not baseline_ok:
                if self.db.response_for_task(task.task_id) is None:
                    self.db.set_task_status(
                        task.task_id, TaskStatus.BLOCKED, f"baseline {base_task.task_id} produced no response"
                    )
                continue
            # Load a fresh, hash-verified copy for every child so no sibling can observe another's mutation.
            snapshot = self._load_snapshot(base_task.task_id)
            self._run_task(plan, binding, adapter, task, parent=snapshot)

    def _load_snapshot(self, task_id: str) -> ContinuationSnapshot:
        response = self.db.response_for_task(task_id)
        if response is None:
            raise RuntimeError(f"task {task_id} has no response")
        ref = self.db.artifacts_for(response.attempt_id)["continuation-state.json"]
        return ContinuationSnapshot.model_validate(self.store.read_json(ref))

    def _run_task(
        self,
        plan: ExperimentPlan,
        binding: ConfigBinding,
        adapter: InferenceAdapter,
        task: ResponseTask,
        parent: ContinuationSnapshot | None,
    ) -> bool:
        if self.db.response_for_task(task.task_id) is not None:
            return True
        prior = self.db.list_attempts(task_id=task.task_id)
        counted = counted_attempts(plan.retry, prior)
        if prior and not may_retry(plan.retry, prior[-1], counted):
            self.db.set_task_status(task.task_id, TaskStatus.FAILED, _failure_reason(prior[-1], len(prior)))
            return False
        messages = build_conversation(plan.scenario, task.round, parent)
        attempt_no, waits = len(prior), list(plan.retry.rate_limit_backoff_s)
        while True:
            attempt_no += 1
            attempt = self._attempt(plan, binding, adapter, task, messages, parent, attempt_no)
            if attempt.status is AttemptStatus.SUCCEEDED:
                return True
            counted += not _rate_limited(plan.retry, attempt)
            if _rate_limited(plan.retry, attempt):
                if not waits:
                    # Waits for this run are used up; the task stays resumable by a later `bench run`.
                    self.db.set_task_status(task.task_id, TaskStatus.FAILED,
                                            _failure_reason(attempt, attempt_no) + " (rate-limit waits used up; "
                                            "a later run resumes it)")
                    return False
                wait = waits.pop(0)
                self.db.set_task_status(task.task_id, TaskStatus.PENDING,
                                        f"rate limited on attempt {attempt_no}; waiting {wait:g}s before retrying")
                self.sleep(wait)
                continue
            if not may_retry(plan.retry, attempt, counted):
                self.db.set_task_status(task.task_id, TaskStatus.FAILED, _failure_reason(attempt, attempt_no))
                return False

    # ----- one attempt -----------------------------------------------------------------------------

    def _attempt(
        self,
        plan: ExperimentPlan,
        binding: ConfigBinding,
        adapter: InferenceAdapter,
        task: ResponseTask,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        attempt_no: int,
    ) -> InvocationAttempt:
        config = binding.configuration
        attempt_id = new_id("att")
        self.db.start_attempt(InvocationAttempt(
            attempt_id=attempt_id, task_id=task.task_id, attempt_no=attempt_no,
            status=AttemptStatus.RUNNING, started_at=utc_now(),
        ))
        self.db.set_task_status(task.task_id, TaskStatus.RUNNING)
        sink = self.store.sink(f"{plan.experiment_id}/artifacts/{attempt_id}")
        ctx = InvocationContext(
            experiment_id=plan.experiment_id, task_id=task.task_id, attempt_id=attempt_id,
            attempt_no=attempt_no, round=task.round,
        )

        outcome = invoke(adapter, config, messages, parent, ctx, sink)
        failure, cost, error = outcome.failure_kind, outcome.cost_incurred, outcome.error
        response: CandidateResponse | None = None
        if outcome.succeeded and outcome.extracted is not None:
            extracted = outcome.extracted
            response = CandidateResponse(
                response_id=new_id("rsp"), task_id=task.task_id, attempt_id=attempt_id,
                experiment_id=plan.experiment_id, configuration_key=config.key,
                repetition=task.repetition, round=task.round,
                final_answer_sha256=sha256_text(extracted.final_answer),
                truncated=extracted.truncated, finish_reason=extracted.finish_reason,
                selection_policy=plan.retry.selection, created_at=utc_now(),
            )

        status = AttemptStatus.SUCCEEDED if response is not None else AttemptStatus.FAILED
        with self.db.transaction():
            self.db.register_artifacts(attempt_id, sink.refs)
            self.db.finish_attempt(
                attempt_id, status=status, failure_kind=failure, cost_incurred=cost, error_summary=error
            )
            if response is not None:
                self.db.insert_response(response)
                self.db.set_task_status(task.task_id, TaskStatus.SUCCEEDED)
            else:
                self.db.set_task_status(task.task_id, TaskStatus.PENDING, f"attempt {attempt_no} failed: {failure}")
        return self.db.get_attempt(attempt_id)


def _failure_reason(attempt: InvocationAttempt, attempts_used: int) -> str:
    return f"{attempt.failure_kind or 'unknown'} after {attempts_used} attempt(s): {attempt.error_summary or ''}".strip()
