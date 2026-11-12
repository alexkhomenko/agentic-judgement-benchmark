"""Tasks, attempts, responses, artifacts, and metric observations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import model_validator

from .common import Frozen


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


class AttemptStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class FailureKind(StrEnum):
    TRANSIENT_TRANSPORT = "transient_transport"
    RATE_LIMITED = "rate_limited"
    AMBIGUOUS_TIMEOUT = "ambiguous_timeout"
    CONTEXT_EXCEEDED = "context_exceeded"
    PROVIDER_ERROR = "provider_error"
    REFUSAL = "refusal"  # provider safety decline; recorded, never retried or rerouted
    INVALID_STATE = "invalid_state"
    HARNESS_ERROR = "harness_error"
    INTERRUPTED = "interrupted"


class CostIncurred(StrEnum):
    NO = "no"
    YES = "yes"
    UNKNOWN = "unknown"


class Observation(Frozen):
    """A measurement. Missing values carry a reason and are never coerced to zero."""

    value: float | int | None
    unit: str
    source: str
    missing_reason: str | None = None

    @model_validator(mode="after")
    def _reason_when_missing(self) -> Observation:
        if self.value is None and not self.missing_reason:
            raise ValueError("a missing observation needs a missing_reason")
        return self

    @classmethod
    def missing(cls, unit: str, source: str, reason: str) -> Observation:
        return cls(value=None, unit=unit, source=source, missing_reason=reason)


class ArtifactRef(Frozen):
    name: str
    relpath: str
    sha256: str
    size_bytes: int
    schema_version: str


class ResponseTask(Frozen):
    task_id: str
    experiment_id: str
    configuration_key: str
    repetition: int
    round: int
    parent_task_id: str | None
    # Operational state; mutable in storage, never part of published evidence.
    status: TaskStatus = TaskStatus.PENDING
    status_reason: str | None = None


class InvocationAttempt(Frozen):
    attempt_id: str
    task_id: str
    attempt_no: int
    status: AttemptStatus
    failure_kind: FailureKind | None = None
    cost_incurred: CostIncurred = CostIncurred.NO
    error_summary: str | None = None
    started_at: str
    finished_at: str | None = None


class CandidateResponse(Frozen):
    response_id: str
    task_id: str
    attempt_id: str
    experiment_id: str
    configuration_key: str
    repetition: int
    round: int
    final_answer_sha256: str
    truncated: bool
    finish_reason: str | None
    selection_policy: str
    created_at: str
