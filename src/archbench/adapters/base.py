"""Adapter contract.

    qualify(config, requirements) -> QualificationReport
    prepare(config, conversation, parent_snapshot, ctx) -> PreparedInvocation
    execute(config, prepared, ctx, artifact_sink) -> NativeExecution
    extract(native_execution) -> ExtractedResponse
    fork_state(prepared, native_execution, extracted) -> ContinuationSnapshot

Common return types carry identifiers, status, final answer, observations, and artifact references.
Provider-specific bodies and continuation state remain native dictionaries. The shared interface does
not impose a common sampling or reasoning scheme.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import Field

from ..domain.common import Frozen, utc_now
from ..domain.configuration import ModelConfiguration
from ..domain.execution import ArtifactRef, CostIncurred, FailureKind, Observation
from ..domain.scenario import Message, Role


class AdapterError(Exception):
    def __init__(self, kind: FailureKind, message: str, cost_incurred: CostIncurred = CostIncurred.NO) -> None:
        super().__init__(message)
        self.kind = kind
        self.cost_incurred = cost_incurred


class InvocationContext(Frozen):
    """Harness metadata for one invocation. Never forwarded to the model as content."""

    experiment_id: str
    task_id: str
    attempt_id: str
    attempt_no: int
    round: int
    purpose: Literal["scored", "qualification", "warmup", "judge", "classification", "extraction"] = "scored"


class ContinuationSnapshot(Frozen):
    """Immutable baseline state from which follow-up branches fork.

    `messages` is the canonical semantic conversation ending with the assistant turn. `native_state`
    holds whatever the adapter needs to continue faithfully (signed thinking blocks, provider response
    references, special tokens). A plain assistant string is not the continuation contract.
    """

    adapter: str
    configuration_key: str
    messages: tuple[Message, ...]
    native_state: dict[str, Any]

    def verify_shape(self) -> None:
        if not self.messages or self.messages[-1].role is not Role.ASSISTANT:
            raise AdapterError(FailureKind.INVALID_STATE, "continuation snapshot must end with an assistant turn")


class PreparedInvocation(Frozen):
    adapter: str
    messages: tuple[Message, ...]
    native_request: dict[str, Any]
    input_tokens: int | None = None
    input_tokens_source: str | None = None


class NativeExecution(Frozen):
    status: Literal["completed", "failed"]
    native_response: dict[str, Any] | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    failure_kind: FailureKind | None = None
    error: dict[str, Any] | None = None
    cost_incurred: CostIncurred = CostIncurred.NO
    observations: dict[str, Observation] = Field(default_factory=dict)


class ExtractedResponse(Frozen):
    final_answer: str
    truncated: bool
    finish_reason: str | None
    reasoning_text: str | None = None


class QualificationRequirements(Frozen):
    """Workload envelope a configuration must support: baseline plus the longest follow-up branch."""

    baseline_prompt_chars: int
    longest_follow_up_chars: int
    output_budget_tokens: int | None
    requires_continuation: bool = True


class QualificationCheck(Frozen):
    name: str
    passed: bool | None  # None: not evaluated
    detail: str


class QualificationReport(Frozen):
    configuration_key: str
    adapter: str
    passed: bool
    checks: tuple[QualificationCheck, ...]
    requirements: QualificationRequirements
    # Evidence of live probe calls made during qualification (redacted wire exchanges, usage, cost).
    probes: tuple[dict[str, Any], ...] = ()
    created_at: str = Field(default_factory=utc_now)

    @classmethod
    def from_checks(
        cls,
        config: ModelConfiguration,
        adapter: str,
        requirements: QualificationRequirements,
        checks: list[QualificationCheck],
        probes: list[dict[str, Any]] | None = None,
    ) -> QualificationReport:
        return cls(
            configuration_key=config.key,
            adapter=adapter,
            passed=all(c.passed is True for c in checks),
            checks=tuple(checks),
            requirements=requirements,
            probes=tuple(probes or ()),
        )


class ArtifactSink(Protocol):
    def write(self, name: str, data: bytes | str) -> ArtifactRef: ...

    def write_json(self, name: str, value: Any) -> ArtifactRef: ...


class InferenceAdapter(Protocol):
    name: str

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport: ...

    def prepare(
        self,
        config: ModelConfiguration,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        ctx: InvocationContext,
    ) -> PreparedInvocation: ...

    def execute(
        self,
        config: ModelConfiguration,
        prepared: PreparedInvocation,
        ctx: InvocationContext,
        sink: ArtifactSink,
    ) -> NativeExecution: ...

    def extract(self, execution: NativeExecution) -> ExtractedResponse: ...

    def fork_state(
        self,
        prepared: PreparedInvocation,
        execution: NativeExecution,
        extracted: ExtractedResponse,
    ) -> ContinuationSnapshot: ...
