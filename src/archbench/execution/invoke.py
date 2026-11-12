"""One adapter invocation with evidence capture.

Shared by candidate generation and by evaluation-side model calls (judges, classifiers) so that every
model call is captured the same way: canonical messages, redacted native request and response, final
output, metrics, and errors are written as immutable artifacts. The caller registers the artifacts and
publishes its own record in one database transaction after the evidence is durable.
"""

from __future__ import annotations

import json
import time

from pydantic import Field

from ..adapters.base import (
    AdapterError,
    ArtifactSink,
    ContinuationSnapshot,
    ExtractedResponse,
    InferenceAdapter,
    InvocationContext,
    NativeExecution,
    PreparedInvocation,
)
from ..adapters.redaction import redact, redact_text
from ..domain.common import Frozen
from ..domain.configuration import ModelConfiguration
from ..domain.execution import CostIncurred, FailureKind, Observation
from ..domain.scenario import Message


class InvocationOutcome(Frozen):
    failure_kind: FailureKind | None = None
    cost_incurred: CostIncurred = CostIncurred.NO
    error: str | None = None
    extracted: ExtractedResponse | None = None
    snapshot: ContinuationSnapshot | None = None
    observations: dict[str, Observation] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.extracted is not None and self.failure_kind is None


def context_violation(config: ModelConfiguration, prepared: PreparedInvocation) -> str | None:
    if prepared.input_tokens is None or config.context_limit is None:
        return None
    reserve = config.explicit_int("max_output_tokens") or 0
    if prepared.input_tokens + reserve > config.context_limit:
        return (
            f"input {prepared.input_tokens} tokens + reserved output {reserve} exceeds "
            f"validated context limit {config.context_limit}"
        )
    return None


def invoke(
    adapter: InferenceAdapter,
    config: ModelConfiguration,
    messages: tuple[Message, ...],
    parent: ContinuationSnapshot | None,
    ctx: InvocationContext,
    sink: ArtifactSink,
    *,
    answer_name: str = "final-answer.txt",
    fork: bool = True,
) -> InvocationOutcome:
    execution: NativeExecution | None = None
    extracted: ExtractedResponse | None = None
    snapshot: ContinuationSnapshot | None = None
    failure: FailureKind | None = None
    cost = CostIncurred.NO
    error: str | None = None
    wall_ms: float | None = None
    dispatched = False
    try:
        prepared = adapter.prepare(config, messages, parent, ctx)
        sink.write_json("canonical-messages.json", [m.model_dump(mode="json") for m in prepared.messages])
        sink.write_json("native-request.json", redact(prepared.native_request))
        violation = context_violation(config, prepared)
        if violation is not None:
            failure, error = FailureKind.CONTEXT_EXCEEDED, violation
        else:
            dispatched = True
            started = time.perf_counter()
            try:
                execution = adapter.execute(config, prepared, ctx, sink)
            finally:
                wall_ms = (time.perf_counter() - started) * 1000
            cost = execution.cost_incurred
            if execution.native_response is not None:
                sink.write_json("native-response.json", redact(execution.native_response))
            if execution.status == "completed":
                extracted = adapter.extract(execution)
                sink.write(answer_name, extracted.final_answer)
                if fork:
                    snapshot = adapter.fork_state(prepared, execution, extracted)
                    snapshot = snapshot.model_copy(update={"configuration_key": config.key})
                    sink.write_json("continuation-state.json", snapshot.model_dump(mode="json"))
                if extracted.reasoning_text is not None:
                    sink.write("reasoning.txt", extracted.reasoning_text)
            else:
                failure = execution.failure_kind or FailureKind.PROVIDER_ERROR
                error = redact_text(json.dumps(execution.error, sort_keys=True)) if execution.error else None
    except AdapterError as exc:
        failure, error = exc.kind, redact_text(str(exc))
        cost = exc.cost_incurred if exc.cost_incurred is not CostIncurred.NO else cost
        extracted = snapshot = None
    except Exception as exc:  # noqa: BLE001 - recorded as evidence, never swallowed silently
        failure, error = FailureKind.HARNESS_ERROR, redact_text(f"{type(exc).__name__}: {exc}")
        extracted = snapshot = None
        if dispatched and execution is None:
            cost = CostIncurred.UNKNOWN

    observations = dict(execution.observations) if execution else {}
    observations["client_wall_ms"] = (
        Observation(value=round(wall_ms, 3), unit="ms", source="controller perf_counter")
        if wall_ms is not None
        else Observation.missing("ms", "controller perf_counter", "request was not dispatched")
    )
    sink.write_json("metrics.json", {k: v.model_dump(mode="json") for k, v in sorted(observations.items())})
    if failure is not None:
        sink.write_json("error.json", {"failure_kind": failure.value, "summary": error})
    return InvocationOutcome(
        failure_kind=failure,
        cost_incurred=cost,
        error=error,
        extracted=extracted,
        snapshot=snapshot,
        observations=observations,
    )
