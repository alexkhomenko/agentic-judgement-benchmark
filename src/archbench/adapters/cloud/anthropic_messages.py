"""Native Anthropic Messages API adapter (official `anthropic` SDK).

Evidence and control:
- Requests go through the SDK, but a capturing transport records every wire exchange after SDK
  serialization: exact request and response bodies, status, and headers (redacted before persistence).
- SDK retries are disabled (`max_retries=0`); every retry is a new harness attempt.
- Only explicitly configured parameters are sent. The adapter never substitutes a model or enables
  server-side fallbacks; a response served by a different model is recorded as a failure.
- Continuation state holds the exact native request messages plus the assistant content blocks
  (including signed thinking blocks), which are replayed unchanged on follow-up branches.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_tokens
    effort             explicit or omitted  -> output_config.effort
    thinking           explicit or omitted  -> thinking
adapter_options:
    timeout_s: <float>                       request timeout (default 600)
    pricing: {input_usd_per_mtok, output_usd_per_mtok, cache_read_usd_per_mtok?,
              cache_write_usd_per_mtok?, source, verified_on}
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2

from ...domain.configuration import ModelConfiguration, ParameterState, Track
from ...domain.execution import CostIncurred, FailureKind, Observation
from ...domain.scenario import Message, Role
from ..base import (
    AdapterError,
    ArtifactSink,
    ContinuationSnapshot,
    ExtractedResponse,
    InvocationContext,
    NativeExecution,
    PreparedInvocation,
    QualificationCheck,
    QualificationReport,
    QualificationRequirements,
)
from .common import CollectingSink, context_check, explicit, failed
from .wire import CapturingTransport, wire_record

STATE_FORMAT = "anthropic-messages-v1"
PROBE_MAX_TOKENS = 2048
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "effort", "thinking"})
TransportFactory = Callable[[], httpx2.BaseTransport]


class AnthropicAdapter:
    name = "anthropic"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        # api_key=None lets the SDK resolve credentials (ANTHROPIC_API_KEY, auth token, or `ant` profile).
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> anthropic.Anthropic:
        timeout = float(str(config.adapter_options.get("timeout_s", 600)))
        return anthropic.Anthropic(
            api_key=self._api_key,
            max_retries=0,
            timeout=timeout,
            http_client=anthropic.DefaultHttpxClient(transport=transport),
        )

    def preflight(self, config: ModelConfiguration) -> str | None:
        """Problem that would make every call fail before dispatch, or None. Sends no request."""
        try:
            client = self._client(config, CapturingTransport(self._transport_factory()))
        except anthropic.CredentialsError as exc:
            return f"Anthropic credentials could not be loaded: {exc}"
        if client.api_key is None and client.auth_token is None and client.credentials is None:
            return ("no Anthropic credentials found: add anthropic.api_key to secrets.local.yaml "
                    "(see secrets.example.yaml), set ANTHROPIC_API_KEY, or run `ant auth login`")
        return None

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        explicit = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
        unsupported = sorted(explicit - SUPPORTED_PARAMETERS)
        model_check, limits = self._model_check(config)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "anthropic",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(
                name="parameter_support",
                passed=not unsupported and "max_output_tokens" in explicit,
                detail=f"unsupported: {unsupported}" if unsupported else
                ("max_output_tokens must be explicit" if "max_output_tokens" not in explicit else "ok"),
            ),
            model_check,
            context_check(config, requirements, limits.get("max_input_tokens")),
        ]
        probes: list[dict[str, Any]] = []
        if requirements.requires_continuation:
            if model_check.passed is not True:
                checks.append(QualificationCheck(name="continuation", passed=None,
                                                 detail="not evaluated: model identity was not verified"))
            else:
                checks.append(self._continuation_check(config, probes))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _continuation_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Live probe: one baseline, then two sibling follow-ups forked from the same snapshot.

        Passes only if both follow-ups complete, each replays the baseline assistant content exactly
        (including thinking-block signatures), neither sees its sibling, and the snapshot is unchanged.
        The probe caps max_tokens at PROBE_MAX_TOKENS; all other parameters are the configuration's own.
        """
        budget = explicit(config, "max_output_tokens")
        probe_tokens = min(budget, PROBE_MAX_TOKENS) if isinstance(budget, int) else PROBE_MAX_TOKENS
        params = dict(config.parameters)
        params["max_output_tokens"] = params["max_output_tokens"].model_copy(update={"value": probe_tokens})
        probe = config.model_copy(update={"parameters": params})

        def run(step: str, messages: tuple[Message, ...], parent: ContinuationSnapshot | None
                ) -> tuple[PreparedInvocation, NativeExecution]:
            ctx = InvocationContext(experiment_id="qualification", task_id=f"qualification:{config.key}",
                                    attempt_id=step, attempt_no=1, round=1 if parent is None else 2,
                                    purpose="qualification")
            sink = CollectingSink()
            prepared = self.prepare(probe, messages, parent, ctx)
            execution = self.execute(probe, prepared, ctx, sink)
            cost = execution.observations.get("api_cost_usd")
            probes.append({
                "step": step, "status": execution.status, "request_id": execution.request_id,
                "failure_kind": execution.failure_kind.value if execution.failure_kind else None,
                "cost_incurred": execution.cost_incurred.value,
                "cost_usd": cost.value if cost else None,
                "max_tokens": probe_tokens,
                "wire": sink.items.get("wire-exchanges.json"),
            })
            return prepared, execution

        base_messages = (Message(role=Role.SYSTEM, content="You are answering a harness qualification probe."),
                         Message(role=Role.USER, content="Reply with exactly one word: baseline"))
        try:
            prepared, execution = run("baseline", base_messages, None)
            if execution.status != "completed":
                return QualificationCheck(name="continuation", passed=False,
                                          detail=f"baseline failed: {execution.failure_kind} {execution.error}")
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = json.dumps(snapshot.native_state, sort_keys=True)
            baseline_content = (execution.native_response or {}).get("content")
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = run(f"branch-{word}", follow, snapshot)
                sent = child.native_request["params"]["messages"]
                if sent[1] != {"role": "assistant", "content": baseline_content}:
                    return QualificationCheck(name="continuation", passed=False,
                                              detail=f"branch {word} did not replay the baseline content exactly")
                if word == "beta" and "alpha" in json.dumps(sent):
                    return QualificationCheck(name="continuation", passed=False, detail="sibling branch leaked")
                if result.status != "completed":
                    return QualificationCheck(name="continuation", passed=False,
                                              detail=f"branch {word} failed: {result.failure_kind} {result.error}")
            if json.dumps(snapshot.native_state, sort_keys=True) != before:
                return QualificationCheck(name="continuation", passed=False, detail="forking mutated baseline state")
        except AdapterError as exc:
            return QualificationCheck(name="continuation", passed=False, detail=str(exc))
        thinking = any(b.get("type") == "thinking" for b in baseline_content or [] if isinstance(b, dict))
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact native replay"
                   + (" (including thinking blocks)" if thinking else ""),
        )

    def _model_check(self, config: ModelConfiguration) -> tuple[QualificationCheck, dict[str, Any]]:
        """Verify model identity and limits against the first-party Models API."""
        transport = CapturingTransport(self._transport_factory())
        try:
            info = self._client(config, transport).models.retrieve(config.model)
        except (TypeError, anthropic.CredentialsError) as exc:  # SDK: no credentials resolved
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"), {}
        except anthropic.APIError as exc:
            return QualificationCheck(name="model_identity", passed=False, detail=f"{type(exc).__name__}: {exc}"), {}
        max_input = getattr(info, "max_input_tokens", None)
        max_output = getattr(info, "max_tokens", None)
        requested = explicit(config, "max_output_tokens")
        problems = []
        if info.id != config.model:
            problems.append(f"API reports id {info.id}")
        if max_input is not None and config.context_limit is not None and config.context_limit > max_input:
            problems.append(f"context_limit {config.context_limit} > max_input_tokens {max_input}")
        if max_output is not None and isinstance(requested, int) and requested > max_output:
            problems.append(f"max_output_tokens {requested} > model max_tokens {max_output}")
        check = QualificationCheck(
            name="model_identity", passed=not problems,
            detail="; ".join(problems) or f"{info.id}: max_input_tokens={max_input} max_tokens={max_output}",
        )
        return check, {"max_input_tokens": max_input, "max_tokens": max_output}

    # ----- invocation ------------------------------------------------------------------------------

    def prepare(
        self,
        config: ModelConfiguration,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        ctx: InvocationContext,
    ) -> PreparedInvocation:
        max_tokens = explicit(config, "max_output_tokens")
        if not isinstance(max_tokens, int):
            raise AdapterError(FailureKind.INVALID_STATE, "max_output_tokens must be an explicit integer")
        system = [m.content for m in messages if m.role is Role.SYSTEM]
        if any(m.role is Role.SYSTEM for m in messages[len(system):]):
            raise AdapterError(FailureKind.INVALID_STATE, "system messages must lead the conversation")
        if parent is not None:
            parent.verify_shape()
            state = parent.native_state
            if state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if state.get("model") != config.model:
                # Thinking blocks are bound to the model that produced them.
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state belongs to a different model")
            if messages[: len(parent.messages)] != parent.messages:
                raise AdapterError(FailureKind.INVALID_STATE, "conversation does not extend the baseline snapshot")
            native_messages = copy.deepcopy(state["messages"])
            new = messages[len(parent.messages):]
        else:
            native_messages = []
            new = messages[len(system):]
        native_messages += [{"role": m.role.value, "content": m.content} for m in new]

        params: dict[str, Any] = {"model": config.model, "max_tokens": max_tokens, "messages": native_messages}
        if parent is not None and parent.native_state.get("system") is not None:
            params["system"] = parent.native_state["system"]
        elif system:
            params["system"] = "\n\n".join(system)
        if (effort := explicit(config, "effort")) is not None:
            params["output_config"] = {"effort": effort}
        if (thinking := explicit(config, "thinking")) is not None:
            params["thinking"] = thinking
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": "POST /v1/messages",
                "sdk": {"package": "anthropic", "version": anthropic.__version__, "max_retries": 0},
                "params": params,
            },
        )

    def execute(
        self,
        config: ModelConfiguration,
        prepared: PreparedInvocation,
        ctx: InvocationContext,
        sink: ArtifactSink,
    ) -> NativeExecution:
        params = prepared.native_request["params"]
        transport = CapturingTransport(self._transport_factory())
        try:
            return self._execute(config, params, transport)
        finally:
            sink.write_json("wire-exchanges.json", wire_record(transport.exchanges))

    def _execute(self, config: ModelConfiguration, params: dict[str, Any], transport: CapturingTransport
                 ) -> NativeExecution:
        def dispatched() -> bool:
            return bool(transport.exchanges)

        try:
            self._client(config, transport).messages.create(**params)
        except (TypeError, anthropic.CredentialsError) as exc:
            if dispatched():
                raise
            # No credential resolved (SDK raises at construction or before sending).
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except anthropic.APITimeoutError as exc:
            # The request may have been processed and billed.
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except anthropic.APIConnectionError as exc:
            cost = CostIncurred.UNKNOWN if dispatched() else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)
        except anthropic.APIStatusError as exc:
            if isinstance(exc, anthropic.RateLimitError):
                kind = FailureKind.RATE_LIMITED
            elif exc.status_code >= 500:
                kind = FailureKind.TRANSIENT_TRANSPORT
            else:
                kind = FailureKind.PROVIDER_ERROR
            return failed(kind, type(exc).__name__, str(exc), CostIncurred.NO,
                           status=exc.status_code, request_id=exc.response.headers.get("request-id"))

        exchange = transport.exchanges[-1]
        sent = json.loads(exchange["request_body"])
        drift = sorted(k for k in set(params) | set(sent) if sent.get(k) != params.get(k))
        if drift:
            raise AdapterError(FailureKind.HARNESS_ERROR, f"SDK changed request fields on the wire: {drift}",
                               CostIncurred.YES)
        body = json.loads(exchange["response_body"])
        request_id = exchange["response_headers"].get("request-id")
        observations = _observations(config, body.get("usage") or {})
        if body.get("model") != config.model:
            return NativeExecution(
                status="failed", native_response=body, request_id=request_id, cost_incurred=CostIncurred.YES,
                failure_kind=FailureKind.PROVIDER_ERROR, observations=observations,
                error={"type": "model_mismatch", "requested": config.model, "served": body.get("model")},
            )
        stop_reason = body.get("stop_reason")
        if stop_reason == "refusal":
            return NativeExecution(
                status="failed", native_response=body, request_id=request_id, finish_reason=stop_reason,
                cost_incurred=CostIncurred.YES, failure_kind=FailureKind.REFUSAL, observations=observations,
                error={"type": "refusal", "stop_details": body.get("stop_details")},
            )
        return NativeExecution(
            status="completed", native_response=body, finish_reason=stop_reason, request_id=request_id,
            cost_incurred=CostIncurred.YES, observations=observations,
        )

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        body = execution.native_response
        if execution.status != "completed" or body is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        blocks = body.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        thinking = [b["thinking"] for b in blocks if b.get("type") == "thinking" and b.get("thinking")]
        return ExtractedResponse(
            final_answer=text,
            truncated=body.get("stop_reason") == "max_tokens",
            finish_reason=body.get("stop_reason"),
            reasoning_text="\n\n".join(thinking) if thinking else None,
        )

    def fork_state(
        self, prepared: PreparedInvocation, execution: NativeExecution, extracted: ExtractedResponse
    ) -> ContinuationSnapshot:
        body = execution.native_response or {}
        params = prepared.native_request["params"]
        return ContinuationSnapshot(
            adapter=self.name,
            configuration_key=str(body.get("model")),
            messages=prepared.messages + (Message(role=Role.ASSISTANT, content=extracted.final_answer),),
            native_state={
                "format": STATE_FORMAT,
                "model": body.get("model"),
                "response_id": body.get("id"),
                "system": params.get("system"),
                # Assistant content is replayed exactly as returned, including signed thinking blocks.
                "messages": copy.deepcopy(params["messages"])
                + [{"role": "assistant", "content": copy.deepcopy(body.get("content") or [])}],
            },
        )


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "anthropic usage"
    obs: dict[str, Observation] = {}
    for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = usage.get(name)
        obs[name] = (Observation(value=int(value), unit="tokens", source=source) if value is not None
                     else Observation.missing("tokens", source, "not reported in usage"))
    # Thinking is billed as output and is included in output_tokens; the API reports no separate count.
    obs["reasoning_tokens"] = Observation.missing("tokens", source, "included in output_tokens; not reported separately")
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")
    rates = {
        "input_tokens": pricing.get("input_usd_per_mtok"),
        "output_tokens": pricing.get("output_usd_per_mtok"),
        "cache_read_input_tokens": pricing.get("cache_read_usd_per_mtok"),
        "cache_creation_input_tokens": pricing.get("cache_write_usd_per_mtok"),
    }
    total = 0.0
    for name, rate in rates.items():
        tokens = obs[name].value
        if tokens is None:
            if name in ("input_tokens", "output_tokens"):
                return Observation.missing("usd", "configuration pricing", f"{name} not reported")
            continue
        if not tokens:
            continue
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", f"no price for {name}")
        total += tokens * rate / 1_000_000
    return Observation(value=round(total, 8), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')})")

