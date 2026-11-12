"""Native OpenAI Responses API adapter (official `openai` SDK, `POST /v1/responses`).

Evidence and control:
- Requests go through the SDK, but a capturing transport records every wire exchange after SDK
  serialization: exact request and response bodies, status, and headers (redacted before persistence).
- SDK retries are disabled (`max_retries=0`); every retry is a new harness attempt.
- Only explicitly configured parameters are sent, plus two fixed harness settings: `store: false`, so
  no response is retained server-side for later retrieval, and `include: ["reasoning.encrypted_content"]`,
  so reasoning items can be replayed statelessly. The adapter never substitutes a model; a response
  served by a different model is recorded as a failure.
- Continuation is stateless and exact: the snapshot holds the native input items sent plus every output
  item returned (reasoning items with their encrypted content, and assistant messages), which follow-up
  branches replay unchanged. `previous_response_id` is never used, so branches cannot share mutable
  server-side state and nothing depends on server-side retention.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_output_tokens (includes reasoning tokens)
    reasoning_effort   explicit or omitted  -> reasoning.effort
    reasoning_summary  explicit or omitted  -> reasoning.summary
    verbosity          explicit or omitted  -> text.verbosity
    temperature        explicit or omitted  -> temperature
adapter_options:
    timeout_s: <float>                       request timeout (default 600)
    replay_reasoning: <bool>                 request encrypted reasoning for replay (default true)
    pricing: {input_usd_per_mtok, output_usd_per_mtok, cached_input_usd_per_mtok?,
              cache_write_30m_usd_per_mtok?, cache_write_12h_usd_per_mtok?,
              long_context?: {threshold_input_tokens, <same rate keys>}, source, verified_on}

Cost assumes cached and cache-write tokens are subsets of `input_tokens`, as the usage breakdown is
documented. A request beyond the long-context threshold without long-context rates has unknown cost.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import httpx2
import openai

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

STATE_FORMAT = "openai-responses-v1"
PROBE_MAX_TOKENS = 2048
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "reasoning_effort", "reasoning_summary", "verbosity",
                                  "temperature"})
TransportFactory = Callable[[], httpx2.BaseTransport]


class OpenAIAdapter:
    """Also the base for providers that serve the same Responses API (see `xai_responses`): subclasses set
    the class attributes below and may override `_client` and `_model_check`."""

    name = "openai"
    provider = "openai"
    base_url: str | None = None  # None: the SDK default
    credentials_hint = ("no OpenAI credentials found: add openai.api_key to secrets.local.yaml "
                        "(see secrets.example.yaml) or set OPENAI_API_KEY")

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        # api_key=None lets the SDK resolve credentials from OPENAI_API_KEY.
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> openai.OpenAI:
        timeout = float(str(config.adapter_options.get("timeout_s", 600)))
        return openai.OpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=timeout,
            http_client=openai.DefaultHttpxClient(transport=transport),
        )

    def preflight(self, config: ModelConfiguration) -> str | None:
        """Problem that would make every call fail before dispatch, or None. Sends no request."""
        try:
            self._client(config, CapturingTransport(self._transport_factory()))
        except openai.OpenAIError:
            return self.credentials_hint
        return None

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
        unsupported = sorted(explicit_names - SUPPORTED_PARAMETERS)
        model_check = self._model_check(config)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == self.provider,
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(
                name="parameter_support",
                passed=not unsupported and "max_output_tokens" in explicit_names,
                detail=f"unsupported: {unsupported}" if unsupported else
                ("max_output_tokens must be explicit" if "max_output_tokens" not in explicit_names else "ok"),
            ),
            model_check,
            # The Models API reports no limits, so the envelope uses the configured (documented) limit.
            context_check(config, requirements, None),
        ]
        probes: list[dict[str, Any]] = []
        if requirements.requires_continuation:
            if model_check.passed is not True:
                checks.append(QualificationCheck(name="continuation", passed=None,
                                                 detail="not evaluated: model identity was not verified"))
            else:
                checks.append(self._continuation_check(config, probes))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _model_check(self, config: ModelConfiguration) -> QualificationCheck:
        transport = CapturingTransport(self._transport_factory())
        try:
            info = self._client(config, transport).models.retrieve(config.model)
        except openai.APIStatusError as exc:
            return QualificationCheck(name="model_identity", passed=False, detail=f"{type(exc).__name__}: {exc}")
        except openai.APIConnectionError as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}")
        except openai.OpenAIError as exc:  # no credentials resolved
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}")
        if info.id != config.model:
            return QualificationCheck(name="model_identity", passed=False, detail=f"API reports id {info.id}")
        return QualificationCheck(name="model_identity", passed=True,
                                  detail=f"{info.id} (owned_by={info.owned_by}); limits not reported by the API")

    def _continuation_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Live probe: one baseline, then two sibling follow-ups forked from the same snapshot.

        Passes only if both follow-ups complete, each replays the baseline output items exactly (including
        encrypted reasoning), neither sees its sibling, and the snapshot is unchanged. The probe caps
        max_output_tokens at PROBE_MAX_TOKENS; all other parameters are the configuration's own.
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
                "max_output_tokens": probe_tokens,
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
            output = (execution.native_response or {}).get("output") or []
            if _replays_reasoning(config) and any(
                i.get("type") == "reasoning" and not i.get("encrypted_content") for i in output
            ):
                return QualificationCheck(name="continuation", passed=False,
                                          detail="reasoning items carry no encrypted_content; stateless replay "
                                                 "would drop the model's reasoning")
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = json.dumps(snapshot.native_state, sort_keys=True)
            base_input = prepared.native_request["params"]["input"]
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = run(f"branch-{word}", follow, snapshot)
                sent = child.native_request["params"]["input"]
                if sent[: len(base_input) + len(output)] != base_input + output:
                    return QualificationCheck(name="continuation", passed=False,
                                              detail=f"branch {word} did not replay the baseline output exactly")
                if word == "beta" and "alpha" in json.dumps(sent):
                    return QualificationCheck(name="continuation", passed=False, detail="sibling branch leaked")
                if result.status != "completed":
                    return QualificationCheck(name="continuation", passed=False,
                                              detail=f"branch {word} failed: {result.failure_kind} {result.error}")
            if json.dumps(snapshot.native_state, sort_keys=True) != before:
                return QualificationCheck(name="continuation", passed=False, detail="forking mutated baseline state")
        except AdapterError as exc:
            return QualificationCheck(name="continuation", passed=False, detail=str(exc))
        reasoning = any(i.get("type") == "reasoning" for i in output)
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact stateless replay"
                   + (" (including encrypted reasoning)" if reasoning else ""),
        )

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
                # Encrypted reasoning is bound to the model that produced it.
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state belongs to a different model")
            if messages[: len(parent.messages)] != parent.messages:
                raise AdapterError(FailureKind.INVALID_STATE, "conversation does not extend the baseline snapshot")
            items = copy.deepcopy(state["input"])
            instructions = state.get("instructions")
            new = messages[len(parent.messages):]
        else:
            items = []
            instructions = "\n\n".join(system) if system else None
            new = messages[len(system):]
        for m in new:
            if m.role is Role.ASSISTANT:
                raise AdapterError(FailureKind.INVALID_STATE,
                                   "assistant turns must come from a continuation snapshot, not plain text")
            items.append({"role": m.role.value, "content": m.content})

        params: dict[str, Any] = {"model": config.model, "input": items, "max_output_tokens": max_tokens,
                                  "store": False}
        if instructions is not None:
            params["instructions"] = instructions
        if _replays_reasoning(config):
            params["include"] = ["reasoning.encrypted_content"]
        reasoning = {k: v for k, v in (("effort", explicit(config, "reasoning_effort")),
                                       ("summary", explicit(config, "reasoning_summary"))) if v is not None}
        if reasoning:
            params["reasoning"] = reasoning
        if (verbosity := explicit(config, "verbosity")) is not None:
            params["text"] = {"verbosity": verbosity}
        if (temperature := explicit(config, "temperature")) is not None:
            params["temperature"] = temperature
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": "POST /v1/responses",
                "sdk": {"package": "openai", "version": openai.__version__, "max_retries": 0},
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
            self._client(config, transport).responses.create(**params)
        except openai.APITimeoutError as exc:
            # The request may have been processed and billed.
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except openai.APIConnectionError as exc:
            cost = CostIncurred.UNKNOWN if dispatched() else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)
        except openai.APIStatusError as exc:
            code = _error_code(exc)
            if exc.status_code == 429 and code == "insufficient_quota":
                kind = FailureKind.PROVIDER_ERROR  # out of credit: retrying cannot help
            elif exc.status_code == 429:
                kind = FailureKind.RATE_LIMITED
            elif code == "context_length_exceeded":
                kind = FailureKind.CONTEXT_EXCEEDED
            elif exc.status_code >= 500:
                kind = FailureKind.TRANSIENT_TRANSPORT
            else:
                kind = FailureKind.PROVIDER_ERROR
            return failed(kind, type(exc).__name__, str(exc), CostIncurred.NO, status=exc.status_code, code=code,
                          request_id=exc.response.headers.get("x-request-id"))
        except openai.OpenAIError as exc:
            if dispatched():
                raise
            # No credential resolved (the SDK raises at client construction, before sending).
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)

        exchange = transport.exchanges[-1]
        sent = json.loads(exchange["request_body"])
        drift = sorted(k for k in set(params) | set(sent) if sent.get(k) != params.get(k))
        if drift:
            raise AdapterError(FailureKind.HARNESS_ERROR, f"SDK changed request fields on the wire: {drift}",
                               CostIncurred.YES)
        body = json.loads(exchange["response_body"])
        request_id = exchange["response_headers"].get("x-request-id")
        observations = _observations(config, body.get("usage") or {})

        def failure(kind: FailureKind, error: dict[str, Any]) -> NativeExecution:
            return NativeExecution(status="failed", native_response=body, request_id=request_id,
                                   finish_reason=_finish_reason(body), cost_incurred=CostIncurred.YES,
                                   failure_kind=kind, observations=observations, error=error)

        if body.get("model") != config.model:
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "model_mismatch", "requested": config.model, "served": body.get("model")})
        status = body.get("status")
        reason = (body.get("incomplete_details") or {}).get("reason")
        refusals = [c.get("refusal") for i in body.get("output") or [] if i.get("type") == "message"
                    for c in i.get("content") or [] if c.get("type") == "refusal"]
        if refusals or (status == "incomplete" and reason == "content_filter"):
            return failure(FailureKind.REFUSAL, {"type": "refusal", "refusals": refusals, "reason": reason})
        if status == "failed":
            return failure(FailureKind.PROVIDER_ERROR, {"type": "response_failed", "error": body.get("error")})
        if status == "incomplete" and reason != "max_output_tokens":
            return failure(FailureKind.PROVIDER_ERROR, {"type": "incomplete", "reason": reason})
        if status not in ("completed", "incomplete"):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "unexpected_status", "status": status})
        return NativeExecution(
            status="completed", native_response=body, finish_reason=_finish_reason(body), request_id=request_id,
            cost_incurred=CostIncurred.YES, observations=observations,
        )

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        body = execution.native_response
        if execution.status != "completed" or body is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        items = body.get("output") or []
        text = "".join(
            c.get("text", "") for i in items if i.get("type") == "message" and i.get("role") == "assistant"
            for c in i.get("content") or [] if c.get("type") == "output_text"
        )
        reasoning = [
            part["text"] for i in items if i.get("type") == "reasoning"
            for part in [*(i.get("summary") or []), *(i.get("content") or [])] if part.get("text")
        ]
        return ExtractedResponse(
            final_answer=text,
            truncated=body.get("status") == "incomplete"
            and (body.get("incomplete_details") or {}).get("reason") == "max_output_tokens",
            finish_reason=_finish_reason(body),
            reasoning_text="\n\n".join(reasoning) if reasoning else None,
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
                "response_id": body.get("id"),  # provenance only: never used for continuation
                "instructions": params.get("instructions"),
                # Output items are replayed exactly as returned, including encrypted reasoning.
                "input": copy.deepcopy(params["input"]) + copy.deepcopy(body.get("output") or []),
            },
        )


def _replays_reasoning(config: ModelConfiguration) -> bool:
    return config.adapter_options.get("replay_reasoning", True) is not False


def _error_code(exc: openai.APIStatusError) -> str | None:
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            return str(error["code"])
    return None


def _finish_reason(body: dict[str, Any]) -> str | None:
    status = body.get("status")
    reason = (body.get("incomplete_details") or {}).get("reason")
    return f"{status}:{reason}" if status == "incomplete" and reason else status


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "openai usage"
    details = usage.get("input_tokens_details") or {}
    writes = details.get("cache_write_tokens_details") or {}
    values = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": details.get("cached_tokens"),
        "cache_write_tokens": details.get("cache_write_tokens"),
        "cache_write_30m_tokens": writes.get("ttl_30m_tokens"),
        "cache_write_12h_tokens": writes.get("ttl_12h_tokens"),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if v is not None
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    configured = _cost(config, obs)
    ticks = usage.get("cost_in_usd_ticks")
    if isinstance(ticks, int):
        # Provider-reported cost (xAI: 1 USD = 10^10 ticks) is the billing record; the configured price table
        # is kept beside it as a cross-check.
        obs["api_cost_usd"] = Observation(value=round(ticks / 1e10, 10), unit="usd",
                                          source="provider-reported usage.cost_in_usd_ticks")
        obs["api_cost_usd_configured"] = configured
    else:
        obs["api_cost_usd"] = configured
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")
    if obs["input_tokens"].value is None or obs["output_tokens"].value is None:
        return Observation.missing("usd", "configuration pricing", "input or output tokens not reported")

    def count(name: str) -> int:
        value = obs[name].value
        return int(value) if isinstance(value, int | float) else 0

    input_tokens = count("input_tokens")
    rates: dict[str, Any] = pricing
    long_context = pricing.get("long_context")
    threshold = long_context.get("threshold_input_tokens") if isinstance(long_context, dict) else None
    if isinstance(long_context, dict) and isinstance(threshold, int):
        if input_tokens > threshold:
            rates = long_context
    elif input_tokens > 272_000:
        return Observation.missing("usd", "configuration pricing",
                                   "request may be priced at long-context rates, which are not configured")
    cached, w30, w12 = count("cached_input_tokens"), count("cache_write_30m_tokens"), count("cache_write_12h_tokens")
    if count("cache_write_tokens") - w30 - w12 > 0:
        return Observation.missing("usd", "configuration pricing", "cache writes reported without a retention breakdown")
    parts = [
        (input_tokens - cached - w30 - w12, "input_usd_per_mtok"),
        (cached, "cached_input_usd_per_mtok"),
        (w30, "cache_write_30m_usd_per_mtok"),
        (w12, "cache_write_12h_usd_per_mtok"),
        (count("output_tokens"), "output_usd_per_mtok"),
    ]
    total = 0.0
    for tokens, key in parts:
        if not tokens:
            continue
        rate = rates.get(key)
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", f"no price for {key}")
        total += tokens * rate / 1_000_000
    return Observation(value=round(total, 8), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')})")
