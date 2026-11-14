"""Native Kimi adapter (Moonshot AI, `POST /v1/chat/completions` on api.moonshot.ai).

Chat Completions is the Kimi platform's primary API; its Responses and Anthropic-compatible Messages
endpoints are alternatives this adapter does not use.

Plain HTTP against the documented REST API: the recorded request is exactly the JSON body on the wire.
- No client-side retries: every retry is a new harness attempt.
- Only explicitly configured parameters are sent. K3-series models fix temperature, top_p, n and the
  penalties, and "passing any other value returns an error", so those are not configurable here.
- Kimi offers no dated model id and non-streaming responses carry no system_fingerprint; qualification
  records the model card from `/v1/models` (`created`, `context_length`, `supports_reasoning`).
- Continuation replays the assistant message exactly as returned, including `reasoning_content`: K3
  "requires the complete assistant message returned by the API to be passed back to `messages` as-is".
- Content filtering is an HTTP 400 with error type `content_filter` (input or output), not a finish
  reason. It is recorded as a refusal with unknown cost, since an output-triggered filter may be billed.

Cost uses configured prices. `usage.prompt_tokens` is the total input; cache reads
(`prompt_tokens_details.cached_tokens`), cache writes (`prompt_tokens_details.cache_write_tokens`) and
uncached input sum to it. Cache writes are priced by TTL, which only the `Msh-Usage-Cache-Write-Tokens-5m`
and `-1h` response headers break down; writes reported without that breakdown leave the cost unknown.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_completion_tokens (includes reasoning tokens)
    reasoning_effort   explicit or omitted  -> reasoning_effort ("low" | "high" | "max"; K3 series only)
    thinking           explicit or omitted  -> thinking.type ("enabled" | "disabled"; K2 series only,
                                               K3 rejects it)
adapter_options:
    base_url: https://api.moonshot.ai/v1 (default)    timeout_s: 960 (default)
    pricing: {input_usd_per_mtok (cache miss), cached_input_usd_per_mtok, cache_write_5m_usd_per_mtok,
              cache_write_1h_usd_per_mtok, output_usd_per_mtok, source, verified_on}
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from typing import Any

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

STATE_FORMAT = "kimi-chat-v1"
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
# The gateway gives up after 900 s without a response (HTTP 504); wait a little longer to receive it.
DEFAULT_TIMEOUT_S = 960
PROBE_MAX_TOKENS = 2048
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "reasoning_effort", "thinking"})
CACHE_WRITE_HEADERS = {"5m": "msh-usage-cache-write-tokens-5m", "1h": "msh-usage-cache-write-tokens-1h"}
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no Moonshot credentials found: add moonshot.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set MOONSHOT_API_KEY")


class KimiAdapter:
    name = "kimi"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        base = str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)).rstrip("/") + "/"
        timeout = float(str(config.adapter_options.get("timeout_s", DEFAULT_TIMEOUT_S)))
        return httpx2.Client(base_url=base, transport=transport, timeout=timeout,
                             headers={"authorization": f"Bearer {self._api_key}", "content-type": "application/json"})

    def preflight(self, config: ModelConfiguration) -> str | None:
        return None if self._api_key else _NO_KEY

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
        unsupported = sorted(explicit_names - SUPPORTED_PARAMETERS)
        model_check, context = self._model_check(config)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "moonshot",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(
                name="parameter_support",
                passed=not unsupported and "max_output_tokens" in explicit_names,
                detail=f"unsupported: {unsupported}" if unsupported else
                ("max_output_tokens must be explicit" if "max_output_tokens" not in explicit_names else "ok"),
            ),
            model_check,
            context_check(config, requirements, context),
        ]
        probes: list[dict[str, Any]] = []
        if requirements.requires_continuation:
            if model_check.passed is not True:
                checks.append(QualificationCheck(name="continuation", passed=None,
                                                 detail="not evaluated: model identity was not verified"))
            else:
                checks.append(self._continuation_check(config, probes))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _model_check(self, config: ModelConfiguration) -> tuple[QualificationCheck, int | None]:
        """Verify the model id, context, and reasoning support against the first-party `/models` list."""
        transport = CapturingTransport(self._transport_factory())
        try:
            with self._client(config, transport) as client:
                response = client.get("models")
        except (AdapterError, httpx2.HTTPError) as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"), None
        if response.status_code != 200:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"HTTP {response.status_code}: {_error_message(response)}"), None
        card = next((m for m in response.json().get("data") or [] if m.get("id") == config.model), None)
        if card is None:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"{config.model} is not listed by /models for this account"), None
        problems = []
        context, reasoning = card.get("context_length"), card.get("supports_reasoning")
        if isinstance(context, int) and config.context_limit is not None and config.context_limit > context:
            problems.append(f"context_limit {config.context_limit} > context_length {context}")
        if reasoning is False and (explicit(config, "reasoning_effort") is not None
                                   or explicit(config, "thinking") is not None):
            problems.append("reasoning parameters set but /models reports supports_reasoning=false")
        detail = (f"{config.model} (owned_by {card.get('owned_by')}, created {card.get('created')}): "
                  f"context_length={context}, supports_reasoning={reasoning}")
        return (QualificationCheck(name="model_identity", passed=not problems, detail="; ".join(problems) or detail),
                context if isinstance(context, int) else None)

    def _continuation_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Live probe: one baseline, then two sibling follow-ups forked from the same snapshot."""
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
            returned = _choice(execution.native_response or {}).get("message") or {}
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = json.dumps(snapshot.native_state, sort_keys=True)
            base_sent = prepared.native_request["body"]["messages"]
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = run(f"branch-{word}", follow, snapshot)
                sent = child.native_request["body"]["messages"]
                if sent[: len(base_sent) + 1] != base_sent + [returned]:
                    return QualificationCheck(name="continuation", passed=False,
                                              detail=f"branch {word} did not replay the baseline message exactly")
                if word == "beta" and "alpha" in json.dumps(sent):
                    return QualificationCheck(name="continuation", passed=False, detail="sibling branch leaked")
                if result.status != "completed":
                    return QualificationCheck(name="continuation", passed=False,
                                              detail=f"branch {word} failed: {result.failure_kind} {result.error}")
            if json.dumps(snapshot.native_state, sort_keys=True) != before:
                return QualificationCheck(name="continuation", passed=False, detail="forking mutated baseline state")
        except AdapterError as exc:
            return QualificationCheck(name="continuation", passed=False, detail=str(exc))
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact message replay"
                   + (" (including reasoning_content)" if returned.get("reasoning_content") else ""),
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
        if parent is not None:
            parent.verify_shape()
            state = parent.native_state
            if state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if state.get("model") != config.model:
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state belongs to a different model")
            if messages[: len(parent.messages)] != parent.messages:
                raise AdapterError(FailureKind.INVALID_STATE, "conversation does not extend the baseline snapshot")
            native = copy.deepcopy(state["messages"])
            new, leading = messages[len(parent.messages):], 0
        else:
            native, new = [], messages
            leading = next((i for i, m in enumerate(messages) if m.role is not Role.SYSTEM), len(messages))
        if any(m.role is Role.SYSTEM for m in new[leading:]):
            raise AdapterError(FailureKind.INVALID_STATE, "system messages must lead the conversation")
        for m in new:
            if m.role is Role.ASSISTANT:
                raise AdapterError(FailureKind.INVALID_STATE,
                                   "assistant turns must come from a continuation snapshot, not plain text")
            native.append({"role": m.role.value, "content": m.content})
        body: dict[str, Any] = {"model": config.model, "messages": native, "max_completion_tokens": max_tokens}
        if (effort := explicit(config, "reasoning_effort")) is not None:
            body["reasoning_effort"] = effort
        if (thinking := explicit(config, "thinking")) is not None:
            body["thinking"] = {"type": thinking}
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": "POST /v1/chat/completions",
                "client": {"package": "httpx2", "version": httpx2.__version__, "retries": 0},
                "body": body,
            },
        )

    def execute(
        self,
        config: ModelConfiguration,
        prepared: PreparedInvocation,
        ctx: InvocationContext,
        sink: ArtifactSink,
    ) -> NativeExecution:
        transport = CapturingTransport(self._transport_factory())
        try:
            return self._execute(config, prepared, transport)
        finally:
            sink.write_json("wire-exchanges.json", wire_record(transport.exchanges))

    def _execute(self, config: ModelConfiguration, prepared: PreparedInvocation, transport: CapturingTransport
                 ) -> NativeExecution:
        body = prepared.native_request["body"]
        try:
            with self._client(config, transport) as client:
                response = client.post("chat/completions", content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        if response.status_code != 200:
            kind, cost = _http_failure(response.status_code, _error_type(response))
            return failed(kind, f"http_{response.status_code}", _error_message(response), cost,
                          status=response.status_code, provider_type=_error_type(response))
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data = response.json()
        observations = _observations(config, data.get("usage") or {}, response.headers)
        choice = _choice(data)
        reason = choice.get("finish_reason")

        def failure(kind: FailureKind, error: dict[str, Any]) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=data.get("id"),
                                   finish_reason=reason, cost_incurred=CostIncurred.YES, failure_kind=kind,
                                   observations=observations, error=error)

        if data.get("model") != config.model:
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "model_mismatch", "requested": config.model, "served": data.get("model")})
        if not choice:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_choices"})
        if reason not in ("stop", "length"):  # tool_calls (no tools are sent), client_disconnected, ...
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason,
                               request_id=data.get("id"), cost_incurred=CostIncurred.YES, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        choice = _choice(data)
        message = choice.get("message") or {}
        return ExtractedResponse(
            final_answer=str(message.get("content") or ""),
            truncated=choice.get("finish_reason") == "length",
            finish_reason=choice.get("finish_reason"),
            reasoning_text=message.get("reasoning_content") or None,
        )

    def fork_state(
        self, prepared: PreparedInvocation, execution: NativeExecution, extracted: ExtractedResponse
    ) -> ContinuationSnapshot:
        data = execution.native_response or {}
        body = prepared.native_request["body"]
        return ContinuationSnapshot(
            adapter=self.name,
            configuration_key=str(data.get("model")),
            messages=prepared.messages + (Message(role=Role.ASSISTANT, content=extracted.final_answer),),
            native_state={
                "format": STATE_FORMAT,
                "model": data.get("model"),
                "response_id": data.get("id"),
                # The assistant message (content and reasoning_content) is replayed exactly as returned.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(_choice(data).get("message") or {})],
            },
        )


def _choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    return first if isinstance(first, dict) else {}


def _error_body(response: httpx2.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except json.JSONDecodeError:
        return {}
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def _error_type(response: httpx2.Response) -> str | None:
    kind = _error_body(response).get("type")
    return str(kind) if kind else None


def _error_message(response: httpx2.Response) -> str:
    message = _error_body(response).get("message")
    return str(message)[:500] if message else response.text[:500]


def _http_failure(status: int, error_type: str | None) -> tuple[FailureKind, CostIncurred]:
    """Map an HTTP error to a failure kind, per the platform's error-code table."""
    if error_type == "content_filter":  # input or output judged high risk; an output hit may be billed
        return FailureKind.REFUSAL, CostIncurred.UNKNOWN
    if status == 429:
        if error_type == "exceeded_current_quota_error":  # insufficient balance or disabled account
            return FailureKind.PROVIDER_ERROR, CostIncurred.NO
        return FailureKind.RATE_LIMITED, CostIncurred.NO  # rate_limit_reached_error, engine_overloaded_error
    if status == 504:  # gateway gave up after 900 s; inference may have run
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status >= 500:  # 500 server_error, 503 server_unavailable: retry after a brief wait
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    # 400 invalid_request_error (including input over the context window), 401, 403, 404: retrying cannot help
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any], headers: Mapping[str, str]
                  ) -> dict[str, Observation]:
    source = "kimi usage"
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", usage.get("cached_tokens"))  # the OpenAPI schema also lists it top-level
    writes = details.get("cache_write_tokens")
    prompt = usage.get("prompt_tokens")
    values = {
        "input_tokens": prompt,
        "cached_input_tokens": cached,
        "cache_write_tokens": writes,
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if v is not None
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    for ttl, header in CACHE_WRITE_HEADERS.items():
        raw = headers.get(header)
        obs[f"cache_write_{ttl}_tokens"] = (
            Observation(value=int(raw), unit="tokens", source=f"response header {header}")
            if raw is not None and raw.strip().isdigit()
            else Observation.missing("tokens", f"response header {header}", "header absent"))
    if isinstance(prompt, int) and isinstance(cached, int) and isinstance(writes, int):
        obs["uncached_input_tokens"] = Observation(value=prompt - cached - writes, unit="tokens",
                                                   source=f"{source}: prompt_tokens - cached_tokens - cache_write_tokens")
    else:
        obs["uncached_input_tokens"] = Observation.missing("tokens", source, "input breakdown not reported")
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")
    uncached, output = obs["uncached_input_tokens"].value, obs["output_tokens"].value
    cached, writes = obs["cached_input_tokens"].value, obs["cache_write_tokens"].value
    if not all(isinstance(v, int) for v in (uncached, output, cached, writes)):
        return Observation.missing("usd", "configuration pricing", "input breakdown or output tokens not reported")
    w5, w1h = obs["cache_write_5m_tokens"].value, obs["cache_write_1h_tokens"].value
    if writes:
        if not (isinstance(w5, int) and isinstance(w1h, int)) or w5 + w1h != writes:
            return Observation.missing("usd", "configuration pricing",
                                       "cache writes reported without a matching TTL breakdown")
    parts = [
        (uncached, "input_usd_per_mtok"),
        (cached, "cached_input_usd_per_mtok"),
        (w5 if writes else 0, "cache_write_5m_usd_per_mtok"),
        (w1h if writes else 0, "cache_write_1h_usd_per_mtok"),
        (output, "output_usd_per_mtok"),
    ]
    total = 0.0
    for tokens, key in parts:
        if not tokens:
            continue
        rate = pricing.get(key)
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", f"no price for {key}")
        total += int(tokens) * rate / 1_000_000
    return Observation(value=round(total, 10), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')})")
