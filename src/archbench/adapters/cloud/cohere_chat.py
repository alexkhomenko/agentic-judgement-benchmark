"""Native Cohere adapter (Chat API v2, `POST /v2/chat` on api.cohere.com).

The native v2 API, not Cohere's OpenAI Compatibility API, which drops Cohere-specific parameters and limits
`reasoning_effort` to none/high.

Plain HTTP against the documented REST API: the recorded request is exactly the JSON body on the wire.
- No client-side retries: every retry is a new harness attempt. (The Python SDK retries twice by default.)
- Only explicitly configured parameters are sent. Cohere's own defaults differ from OpenAI-style APIs
  (temperature 0.3, p 0.75, reasoning on for models that support it).
- SERVER-SIDE PROMPT CHANGE: `safety_mode` selects a safety instruction that Cohere inserts into the prompt.
  The default is CONTEXTUAL and Command A models do not support OFF, so every candidate prompt is extended by
  Cohere's published contextual-safety instruction. Configurations record this; it cannot be avoided.
- Model ids are dated snapshots (e.g. `command-a-plus-05-2026`), but responses do not echo the model.
  Qualification verifies the id against the first-party model endpoint (`GET /v1/models/{model}`); individual
  responses cannot be checked.
- `max_tokens` above the model's maximum is silently capped, so the configuration must state the model's
  output limit (`adapter_options.max_output_tokens_limit`) and qualification rejects a larger budget.
- Continuation replays the assistant message exactly as returned: its content blocks (`thinking` and `text`)
  and any other fields the response message carries.
- A 200 response can still be a failed generation: finish_reason ERROR or TIMEOUT. HTTP 498 means a
  deny-listed token in the request or response, recorded as a refusal with unknown cost.

Cost uses configured prices applied to `usage.billed_units` (Cohere does not bill the tokens it adds, so
billed input is lower than `usage.tokens.input_tokens`). `usage.tokens.reasoning_tokens` (undocumented, but
returned) is part of `usage.tokens.output_tokens`, and billed output includes thinking: observed live on
2026-11-16 (82 of 87 output tokens were reasoning; 83 billed). `usage.cached_tokens` is recorded but not
priced separately. Responses carry `x-debug-trace-id` (recorded as the request id) and trial-limit headers
(`x-trial-endpoint-call-remaining`, `x-endpoint-monthly-call-limit`), kept in the wire capture.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens       explicit, required   -> max_tokens
    thinking                explicit or omitted  -> thinking.type ("enabled" | "disabled")
    thinking_token_budget   explicit or omitted  -> thinking.token_budget (requires thinking "enabled")
    safety_mode             explicit or omitted  -> safety_mode ("CONTEXTUAL" | "STRICT" | "OFF")
    temperature             explicit or omitted  -> temperature
    top_p                   explicit or omitted  -> p
    top_k                   explicit or omitted  -> k
    frequency_penalty       explicit or omitted  -> frequency_penalty
    presence_penalty        explicit or omitted  -> presence_penalty
    seed                    explicit or omitted  -> seed (best effort)
adapter_options:
    base_url: https://api.cohere.com (default)    timeout_s: 900 (default)
    max_output_tokens_limit: the model's documented maximum output (required)
    pricing: {input_usd_per_mtok, output_usd_per_mtok, source, verified_on} (per billed token)
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote, urlsplit

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

STATE_FORMAT = "cohere-chat-v2"
DEFAULT_BASE_URL = "https://api.cohere.com"
FIRST_PARTY_HOSTS = frozenset({"api.cohere.com", "api.cohere.ai"})
DEFAULT_TIMEOUT_S = 900
PROBE_MAX_TOKENS = 2048
NATIVE_NAMES = {"temperature": "temperature", "top_p": "p", "top_k": "k", "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty", "seed": "seed", "safety_mode": "safety_mode"}
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "thinking", "thinking_token_budget", *NATIVE_NAMES})
CONTENT_TYPES = frozenset({"text", "thinking"})
REQUEST_ID_HEADERS = ("x-debug-trace-id",)  # undocumented but sent (seen 2026-11-16); falls back to the response `id`
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no Cohere credentials found: add cohere.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set CO_API_KEY")


class CohereAdapter:
    name = "cohere"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        timeout = float(str(config.adapter_options.get("timeout_s", DEFAULT_TIMEOUT_S)))
        return httpx2.Client(base_url=_base_url(config), transport=transport, timeout=timeout,
                             headers={"authorization": f"Bearer {self._api_key}", "content-type": "application/json"})

    def preflight(self, config: ModelConfiguration) -> str | None:
        return None if self._api_key else _NO_KEY

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        parameter_detail = _parameter_problem(config) or "ok"
        base = _base_url(config)
        host, path = (urlsplit(base).hostname or ""), urlsplit(base).path.strip("/")
        endpoint_ok = base.startswith("https://") and host in FIRST_PARTY_HOSTS and not path
        static = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "cohere",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(name="endpoint", passed=endpoint_ok,
                               detail=base if endpoint_ok else f"{base} is not the first-party Cohere API root"),
            QualificationCheck(name="parameter_support", passed=parameter_detail == "ok", detail=parameter_detail),
        ]
        probes: list[dict[str, Any]] = []
        static_failures = [c.name for c in static if c.passed is not True]
        if static_failures:  # never spend calls (Cohere caps them monthly) on a configuration that cannot qualify
            model_check = QualificationCheck(name="model_identity", passed=None,
                                             detail=f"not evaluated: {', '.join(static_failures)} failed")
            context = None
        else:
            model_check, context, card = self._model_check(config)
            probes.append({"step": "model_card", "card": card})
        checks = [*static, model_check, context_check(config, requirements, context)]
        live_name = "continuation" if requirements.requires_continuation else "chat_probe"
        if model_check.passed is True:
            checks.append(self._live_check(config, probes, branches=requirements.requires_continuation))
        else:
            checks.append(QualificationCheck(name=live_name, passed=None,
                                             detail="not evaluated: model identity was not verified"))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _model_check(self, config: ModelConfiguration
                     ) -> tuple[QualificationCheck, int | None, dict[str, Any] | None]:
        """Verify the dated id, chat support, and context length against the first-party model endpoint."""
        transport = CapturingTransport(self._transport_factory())
        try:
            with self._client(config, transport) as client:
                response = client.get(f"v1/models/{quote(config.model, safe='')}")
        except (AdapterError, httpx2.HTTPError) as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"), None, None
        if response.status_code != 200:
            return (QualificationCheck(name="model_identity", passed=False,
                                       detail=f"HTTP {response.status_code}: {_error_message(response)}"), None, None)
        card = response.json()
        if not isinstance(card, dict):
            return QualificationCheck(name="model_identity", passed=False, detail="model card is not an object"), None, None
        problems = []
        if card.get("name") != config.model:
            problems.append(f"model card names {card.get('name')!r}, not {config.model!r}")
        endpoints = card.get("endpoints")
        if isinstance(endpoints, list) and "chat" not in endpoints:
            problems.append(f"model does not serve chat (endpoints {endpoints})")
        if card.get("is_deprecated") is True:
            problems.append("model is deprecated")
        context = card.get("context_length")
        context = int(context) if isinstance(context, int | float) else None
        if context is not None and config.context_limit is not None and config.context_limit > context:
            problems.append(f"context_limit {config.context_limit} > context_length {context}")
        detail = (f"{card.get('name')} (dated snapshot): endpoints={endpoints}, context_length={context}, "
                  f"features={card.get('features')}, sampling_defaults={card.get('sampling_defaults')}")
        return (QualificationCheck(name="model_identity", passed=not problems, detail="; ".join(problems) or detail),
                context, card)

    def _live_check(self, config: ModelConfiguration, probes: list[dict[str, Any]], *, branches: bool
                    ) -> QualificationCheck:
        """Live probe with the configured parameters: one baseline, then (when the scenario branches) two
        sibling follow-ups forked from the same snapshot."""
        name = "continuation" if branches else "chat_probe"
        budget = explicit(config, "max_output_tokens")
        probe_tokens = min(budget, PROBE_MAX_TOKENS) if isinstance(budget, int) else PROBE_MAX_TOKENS
        params = dict(config.parameters)
        params["max_output_tokens"] = params["max_output_tokens"].model_copy(update={"value": probe_tokens})
        if "thinking_token_budget" in params and params["thinking_token_budget"].state is ParameterState.EXPLICIT:
            value = min(int(str(params["thinking_token_budget"].value)), probe_tokens // 2)
            params["thinking_token_budget"] = params["thinking_token_budget"].model_copy(update={"value": value})
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
                return QualificationCheck(name=name, passed=False,
                                          detail=f"baseline probe failed: {execution.failure_kind} {execution.error}")
            returned = (execution.native_response or {}).get("message") or {}
            thinking = any(isinstance(b, dict) and b.get("type") == "thinking" for b in returned.get("content") or [])
            if not branches:
                return QualificationCheck(name=name, passed=True,
                                          detail="baseline completed with the configured parameters"
                                                 + (" (with thinking blocks)" if thinking else ""))
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
            return QualificationCheck(name=name, passed=False, detail=str(exc))
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact message replay"
                   + (" (including thinking blocks)" if thinking else ""),
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
        body: dict[str, Any] = {"model": config.model, "messages": native, "max_tokens": max_tokens}
        thinking: dict[str, Any] = {}
        if (kind := explicit(config, "thinking")) is not None:
            thinking["type"] = kind
        if (budget := explicit(config, "thinking_token_budget")) is not None:
            thinking["token_budget"] = budget
        if thinking:
            body["thinking"] = thinking
        for name, native_name in NATIVE_NAMES.items():
            if (value := explicit(config, name)) is not None:
                body[native_name] = value
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": "POST /v2/chat",
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
                response = client.post("v2/chat", content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        header_id = _request_id(response.headers)
        if response.status_code != 200:
            kind, cost = _http_failure(response.status_code)
            return failed(kind, f"http_{response.status_code}", _error_message(response), cost,
                          status=response.status_code, request_id=header_id or _error_body(response).get("id"))
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data = response.json()
        observations = _observations(config, data.get("usage") or {})
        reason = data.get("finish_reason")
        request_id = header_id or data.get("id")

        def failure(kind: FailureKind, error: dict[str, Any], cost: CostIncurred = CostIncurred.YES
                    ) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=request_id,
                                   finish_reason=reason, cost_incurred=cost, failure_kind=kind,
                                   observations=observations, error=error)

        message = data.get("message")
        if not isinstance(message, dict):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_message"})
        if reason == "TIMEOUT":  # the server's time limit; the generation may be partial and billed
            return failure(FailureKind.AMBIGUOUS_TIMEOUT, {"type": "finish_reason", "reason": reason},
                           CostIncurred.UNKNOWN)
        if reason == "ERROR":
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason},
                           CostIncurred.UNKNOWN)
        if reason not in ("COMPLETE", "MAX_TOKENS"):  # TOOL_CALL, STOP_SEQUENCE (none are sent), undocumented
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason})
        content = message.get("content")
        if not isinstance(content, list) or any(not isinstance(b, dict) or b.get("type") not in CONTENT_TYPES
                                                for b in content):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "unexpected_content", "content_types": sorted(
                {str(b.get("type")) if isinstance(b, dict) else type(b).__name__ for b in content or []})})
        if message.get("tool_calls"):  # no tools are sent
            return failure(FailureKind.PROVIDER_ERROR, {"type": "unexpected_tool_calls"})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason,
                               request_id=request_id, cost_incurred=CostIncurred.YES, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        content = (data.get("message") or {}).get("content") or []
        text = "".join(str(b.get("text") or "") for b in content if b.get("type") == "text")
        thoughts = "".join(str(b.get("thinking") or "") for b in content if b.get("type") == "thinking")
        return ExtractedResponse(
            final_answer=text,
            truncated=data.get("finish_reason") == "MAX_TOKENS",
            finish_reason=data.get("finish_reason"),
            reasoning_text=thoughts or None,
        )

    def fork_state(
        self, prepared: PreparedInvocation, execution: NativeExecution, extracted: ExtractedResponse
    ) -> ContinuationSnapshot:
        data = execution.native_response or {}
        body = prepared.native_request["body"]
        return ContinuationSnapshot(
            adapter=self.name,
            configuration_key=str(body["model"]),
            messages=prepared.messages + (Message(role=Role.ASSISTANT, content=extracted.final_answer),),
            native_state={
                "format": STATE_FORMAT,
                "model": body["model"],
                "response_id": data.get("id"),
                # The assistant message (thinking and text blocks) is replayed exactly as returned.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(data.get("message") or {})],
            },
        )


def _base_url(config: ModelConfiguration) -> str:
    return str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)).rstrip("/") + "/"


def _parameter_problem(config: ModelConfiguration) -> str | None:
    names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
    if unsupported := sorted(names - SUPPORTED_PARAMETERS):
        return f"unsupported: {unsupported}"
    limit = config.adapter_options.get("max_output_tokens_limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        return "adapter_options.max_output_tokens_limit must state the model's maximum output (Cohere caps silently)"
    budget = explicit(config, "max_output_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= limit:
        return f"max_output_tokens must be an explicit integer in 1..{limit}"
    thinking = explicit(config, "thinking")
    if thinking is not None and thinking not in ("enabled", "disabled"):
        return "thinking must be 'enabled' or 'disabled'"
    token_budget = explicit(config, "thinking_token_budget")
    if token_budget is not None:
        if thinking != "enabled":
            return "thinking_token_budget requires thinking: enabled"
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or not 1 <= token_budget < budget:
            return "thinking_token_budget must be a positive integer below max_output_tokens"
    return None


def _request_id(headers: Mapping[str, str]) -> str | None:
    return next((headers[h] for h in REQUEST_ID_HEADERS if headers.get(h)), None)


def _error_body(response: httpx2.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def _error_message(response: httpx2.Response) -> str:
    message = _error_body(response).get("message")
    return str(message)[:500] if message else response.text[:500]


def _http_failure(status: int) -> tuple[FailureKind, CostIncurred]:
    if status == 498:  # a deny-listed token in the request or the generated response
        return FailureKind.REFUSAL, CostIncurred.UNKNOWN
    if status == 429:
        return FailureKind.RATE_LIMITED, CostIncurred.NO
    if status in (499, 504):  # cancelled or timed out after dispatch; inference may have run
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status in (500, 502, 503):
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    # 400, 401, 402 billing limit, 403, 404 unknown model, 422 malformed, 501: retrying cannot help
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "cohere usage"
    raw_billed, raw_tokens = usage.get("billed_units"), usage.get("tokens")
    billed: dict[str, Any] = raw_billed if isinstance(raw_billed, dict) else {}
    tokens: dict[str, Any] = raw_tokens if isinstance(raw_tokens, dict) else {}
    values = {
        "input_tokens": tokens.get("input_tokens"),
        "output_tokens": tokens.get("output_tokens"),
        "billed_input_tokens": billed.get("input_tokens"),
        "billed_output_tokens": billed.get("output_tokens"),
        "cached_input_tokens": usage.get("cached_tokens"),
        "reasoning_tokens": tokens.get("reasoning_tokens"),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if v is not None
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")
    parts = [(obs["billed_input_tokens"].value, "input_usd_per_mtok"),
             (obs["billed_output_tokens"].value, "output_usd_per_mtok")]
    if not all(isinstance(tokens, int) for tokens, _ in parts):
        return Observation.missing("usd", "configuration pricing", "billed units not reported")
    total = 0.0
    for tokens, key in parts:
        if not tokens:
            continue
        rate = pricing.get(key)
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", f"no price for {key}")
        total += int(tokens) * rate / 1_000_000
    return Observation(value=round(total, 10), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')}), billed units")
