"""Native GLM adapter (Z.ai / Zhipu AI, `POST /api/paas/v4/chat/completions` on api.z.ai).

Plain HTTP against the documented REST API: the recorded request is exactly the JSON body on the wire.
- No client-side retries: every retry is a new harness attempt.
- Only explicitly configured parameters are sent.
- Pay-as-you-go endpoint only. The Coding Plan endpoints (`/api/coding/paas/v4`, `/api/anthropic`) are
  restricted to supported coding tools and bill against plan quota; qualification rejects them.
- GLM model ids are undated and Z.ai documents no model-list endpoint, so model identity is established by
  a live probe whose response must echo the configured id; every response's `model` is checked as well.
- Continuation replays the assistant message exactly as returned, including `reasoning_content`. On the
  standard endpoint `thinking.clear_thinking` defaults to true and the server ignores reasoning from prior
  turns; set `clear_thinking: false` to have it kept ("preserved thinking"), which then requires the
  complete, unmodified `reasoning_content`. Exact replay satisfies both.
- Errors carry a string business code (`{"error": {"code": "1301", "message": ...}}`). Code 1301 (unsafe
  content) is a refusal with unknown cost; a 200 response with `finish_reason: sensitive` is a billed refusal.

Cost uses configured prices. Caching is implicit: `usage.prompt_tokens` is the total input, of which
`prompt_tokens_details.cached_tokens` were cache hits. Cache storage is not reported per request (it is
free at the time of writing). Reasoning tokens are not broken out in usage.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_tokens (1..131072)
    thinking           explicit or omitted  -> thinking.type ("enabled" | "disabled"; GLM-5.3 accepts only
                                               "enabled")
    clear_thinking     explicit or omitted  -> thinking.clear_thinking (bool)
    reasoning_effort   explicit or omitted  -> reasoning_effort (GLM-5.3: "low" | "high" | "max")
    temperature        explicit or omitted  -> temperature
    top_p              explicit or omitted  -> top_p
adapter_options:
    base_url: https://api.z.ai/api/paas/v4 (default; mainland: https://open.bigmodel.cn/api/paas/v4)
    timeout_s: 900 (default; Z.ai documents no server-side limit)
    pricing: {input_usd_per_mtok (cache miss), cached_input_usd_per_mtok, output_usd_per_mtok, source, verified_on}
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
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

STATE_FORMAT = "glm-chat-v1"
DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
CODING_PLAN_PATHS = ("/api/coding/", "/api/anthropic")
DEFAULT_TIMEOUT_S = 900
MAX_OUTPUT_TOKENS = 131072
PROBE_MAX_TOKENS = 2048
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "thinking", "clear_thinking", "reasoning_effort",
                                  "temperature", "top_p"})
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no Z.ai credentials found: add zhipu.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set ZAI_API_KEY")


class GlmAdapter:
    name = "glm"

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
        explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
        unsupported = sorted(explicit_names - SUPPORTED_PARAMETERS)
        budget = explicit(config, "max_output_tokens")
        if unsupported:
            parameter_detail = f"unsupported: {unsupported}"
        elif not isinstance(budget, int) or not 1 <= budget <= MAX_OUTPUT_TOKENS:
            parameter_detail = f"max_output_tokens must be an explicit integer in 1..{MAX_OUTPUT_TOKENS}"
        else:
            parameter_detail = "ok"
        base = _base_url(config)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "zhipu",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(
                name="endpoint",
                passed=not any(p in base for p in CODING_PLAN_PATHS),
                detail=f"{base} is a Coding Plan endpoint; use the pay-as-you-go API"
                if any(p in base for p in CODING_PLAN_PATHS) else base,
            ),
            QualificationCheck(name="parameter_support", passed=parameter_detail == "ok", detail=parameter_detail),
            context_check(config, requirements, None),
        ]
        probes: list[dict[str, Any]] = []
        static_failures = [c.name for c in checks[:3] if c.passed is not True]
        if static_failures:  # never spend on a probe that cannot qualify, or against Coding Plan quota
            checks.append(QualificationCheck(name="model_identity", passed=None,
                                             detail=f"not evaluated: {', '.join(static_failures)} failed"))
        else:
            checks.extend(self._live_checks(config, requirements, probes))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _live_checks(self, config: ModelConfiguration, requirements: QualificationRequirements,
                     probes: list[dict[str, Any]]) -> list[QualificationCheck]:
        """Live probes: a baseline that proves the served model id, then (when the scenario branches) two
        sibling follow-ups forked from the same snapshot."""
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
        except AdapterError as exc:
            return [QualificationCheck(name="model_identity", passed=False, detail=str(exc))]
        if execution.status != "completed":
            error = execution.error or {}
            passed = None if error.get("type") == "credentials" else False
            return [QualificationCheck(name="model_identity", passed=passed,
                                       detail=f"baseline probe failed: {execution.failure_kind} {error}")]
        data = execution.native_response or {}
        checks = [QualificationCheck(
            name="model_identity", passed=True,
            detail=f"served model {data.get('model')} (undated id; response id {data.get('id')}, "
                   f"created {data.get('created')})",
        )]
        if requirements.requires_continuation:
            checks.append(self._continuation_check(prepared, execution, run))
        return checks

    def _continuation_check(
        self, prepared: PreparedInvocation, execution: NativeExecution,
        run: Callable[[str, tuple[Message, ...], ContinuationSnapshot | None],
                      tuple[PreparedInvocation, NativeExecution]],
    ) -> QualificationCheck:
        returned = _choice(execution.native_response or {}).get("message") or {}
        try:
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
        body: dict[str, Any] = {"model": config.model, "messages": native, "max_tokens": max_tokens}
        thinking: dict[str, Any] = {}
        if (kind := explicit(config, "thinking")) is not None:
            thinking["type"] = kind
        if (clear := explicit(config, "clear_thinking")) is not None:
            thinking["clear_thinking"] = clear
        if thinking:
            body["thinking"] = thinking
        for name in ("reasoning_effort", "temperature", "top_p"):
            if (value := explicit(config, name)) is not None:
                body[name] = value
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": "POST /api/paas/v4/chat/completions",
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
            code = _error_code(response)
            kind, cost = _http_failure(response.status_code, code)
            return failed(kind, f"http_{response.status_code}", _error_message(response), cost,
                          status=response.status_code, provider_code=code)
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data = response.json()
        observations = _observations(config, data.get("usage") or {})
        choice = _choice(data)
        reason = choice.get("finish_reason")
        request_id = data.get("request_id") or data.get("id")

        def failure(kind: FailureKind, error: dict[str, Any]) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=request_id,
                                   finish_reason=reason, cost_incurred=CostIncurred.YES, failure_kind=kind,
                                   observations=observations, error=error)

        if data.get("model") != config.model:
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "model_mismatch", "requested": config.model, "served": data.get("model")})
        if not choice:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_choices"})
        if reason == "sensitive":  # output stopped by Z.ai's content safety
            return failure(FailureKind.REFUSAL, {"type": "finish_reason", "reason": reason})
        if reason == "model_context_window_exceeded":
            return failure(FailureKind.CONTEXT_EXCEEDED, {"type": "finish_reason", "reason": reason})
        if reason == "network_error":  # inference ended abnormally on the provider side
            return failure(FailureKind.TRANSIENT_TRANSPORT, {"type": "finish_reason", "reason": reason})
        if reason not in ("stop", "length"):  # tool_calls (no tools are sent), undocumented values
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason,
                               request_id=request_id, cost_incurred=CostIncurred.YES, observations=observations)

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


def _base_url(config: ModelConfiguration) -> str:
    return str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)).rstrip("/") + "/"


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


def _error_code(response: httpx2.Response) -> str | None:
    code = _error_body(response).get("code")
    return str(code) if code is not None else None


def _error_message(response: httpx2.Response) -> str:
    message = _error_body(response).get("message")
    return str(message)[:500] if message else response.text[:500]


def _http_failure(status: int, code: str | None) -> tuple[FailureKind, CostIncurred]:
    """Map an HTTP error to a failure kind, per the platform's business-code table."""
    if code == "1301":  # unsafe or sensitive content
        return FailureKind.REFUSAL, CostIncurred.UNKNOWN
    if code == "1261":  # prompt too long
        return FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO
    if code in ("1302", "1305"):  # rate limit; service temporarily overloaded
        return FailureKind.RATE_LIMITED, CostIncurred.NO
    if status == 429:  # 1113 insufficient balance, 1308-1321 plan and spend limits: waiting does not help
        return FailureKind.PROVIDER_ERROR, CostIncurred.NO
    if status == 504:  # a gateway gave up; inference may have run
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status >= 500:  # 1200 API call error, 1230 call process error, 1234 network error
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    # 400 (1210-1222 parameters and model), 401 (1000-1005 auth), 403 (1220): retrying cannot help
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "glm usage"
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    prompt = usage.get("prompt_tokens")
    values = {
        "input_tokens": prompt,
        "cached_input_tokens": cached,
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if v is not None
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    if isinstance(prompt, int) and isinstance(cached, int):
        obs["uncached_input_tokens"] = Observation(value=prompt - cached, unit="tokens",
                                                   source=f"{source}: prompt_tokens - cached_tokens")
    else:
        obs["uncached_input_tokens"] = Observation.missing("tokens", source, "input breakdown not reported")
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")
    parts = [
        (obs["uncached_input_tokens"].value, "input_usd_per_mtok"),
        (obs["cached_input_tokens"].value, "cached_input_usd_per_mtok"),
        (obs["output_tokens"].value, "output_usd_per_mtok"),
    ]
    if not all(isinstance(tokens, int) for tokens, _ in parts):
        return Observation.missing("usd", "configuration pricing", "input breakdown or output tokens not reported")
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
