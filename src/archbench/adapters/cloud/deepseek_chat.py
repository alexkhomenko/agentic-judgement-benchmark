"""Native DeepSeek adapter (`POST /chat/completions` on api.deepseek.com).

Chat Completions is DeepSeek's native format. Its Responses endpoint is a compatibility layer that
silently ignores `store` and `include` and merges replayed reasoning text into the assistant message,
which is not how DeepSeek treats reasoning natively, so this adapter does not use it.

Plain HTTP against the documented REST API: the recorded request is exactly the JSON body on the wire.
- No client-side retries: every retry is a new harness attempt.
- Only explicitly configured parameters are sent.
- DeepSeek offers no dated model id (`deepseek-flash` is served by a named build that DeepSeek changes);
  qualification records the build name from `/models`, and every response's `system_fingerprint` is kept
  in the native response.
- Continuation replays the assistant message exactly as returned. Without tools, DeepSeek documents that
  earlier `reasoning_content` "will be ignored and will not be concatenated into the context".

Pricing is time-of-day dependent: peak and off-peak rates (off-peak is a fraction of peak), where peak is
fixed UTC hour windows on weekdays, excluding Chinese public holidays. Cost uses the request's send time.
A weekday peak-hour request on a date the configured holiday list does not cover has unknown cost; the
peak-rate figure is recorded beside it as api_cost_usd_if_peak.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_tokens (includes reasoning tokens)
    thinking           explicit or omitted  -> thinking.type ("enabled" | "disabled")
    reasoning_effort   explicit or omitted  -> reasoning_effort
    temperature, top_p explicit or omitted  -> temperature, top_p (no effect in thinking mode, per the docs)
adapter_options:
    base_url: https://api.deepseek.com (default)    timeout_s: 900 (default)
    pricing: {input_cache_miss_usd_per_mtok, input_cache_hit_usd_per_mtok, output_usd_per_mtok,   (peak rates)
              off_peak_multiplier, peak_hours_utc: [[start, end], ...], peak_weekdays: [0..6],
              holidays: [YYYY-MM-DD, ...], holidays_cover_through: YYYY-MM-DD | null, source, verified_on}
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
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

STATE_FORMAT = "deepseek-chat-v1"
DEFAULT_BASE_URL = "https://api.deepseek.com"
PROBE_MAX_TOKENS = 2048
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "thinking", "reasoning_effort", "temperature", "top_p"})
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no DeepSeek credentials found: add deepseek.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set DEEPSEEK_API_KEY")


class DeepSeekAdapter:
    name = "deepseek"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        base = str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        timeout = float(str(config.adapter_options.get("timeout_s", 900)))
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
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "deepseek",
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
        """Verify the model id, limits, and served build against the first-party `/models` list."""
        transport = CapturingTransport(self._transport_factory())
        try:
            with self._client(config, transport) as client:
                response = client.get("/models")
        except (AdapterError, httpx2.HTTPError) as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"), None
        if response.status_code != 200:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"HTTP {response.status_code}: {_error_message(response)}"), None
        card = next((m for m in response.json().get("data") or [] if m.get("id") == config.model), None)
        if card is None:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"{config.model} is not listed by /models"), None
        problems = []
        context, max_output = card.get("context_window"), card.get("max_output_tokens")
        requested = explicit(config, "max_output_tokens")
        if isinstance(context, int) and config.context_limit is not None and config.context_limit > context:
            problems.append(f"context_limit {config.context_limit} > context_window {context}")
        if isinstance(max_output, int) and isinstance(requested, int) and requested > max_output:
            problems.append(f"max_output_tokens {requested} > model max_output_tokens {max_output}")
        effort = card.get("effort") or {}
        detail = (f"{config.model} served by {card.get('name')}: context_window={context}, "
                  f"max_output_tokens={max_output}, effort default {effort.get('default_level')} "
                  f"of {effort.get('supported_levels')}")
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
        fingerprint = (execution.native_response or {}).get("system_fingerprint")
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact message replay"
                   + (" (including reasoning_content, which DeepSeek ignores without tools)"
                      if returned.get("reasoning_content") else "")
                   + f"; system_fingerprint {fingerprint}",
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
        if (thinking := explicit(config, "thinking")) is not None:
            body["thinking"] = {"type": thinking}
        for name in ("reasoning_effort", "temperature", "top_p"):
            if (value := explicit(config, name)) is not None:
                body[name] = value
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": "POST /chat/completions",
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
        sent_at = datetime.now(UTC)
        try:
            with self._client(config, transport) as client:
                response = client.post("/chat/completions", content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        if response.status_code != 200:
            if response.status_code == 429:
                kind = FailureKind.RATE_LIMITED
            elif response.status_code >= 500:  # 500 server error, 503 overloaded: retry after a brief wait
                kind = FailureKind.TRANSIENT_TRANSPORT
            else:  # 400, 401, 402 insufficient balance, 422: retrying cannot help
                kind = FailureKind.PROVIDER_ERROR
            return failed(kind, f"http_{response.status_code}", _error_message(response), CostIncurred.NO,
                          status=response.status_code)
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data = response.json()
        observations = _observations(config, data.get("usage") or {}, sent_at)
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
        if reason == "content_filter":
            return failure(FailureKind.REFUSAL, {"type": "content_filter"})
        if reason == "insufficient_system_resource":
            # The answer was cut short by provider capacity, not by the request: a retry can succeed.
            return failure(FailureKind.TRANSIENT_TRANSPORT, {"type": reason})
        if reason not in ("stop", "length"):
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
                "system_fingerprint": data.get("system_fingerprint"),
                # The assistant message is replayed exactly as returned.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(_choice(data).get("message") or {})],
            },
        )


def _choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    return first if isinstance(first, dict) else {}


def _error_message(response: httpx2.Response) -> str:
    try:
        body = response.json()
    except json.JSONDecodeError:
        return response.text[:500]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])[:500]
    return response.text[:500]


def _observations(config: ModelConfiguration, usage: dict[str, Any], sent_at: datetime) -> dict[str, Observation]:
    source = "deepseek usage"
    values = {
        "input_tokens": usage.get("prompt_tokens"),
        "cached_input_tokens": usage.get("prompt_cache_hit_tokens"),
        "uncached_input_tokens": usage.get("prompt_cache_miss_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if v is not None
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    obs.update(_cost(config, values, sent_at))
    return obs


def pricing_period(pricing: dict[str, Any], at: datetime) -> str:
    """'off_peak', 'peak', or 'unknown' (a weekday peak hour on a date the holiday list does not cover)."""
    at = at.astimezone(UTC)
    windows = pricing.get("peak_hours_utc") or []
    weekdays = pricing.get("peak_weekdays", [0, 1, 2, 3, 4])
    in_window = at.weekday() in weekdays and any(start <= at.hour < end for start, end in windows)
    if not in_window:
        return "off_peak"
    day = at.date()
    if day.isoformat() in (pricing.get("holidays") or []):
        return "off_peak"
    covered = pricing.get("holidays_cover_through")
    if isinstance(covered, str) and day <= date.fromisoformat(covered):
        return "peak"
    return "unknown"


def _cost(config: ModelConfiguration, tokens: dict[str, Any], sent_at: datetime) -> dict[str, Observation]:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return {"api_cost_usd": Observation.missing("usd", "configuration pricing", "no pricing in configuration")}
    hit, miss, output = tokens["cached_input_tokens"], tokens["uncached_input_tokens"], tokens["output_tokens"]
    if not all(isinstance(v, int) for v in (hit, miss, output)):
        return {"api_cost_usd": Observation.missing("usd", "configuration pricing",
                                                    "cache hit/miss or output tokens not reported")}
    rates = [pricing.get(k) for k in ("input_cache_hit_usd_per_mtok", "input_cache_miss_usd_per_mtok",
                                      "output_usd_per_mtok")]
    if not all(isinstance(r, int | float) for r in rates):
        return {"api_cost_usd": Observation.missing("usd", "configuration pricing", "incomplete pricing")}
    peak = sum(n * r for n, r in zip((hit, miss, output), rates)) / 1_000_000
    period = pricing_period(pricing, sent_at)
    source = f"configuration pricing ({pricing.get('source', 'unspecified source')}), {period} at {sent_at.isoformat()}"
    multiplier = pricing.get("off_peak_multiplier", 1)
    if period == "peak":
        return {"api_cost_usd": Observation(value=round(peak, 10), unit="usd", source=source)}
    if period == "off_peak" and isinstance(multiplier, int | float):
        return {"api_cost_usd": Observation(value=round(peak * multiplier, 10), unit="usd", source=source)}
    return {
        "api_cost_usd": Observation.missing(
            "usd", source, "weekday peak hour on a date the holiday list does not cover: peak or holiday off-peak"),
        "api_cost_usd_if_peak": Observation(value=round(peak, 10), unit="usd", source=f"{source}; assumes peak"),
    }
