"""Native Microsoft MAI adapter (MAI-Thinking-1 on Microsoft Foundry, `POST {resource}/mai/v1/chat/completions`).

Microsoft's own model family, called on the MAI route of a Foundry resource that you deploy into (GlobalStandard
only). Not Azure OpenAI: other vendors' models on Foundry are called through their own first-party APIs, and
the bias report maps `provider: microsoft` to Microsoft.

Plain HTTP against the documented REST API: the recorded request is exactly the JSON body on the wire.
- PREVIEW: no SLA. The default (Low) quota tier is 0 tokens per minute; a subscription needs the Medium or High
  tier before any call succeeds, and how a tier is granted is not documented.
- ENDPOINT: `adapter_options.base_url` is your resource endpoint (`https://<resource>.services.ai.azure.com`).
  `model` is your DEPLOYMENT name, not the model name. Authentication is the resource key in the `api-key` header.
- MODEL IDENTITY: the deployment's model and version are fixed when it is created, and cannot be read back
  with an API key (only through Azure Resource Manager with Entra ID). The configuration declares them
  (`served_model`, `model_version`); every response's `model` must be the deployment name or `served_model`
  (Microsoft's examples show both). The version is recorded as declared, not verified.
- Only `max_completion_tokens` is documented (it includes reasoning tokens; `max_tokens` is a 400). No sampling,
  reasoning-effort or seed parameters are documented, so none are configurable.
- REASONING: with `reasoning_display: "encrypted"` the response carries opaque `message.reasoning.encrypted_content`,
  which Microsoft says to return "exactly as you received it" with the complete assistant message. The adapter
  requests it on every call (`adapter_options.replay_reasoning`, default true) and replays assistant messages
  exactly as returned. Without it, `reasoning` is null and nothing carries across turns.
- CONTENT SAFETY: input and output are always screened. A block is HTTP 400 (input and output blocks are not
  distinguished, and the error body is not documented) or `finish_reason` "content filter"; both are refusals.
- Tool calls arrive with `finish_reason: "stop"`, so `tool_calls` is checked directly (no tools are sent).

Cost uses configured prices. `usage.prompt_tokens` includes `prompt_tokens_details.cached_tokens` (Microsoft's
stream example: total = prompt + completion). No cached-input price is published, so a response with cached
tokens has unknown cost unless the configuration states one.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_completion_tokens (includes reasoning)
adapter_options:
    base_url: https://<resource>.services.ai.azure.com (required)
    served_model: the model name responses may report (e.g. "MAI-Thinking-1", required)
    model_version: the version the deployment was created with (required; declared, not verifiable by key)
    max_output_tokens_limit: the model's documented maximum output (required)
    replay_reasoning: true (default)        timeout_s: 900 (default; no server-side limit documented)
    pricing: {input_usd_per_mtok, output_usd_per_mtok, cached_input_usd_per_mtok (optional), source, verified_on}
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

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

STATE_FORMAT = "mai-chat-completions"
ROUTE = "mai/v1/chat/completions"
DEFAULT_TIMEOUT_S = 900
PROBE_MAX_TOKENS = 2048
RESOURCE_HOST = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}\.services\.ai\.azure\.com$")
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens"})
# Undocumented; the first one present is recorded, falling back to the completion id.
REQUEST_ID_HEADERS = ("apim-request-id", "x-request-id", "x-ms-request-id")
SAFETY_MARKERS = ("safety", "content filter", "content_filter", "contentfilter", "responsibleai")
TransportFactory = Callable[[], httpx2.BaseTransport]
CONTEXT_MARKERS = ("context window", "context length", "context_length", "too long", "maximum context")
_NO_KEY = ("no Microsoft Foundry key found: add microsoft.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set AZURE_API_KEY")


class MaiAdapter:
    name = "mai"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        timeout = float(str(config.adapter_options.get("timeout_s", DEFAULT_TIMEOUT_S)))
        return httpx2.Client(base_url=_base_url(config), transport=transport, timeout=timeout,
                             headers={"api-key": self._api_key, "content-type": "application/json"})

    def preflight(self, config: ModelConfiguration) -> str | None:
        return None if self._api_key else _NO_KEY

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        parameter_detail = _parameter_problem(config) or "ok"
        endpoint_detail = _endpoint_problem(config)
        declared = {k: config.adapter_options.get(k) for k in ("served_model", "model_version")}
        static = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "microsoft",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(name="endpoint", passed=endpoint_detail is None,
                               detail=endpoint_detail or _base_url(config) + ROUTE),
            QualificationCheck(name="first_party_model",
                               passed=isinstance(declared["served_model"], str)
                               and str(declared["served_model"]).startswith("MAI-")
                               and isinstance(declared["model_version"], str),
                               detail=f"declared deployment model {declared['served_model']!r} version "
                                      f"{declared['model_version']!r}: only Microsoft's MAI models are benchmarked "
                                      "through Foundry"),
            QualificationCheck(name="parameter_support", passed=parameter_detail == "ok", detail=parameter_detail),
        ]
        checks = [*static, context_check(config, requirements, None)]
        probes: list[dict[str, Any]] = []
        static_failures = [c.name for c in static if c.passed is not True]
        live_name = "continuation" if requirements.requires_continuation else "chat_probe"
        if static_failures:
            checks.append(QualificationCheck(name="model_identity", passed=None,
                                             detail=f"not evaluated: {', '.join(static_failures)} failed"))
            checks.append(QualificationCheck(name=live_name, passed=None,
                                             detail=f"not evaluated: {', '.join(static_failures)} failed"))
        else:
            checks.extend(self._live_checks(config, probes, branches=requirements.requires_continuation))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _live_checks(self, config: ModelConfiguration, probes: list[dict[str, Any]], *, branches: bool
                     ) -> list[QualificationCheck]:
        """The baseline probe establishes model identity (the response must name the deployment or the declared
        model); when the scenario branches, two sibling follow-ups fork from the same snapshot."""
        name = "continuation" if branches else "chat_probe"
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
        version = config.adapter_options.get("model_version")
        try:
            prepared, execution = run("baseline", base_messages, None)
        except AdapterError as exc:
            return [QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"),
                    QualificationCheck(name=name, passed=False, detail=str(exc))]
        served = (execution.native_response or {}).get("model")
        if execution.status != "completed":
            failure = f"baseline probe failed: {execution.failure_kind} {execution.error}"
            if execution.failure_kind is FailureKind.RATE_LIMITED:
                failure += " (the default MAI quota tier is 0 tokens per minute)"
            identity = (QualificationCheck(name="model_identity", passed=False,
                                           detail=f"response named {served!r}") if execution.error
                        and execution.error.get("type") == "model_mismatch"
                        else QualificationCheck(name="model_identity", passed=None, detail="not evaluated: " + failure))
            return [identity, QualificationCheck(name=name, passed=False, detail=failure)]
        identity = QualificationCheck(
            name="model_identity", passed=True,
            detail=f"deployment {config.model!r} answered as {served!r}; declared version {version!r} "
                   "(not verifiable with an API key)")
        returned = _message(execution.native_response or {})
        reasoning = isinstance(returned.get("reasoning"), dict)
        if _replays_reasoning(config) and not (reasoning and returned["reasoning"].get("encrypted_content")):
            return [identity, QualificationCheck(
                name=name, passed=False,
                detail="reasoning_display was 'encrypted' but the message carries no reasoning.encrypted_content; "
                       "replay would drop the model's reasoning")]
        if not branches:
            return [identity, QualificationCheck(name=name, passed=True,
                                                 detail="baseline completed with the configured parameters"
                                                        + (" (with encrypted reasoning)" if reasoning else ""))]
        try:
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = json.dumps(snapshot.native_state, sort_keys=True)
            base_sent = prepared.native_request["body"]["messages"]
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = run(f"branch-{word}", follow, snapshot)
                sent = child.native_request["body"]["messages"]
                if sent[: len(base_sent) + 1] != base_sent + [returned]:
                    return [identity, QualificationCheck(name=name, passed=False, detail=(
                        f"branch {word} did not replay the baseline message exactly"))]
                if word == "beta" and "alpha" in json.dumps(sent):
                    return [identity, QualificationCheck(name=name, passed=False, detail="sibling branch leaked")]
                if result.status != "completed":
                    return [identity, QualificationCheck(name=name, passed=False, detail=(
                        f"branch {word} failed: {result.failure_kind} {result.error}"))]
            if json.dumps(snapshot.native_state, sort_keys=True) != before:
                return [identity, QualificationCheck(name=name, passed=False, detail="forking mutated baseline state")]
        except AdapterError as exc:
            return [identity, QualificationCheck(name=name, passed=False, detail=str(exc))]
        return [identity, QualificationCheck(
            name=name, passed=True,
            detail="baseline forked into two independent branches with exact message replay"
                   + (" (including encrypted reasoning)" if reasoning else ""))]

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
            # Encrypted reasoning is opaque; keep it on the deployment that produced it.
            if state.get("model") != config.model or state.get("base_url") != _base_url(config):
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state belongs to a different deployment")
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
        if _replays_reasoning(config):
            body["reasoning_display"] = "encrypted"
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": f"POST /{ROUTE}",
                "base_url": _base_url(config),
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
                response = client.post(ROUTE, content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        header_id = _request_id(response.headers)
        if response.status_code != 200:
            message = _error_message(response)
            kind, cost = _http_failure(response.status_code, message)
            return failed(kind, f"http_{response.status_code}", message, cost,
                          status=response.status_code, request_id=header_id)
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data = response.json()
        observations = _observations(config, data.get("usage") or {})
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
        reason = choice.get("finish_reason")
        request_id = header_id or data.get("id")

        def failure(kind: FailureKind, error: dict[str, Any], cost: CostIncurred = CostIncurred.YES
                    ) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=request_id,
                                   finish_reason=reason, cost_incurred=cost, failure_kind=kind,
                                   observations=observations, error=error)

        served = data.get("model")
        accepted = {config.model, config.adapter_options.get("served_model")}
        if served not in accepted:  # never accept an answer from another model or deployment
            return failure(FailureKind.PROVIDER_ERROR, {"type": "model_mismatch", "served": served,
                                                        "accepted": sorted(str(a) for a in accepted)})
        if not choice:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "choices", "count": len(choices or [])})
        returned = choice.get("message")
        if not isinstance(returned, dict):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_message"})
        if reason in ("content filter", "content_filter") or returned.get("refusal"):
            return failure(FailureKind.REFUSAL, {"type": "content_safety", "reason": reason,
                                                 "refusal": returned.get("refusal")})
        if returned.get("tool_calls"):  # reported with finish_reason "stop"; no tools are sent
            return failure(FailureKind.PROVIDER_ERROR, {"type": "unexpected_tool_calls"})
        if reason not in ("stop", "length"):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason})
        if not isinstance(returned.get("content"), str):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_content"})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason,
                               request_id=request_id, cost_incurred=CostIncurred.YES, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        reason = (data.get("choices") or [{}])[0].get("finish_reason")
        return ExtractedResponse(
            final_answer=str(_message(data).get("content") or ""),
            truncated=reason == "length",
            finish_reason=reason,
            reasoning_text=None,  # MAI returns reasoning only as opaque encrypted content
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
                "base_url": prepared.native_request["base_url"],
                "served_model": data.get("model"),
                "response_id": data.get("id"),
                # The complete assistant message (content, refusal, encrypted reasoning) is replayed exactly.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(_message(data))],
            },
        )


def _base_url(config: ModelConfiguration) -> str:
    return str(config.adapter_options.get("base_url", "")).rstrip("/") + "/"


def _endpoint_problem(config: ModelConfiguration) -> str | None:
    base = str(config.adapter_options.get("base_url", ""))
    parts = urlsplit(base)
    if parts.scheme != "https" or not RESOURCE_HOST.match(parts.hostname or "") or parts.path.strip("/"):
        return (f"adapter_options.base_url {base!r} must be your Foundry resource endpoint, "
                "https://<resource>.services.ai.azure.com (no path)")
    return None


def _replays_reasoning(config: ModelConfiguration) -> bool:
    return config.adapter_options.get("replay_reasoning", True) is not False


def _message(data: Mapping[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def _parameter_problem(config: ModelConfiguration) -> str | None:
    names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
    if unsupported := sorted(names - SUPPORTED_PARAMETERS):
        return f"unsupported (MAI documents only max_completion_tokens): {unsupported}"
    if not isinstance(config.adapter_options.get("replay_reasoning", True), bool):
        return "adapter_options.replay_reasoning must be true or false"
    limit = config.adapter_options.get("max_output_tokens_limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        return "adapter_options.max_output_tokens_limit must state the model's maximum output"
    budget = explicit(config, "max_output_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= limit:
        return f"max_output_tokens must be an explicit integer in 1..{limit}"
    return None


def _request_id(headers: Mapping[str, str]) -> str | None:
    return next((headers[h] for h in REQUEST_ID_HEADERS if headers.get(h)), None)


def _error_message(response: httpx2.Response) -> str:
    """The error body is not documented; keep the OpenAI-style `error` object when present, else the raw text."""
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return response.text[:500]
    error = body.get("error") if isinstance(body, dict) else None
    return json.dumps(error if isinstance(error, dict) else body, sort_keys=True)[:500]


def _http_failure(status: int, message: str = "") -> tuple[FailureKind, CostIncurred]:
    text = message.lower()
    if status == 400 and any(m in text for m in SAFETY_MARKERS):
        # Input and output blocks share this status; an output block means the model ran.
        return FailureKind.REFUSAL, CostIncurred.UNKNOWN
    if status == 400 and any(m in text for m in CONTEXT_MARKERS):
        return FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO
    if status == 429:
        return FailureKind.RATE_LIMITED, CostIncurred.NO
    if status in (408, 504):
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status in (500, 502, 503):  # 503 includes the safety service being unavailable
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    # 400 invalid request, 401 bad key, 403 no access to the model, 404 wrong deployment, 422 schema
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "mai usage"
    raw_prompt, raw_completion = usage.get("prompt_tokens_details"), usage.get("completion_tokens_details")
    prompt_details: dict[str, Any] = raw_prompt if isinstance(raw_prompt, dict) else {}
    completion_details: dict[str, Any] = raw_completion if isinstance(raw_completion, dict) else {}
    values = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "cached_input_tokens": prompt_details.get("cached_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),  # not documented; recorded if present
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if isinstance(v, int | float)
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")
    prompt, output = obs["input_tokens"].value, obs["output_tokens"].value
    if not isinstance(prompt, int) or not isinstance(output, int):
        return Observation.missing("usd", "configuration pricing", "token counts not reported")
    cached = obs["cached_input_tokens"].value
    cached = int(cached) if isinstance(cached, int) else 0
    parts = [(prompt - cached, "input_usd_per_mtok"), (cached, "cached_input_usd_per_mtok"),
             (output, "output_usd_per_mtok")]
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
