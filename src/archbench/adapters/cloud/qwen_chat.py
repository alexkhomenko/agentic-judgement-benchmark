"""Native Qwen adapter (Alibaba Cloud Model Studio, OpenAI-compatible `POST /compatible-mode/v1/chat/completions`).

Model Studio's OpenAI-compatible Chat Completions endpoint is the one its docs recommend; the DashScope-native
endpoint uses a different body shape (`input`/`parameters`) and documents no `model` field in its response.

Plain HTTP against the documented REST API: the recorded request is exactly the JSON body on the wire.
- No client-side retries: every retry is a new harness attempt.
- Only explicitly configured parameters are sent, plus `stream: true` and `stream_options.include_usage`.
  Streaming is required, not a preference: a non-streaming request that reaches the server timeout (300 s or
  more) "returns the generated content instead of an error", which would be a silent truncation. The raw
  event stream is the captured response body; `native_response` is the completion reassembled from it.
- Pay-as-you-go endpoints only: the shared regional hosts (`dashscope-intl.aliyuncs.com` for Singapore,
  `dashscope.aliyuncs.com` for Beijing) and workspace hosts (`{WorkspaceId}.{region}.maas.aliyuncs.com`).
  Token Plan, trial, and Coding Plan hosts are rejected at qualification. API keys, model lists, and prices
  are per region, so the configured pricing must name the region the endpoint serves.
- Model ids such as `qwen4-max` may be movable aliases with no dated snapshot. Every stream chunk's `model`
  must equal the expected served id (`adapter_options.served_model`, default: the requested id).
- Continuation replays the assistant message exactly as reassembled, including `reasoning_content`. Qwen4
  defaults to `preserve_thinking: true`, which requires all earlier reasoning to be sent back in that field.
- Content moderation (`data_inspection_failed`, `DataInspectionFailed`, `CustomRoleBlocked`) is a refusal with
  unknown cost, whether it arrives as an HTTP error or inside the stream.

Cost uses configured prices. `usage.prompt_tokens` is the total input; cache reads
(`prompt_tokens_details.cached_tokens`), cache writes (`prompt_tokens_details.cache_write_tokens`) and uncached
input sum to it. `cache_creation_input_tokens` duplicates `cache_write_tokens` and is not added. A field omitted
from a reported `prompt_tokens_details` counts as zero; a usage object without `prompt_tokens_details` leaves
the breakdown and cost unknown. Writes are priced by TTL from `cache_creation.ephemeral_5m_input_tokens` and
`ephemeral_1h_input_tokens`; writes without that breakdown leave the cost unknown. The adapter sends no cache
markers, so writes are not expected. `completion_tokens` includes reasoning tokens.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens   explicit, required   -> max_tokens (thinking plus answer; 1..262144, or 1..131072 with
                                                thinking disabled)
    enable_thinking     explicit or omitted  -> enable_thinking (bool)
    reasoning_effort    explicit or omitted  -> reasoning_effort (qwen4-max: "none" | "low" | "high" | "xhigh")
    thinking_budget     explicit or omitted  -> thinking_budget (int >= 1; not together with reasoning_effort)
    preserve_thinking   explicit or omitted  -> preserve_thinking (bool)
    temperature, top_p, top_k, presence_penalty, repetition_penalty, seed
                        explicit or omitted  -> same-named fields
adapter_options:
    base_url: https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (default, Singapore)
    served_model: expected `model` in responses (default: the configured model)
    timeout_s: 300 (default; bounds connecting and each read between stream chunks, not the whole stream)
    pricing: {region, input_usd_per_mtok (cache miss), cached_input_usd_per_mtok, cache_write_5m_usd_per_mtok,
              cache_write_1h_usd_per_mtok, output_usd_per_mtok, source, verified_on}
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
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

STATE_FORMAT = "qwen-chat-v1"
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
COMPATIBLE_PATH = "/compatible-mode/v1"
SHARED_HOSTS = {"dashscope-intl.aliyuncs.com": "ap-southeast-1", "dashscope.aliyuncs.com": "cn-beijing"}
WORKSPACE_HOST = re.compile(r"^(?P<prefix>[a-z0-9-]+)\.(?P<region>[a-z]{2}-[a-z]+-\d+)\.maas\.aliyuncs\.com$")
RESTRICTED_PREFIXES = ("token-plan", "trial")  # plan quota; no SLA
DEFAULT_TIMEOUT_S = 300
MAX_OUTPUT_TOKENS = 262144
MAX_OUTPUT_TOKENS_NO_THINKING = 131072
PROBE_MAX_TOKENS = 2048
PASSTHROUGH = ("enable_thinking", "reasoning_effort", "thinking_budget", "preserve_thinking", "temperature",
               "top_p", "top_k", "presence_penalty", "repetition_penalty", "seed")
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", *PASSTHROUGH})
REFUSAL_CODES = frozenset({"data_inspection_failed", "datainspectionfailed", "customroleblocked"})
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no Model Studio credentials found: add alibaba.api_key to secrets.local.yaml (see "
           "secrets.example.yaml) or set DASHSCOPE_API_KEY")


class QwenAdapter:
    name = "qwen"

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
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "alibaba",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            _endpoint_check(config),
            QualificationCheck(name="parameter_support", passed=parameter_detail == "ok", detail=parameter_detail),
            context_check(config, requirements, None),
        ]
        probes: list[dict[str, Any]] = []
        static_failures = [c.name for c in checks[:3] if c.passed is not True]
        if static_failures:  # never spend on a probe that cannot qualify, or against plan quota
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
        alias = "" if data.get("model") == config.model else f" for requested id {config.model}"
        checks = [QualificationCheck(
            name="model_identity", passed=True,
            detail=f"served model {data.get('model')}{alias} (no dated snapshot; response id {data.get('id')}, "
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
        body: dict[str, Any] = {"model": config.model, "messages": native, "max_tokens": max_tokens,
                                "stream": True, "stream_options": {"include_usage": True}}
        for name in PASSTHROUGH:
            if (value := explicit(config, name)) is not None:
                body[name] = value
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": f"POST {COMPATIBLE_PATH}/chat/completions",
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
            error = _error_body(response)
            code = _error_code(error)
            kind, cost = _http_failure(response.status_code, code)
            message = str(error.get("message") or response.text)[:500]
            return failed(kind, f"http_{response.status_code}", message, cost, status=response.status_code,
                          provider_code=code, request_id=response.headers.get("x-request-id"))
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data, problem = _assemble(response.text)
        observations = _observations(config, data.get("usage") or {})
        choice = _choice(data)
        reason = choice.get("finish_reason")
        request_id = response.headers.get("x-request-id") or data.get("id")

        def failure(kind: FailureKind, error: dict[str, Any], cost: CostIncurred = CostIncurred.YES
                    ) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=request_id,
                                   finish_reason=reason, cost_incurred=cost, failure_kind=kind,
                                   observations=observations, error=error)

        if problem is not None:
            if problem["type"] == "stream_error":
                refused = _error_code(problem["error"]) in REFUSAL_CODES
                return failure(FailureKind.REFUSAL if refused else FailureKind.PROVIDER_ERROR, problem,
                               CostIncurred.UNKNOWN)
            if problem["type"] == "incomplete_stream":  # the stream ended before finish_reason and [DONE]
                return failure(FailureKind.TRANSIENT_TRANSPORT, problem, CostIncurred.UNKNOWN)
            return failure(FailureKind.PROVIDER_ERROR, problem)
        expected = str(config.adapter_options.get("served_model", config.model))
        served = data["stream"]["models"]
        if served != [expected]:
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "model_mismatch", "requested": config.model, "expected": expected,
                            "served": served})
        if reason == "content_filter":  # not documented for Qwen; mapped defensively
            return failure(FailureKind.REFUSAL, {"type": "finish_reason", "reason": reason})
        if reason not in ("stop", "length"):  # tool_calls (no tools are sent), undocumented values
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason})
        output = observations["output_tokens"].value
        if isinstance(output, int) and output > int(body["max_tokens"]):
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "output_budget_exceeded", "max_tokens": body["max_tokens"], "output_tokens": output})
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
                "model": body["model"],
                "served_model": data.get("model"),
                "response_id": data.get("id"),
                # The reassembled assistant message (content and reasoning_content) is replayed exactly.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(_choice(data).get("message") or {})],
            },
        )


def _base_url(config: ModelConfiguration) -> str:
    return str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)).rstrip("/") + "/"


def _endpoint(base_url: str) -> tuple[str | None, str | None]:
    """Return (region, problem) for a configured base URL."""
    parts = urlsplit(base_url)
    host, path = (parts.hostname or "").lower(), parts.path.rstrip("/")
    if parts.scheme != "https":
        return None, f"{base_url} is not https"
    if path != COMPATIBLE_PATH:
        return None, f"path {path or '/'} is not the OpenAI-compatible {COMPATIBLE_PATH}"
    if host in SHARED_HOSTS:
        return SHARED_HOSTS[host], None
    match = WORKSPACE_HOST.match(host)
    if match is None:
        return None, f"{host} is not a pay-as-you-go Model Studio endpoint (Coding Plan and unknown hosts are rejected)"
    if match["prefix"] in RESTRICTED_PREFIXES:
        return None, f"{host} is a {match['prefix']} endpoint; use a workspace or shared pay-as-you-go endpoint"
    return match["region"], None


def _endpoint_check(config: ModelConfiguration) -> QualificationCheck:
    base = _base_url(config)
    region, problem = _endpoint(base)
    if problem is not None:
        return QualificationCheck(name="endpoint", passed=False, detail=problem)
    pricing = config.adapter_options.get("pricing")
    priced = pricing.get("region") if isinstance(pricing, dict) else None
    if isinstance(pricing, dict) and priced != region:
        return QualificationCheck(name="endpoint", passed=False,
                                  detail=f"{base} serves {region} but pricing is for {priced}; prices are per region")
    return QualificationCheck(name="endpoint", passed=True, detail=f"{base} ({region})")


def _parameter_problem(config: ModelConfiguration) -> str | None:
    names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
    if unsupported := sorted(names - SUPPORTED_PARAMETERS):
        return f"unsupported: {unsupported}"
    no_thinking = explicit(config, "enable_thinking") is False or explicit(config, "reasoning_effort") == "none"
    cap = MAX_OUTPUT_TOKENS_NO_THINKING if no_thinking else MAX_OUTPUT_TOKENS
    budget = explicit(config, "max_output_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= cap:
        return f"max_output_tokens must be an explicit integer in 1..{cap}"
    if {"reasoning_effort", "thinking_budget"} <= names:
        return "reasoning_effort and thinking_budget cannot both be set"
    thinking_budget = explicit(config, "thinking_budget")
    if thinking_budget is not None and (isinstance(thinking_budget, bool) or not isinstance(thinking_budget, int)
                                        or thinking_budget < 1):
        return "thinking_budget must be a positive integer"
    return None


def _choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    return first if isinstance(first, dict) else {}


def _events(text: str) -> list[str]:
    """Server-sent events: the joined `data:` lines of each blank-line-separated event."""
    events: list[str] = []
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw:
            if lines:
                events.append("\n".join(lines))
                lines = []
        elif raw.startswith("data:"):
            lines.append(raw[5:].removeprefix(" "))
    if lines:
        events.append("\n".join(lines))
    return events


def _assemble(text: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Reassemble a streamed chat completion. Returns the completion and the first problem found, if any."""
    content: list[str] = []
    reasoning: list[str] = []
    models: list[str] = []
    unexpected: set[str] = set()
    role, finish, done, chunks = None, None, False, 0
    data: dict[str, Any] = {"id": None, "object": "chat.completion", "created": None, "model": None,
                            "choices": [], "usage": None}

    def result(problem: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
        message: dict[str, Any] = {"role": role or "assistant", "content": "".join(content)}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        data["model"] = models[0] if models else None
        data["choices"] = [{"index": 0, "finish_reason": finish, "message": message}]
        data["stream"] = {"chunks": chunks, "done": done, "models": models}
        return data, problem

    for event in _events(text):
        if event.strip() == "[DONE]":
            done = True
            continue
        try:
            chunk = json.loads(event)
        except json.JSONDecodeError:
            chunk = None
        if not isinstance(chunk, dict):
            return result({"type": "malformed_stream", "event": event[:500]})
        chunks += 1
        if "error" in chunk or ("code" in chunk and "choices" not in chunk):
            error = chunk.get("error", chunk)
            return result({"type": "stream_error", "error": error if isinstance(error, dict) else {"message": error}})
        data["id"] = data["id"] or chunk.get("id")
        data["created"] = data["created"] or chunk.get("created")
        if (model := chunk.get("model")) is not None and model not in models:
            models.append(model)
        if chunk.get("usage"):
            data["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                unexpected.add("choices[index != 0]")
                continue
            for key, value in (choice.get("delta") or {}).items():
                if value is None or value == "":
                    continue
                if key == "role":
                    role = value
                elif key == "content" and isinstance(value, str):
                    content.append(value)
                elif key == "reasoning_content" and isinstance(value, str):
                    reasoning.append(value)
                else:  # tool calls, encrypted reasoning, or anything else that exact replay cannot carry
                    unexpected.add(f"delta.{key}")
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if unexpected:
        return result({"type": "unexpected_stream_fields", "fields": sorted(unexpected)})
    if finish is None or not done:
        return result({"type": "incomplete_stream", "finish_reason": finish, "done": done})
    return result(None)


def _error_body(response: httpx2.Response) -> dict[str, Any]:
    """`{"error": {...}}` on the OpenAI-compatible endpoint; `{"code", "message", "request_id"}` otherwise."""
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(body, dict):
        return {}
    error = body.get("error")
    return error if isinstance(error, dict) else body


def _error_code(error: dict[str, Any]) -> str | None:
    code = error.get("code")
    return str(code).lower() if code is not None else None


def _http_failure(status: int, code: str | None) -> tuple[FailureKind, CostIncurred]:
    if code in REFUSAL_CODES:  # content moderation on input or output
        return FailureKind.REFUSAL, CostIncurred.UNKNOWN
    if status == 429:
        return FailureKind.RATE_LIMITED, CostIncurred.NO
    if status == 504:  # a gateway gave up; inference may have run
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status >= 500:
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    # 400 parameters, model access, arrears; 401 key (keys are per region); 403; 404 model not in region
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "qwen usage"
    raw_details = usage.get("prompt_tokens_details")
    details: dict[str, Any] | None = raw_details if isinstance(raw_details, dict) else None
    creation = (details or {}).get("cache_creation")
    creation = creation if isinstance(creation, dict) else {}
    prompt = usage.get("prompt_tokens")
    values = {
        "input_tokens": prompt,
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cache_write_5m_tokens": creation.get("ephemeral_5m_input_tokens"),
        "cache_write_1h_tokens": creation.get("ephemeral_1h_input_tokens"),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if v is not None
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    breakdown: dict[str, int | None] = {}
    for name, key in (("cached_input_tokens", "cached_tokens"), ("cache_write_tokens", "cache_write_tokens")):
        if details is None:
            obs[name] = Observation.missing("tokens", source, "prompt_tokens_details not reported")
            breakdown[name] = None
        elif details.get(key) is None:
            obs[name] = Observation(value=0, unit="tokens", source=f"{source}: {key} omitted from prompt_tokens_details")
            breakdown[name] = 0
        else:
            obs[name] = Observation(value=int(details[key]), unit="tokens", source=source)
            breakdown[name] = int(details[key])
    cached, writes = breakdown["cached_input_tokens"], breakdown["cache_write_tokens"]
    if isinstance(prompt, int) and cached is not None and writes is not None:
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
        w5, w1h = w5 or 0, w1h or 0
        if w5 + w1h != writes:
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
