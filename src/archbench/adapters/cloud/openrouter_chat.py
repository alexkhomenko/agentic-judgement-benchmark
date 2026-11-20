"""OpenRouter chat-completions adapter, pinned to the model vendor's own serving infrastructure
(`POST https://openrouter.ai/api/v1/chat/completions`, `Authorization: Bearer`).

This adapter exists for vendors whose first-party API is unavailable to us (Qwen: the Alibaba Model Studio
account is closed). It reaches the SAME vendor's servers through OpenRouter and nothing else: `provider` in the
configuration names the vendor (so the bias report attributes results to it), and only the routes in `ROUTES`
are allowed, each the vendor's own OpenRouter provider. Every request carries
`provider: {only: [<vendor slug>], allow_fallbacks: false, require_parameters: true}`: no other provider, no
fallback, and no routing at all if a sent parameter is not supported. Model-level fallbacks (`models`, `route`)
and model variants (`:nitro`, `:floor`, ...) are never used.

Plain HTTP against the documented REST API: the recorded request is exactly the JSON body on the wire.
- No client-side retries: every retry is a new harness attempt. OpenRouter may retry within the pinned provider;
  `openrouter_metadata.attempt` (kept in the native response) shows when it did.
- Every request opts into routing metadata (`X-OpenRouter-Metadata: enabled`). The served provider is read from
  `openrouter_metadata` (the selected endpoint, else the last attempt) and must be the pinned vendor; successful
  responses carry no documented top-level `provider`. A response that does not say who served it is a failure.
- Only explicitly configured parameters are sent; omitted ones are left to the provider. OpenRouter's own default
  when `reasoning` is omitted is undocumented, so a native default can only be reproduced by sending it.
- Reasoning arrives as `message.reasoning` and `message.reasoning_details`. Continuation replays the assistant
  message exactly as returned. Whether that reaches the vendor is not documented: qualification checks it.
- UPSTREAM CHECK: OpenRouter translates the request into the vendor's format. Qualification sends one streamed
  follow-up with `debug.echo_upstream_body` (streaming only, best-effort redacted) and checks the body OpenRouter
  forwarded: the vendor model id must be `adapter_options.upstream_model`; parameters the configuration omits must
  not be injected; `max_tokens` must be unchanged; replayed reasoning must be forwarded; `enable_search` must not be
  on; and every key in `adapter_options.upstream_expect` must match. The echo is recorded in the probes.
- Cancellation is not supported for Alibaba: an abandoned request runs, and is billed, to completion.
- Refusals: OpenRouter moderation (HTTP 403 with flagged metadata), a vendor content block (Alibaba's
  `data_inspection_failed`), or finish_reason `content_filter`.

Cost uses configured prices, which qualification compares with the endpoint's live OpenRouter pricing.
`usage.prompt_tokens` is the total input: cache reads (`prompt_tokens_details.cached_tokens`), cache writes
(`prompt_tokens_details.cache_write_tokens`, reported only for explicit caching, which this adapter never
requests) and uncached input sum to it. OpenRouter's own charge (`usage.cost`, in credits) is recorded beside it.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_tokens (includes reasoning tokens)
    reasoning_effort   explicit or omitted  -> reasoning.effort
    temperature, top_p, top_k, presence_penalty   explicit or omitted -> same names
adapter_options:
    upstream_model: the vendor's model id OpenRouter forwards (required; from the endpoint name)
    upstream_expect: {key: value} that must appear in the forwarded body (optional)
    timeout_s: 900 (default)
    pricing: {input_usd_per_mtok, cached_input_usd_per_mtok, cache_write_usd_per_mtok, output_usd_per_mtok,
              source, verified_on}
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Route:
    slug: str  # OpenRouter provider slug, sent in provider.only
    name: str  # provider name OpenRouter reports (endpoints JSON provider_name, routing metadata)


# configuration `provider` (the vendor) -> the vendor's own OpenRouter provider. Never a third-party host.
ROUTES = {"alibaba": Route(slug="alibaba", name="Alibaba")}

STATE_FORMAT = "openrouter-chat-v1"
BASE_URL = "https://openrouter.ai/api/v1/"
CHAT_PATH = "chat/completions"
DEFAULT_TIMEOUT_S = 900
PROBE_MAX_TOKENS = 2048
METADATA_HEADER = {"x-openrouter-metadata": "enabled"}
SAMPLING = ("temperature", "top_p", "top_k", "presence_penalty")
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "reasoning_effort", *SAMPLING})
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")  # where a vendor body may carry earlier reasoning
UPSTREAM_REPORTED = ("model", "max_tokens", "max_completion_tokens", "enable_thinking", "reasoning_effort",
                     "thinking_budget", "preserve_thinking", "enable_search", *SAMPLING, "stream")
CONTEXT_OVERFLOW = re.compile(r"(?i)context.?(length|window)|maximum context|too (long|many tokens)|range of input")
VENDOR_BLOCK = re.compile(r"(?i)data_inspection_failed|inappropriate content")
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no OpenRouter key found: add openrouter.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set OPENROUTER_API_KEY")


class OpenRouterAdapter:
    name = "openrouter"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        timeout = float(str(config.adapter_options.get("timeout_s", DEFAULT_TIMEOUT_S)))
        # No HTTP-Referer or X-OpenRouter-Title: nothing identifying is sent beyond the key.
        return httpx2.Client(base_url=BASE_URL, transport=transport, timeout=timeout,
                             headers={"authorization": f"Bearer {self._api_key}",
                                      "content-type": "application/json", **METADATA_HEADER})

    def preflight(self, config: ModelConfiguration) -> str | None:
        return None if self._api_key else _NO_KEY

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        parameter_check = _parameter_check(config)
        probes: list[dict[str, Any]] = []
        model_check, context = self._model_check(config, probes)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=(config.adapter == self.name and config.track is Track.CLOUD and config.provider in ROUTES
                        and ":" not in config.model),
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider} "
                       f"model={config.model} (allowed vendors: {sorted(ROUTES)}; no model variants)",
            ),
            parameter_check,
            model_check,
            context_check(config, requirements, context),
        ]
        # Probes are paid requests built from the configured parameters: send none for an invalid configuration.
        live: list[tuple[str, Callable[[], QualificationCheck]]] = [
            ("upstream_request", lambda: self._upstream_check(config, probes))]
        if requirements.requires_continuation:
            live.append(("continuation", lambda: self._continuation_check(config, probes)))
        for name, run in live:
            if parameter_check.passed is not True or checks[0].passed is not True:
                checks.append(QualificationCheck(name=name, passed=None,
                                                 detail="not evaluated: configuration parameters are invalid"))
            elif model_check.passed is not True:
                checks.append(QualificationCheck(name=name, passed=None,
                                                 detail="not evaluated: model identity was not verified"))
            else:
                checks.append(run())
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _model_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]
                     ) -> tuple[QualificationCheck, int | None]:
        """The model's OpenRouter endpoints: exactly one from the vendor, serving `upstream_model`, with enough
        context and output, every sent parameter supported, and the configured prices."""
        route = ROUTES.get(config.provider)
        upstream = config.adapter_options.get("upstream_model")
        if route is None or ":" in config.model:
            return QualificationCheck(name="model_identity", passed=False, detail="no allowed route"), None
        if not isinstance(upstream, str) or not upstream:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail="adapter_options.upstream_model is required"), None
        if config.context_limit is None:
            return QualificationCheck(name="model_identity", passed=False, detail="context_limit is required"), None
        transport = CapturingTransport(self._transport_factory())
        try:
            with self._client(config, transport) as client:
                response = client.get(f"models/{config.model}/endpoints")
        except (AdapterError, httpx2.HTTPError) as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"), None
        finally:
            probes.append({"step": "endpoints", "wire": wire_record(transport.exchanges)})
        if response.status_code != 200:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"HTTP {response.status_code}: {_error_message(response)}"), None
        data = response.json().get("data") or {}
        endpoints = [e for e in data.get("endpoints") or [] if isinstance(e, dict)]
        ours = [e for e in endpoints if str(e.get("tag") or "").split("/")[0] == route.slug]
        problems: list[str] = []
        if data.get("id") != config.model:
            problems.append(f"endpoints list is for {data.get('id')!r}, not {config.model!r}")
        if len(ours) != 1:
            names = [e.get("name") for e in endpoints]
            detail = f"expected exactly one {route.name} endpoint for {config.model}, found {len(ours)} in {names}"
            return QualificationCheck(name="model_identity", passed=False, detail=detail), None
        endpoint = ours[0]
        probes.append({"step": "endpoint", "endpoint": endpoint})
        served = str(endpoint.get("name") or "").split("|")[-1].strip().rsplit("/", 1)[-1]
        if served != upstream:
            problems.append(f"endpoint {endpoint.get('name')!r} serves {served!r}, not upstream_model {upstream!r}")
        if endpoint.get("provider_name") != route.name:
            problems.append(f"endpoint provider is {endpoint.get('provider_name')!r}, not {route.name!r}")
        context, output_cap = endpoint.get("context_length"), endpoint.get("max_completion_tokens")
        if isinstance(context, int) and config.context_limit > context:
            problems.append(f"context_limit {config.context_limit} > endpoint context_length {context}")
        requested = explicit(config, "max_output_tokens")
        if isinstance(output_cap, int) and isinstance(requested, int) and requested > output_cap:
            problems.append(f"max_output_tokens {requested} > endpoint max_completion_tokens {output_cap}")
        supported = set(endpoint.get("supported_parameters") or [])
        if missing := sorted(set(_wire_parameters(config)) - supported):
            problems.append(f"endpoint does not support {missing}")
        problems.extend(_pricing_problems(config, endpoint.get("pricing") or {}))
        detail = (f"{endpoint.get('name')} ({endpoint.get('provider_name')}, quantization "
                  f"{endpoint.get('quantization')}, status {endpoint.get('status')}): context_length={context}, "
                  f"max_completion_tokens={output_cap}; prices match configuration")
        return (QualificationCheck(name="model_identity", passed=not problems, detail="; ".join(problems) or detail),
                context if isinstance(context, int) else None)

    def _probe_config(self, config: ModelConfiguration) -> ModelConfiguration:
        budget = explicit(config, "max_output_tokens")
        params = dict(config.parameters)
        params["max_output_tokens"] = params["max_output_tokens"].model_copy(
            update={"value": min(budget, PROBE_MAX_TOKENS) if isinstance(budget, int) else PROBE_MAX_TOKENS})
        return config.model_copy(update={"parameters": params})

    def _run_probe(self, config: ModelConfiguration, probes: list[dict[str, Any]], step: str,
                   messages: tuple[Message, ...], parent: ContinuationSnapshot | None
                   ) -> tuple[PreparedInvocation, NativeExecution]:
        ctx = InvocationContext(experiment_id="qualification", task_id=f"qualification:{config.key}",
                                attempt_id=step, attempt_no=1, round=1 if parent is None else 2,
                                purpose="qualification")
        sink = CollectingSink()
        prepared = self.prepare(config, messages, parent, ctx)
        execution = self.execute(config, prepared, ctx, sink)
        cost = execution.observations.get("api_cost_usd")
        probes.append({
            "step": step, "status": execution.status, "request_id": execution.request_id,
            "failure_kind": execution.failure_kind.value if execution.failure_kind else None,
            "cost_incurred": execution.cost_incurred.value, "cost_usd": cost.value if cost else None,
            "max_output_tokens": explicit(config, "max_output_tokens"),
            "wire": sink.items.get("wire-exchanges.json"),
        })
        return prepared, execution

    def _upstream_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """A baseline, then one streamed follow-up (configured max_output_tokens, replayed assistant message) with
        `debug.echo_upstream_body`: the body OpenRouter forwarded to the vendor must match the configuration."""
        base_messages = (Message(role=Role.SYSTEM, content="You are answering a harness qualification probe."),
                         Message(role=Role.USER, content="Reply with exactly one word: baseline"))
        try:
            prepared, execution = self._run_probe(self._probe_config(config), probes, "upstream-baseline",
                                                  base_messages, None)
            if execution.status != "completed":
                return QualificationCheck(name="upstream_request", passed=False,
                                          detail=f"baseline failed: {execution.failure_kind} {execution.error}")
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            follow = snapshot.messages + (Message(role=Role.USER, content="Reply with exactly one word: gamma"),)
            ctx = InvocationContext(experiment_id="qualification", task_id=f"qualification:{config.key}",
                                    attempt_id="upstream-echo", attempt_no=1, round=2, purpose="qualification")
            body = {**self.prepare(config, follow, snapshot, ctx).native_request["body"],
                    "stream": True, "debug": {"echo_upstream_body": True}}
        except AdapterError as exc:
            return QualificationCheck(name="upstream_request", passed=False, detail=str(exc))
        transport = CapturingTransport(self._transport_factory())
        record: dict[str, Any] = {"step": "upstream-echo", "max_output_tokens": explicit(config, "max_output_tokens")}
        try:
            with self._client(config, transport) as client:
                response = client.post(CHAT_PATH, content=json.dumps(body))
        except (AdapterError, httpx2.HTTPError) as exc:
            return QualificationCheck(name="upstream_request", passed=None, detail=f"not evaluated: {exc}")
        finally:
            record["wire"] = wire_record(transport.exchanges)
            probes.append(record)
        if response.status_code != 200:
            return QualificationCheck(name="upstream_request", passed=False,
                                      detail=f"echo probe: HTTP {response.status_code}: {_error_message(response)}")
        chunks = _sse_chunks(response.text)
        debug = next((c for c in chunks if isinstance(c.get("debug"), dict)), None)
        echo = (debug or {}).get("debug", {}).get("echo_upstream_body")
        if not isinstance(echo, dict):
            return QualificationCheck(name="upstream_request", passed=None,
                                      detail="not evaluated: the stream carried no debug.echo_upstream_body chunk")
        record.update(upstream_provider=(debug or {}).get("provider"), upstream_body=echo)
        returned = _choice(execution.native_response or {}).get("message") or {}
        problems = _upstream_problems(config, echo, (debug or {}).get("provider"), bool(_reasoning_text(returned)))
        reported = {k: echo[k] for k in UPSTREAM_REPORTED if k in echo}
        detail = (f"forwarded to {(debug or {}).get('provider')}: {json.dumps(reported, sort_keys=True)}; "
                  f"top-level keys {sorted(echo)}")
        return QualificationCheck(name="upstream_request", passed=not problems,
                                  detail="; ".join(problems) + f" ({detail})" if problems else detail)

    def _continuation_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Live probe: one baseline, then two sibling follow-ups forked from the same snapshot."""
        probe = self._probe_config(config)
        base_messages = (Message(role=Role.SYSTEM, content="You are answering a harness qualification probe."),
                         Message(role=Role.USER, content="Reply with exactly one word: baseline"))
        try:
            prepared, execution = self._run_probe(probe, probes, "baseline", base_messages, None)
            if execution.status != "completed":
                return QualificationCheck(name="continuation", passed=False,
                                          detail=f"baseline failed: {execution.failure_kind} {execution.error}")
            returned = _choice(execution.native_response or {}).get("message") or {}
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = json.dumps(snapshot.native_state, sort_keys=True)
            base_sent = prepared.native_request["body"]["messages"]
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = self._run_probe(probe, probes, f"branch-{word}", follow, snapshot)
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
                   + (" (including reasoning and reasoning_details)" if _reasoning_text(returned) else ""),
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
        route = ROUTES.get(config.provider)
        if route is None or ":" in config.model:
            raise AdapterError(FailureKind.INVALID_STATE,
                               f"no OpenRouter route for provider {config.provider!r} and model {config.model!r}")
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
        for name in SAMPLING:
            if (value := explicit(config, name)) is not None:
                body[name] = value
        if (effort := explicit(config, "reasoning_effort")) is not None:
            body["reasoning"] = {"effort": effort}
        body["provider"] = {"only": [route.slug], "allow_fallbacks": False, "require_parameters": True}
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": f"POST /api/v1/{CHAT_PATH}",
                "client": {"package": "httpx2", "version": httpx2.__version__, "retries": 0},
                "headers": dict(METADATA_HEADER),
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
                response = client.post(CHAT_PATH, content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:  # Alibaba requests cannot be cancelled: the model keeps running
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        request_id = response.headers.get("x-generation-id")
        if response.status_code != 200:
            error = _error(response)
            kind, cost = _status_failure(response.status_code, error)
            return failed(kind, f"http_{response.status_code}", _error_message(response), cost,
                          status=response.status_code, request_id=request_id, metadata=error.get("metadata"))
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        try:
            data = response.json()  # leading keep-alive whitespace is valid JSON whitespace
        except json.JSONDecodeError:
            return failed(FailureKind.PROVIDER_ERROR, "invalid_body", response.text[:500], CostIncurred.UNKNOWN,
                          request_id=request_id)
        request_id = data.get("id") or request_id
        observations = _observations(config, data.get("usage") or {})
        choice = _choice(data)
        reason = choice.get("finish_reason")
        message = choice.get("message") or {}

        def failure(kind: FailureKind, error: dict[str, Any], cost: CostIncurred = CostIncurred.YES
                    ) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=request_id,
                                   finish_reason=reason, cost_incurred=cost, failure_kind=kind,
                                   observations=observations, error=error)

        # An error after the provider connected arrives with HTTP 200: as a bare error body, or as
        # finish_reason "error" beside it. Tokens may have been generated.
        if isinstance(data.get("error"), dict) or reason == "error":
            body_error: dict[str, Any] = data["error"] if isinstance(data.get("error"), dict) else {}
            code = body_error["code"] if isinstance(body_error.get("code"), int) else 502
            kind, _ = _status_failure(code, body_error)
            billed = CostIncurred.YES if (data.get("usage") or {}).get("completion_tokens") else CostIncurred.UNKNOWN
            return failure(kind, {"type": "error_in_body", "code": body_error.get("code"),
                                  "message": str(body_error.get("message") or "")[:500],
                                  "metadata": body_error.get("metadata")}, billed)
        if data.get("model") != config.model:
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "model_mismatch", "requested": config.model, "served": data.get("model")})
        route = ROUTES[config.provider]
        served = _served_provider(data)
        if served != route.name:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "provider_mismatch", "expected": route.name,
                                                        "served": served})
        if not choice:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_choices"})
        if reason == "content_filter" or message.get("refusal"):
            return failure(FailureKind.REFUSAL, {"type": "content_filter", "refusal": message.get("refusal"),
                                                 "native_finish_reason": choice.get("native_finish_reason")})
        if reason not in ("stop", "length"):  # tool_calls (no tools are sent), ...
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason,
                                                        "native_finish_reason": choice.get("native_finish_reason")})
        # Without reasoning in the message a follow-up would silently drop it from the model's context. A missing
        # reasoning text is acceptable only when the response reports zero reasoning tokens; unreported is not zero.
        if observations["reasoning_tokens"].value != 0 and not _reasoning_text(message):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "missing_reasoning",
                                                        "reasoning_tokens": observations["reasoning_tokens"].value})
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
            reasoning_text=_reasoning_text(message),
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
                "served_by": _served_provider(data),
                # The assistant message (content, reasoning, reasoning_details) is replayed exactly as returned.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(_choice(data).get("message") or {})],
            },
        )


def _wire_parameters(config: ModelConfiguration) -> list[str]:
    names = ["max_tokens"] + [n for n in SAMPLING if explicit(config, n) is not None]
    return names + (["reasoning"] if explicit(config, "reasoning_effort") is not None else [])


def _parameter_check(config: ModelConfiguration) -> QualificationCheck:
    explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
    problems = []
    if unsupported := sorted(explicit_names - SUPPORTED_PARAMETERS):
        problems.append(f"unsupported: {unsupported}")
    if "max_output_tokens" not in explicit_names:
        problems.append("max_output_tokens must be explicit")
    effort = explicit(config, "reasoning_effort")
    if effort is not None and effort not in REASONING_EFFORTS:
        problems.append(f"reasoning_effort {effort!r} is not one of {sorted(REASONING_EFFORTS)}")
    return QualificationCheck(name="parameter_support", passed=not problems, detail="; ".join(problems) or "ok")


def _pricing_problems(config: ModelConfiguration, live: dict[str, Any]) -> list[str]:
    """OpenRouter publishes USD per token as strings; the configuration holds USD per million tokens."""
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return ["no pricing in configuration"]
    problems = []
    for live_key, key in (("prompt", "input_usd_per_mtok"), ("completion", "output_usd_per_mtok"),
                          ("input_cache_read", "cached_input_usd_per_mtok"),
                          ("input_cache_write", "cache_write_usd_per_mtok")):
        try:
            published = float(live[live_key]) * 1_000_000 if live.get(live_key) is not None else None
        except (TypeError, ValueError):
            published = None
        configured = pricing.get(key)
        if published is None and configured is None:
            continue
        if published is None or not isinstance(configured, int | float) or abs(published - configured) > 1e-6:
            problems.append(f"{key}: configured {configured}, OpenRouter {published}")
    return problems


def _upstream_problems(config: ModelConfiguration, echo: dict[str, Any], provider: Any, reasoned: bool) -> list[str]:
    route = ROUTES[config.provider]
    problems = []
    if provider is not None and str(provider) != route.name:
        problems.append(f"forwarded to {provider!r}, not {route.name!r}")
    if echo.get("model") != config.adapter_options.get("upstream_model"):
        problems.append(f"vendor model {echo.get('model')!r}, not upstream_model "
                        f"{config.adapter_options.get('upstream_model')!r}")
    for name in SAMPLING:  # an omitted parameter must be left to the vendor, not filled in by OpenRouter
        configured = explicit(config, name)
        if configured is None and name in echo:
            problems.append(f"{name} omitted in configuration but forwarded as {echo[name]!r}")
        elif configured is not None and echo.get(name) != configured:
            problems.append(f"{name} {configured!r} forwarded as {echo.get(name)!r}")
    budget = explicit(config, "max_output_tokens")
    for key in ("max_tokens", "max_completion_tokens"):
        if key in echo and echo[key] != budget:
            problems.append(f"{key} forwarded as {echo[key]!r}, configured {budget}")
    if echo.get("enable_search"):
        problems.append("enable_search is on")
    if reasoned:
        assistants = [m for m in echo.get("messages") or [] if isinstance(m, dict) and m.get("role") == "assistant"]
        if not any(m.get(k) for m in assistants for k in REASONING_KEYS):
            problems.append("earlier reasoning is not forwarded to the vendor (it was returned and replayed)")
    expect = config.adapter_options.get("upstream_expect") or {}
    for key, value in (expect.items() if isinstance(expect, dict) else ()):
        if echo.get(key) != value:
            problems.append(f"upstream_expect {key}={value!r}, forwarded {echo.get(key)!r}")
    return problems


def _sse_chunks(text: str) -> list[dict[str, Any]]:
    """Data lines of an SSE stream as JSON objects; comment lines (`: OPENROUTER PROCESSING`) are skipped."""
    chunks = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            chunks.append(value)
    return chunks


def _served_provider(data: dict[str, Any]) -> str | None:
    """The provider that produced this response, from routing metadata: the selected endpoint, else the last
    attempt. A top-level `provider` is accepted only when the metadata names none."""
    meta = data.get("openrouter_metadata")
    if isinstance(meta, dict):
        available = (meta.get("endpoints") or {}).get("available") or []
        selected = [e.get("provider") for e in available if isinstance(e, dict) and e.get("selected")]
        if len(selected) == 1 and isinstance(selected[0], str):
            return selected[0]
        attempts = [a for a in meta.get("attempts") or [] if isinstance(a, dict)]
        last = attempts[-1].get("provider") if attempts else None
        if isinstance(last, str):
            return last
    provider = data.get("provider")
    return provider if isinstance(provider, str) else None


def _reasoning_text(message: dict[str, Any]) -> str | None:
    text = message.get("reasoning")
    if isinstance(text, str) and text:
        return text
    parts = [d.get("text") or d.get("summary") for d in message.get("reasoning_details") or [] if isinstance(d, dict)]
    joined = "".join(p for p in parts if isinstance(p, str))
    return joined or None


def _choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    return first if isinstance(first, dict) else {}


def _error(response: httpx2.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except json.JSONDecodeError:
        return {}
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def _error_message(response: httpx2.Response) -> str:
    message = _error(response).get("message")
    return str(message)[:500] if message else response.text[:500]


def _status_failure(status: int, error: dict[str, Any]) -> tuple[FailureKind, CostIncurred]:
    """Map OpenRouter's error codes (errors-and-debugging page, read 2026-11-20) to a failure kind."""
    metadata: dict[str, Any] = error["metadata"] if isinstance(error.get("metadata"), dict) else {}
    text = f"{error.get('message') or ''} {metadata.get('raw') or ''}"
    if status == 403 and (metadata.get("reasons") or metadata.get("flagged_input")):
        return FailureKind.REFUSAL, CostIncurred.NO  # OpenRouter moderation: blocked before the provider
    if VENDOR_BLOCK.search(text):
        return FailureKind.REFUSAL, CostIncurred.UNKNOWN  # the vendor's content inspection; an output hit may bill
    if status in (400, 413) and CONTEXT_OVERFLOW.search(text):
        return FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO
    if status == 429:
        return FailureKind.RATE_LIMITED, CostIncurred.NO
    if status in (408, 504, 524):  # timed out; the model may have run (Alibaba requests are not cancellable)
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status in (502, 503, 529):  # 502 provider down ("failed generations are not billed"), 503 no eligible provider
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    if status >= 500:
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN
    # 400 invalid request, 401 key, 402 credits, 403 permission or guardrail, 404 unknown model: retrying cannot help
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "openrouter usage"
    details = usage.get("prompt_tokens_details")
    prompt = usage.get("prompt_tokens")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    # cache_write_tokens is reported only for explicit caching (cache_control), which is never sent.
    writes = details.get("cache_write_tokens", 0) if isinstance(details, dict) else None
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
    if isinstance(prompt, int) and isinstance(cached, int) and isinstance(writes, int):
        obs["uncached_input_tokens"] = Observation(value=prompt - cached - writes, unit="tokens",
                                                   source=f"{source}: prompt_tokens - cached_tokens - cache_write_tokens")
    else:
        obs["uncached_input_tokens"] = Observation.missing("tokens", source, "input breakdown not reported")
    charged = usage.get("cost")
    obs["openrouter_charge"] = (
        Observation(value=float(charged), unit="credits", source="openrouter usage.cost")
        if isinstance(charged, int | float) else Observation.missing("credits", "openrouter usage.cost", "not reported"))
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")
    parts = [
        (obs["uncached_input_tokens"].value, "input_usd_per_mtok"),
        (obs["cached_input_tokens"].value, "cached_input_usd_per_mtok"),
        (obs["cache_write_tokens"].value, "cache_write_usd_per_mtok"),
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
