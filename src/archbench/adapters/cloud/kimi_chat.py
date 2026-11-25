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
uncached input sum to it. The documented response omits `cached_tokens` when nothing was read from the cache
(it reports `prompt_tokens_details` with only `cache_write_tokens`); that is recorded as zero cache reads, while
a usage object with no `prompt_tokens_details` leaves the breakdown unknown. Symmetrically, a reported
`prompt_tokens_details` without `cache_write_tokens`, sent with neither cache-write header, is zero writes. Cache writes are priced by TTL, which only the `Msh-Usage-Cache-Write-Tokens-5m`
and `-1h` response headers break down; writes reported without that breakdown leave the cost unknown.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_completion_tokens (includes reasoning tokens)
    reasoning_effort   explicit or omitted  -> reasoning_effort ("low" | "high" | "max"; K3 series only)
    thinking           explicit or omitted  -> thinking.type ("enabled" | "disabled"; K2 series only,
                                               K3 rejects it)
    web_search         explicit or omitted  -> tools: [value] (the official Formula `web-search` tool declaration,
                                               sent verbatim: the entry returned by
                                               `GET /v1/formulas/moonshot/web-search:latest/tools`,
                                               {type: function, function: {name: web_search, description, parameters}})
adapter_options:
    base_url: https://api.moonshot.ai/v1 (default)    timeout_s: 960 (default)
    web_search_formula: moonshot/web-search:latest (default)   formula URI that executes the web_search tool calls
    web_search_max_rounds: 3 (default)              tool-call rounds serviced before the loop counts as incomplete
    pricing: {input_usd_per_mtok (cache miss), cached_input_usd_per_mtok, cache_write_5m_usd_per_mtok,
              cache_write_1h_usd_per_mtok, output_usd_per_mtok, web_search_usd_per_request?, source, verified_on}

Web search (verified 2026-11-24):
- The `$web_search` builtin_function is retired: Moonshot's guide listed it as "Retires on October 20, 2026"
  (https://platform.kimi.ai/docs/guide/use-web-search, snapshots of 2026-10-06 and 2026-10-27 on web.archive.org),
  the page now documents only the standalone REST search APIs, and the Chat Completions `ToolDefinition` accepts
  only `type: function` (https://platform.kimi.ai/docs/api/chat). A builtin_function value is rejected.
- For kimi-k3.1-pro Moonshot directs web search to the Formula official-tools channel on Chat Completions
  (https://platform.kimi.ai/docs/guide/use-official-tools: "verified with kimi-k3.1-pro"). The search is run by
  Moonshot, but the round trip is client-side: the model ends with `finish_reason: tool_calls`; each call's
  `function` ({name, arguments}, verbatim) is posted to `POST /v1/formulas/{uri}/fibers`; the fiber's
  `context.output` or `context.encrypted_output` goes back as a `role: tool` message with the matching
  `tool_call_id`, after the complete assistant message (reasoning_content and tool_calls unchanged: "Return the
  complete assistant message unchanged in multi-turn conversations and tool calls",
  https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart). The K3.1 Pro guide also says web search "is being
  updated and is not recommended for production workflows in the near term".
- execute() runs this loop, at most web_search_max_rounds tool rounds; a model still calling tools after that, a
  call to any other tool, or a failed fiber is a recorded failure (never a partial answer). Every chat and fiber
  exchange is in wire-exchanges.json. The native response is the final chat completion plus
  `harness_web_search_rounds`: each earlier completion with its fiber bodies and the tool messages sent back.
- Billing (https://platform.kimi.ai/docs/pricing/websearch): $0.005 per Formula web-search execution; failed
  requests and empty results are not billed. web_search_requests counts fibers that succeeded with a non-empty
  result (no usage field reports searches). Search results are billed as input tokens on the next chat call, so
  usage, cache headers and cost are summed over every chat completion of the loop.
- Continuation replays the whole loop: the request messages, each tool-calling assistant message and its tool
  messages, then the final assistant message, exactly as sent and returned. Follow-up turns offer the tool again.
"""

from __future__ import annotations

import copy
import json
import re
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
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "reasoning_effort", "thinking", "web_search"})
DEFAULT_SEARCH_FORMULA = "moonshot/web-search:latest"
DEFAULT_SEARCH_ROUNDS = 3
ROUNDS_KEY = "harness_web_search_rounds"
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
        parameter_check = QualificationCheck(
            name="parameter_support",
            passed=not unsupported and "max_output_tokens" in explicit_names,
            detail=f"unsupported: {unsupported}" if unsupported else
            ("max_output_tokens must be explicit" if "max_output_tokens" not in explicit_names else "ok"),
        )
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "moonshot",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            parameter_check,
            model_check,
            context_check(config, requirements, context),
        ]
        probes: list[dict[str, Any]] = []
        if requirements.requires_continuation:
            # Probes are paid requests built from the configured parameters: send none for an invalid configuration.
            if parameter_check.passed is not True:
                checks.append(QualificationCheck(name="continuation", passed=None,
                                                 detail="not evaluated: configuration parameters are invalid"))
            elif model_check.passed is not True:
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
            # The baseline request, any web search tool rounds, then the returned message.
            replayed = snapshot.native_state["messages"]
            if replayed[: len(base_sent)] != base_sent or replayed[-1:] != [returned]:
                return QualificationCheck(name="continuation", passed=False,
                                          detail="snapshot does not hold the baseline request and returned message")
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = run(f"branch-{word}", follow, snapshot)
                sent = child.native_request["body"]["messages"]
                if sent[: len(replayed)] != replayed:
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
        if (web_search := explicit(config, "web_search")) is not None:
            _search_tool_name(web_search)
            _search_options(config)
            body["tools"] = [copy.deepcopy(web_search)]
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
        if "tools" in body:
            return self._execute_searching(config, body, transport)
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

    def _execute_searching(self, config: ModelConfiguration, first: dict[str, Any], transport: CapturingTransport
                           ) -> NativeExecution:
        """The Formula web search loop: chat completions, with each web_search call run as a fiber in between."""
        tool_name = _search_tool_name(first["tools"][0])
        formula, max_rounds = _search_options(config)
        rounds: list[dict[str, Any]] = []  # tool-calling completions, their fibers, and the tool messages sent back
        usages: list[tuple[dict[str, Any], Mapping[str, str]]] = []
        searches = 0

        def stop(kind: FailureKind, error_type: str, message: str, cost: CostIncurred, **extra: Any
                 ) -> NativeExecution:
            if not rounds:  # nothing billed yet: exactly the single-request failure
                return failed(kind, error_type, message, cost, **extra)
            unpriced = None if cost is CostIncurred.NO else f"{error_type} after billed requests; its cost is unknown"
            return NativeExecution(
                status="failed", native_response={ROUNDS_KEY: rounds}, request_id=extra.get("request_id"),
                cost_incurred=CostIncurred.YES, failure_kind=kind,
                observations=_loop_observations(config, usages, searches, unpriced=unpriced),
                error={"type": error_type, "message": message, **extra})

        try:
            client = self._client(config, transport)
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        messages = copy.deepcopy(first["messages"])
        with client:
            for completed_rounds in range(max_rounds + 1):
                body = {**first, "messages": messages} if rounds else first
                try:
                    response = client.post("chat/completions", content=json.dumps(body))
                except httpx2.TimeoutException as exc:
                    return stop(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
                except httpx2.TransportError as exc:
                    sent = transport.exchanges and "status" not in transport.exchanges[-1]
                    return stop(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc),
                                CostIncurred.UNKNOWN if sent else CostIncurred.NO)
                if response.status_code != 200:
                    kind, cost = _http_failure(response.status_code, _error_type(response))
                    return stop(kind, f"http_{response.status_code}", _error_message(response), cost,
                                status=response.status_code, provider_type=_error_type(response))
                if json.loads(transport.exchanges[-1]["request_body"]) != body:
                    raise AdapterError(FailureKind.HARNESS_ERROR,
                                       "request body on the wire differs from the prepared body", CostIncurred.YES)
                data = response.json()
                usages.append((data.get("usage") or {}, response.headers))
                choice = _choice(data)
                reason = choice.get("finish_reason")

                def failure(kind: FailureKind, error: dict[str, Any], data: dict[str, Any] = data,
                            reason: Any = reason) -> NativeExecution:
                    return NativeExecution(
                        status="failed", native_response={**data, ROUNDS_KEY: rounds}, request_id=data.get("id"),
                        finish_reason=reason, cost_incurred=CostIncurred.YES, failure_kind=kind,
                        observations=_loop_observations(config, usages, searches), error=error)

                if data.get("model") != config.model:
                    return failure(FailureKind.PROVIDER_ERROR,
                                   {"type": "model_mismatch", "requested": config.model, "served": data.get("model")})
                if not choice:
                    return failure(FailureKind.PROVIDER_ERROR, {"type": "no_choices"})
                if reason in ("stop", "length"):
                    return NativeExecution(
                        status="completed", native_response={**data, ROUNDS_KEY: rounds}, finish_reason=reason,
                        request_id=data.get("id"), cost_incurred=CostIncurred.YES,
                        observations=_loop_observations(config, usages, searches))
                if reason != "tool_calls":  # client_disconnected, undocumented values
                    return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason})
                message = choice.get("message") or {}
                calls = message.get("tool_calls") or []
                if completed_rounds == max_rounds:
                    return failure(FailureKind.PROVIDER_ERROR, {
                        "type": "incomplete",
                        "detail": f"the model still called tools after {max_rounds} web search rounds"})
                if not calls:
                    return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason,
                                                                "detail": "tool_calls finish without tool_calls"})
                record: dict[str, Any] = {"chat_completion": data, "fibers": [], "tool_messages": []}
                rounds.append(record)
                for call in calls:
                    function = call.get("function") or {}
                    if function.get("name") != tool_name:
                        return stop(FailureKind.PROVIDER_ERROR, "unexpected_tool_call",
                                    f"the model called {function.get('name')!r}; only {tool_name!r} is offered",
                                    CostIncurred.NO, request_id=data.get("id"))
                    try:  # name and arguments are passed through verbatim
                        fiber_response = client.post(f"formulas/{formula}/fibers", content=json.dumps(
                            {"name": function.get("name"), "arguments": function.get("arguments")}))
                    except httpx2.TimeoutException as exc:
                        return stop(FailureKind.AMBIGUOUS_TIMEOUT, "tool_timeout", str(exc), CostIncurred.UNKNOWN,
                                    request_id=data.get("id"))
                    except httpx2.TransportError as exc:
                        return stop(FailureKind.TRANSIENT_TRANSPORT, "tool_connection", str(exc),
                                    CostIncurred.UNKNOWN, request_id=data.get("id"))
                    if fiber_response.status_code != 200:  # failed requests are not billed
                        kind, _ = _http_failure(fiber_response.status_code, _error_type(fiber_response))
                        return stop(kind, f"tool_http_{fiber_response.status_code}", _error_message(fiber_response),
                                    CostIncurred.NO, status=fiber_response.status_code,
                                    provider_type=_error_type(fiber_response), request_id=data.get("id"))
                    fiber = fiber_response.json()
                    record["fibers"].append(fiber)
                    if not isinstance(fiber, dict) or fiber.get("status") != "succeeded":
                        status = fiber.get("status") if isinstance(fiber, dict) else None
                        return stop(FailureKind.PROVIDER_ERROR, "tool_error", f"fiber status {status!r}",
                                    CostIncurred.NO, fiber_status=status, request_id=data.get("id"))
                    context = fiber.get("context") or {}
                    result = context.get("output") or context.get("encrypted_output") or ""
                    if result:  # empty results are not billed
                        searches += 1
                    record["tool_messages"].append({
                        "role": "tool", "tool_call_id": call.get("id"),
                        "content": result if isinstance(result, str) else json.dumps(result)})
                # The complete assistant message (reasoning_content and tool_calls) is sent back unchanged.
                messages = messages + [copy.deepcopy(message)] + copy.deepcopy(record["tool_messages"])
        raise AssertionError("unreachable: the loop returns on its last round")  # pragma: no cover

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        choice = _choice(data)
        message = choice.get("message") or {}
        # Reasoning from web search tool rounds precedes the final message's; the answer is the final content only.
        reasoning = [str(m["reasoning_content"]) for m in _round_messages(data) + [message]
                     if m.get("reasoning_content")]
        return ExtractedResponse(
            final_answer=str(message.get("content") or ""),
            truncated=choice.get("finish_reason") == "length",
            finish_reason=choice.get("finish_reason"),
            reasoning_text="\n\n".join(reasoning) or None,
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
                # The assistant message (content and reasoning_content) is replayed exactly as returned, after any
                # web search tool rounds exactly as they were sent.
                "messages": copy.deepcopy(body["messages"]) + _transcript(data)
                + [copy.deepcopy(_choice(data).get("message") or {})],
            },
        )


def _search_tool_name(value: Any) -> str:
    """The function name of a Formula web-search tool declaration; anything else is an invalid configuration."""
    if isinstance(value, dict) and value.get("type") == "builtin_function":
        raise AdapterError(FailureKind.INVALID_STATE,
                           "the $web_search builtin_function retired on 2026-10-20; use the Formula web-search tool "
                           "declaration from GET /v1/formulas/moonshot/web-search:latest/tools")
    function = value.get("function") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("type") != "function" or not isinstance(function, dict)
            or not isinstance(function.get("name"), str) or not function["name"]
            or not isinstance(function.get("parameters"), dict)):
        raise AdapterError(FailureKind.INVALID_STATE, "web_search must be the Formula web-search tool declaration "
                                                      "{type: function, function: {name, description, parameters}}")
    return str(function["name"])


def _search_options(config: ModelConfiguration) -> tuple[str, int]:
    formula = config.adapter_options.get("web_search_formula", DEFAULT_SEARCH_FORMULA)
    rounds = config.adapter_options.get("web_search_max_rounds", DEFAULT_SEARCH_ROUNDS)
    if not isinstance(formula, str) or not re.fullmatch(r"moonshot/web-search(:[\w.-]+)?", formula):
        raise AdapterError(FailureKind.INVALID_STATE,
                           "web_search_formula must be the official web-search formula "
                           "(e.g. moonshot/web-search:latest)")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise AdapterError(FailureKind.INVALID_STATE, "web_search_max_rounds must be a positive integer")
    return formula, rounds


def _rounds(data: dict[str, Any]) -> list[dict[str, Any]]:
    rounds = data.get(ROUNDS_KEY) or []
    return [r for r in rounds if isinstance(r, dict)]


def _round_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [_choice(r.get("chat_completion") or {}).get("message") or {} for r in _rounds(data)]


def _transcript(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The web search tool rounds as sent: each tool-calling assistant message, then its tool messages."""
    messages: list[dict[str, Any]] = []
    for record, message in zip(_rounds(data), _round_messages(data)):
        messages += [copy.deepcopy(message)] + copy.deepcopy(record.get("tool_messages") or [])
    return messages


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
    if status == 499:  # connection closed before the response was returned; inference may already have run
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN
    if status == 504:  # gateway gave up after 900 s; inference may have run
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status >= 500:  # 500 server_error, 503 server_unavailable: retry after a brief wait
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    # 400 invalid_request_error (including input over the context window), 401, 403, 404: retrying cannot help
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any], headers: Mapping[str, str]
                  ) -> dict[str, Observation]:
    obs = _token_observations(usage, headers)
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _loop_observations(config: ModelConfiguration, usages: list[tuple[dict[str, Any], Mapping[str, str]]],
                       searches: int, *, unpriced: str | None = None) -> dict[str, Observation]:
    """Token counts summed over every chat completion of a web search loop, plus the searches it ran."""
    per_call = [_token_observations(usage, headers) for usage, headers in usages]
    obs: dict[str, Observation] = {}
    for name, first in per_call[0].items():
        if len(per_call) == 1:
            obs[name] = first
            continue
        values = [call[name].value for call in per_call]
        source = f"{first.source} (summed over {len(per_call)} chat completions)"
        missing = next((call[name].missing_reason for call in per_call if call[name].value is None), None)
        obs[name] = (Observation(value=sum(int(v or 0) for v in values), unit=first.unit, source=source)
                     if missing is None else
                     Observation.missing(first.unit, source, f"not reported for every chat completion: {missing}"))
    obs["web_search_requests"] = Observation(
        value=searches, unit="requests",
        source="counted from Formula web-search fibers that succeeded with a non-empty result")
    obs["api_cost_usd"] = (_cost(config, obs) if unpriced is None
                           else Observation.missing("usd", "configuration pricing", unpriced))
    return obs


def _token_observations(usage: dict[str, Any], headers: Mapping[str, str]) -> dict[str, Observation]:
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
    if cached is None and isinstance(writes, int):
        # The documented response reports the cache breakdown as prompt_tokens_details with only
        # cache_write_tokens: cached_tokens is omitted when nothing was read from the cache.
        cached = 0
        obs["cached_input_tokens"] = Observation(
            value=0, unit="tokens",
            source=f"{source}: cached_tokens omitted from a reported prompt_tokens_details (no cache reads)")
    if (writes is None and isinstance(cached, int) and details
            and not any(h in headers for h in CACHE_WRITE_HEADERS.values())):
        # Seen 2026-11-25 on a turn that wrote nothing: prompt_tokens_details carries only cached_tokens and
        # neither cache-write header is sent. Both absences together mean zero writes.
        writes = 0
        obs["cache_write_tokens"] = Observation(
            value=0, unit="tokens",
            source=f"{source}: cache_write_tokens and cache-write headers both absent (no cache writes)")
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
    if (searches := obs.get("web_search_requests")) is not None and searches.value:
        rate = pricing.get("web_search_usd_per_request")
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", "no price for web_search_requests")
        total += int(searches.value) * rate
    return Observation(value=round(total, 10), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')})")
