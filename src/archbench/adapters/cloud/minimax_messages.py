"""Native MiniMax adapter (Anthropic-compatible Messages API, `POST /anthropic/v1/messages` on api.minimax.io).

MiniMax marks its Anthropic-compatible API as the recommended one for the M3 series. It returns signed
`thinking` blocks that its request schema accepts back unchanged; the OpenAI-compatible request schema does
not document `reasoning_content` on input, and the native ChatCompletion v2 API has no effort control.

Plain HTTP, not the `anthropic` SDK: the recorded request is exactly the JSON body on the wire, and no
Anthropic credential can reach MiniMax (the SDK reads ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, and the
`ant` profile on its own).
- No client-side retries: every retry is a new harness attempt.
- Only explicitly configured parameters are sent. MiniMax ignores `temperature` (always 1), `top_p` (always
  0.95) and `top_k`, so they are not configurable: sending them would misrepresent what ran.
- Pay-as-you-go keys only. M Plan / Token Plan subscription keys use quota windows and plan-level model
  rerouting; the endpoint cannot tell them apart, so the key in secrets must be a pay-as-you-go key.
- Model ids are undated aliases, and MiniMax documents silent rerouting of retired ids, so every response's
  `model` must equal the configured id. Qualification proves the served id with a live probe.
- Continuation replays the assistant content exactly as returned, including signed thinking blocks ("preserve
  them unchanged in later turns"). Thinking is adaptive: some turns have no thinking block.
- Example responses carry `base_resp: {status_code, status_msg}`; a nonzero status in a 200 body is a failure,
  mapped by MiniMax's error-code table.

Cost uses configured prices. `usage.input_tokens` counts new input only; `cache_read_input_tokens` and
`cache_creation_input_tokens` are separate. Caching is automatic (prompts of 512+ tokens) and MiniMax lists
no cache-write price; writes without a configured price leave the cost unknown. Thinking counts toward
`max_tokens` and is billed as output; `usage.output_tokens_details.thinking_tokens` (undocumented, present when
the turn thought; seen 2026-11-16) is recorded as reasoning_tokens. Every 200 body seen carries
`base_resp: {"status_code": 0}`. Input usage includes roughly 190 tokens beyond the visible prompt, apparently a
server-side template or system prompt (a two-message probe reported 211 input tokens).

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_tokens (1..524288; thinking included)
    effort             explicit or omitted  -> output_config.effort ("low" | "medium" | "high" | "xhigh" | "max")
    thinking           explicit or omitted  -> thinking.type ("adaptive" | "disabled"; M3.1-Flash accepts only
                                               "adaptive")
    web_search         explicit or omitted  -> tools: [value] (the server tool, sent verbatim:
                                               {type: web_search_20250305, name: web_search})
adapter_options:
    pause_turn_max_resumes: 3 (default)   resumes of a paused server search loop before it counts as incomplete
    base_url: https://api.minimax.io/anthropic (default; mainland: https://api.minimax.cn/anthropic)
    timeout_s: 900 (default; MiniMax documents no non-streaming limit)
    pricing: {region, input_usd_per_mtok, cache_read_usd_per_mtok, cache_write_usd_per_mtok?,
              output_usd_per_mtok, web_search_usd_per_request?, source, verified_on}

Web search (Server Tools, Beta; verified 2026-11-24, https://platform.minimax.io/docs/guides/server-tools): the
`web_search` server tool runs on MiniMax's servers inside the one request; the model decides whether and how often
to search. The content then holds `server_tool_use` (input.query) and `web_search_tool_result` blocks among the
thinking and text blocks; continuation replays them unchanged with the rest of the content (MiniMax's own
pause_turn guidance sends that content back as-is). The documented tool has no options (no max_uses). Usage in the
documented example reports no search count, so web_search_requests is `usage.server_tool_use.web_search_requests`
when present, else the number of `server_tool_use` blocks named web_search. MiniMax prices web_search at
$0.01 per request (https://platform.minimax.io/docs/pricing/overview, Server Tools), read here as per search.
`stop_reason: pause_turn` (the server search loop reached its limit before a final answer) is resumed as the
Anthropic protocol documents: the same request is sent again with the paused assistant content appended (no extra
user message), up to `adapter_options.pause_turn_max_resumes` times (default 3); still paused after that is an
incomplete failure. A resumed turn's native response is the last body with `content` holding the whole assistant
content and `usage` summed over every call, the earlier raw bodies under `harness_pause_turns`; searches are
counted per call and summed. The final answer is the text blocks only, which include any lead-in text written before a search;
search results stay in the native response.
"""

from __future__ import annotations

import copy
import json
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
from .common import CollectingSink, context_check, explicit, failed, merge_pause_turns
from .wire import CapturingTransport, wire_record

STATE_FORMAT = "minimax-messages-v1"
DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"
HOSTS = {"api.minimax.io": "global", "api.minimax.cn": "mainland"}  # keys are per region
API_PATH = "/anthropic"
DEFAULT_TIMEOUT_S = 900
MAX_OUTPUT_TOKENS = 524288
PROBE_MAX_TOKENS = 2048
EFFORTS = ("low", "medium", "high", "xhigh", "max")
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "effort", "thinking", "web_search"})
CONTENT_TYPES = frozenset({"text", "thinking"})
SEARCH_CONTENT_TYPES = CONTENT_TYPES | {"server_tool_use", "web_search_tool_result"}
DEFAULT_PAUSE_RESUMES = 3
REQUEST_ID_HEADERS = ("trace-id",)  # equals the error body's request_id (seen 2026-11-16); else the response `id`
# base_resp.status_code values from MiniMax's error-code table.
BASE_RESP_FAILURES = {
    1002: FailureKind.RATE_LIMITED,       # rate limit
    1026: FailureKind.REFUSAL,            # input flagged as sensitive
    1027: FailureKind.REFUSAL,            # output flagged as sensitive
    2064: FailureKind.TRANSIENT_TRANSPORT,  # server overloaded
}
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no MiniMax credentials found: add minimax.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set MINIMAX_API_KEY")


class MiniMaxAdapter:
    name = "minimax"

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
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "minimax",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            _endpoint_check(config),
            QualificationCheck(name="parameter_support", passed=parameter_detail == "ok", detail=parameter_detail),
            context_check(config, requirements, None),
        ]
        probes: list[dict[str, Any]] = []
        static_failures = [c.name for c in checks[:3] if c.passed is not True]
        if static_failures:  # never spend on a probe that cannot qualify
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
            detail=f"served model {data.get('model')} (undated alias; response id {data.get('id')})",
        )]
        if requirements.requires_continuation:
            checks.append(self._continuation_check(prepared, execution, run))
        return checks

    def _continuation_check(
        self, prepared: PreparedInvocation, execution: NativeExecution,
        run: Callable[[str, tuple[Message, ...], ContinuationSnapshot | None],
                      tuple[PreparedInvocation, NativeExecution]],
    ) -> QualificationCheck:
        returned = {"role": "assistant", "content": (execution.native_response or {}).get("content") or []}
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
        thinking = any(isinstance(b, dict) and b.get("type") == "thinking" for b in returned["content"])
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact content replay"
                   + (" (including signed thinking blocks)" if thinking else " (baseline had no thinking block)"),
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
        leading = next((i for i, m in enumerate(messages) if m.role is not Role.SYSTEM), len(messages))
        system = [m.content for m in messages[:leading]]
        if any(m.role is Role.SYSTEM for m in messages[leading:]):
            raise AdapterError(FailureKind.INVALID_STATE, "system messages must lead the conversation")
        if parent is not None:
            parent.verify_shape()
            state = parent.native_state
            if state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if state.get("model") != config.model:  # signed thinking blocks belong to the model that wrote them
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state belongs to a different model")
            if messages[: len(parent.messages)] != parent.messages:
                raise AdapterError(FailureKind.INVALID_STATE, "conversation does not extend the baseline snapshot")
            native = copy.deepcopy(state["messages"])
            new = messages[len(parent.messages):]
        else:
            native, new = [], messages[len(system):]
        for m in new:
            if m.role is Role.ASSISTANT:
                raise AdapterError(FailureKind.INVALID_STATE,
                                   "assistant turns must come from a continuation snapshot, not plain text")
            native.append({"role": m.role.value, "content": m.content})
        body: dict[str, Any] = {"model": config.model, "max_tokens": max_tokens, "messages": native}
        if parent is not None and parent.native_state.get("system") is not None:
            body["system"] = parent.native_state["system"]
        elif parent is None and system:
            body["system"] = "\n\n".join(system)
        if (effort := explicit(config, "effort")) is not None:
            body["output_config"] = {"effort": effort}
        if (thinking := explicit(config, "thinking")) is not None:
            body["thinking"] = {"type": thinking}
        if (web_search := explicit(config, "web_search")) is not None:
            if problem := _search_tool_problem(web_search):
                raise AdapterError(FailureKind.INVALID_STATE, problem)
            body["tools"] = [copy.deepcopy(web_search)]
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": f"POST {API_PATH}/v1/messages",
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
        searching = "tools" in body
        resumes = config.adapter_options.get("pause_turn_max_resumes", DEFAULT_PAUSE_RESUMES)
        if not isinstance(resumes, int) or isinstance(resumes, bool) or resumes < 0:
            raise AdapterError(FailureKind.INVALID_STATE, "pause_turn_max_resumes must be a non-negative integer")
        calls: list[dict[str, Any]] = []
        header_id: str | None = None
        current = body
        while True:
            outcome = self._post(config, current, transport)
            if isinstance(outcome, NativeExecution):
                if not calls:
                    return outcome
                merged = merge_pause_turns(calls)  # the paused calls were billed and stay in the evidence
                return outcome.model_copy(update={
                    "cost_incurred": CostIncurred.YES, "native_response": merged,
                    "observations": _observations(config, merged.get("usage") or {},
                                                  search=_summed_search(calls) if searching else None),
                    "error": {**(outcome.error or {}), "after_pause_turns": len(calls)}})
            data, header_id = outcome
            calls.append(data)
            if not (searching and data.get("stop_reason") == "pause_turn") or len(calls) > resumes \
                    or data.get("model") != config.model:
                break
            current = {**body, "messages": copy.deepcopy(body["messages"])
                       + [{"role": "assistant", "content": copy.deepcopy(merge_pause_turns(calls)["content"])}]}
        data = merge_pause_turns(calls)
        observations = _observations(config, data.get("usage") or {},
                                     search=_summed_search(calls) if searching else None)
        reason = data.get("stop_reason")
        request_id = header_id or data.get("id")

        def failure(kind: FailureKind, error: dict[str, Any], cost: CostIncurred = CostIncurred.YES
                    ) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=request_id,
                                   finish_reason=reason, cost_incurred=cost, failure_kind=kind,
                                   observations=observations, error=error)

        base = data.get("base_resp")
        if isinstance(base, dict) and base.get("status_code") not in (None, 0):
            code = base.get("status_code")
            kind = BASE_RESP_FAILURES.get(code, FailureKind.PROVIDER_ERROR) if isinstance(code, int) \
                else FailureKind.PROVIDER_ERROR
            return failure(kind, {"type": "base_resp", "status_code": code, "status_msg": base.get("status_msg")},
                           CostIncurred.YES if len(calls) > 1 else CostIncurred.UNKNOWN)
        if data.get("model") != config.model:
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "model_mismatch", "requested": config.model, "served": data.get("model")})
        if reason == "pause_turn" and searching:  # still paused after every resume
            return failure(FailureKind.PROVIDER_ERROR, {"type": "incomplete", "reason": reason,
                                                        "detail": f"server tool loop still paused after {resumes} "
                                                                  "resumes"})
        if reason not in ("end_turn", "max_tokens"):  # tool_use, pause_turn (no tools are sent), undocumented
            return failure(FailureKind.PROVIDER_ERROR, {"type": "stop_reason", "reason": reason})
        content = data.get("content")
        allowed = SEARCH_CONTENT_TYPES if searching else CONTENT_TYPES
        if not isinstance(content, list) or any(not isinstance(b, dict) or b.get("type") not in allowed
                                                for b in content):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "unexpected_content", "content_types": sorted(
                {str(b.get("type")) if isinstance(b, dict) else type(b).__name__ for b in content or []})})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason,
                               request_id=request_id, cost_incurred=CostIncurred.YES, observations=observations)

    def _post(self, config: ModelConfiguration, body: dict[str, Any], transport: CapturingTransport
              ) -> NativeExecution | tuple[dict[str, Any], str | None]:
        """One Messages call: the response JSON and request id, or the failure."""
        before = len(transport.exchanges)
        try:
            with self._client(config, transport) as client:
                response = client.post("v1/messages", content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if len(transport.exchanges) > before else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        header_id = _request_id(response.headers)
        if response.status_code != 200:
            error = _error_body(response)
            kind, cost = _http_failure(response.status_code)
            return failed(kind, f"http_{response.status_code}", str(error.get("message") or response.text)[:500], cost,
                          status=response.status_code, provider_type=error.get("type"),
                          request_id=header_id or error.get("request_id"))
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data: dict[str, Any] = response.json()
        return data, header_id

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        blocks = data.get("content") or []
        text = "".join(str(b.get("text") or "") for b in blocks if b.get("type") == "text")
        thinking = [str(b["thinking"]) for b in blocks if b.get("type") == "thinking" and b.get("thinking")]
        return ExtractedResponse(
            final_answer=text,
            truncated=data.get("stop_reason") == "max_tokens",
            finish_reason=data.get("stop_reason"),
            reasoning_text="\n\n".join(thinking) or None,
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
                "response_id": data.get("id"),
                "system": body.get("system"),
                # Assistant content is replayed exactly as returned, including signed thinking blocks.
                "messages": copy.deepcopy(body["messages"])
                + [{"role": "assistant", "content": copy.deepcopy(data.get("content") or [])}],
            },
        )


def _base_url(config: ModelConfiguration) -> str:
    return str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)).rstrip("/") + "/"


def _endpoint_check(config: ModelConfiguration) -> QualificationCheck:
    base = _base_url(config)
    parts = urlsplit(base)
    region = HOSTS.get((parts.hostname or "").lower())
    if parts.scheme != "https" or region is None or parts.path.rstrip("/") != API_PATH:
        return QualificationCheck(name="endpoint", passed=False,
                                  detail=f"{base} is not MiniMax's Anthropic-compatible API "
                                         f"(https://api.minimax.io{API_PATH} or https://api.minimax.cn{API_PATH})")
    pricing = config.adapter_options.get("pricing")
    priced = pricing.get("region") if isinstance(pricing, dict) else None
    if isinstance(pricing, dict) and priced != region:
        return QualificationCheck(name="endpoint", passed=False,
                                  detail=f"{base} is the {region} platform but pricing is for {priced}")
    return QualificationCheck(name="endpoint", passed=True, detail=f"{base} ({region})")


def _parameter_problem(config: ModelConfiguration) -> str | None:
    names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
    if unsupported := sorted(names - SUPPORTED_PARAMETERS):
        return f"unsupported: {unsupported} (MiniMax ignores sampling parameters)"
    budget = explicit(config, "max_output_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= MAX_OUTPUT_TOKENS:
        return f"max_output_tokens must be an explicit integer in 1..{MAX_OUTPUT_TOKENS}"
    effort = explicit(config, "effort")
    if effort is not None and effort not in EFFORTS:
        return f"effort must be one of {', '.join(EFFORTS)}"
    thinking = explicit(config, "thinking")
    if thinking is not None and thinking not in ("adaptive", "disabled"):
        return "thinking must be 'adaptive' or 'disabled'"
    web_search = explicit(config, "web_search")
    return _search_tool_problem(web_search) if web_search is not None else None


def _search_tool_problem(value: Any) -> str | None:
    if (not isinstance(value, dict) or not str(value.get("type", "")).startswith("web_search_")
            or value.get("name") != "web_search"):
        return "web_search must be the web_search server tool {type: web_search_20250305, name: web_search}"
    return None


def _summed_search(calls: list[dict[str, Any]]) -> Observation:
    """Searches over every call of a (possibly resumed) turn, each counted on its own."""
    counts = [_search_count(call) for call in calls]
    if len(counts) == 1:
        return counts[0]
    return Observation(value=sum(int(c.value or 0) for c in counts), unit="requests",
                       source=f"summed over {len(counts)} calls: " + "; ".join(sorted({c.source for c in counts})))


def _search_count(data: dict[str, Any]) -> Observation:
    """Searches in one response: the usage count if MiniMax reports one, else the server_tool_use blocks."""
    usage = data.get("usage")
    server = usage.get("server_tool_use") if isinstance(usage, dict) else None
    reported = server.get("web_search_requests") if isinstance(server, dict) else None
    if isinstance(reported, int):
        return Observation(value=reported, unit="requests", source="minimax usage: server_tool_use.web_search_requests")
    content = data.get("content")
    calls = [b for b in content if isinstance(b, dict) and b.get("type") == "server_tool_use"
             and b.get("name") == "web_search"] if isinstance(content, list) else []
    return Observation(value=len(calls), unit="requests",
                       source="counted from server_tool_use web_search blocks (usage reports no search count)")


def _request_id(headers: Mapping[str, str]) -> str | None:
    return next((headers[h] for h in REQUEST_ID_HEADERS if headers.get(h)), None)


def _error_body(response: httpx2.Response) -> dict[str, Any]:
    """`{"type": "error", "request_id": ..., "error": {"type", "message"}}`."""
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(body, dict):
        return {}
    error = body.get("error")
    merged = dict(error) if isinstance(error, dict) else {}
    if body.get("request_id"):
        merged.setdefault("request_id", body["request_id"])
    return merged


def _http_failure(status: int) -> tuple[FailureKind, CostIncurred]:
    if status == 429:  # rate_limit_error
        return FailureKind.RATE_LIMITED, CostIncurred.NO
    if status == 504:  # a gateway gave up; inference may have run
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status >= 500:  # 500 api_error, 529 overloaded_error ("retryable")
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    # 400 invalid request (including thinking disabled on M3.1-Flash), 401 key or region, 402 insufficient
    # balance (code 1008), 403, 404, 413
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any], *, search: Observation | None = None
                  ) -> dict[str, Observation]:
    source = "minimax usage"
    obs: dict[str, Observation] = {}
    for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = usage.get(name)
        obs[name] = (Observation(value=int(value), unit="tokens", source=source) if value is not None
                     else Observation.missing("tokens", source, "not reported in usage"))
    details = usage.get("output_tokens_details")
    thinking = details.get("thinking_tokens") if isinstance(details, dict) else None
    obs["reasoning_tokens"] = (
        Observation(value=int(thinking), unit="tokens", source=f"{source}: output_tokens_details.thinking_tokens")
        if thinking is not None else Observation.missing("tokens", source, "no thinking_tokens in usage"))
    if search is not None:
        obs["web_search_requests"] = search
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
        total += int(tokens) * rate / 1_000_000
    if (searches := obs.get("web_search_requests")) is not None and searches.value:
        rate = pricing.get("web_search_usd_per_request")
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", "no price for web_search_requests")
        total += int(searches.value) * rate
    return Observation(value=round(total, 10), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')})")
