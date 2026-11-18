"""Native Microsoft AI adapter: MAI models deployed in your own Microsoft Foundry resource, called on the MAI
chat-completions route (`POST {resource}/mai/v1/chat/completions`, `api-key` header).

Plain HTTP against the REST API: the recorded request is exactly the JSON body on the wire.
- No client-side retries: every retry is a new harness attempt.
- `model` is the DEPLOYMENT name in your resource. Responses name the model, not the deployment, and not in the
  requested case (deployment `MAI-Thinking-1` is answered as `mai-thinking-1`, observed 2026-11-18), so the echo is
  compared case-insensitively against the configured `served_model` and the deployment name.
- No `/models` route exists on `/mai/v1` (404, observed 2026-11-18) and the deployment's model version is not
  readable with an API key. Qualification therefore verifies identity with one tiny live completion, and the
  configured `model_version` is declared, not verified.
- Only explicitly configured parameters are sent. Unknown request fields are silently ignored by the service
  (observed), so a silently ignored parameter could not be told apart from an honoured one: only
  `max_completion_tokens` is documented and supported. `reasoning_effort` is validated by the service
  (none, minimal, low, medium, high, xhigh, max; anything else is a 422) but undocumented, so it is not offered.
- Reasoning is never returned in clear. With `replay_reasoning: true` the adapter sends `reasoning_display:
  "encrypted"` (the only accepted value), and the assistant message comes back with
  `reasoning.encrypted_content`. Continuation replays the assistant message exactly as returned; the service
  re-reads the blob (prompt tokens grow by about the reasoning length) and rejects an altered blob with a 400.
  Without it, `reasoning` is null and earlier reasoning is not carried into follow-ups.
- Content safety: a block is a refusal (HTTP 400 with code `content_filter`, finish_reason `content_filter`, or a
  non-null `message.refusal`).

Every response so far reports a couple of cached prompt tokens even on a first request. Microsoft publishes no
cached-input price, so without `cached_input_usd_per_mtok` such a response has unknown cost; the upper bound
(cached tokens at the full input rate) is recorded beside it as api_cost_usd_if_uncached.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_completion_tokens (includes reasoning tokens)
adapter_options:
    base_url: https://<resource>.services.ai.azure.com (required)    timeout_s: 900 (default)
    served_model: the model name responses report (required)         model_version: declared deployment version
    max_output_tokens_limit: model output cap                         replay_reasoning: true | false (default)
    pricing: {input_usd_per_mtok, output_usd_per_mtok, cached_input_usd_per_mtok (optional), source, verified_on}
"""

from __future__ import annotations

import copy
import json
import re
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

STATE_FORMAT = "mai-chat-v1"
CHAT_PATH = "/mai/v1/chat/completions"
PROBE_MAX_TOKENS = 2048
IDENTITY_PROBE_TOKENS = 16
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens"})
REFUSAL_CODES = frozenset({"content_filter", "content_policy_violation"})
CONTEXT_OVERFLOW = re.compile(r"(?i)context.?(length|window)|maximum context|too (long|many tokens)")
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no Microsoft Foundry key found: add microsoft.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set AZURE_INFERENCE_CREDENTIAL")


class MaiAdapter:
    name = "mai"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    @staticmethod
    def _base_url(config: ModelConfiguration) -> str | None:
        base = str(config.adapter_options.get("base_url") or "").rstrip("/")
        return base if base.startswith("https://") and "<" not in base else None

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        base = self._base_url(config)
        if base is None:
            raise AdapterError(FailureKind.PROVIDER_ERROR,
                               "adapter_options.base_url must be your Foundry resource endpoint (https://...)")
        timeout = float(str(config.adapter_options.get("timeout_s", 900)))
        # The key travels only in `api-key`; nothing is read from AZURE_OPENAI_* or OPENAI_* settings.
        return httpx2.Client(base_url=base, transport=transport, timeout=timeout,
                             headers={"api-key": self._api_key, "content-type": "application/json"})

    def preflight(self, config: ModelConfiguration) -> str | None:
        if not self._api_key:
            return _NO_KEY
        if self._base_url(config) is None:
            return "adapter_options.base_url must be your Foundry resource endpoint (https://...)"
        return None

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
        unsupported = sorted(explicit_names - SUPPORTED_PARAMETERS)
        parameter_check = QualificationCheck(
            name="parameter_support",
            passed=not unsupported and "max_output_tokens" in explicit_names,
            detail=f"unsupported: {unsupported}" if unsupported else
            ("max_output_tokens must be explicit" if "max_output_tokens" not in explicit_names else "ok"),
        )
        probes: list[dict[str, Any]] = []
        model_check = self._model_check(config, probes)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "microsoft",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            parameter_check,
            model_check,
            context_check(config, requirements, None),
        ]
        if requirements.requires_continuation:
            # Probes are built from the configured parameters: send none for an invalid configuration.
            if parameter_check.passed is not True:
                checks.append(QualificationCheck(name="continuation", passed=None,
                                                 detail="not evaluated: configuration parameters are invalid"))
            elif model_check.passed is not True:
                checks.append(QualificationCheck(name="continuation", passed=None,
                                                 detail="not evaluated: model identity was not verified"))
            else:
                checks.append(self._continuation_check(config, probes))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _model_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Static limits first, then one tiny completion: the deployment must exist and serve `served_model`."""
        served = config.adapter_options.get("served_model")
        if not isinstance(served, str) or not served:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail="adapter_options.served_model is required (the model responses name)")
        limit = config.adapter_options.get("max_output_tokens_limit")
        requested = explicit(config, "max_output_tokens")
        if isinstance(limit, int) and isinstance(requested, int) and requested > limit:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"max_output_tokens {requested} > model max_output_tokens {limit}")
        if config.context_limit is None:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail="context_limit is required: the service publishes no model limits")
        if (problem := self.preflight(config)) is not None:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {problem}")
        body = {"model": config.model, "max_completion_tokens": IDENTITY_PROBE_TOKENS,
                "messages": [{"role": "user", "content": "Reply with exactly one word: ok"}]}
        transport = CapturingTransport(self._transport_factory())
        try:
            with self._client(config, transport) as client:
                response = client.post(CHAT_PATH, content=json.dumps(body))
        except (AdapterError, httpx2.HTTPError) as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}")
        finally:
            probes.append({"step": "identity", "max_output_tokens": IDENTITY_PROBE_TOKENS,
                           "wire": wire_record(transport.exchanges)})
        if response.status_code != 200:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"deployment {config.model}: HTTP {response.status_code}: "
                                             f"{_error_message(response)}")
        data = response.json()
        probes[-1].update({"status": "completed", "request_id": data.get("id"),
                           "cost_usd": _observations(config, data.get("usage") or {})["api_cost_usd"].value})
        echoed = data.get("model")
        if not _same_model(config, echoed):
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"deployment {config.model} serves {echoed}, expected {served}")
        version = config.adapter_options.get("model_version")
        return QualificationCheck(
            name="model_identity", passed=True,
            detail=f"deployment {config.model} serves {echoed}; context_limit {config.context_limit} and "
                   f"output limit {limit} from configuration; model_version {version or 'not declared'} "
                   "is declared, not verifiable with an API key")

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
        replay = bool(config.adapter_options.get("replay_reasoning"))
        try:
            prepared, execution = run("baseline", base_messages, None)
            if execution.status != "completed":
                return QualificationCheck(name="continuation", passed=False,
                                          detail=f"baseline failed: {execution.failure_kind} {execution.error}")
            returned = _choice(execution.native_response or {}).get("message") or {}
            if replay and not _encrypted_reasoning(returned):
                return QualificationCheck(name="continuation", passed=False,
                                          detail="replay_reasoning is on but the baseline returned no encrypted reasoning")
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
                   + (" (including encrypted reasoning, accepted by the service)" if replay else
                      " (reasoning not replayed: replay_reasoning is off)"),
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
            if state.get("deployment") != config.model:
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
        if config.adapter_options.get("replay_reasoning"):
            body["reasoning_display"] = "encrypted"
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": f"POST {CHAT_PATH}",
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
                response = client.post(CHAT_PATH, content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "configuration", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            # Includes the intermittent TLS EOF seen on Foundry before any request byte is accepted.
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        if response.status_code != 200:
            kind, cost = _status_failure(response)
            return failed(kind, f"http_{response.status_code}", _error_message(response), cost,
                          status=response.status_code, request_id=_request_id(response))
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data = response.json()
        observations = _observations(config, data.get("usage") or {})
        choice = _choice(data)
        reason = choice.get("finish_reason")
        message = choice.get("message") or {}

        def failure(kind: FailureKind, error: dict[str, Any]) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=data.get("id"),
                                   finish_reason=reason, cost_incurred=CostIncurred.YES, failure_kind=kind,
                                   observations=observations, error=error)

        if not _same_model(config, data.get("model")):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "model_mismatch", "requested": config.model,
                                                        "expected": config.adapter_options.get("served_model"),
                                                        "served": data.get("model")})
        if not choice:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_choices"})
        if reason in ("content_filter", "content filter") or message.get("refusal"):
            return failure(FailureKind.REFUSAL, {"type": "content_filter", "refusal": message.get("refusal")})
        if reason not in ("stop", "length"):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason})
        reasoned = (observations["reasoning_tokens"].value or 0) > 0
        if config.adapter_options.get("replay_reasoning") and reasoned and not _encrypted_reasoning(message):
            # Without the blob a follow-up would silently drop the model's reasoning from its context.
            return failure(FailureKind.PROVIDER_ERROR, {"type": "missing_encrypted_reasoning"})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason,
                               request_id=data.get("id"), cost_incurred=CostIncurred.YES, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        choice = _choice(data)
        message = choice.get("message") or {}
        reasoning = message.get("reasoning")
        text = (reasoning.get("content") or reasoning.get("summary")) if isinstance(reasoning, dict) else None
        return ExtractedResponse(
            final_answer=str(message.get("content") or ""),
            truncated=choice.get("finish_reason") == "length",
            finish_reason=choice.get("finish_reason"),
            reasoning_text=text if isinstance(text, str) and text else None,
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
                "deployment": body["model"],
                "model": data.get("model"),
                "response_id": data.get("id"),
                # The assistant message, including any encrypted reasoning, is replayed exactly as returned.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(_choice(data).get("message") or {})],
            },
        )


def _same_model(config: ModelConfiguration, echoed: Any) -> bool:
    if not isinstance(echoed, str):
        return False
    accepted = {config.model, config.adapter_options.get("served_model")}
    return echoed.casefold() in {str(a).casefold() for a in accepted if a}


def _encrypted_reasoning(message: dict[str, Any]) -> bool:
    reasoning = message.get("reasoning")
    return isinstance(reasoning, dict) and bool(reasoning.get("encrypted_content"))


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


def _request_id(response: httpx2.Response) -> str | None:
    return _error(response).get("request_id") or response.headers.get("apim-request-id")


def _status_failure(response: httpx2.Response) -> tuple[FailureKind, CostIncurred]:
    status, error = response.status_code, _error(response)
    if status == 400 and (error.get("code") in REFUSAL_CODES or error.get("type") in REFUSAL_CODES):
        return FailureKind.REFUSAL, CostIncurred.UNKNOWN  # whether a blocked prompt is billed is not documented
    if status in (400, 413) and CONTEXT_OVERFLOW.search(str(error.get("message") or "")):
        return FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO
    if status == 429:  # also the default Low quota tier (0 tokens per minute)
        return FailureKind.RATE_LIMITED, CostIncurred.NO
    if status in (408, 504):  # the request ran too long; the model may have run
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status >= 500:
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    # 400 invalid request (including an altered reasoning blob), 401/403 key or access, 404 unknown deployment,
    # 422 field validation: retrying cannot help
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "mai usage"
    prompt = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    values = {
        "input_tokens": prompt,
        "cached_input_tokens": cached,
        "uncached_input_tokens": prompt - cached if isinstance(prompt, int) and isinstance(cached, int) else None,
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if v is not None
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    obs.update(_cost(config, values))
    return obs


def _cost(config: ModelConfiguration, tokens: dict[str, Any]) -> dict[str, Observation]:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return {"api_cost_usd": Observation.missing("usd", "configuration pricing", "no pricing in configuration")}
    prompt, output = tokens["input_tokens"], tokens["output_tokens"]
    cached = tokens["cached_input_tokens"] or 0
    if not isinstance(prompt, int) or not isinstance(output, int):
        return {"api_cost_usd": Observation.missing("usd", "configuration pricing",
                                                    "input or output tokens not reported")}
    rate_in, rate_out = pricing.get("input_usd_per_mtok"), pricing.get("output_usd_per_mtok")
    if not isinstance(rate_in, int | float) or not isinstance(rate_out, int | float):
        return {"api_cost_usd": Observation.missing("usd", "configuration pricing", "incomplete pricing")}
    source = f"configuration pricing ({pricing.get('source', 'unspecified source')})"
    uncached_cost = (prompt * rate_in + output * rate_out) / 1_000_000
    rate_cached = pricing.get("cached_input_usd_per_mtok")
    if cached == 0:
        return {"api_cost_usd": Observation(value=round(uncached_cost, 10), unit="usd", source=source)}
    if isinstance(rate_cached, int | float):
        value = ((prompt - cached) * rate_in + cached * rate_cached + output * rate_out) / 1_000_000
        return {"api_cost_usd": Observation(value=round(value, 10), unit="usd", source=source)}
    return {
        "api_cost_usd": Observation.missing("usd", source, f"{cached} cached input tokens and no cached-input price"),
        "api_cost_usd_if_uncached": Observation(value=round(uncached_cost, 10), unit="usd",
                                                source=f"{source}; upper bound, cached tokens at the input rate"),
    }
