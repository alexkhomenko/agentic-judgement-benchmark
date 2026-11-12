"""Native Mistral API adapter (`POST /v1/chat/completions` on api.mistral.ai).

Plain HTTP against the documented REST API, for evidence fidelity: the recorded request is exactly the
JSON body on the wire, and reasoning chunks (including their `signature`, which Mistral documents as
"Signature to replay some reasoning blocks across turns") are replayed byte for byte.
- No client-side retries: every retry is a new harness attempt.
- Only explicitly configured parameters are sent (`safe_prompt` and `stream` keep their false defaults).
- Pin a dated model id: Mistral warns that aliases ("-latest", major-only) update silently. A response
  that names a different model is a failure.
- Continuation replays the exact native messages, including the assistant message as returned, with its
  `thinking` chunks: the reasoning guide says "Do not strip ThinkChunk from assistant messages".

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_tokens
    reasoning_effort   explicit or omitted  -> reasoning_effort
    temperature        explicit or omitted  -> temperature
    top_p              explicit or omitted  -> top_p
    seed               explicit or omitted  -> random_seed
adapter_options:
    base_url: https://api.mistral.ai (default)    timeout_s: <float> (default 600)
    pricing: {input_usd_per_mtok, output_usd_per_mtok, cached_input_usd_per_mtok?, source, verified_on}

Usage: the chat schema reports prompt, completion, and total tokens only. Cached tokens are priced when a
response reports them (`prompt_tokens_details.cached_tokens` or `num_cached_tokens`); this adapter never
sends `prompt_cache_key`, so caching is not requested.
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

STATE_FORMAT = "mistral-chat-v1"
DEFAULT_BASE_URL = "https://api.mistral.ai"
PROBE_MAX_TOKENS = 2048
NATIVE_NAMES = {"max_output_tokens": "max_tokens", "reasoning_effort": "reasoning_effort",
                "temperature": "temperature", "top_p": "top_p", "seed": "random_seed"}
SUPPORTED_PARAMETERS = frozenset(NATIVE_NAMES)
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no Mistral credentials found: add mistral.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set MISTRAL_API_KEY")


class MistralAdapter:
    name = "mistral"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        base = str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        timeout = float(str(config.adapter_options.get("timeout_s", 600)))
        return httpx2.Client(base_url=base, transport=transport, timeout=timeout,
                             headers={"authorization": f"Bearer {self._api_key}", "content-type": "application/json"})

    def preflight(self, config: ModelConfiguration) -> str | None:
        return None if self._api_key else _NO_KEY

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
        unsupported = sorted(explicit_names - SUPPORTED_PARAMETERS)
        model_check, max_context = self._model_check(config)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "mistral",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(
                name="parameter_support",
                passed=not unsupported and "max_output_tokens" in explicit_names,
                detail=f"unsupported: {unsupported}" if unsupported else
                ("max_output_tokens must be explicit" if "max_output_tokens" not in explicit_names else "ok"),
            ),
            model_check,
            context_check(config, requirements, max_context),
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
        """Verify model identity, capability, limits, and deprecation against the first-party Models API."""
        transport = CapturingTransport(self._transport_factory())
        try:
            with self._client(config, transport) as client:
                response = client.get(f"/v1/models/{config.model}")
        except (AdapterError, httpx2.HTTPError) as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"), None
        if response.status_code != 200:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"HTTP {response.status_code}: {_error_message(response)}"), None
        card = response.json()
        problems = []
        if card.get("id") != config.model:
            problems.append(f"API reports id {card.get('id')} (aliases {card.get('aliases')}); pin that id")
        if not (card.get("capabilities") or {}).get("completion_chat"):
            problems.append("model does not support chat completion")
        max_context = card.get("max_context_length")
        if isinstance(max_context, int) and config.context_limit is not None and config.context_limit > max_context:
            problems.append(f"context_limit {config.context_limit} > max_context_length {max_context}")
        deprecation = card.get("deprecation")
        detail = (f"{card.get('id')}: max_context_length={max_context}, "
                  f"default_model_temperature={card.get('default_model_temperature')}, aliases={card.get('aliases')}"
                  + (f"; DEPRECATED {deprecation} (replacement {card.get('deprecation_replacement_model')})"
                     if deprecation else ""))
        return (QualificationCheck(name="model_identity", passed=not problems, detail="; ".join(problems) or detail),
                max_context if isinstance(max_context, int) else None)

    def _continuation_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Live probe: one baseline, then two sibling follow-ups forked from the same snapshot.

        Passes only if both follow-ups complete, each replays the baseline assistant message exactly
        (including thinking chunks and their signatures), neither sees its sibling, and the snapshot is
        unchanged.
        """
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
        content = returned.get("content")
        chunks = [c for c in content if isinstance(c, dict)] if isinstance(content, list) else []
        thinking = any(c.get("type") == "thinking" for c in chunks)
        signed = any(c.get("type") == "thinking" and c.get("signature") for c in chunks)
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact message replay"
                   + (" (including signed thinking chunks)" if signed else
                      " (including thinking chunks)" if thinking else " (no thinking chunk returned)"),
        )

    # ----- invocation ------------------------------------------------------------------------------

    def prepare(
        self,
        config: ModelConfiguration,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        ctx: InvocationContext,
    ) -> PreparedInvocation:
        if not isinstance(explicit(config, "max_output_tokens"), int):
            raise AdapterError(FailureKind.INVALID_STATE, "max_output_tokens must be an explicit integer")
        if parent is not None:
            parent.verify_shape()
            state = parent.native_state
            if state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if state.get("model") != config.model:
                # Reasoning signatures are bound to the model that produced them.
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
        body: dict[str, Any] = {"model": config.model, "messages": native}
        for name, native_name in NATIVE_NAMES.items():
            if (value := explicit(config, name)) is not None:
                body[native_name] = value
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
                response = client.post("/v1/chat/completions", content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:
            # The request may have been processed and billed.
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        if response.status_code != 200:
            if response.status_code == 429:
                kind = FailureKind.RATE_LIMITED
            elif response.status_code >= 500:
                kind = FailureKind.TRANSIENT_TRANSPORT
            else:
                kind = FailureKind.PROVIDER_ERROR
            return failed(kind, f"http_{response.status_code}", _error_message(response), CostIncurred.NO,
                          status=response.status_code)
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data = response.json()
        observations = _observations(config, data.get("usage") or {})
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
        if reason == "model_length":
            # Generation hit the model's context window, which qualification should have ruled out.
            return failure(FailureKind.CONTEXT_EXCEEDED, {"type": "model_length"})
        if reason not in ("stop", "length"):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason,
                               request_id=data.get("id"), cost_incurred=CostIncurred.YES, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        choice = _choice(data)
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, str) or content is None:
            text, thoughts = content or "", []
        else:
            text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
            thoughts = [part.get("text", "") for c in content if c.get("type") == "thinking"
                        for part in c.get("thinking") or [] if part.get("type") == "text"]
        return ExtractedResponse(
            final_answer=text,
            truncated=choice.get("finish_reason") == "length",
            finish_reason=choice.get("finish_reason"),
            reasoning_text="".join(thoughts) or None,
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
                # The assistant message is replayed exactly as returned, including thinking chunks.
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
    if isinstance(body, dict):
        for key in ("message", "detail", "error"):
            if body.get(key):
                return json.dumps(body[key])[:500] if not isinstance(body[key], str) else body[key][:500]
    return response.text[:500]


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "mistral usage"
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", usage.get("num_cached_tokens"))
    values = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "cached_input_tokens": cached,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
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
    input_tokens, output_tokens = obs["input_tokens"].value, obs["output_tokens"].value
    if not isinstance(input_tokens, int | float) or not isinstance(output_tokens, int | float):
        return Observation.missing("usd", "configuration pricing", "input or output tokens not reported")
    cached_value = obs["cached_input_tokens"].value
    cached = int(cached_value) if isinstance(cached_value, int | float) else 0
    total = 0.0
    for tokens, key in ((int(input_tokens) - cached, "input_usd_per_mtok"), (cached, "cached_input_usd_per_mtok"),
                        (int(output_tokens), "output_usd_per_mtok")):
        if not tokens:
            continue
        rate = pricing.get(key)
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", f"no price for {key}")
        total += tokens * rate / 1_000_000
    return Observation(value=round(total, 8), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')})")
