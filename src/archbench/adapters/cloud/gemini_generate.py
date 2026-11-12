"""Native Gemini Developer API adapter (`POST /{version}/models/{model}:generateContent`).

Plain HTTP against the documented REST API rather than the `google-genai` SDK, for evidence fidelity:
- The recorded native request is exactly the JSON body on the wire. The SDK would convert field names
  and decode thought signatures to bytes and re-encode them; this adapter never touches them.
- No client-side retries: every retry is a new harness attempt.
- Only explicitly configured parameters are sent. A response from a different model is a failure.
- Continuation is stateless and exact: the snapshot holds the contents sent plus the model's returned
  content (every part, including thought signatures), which follow-up branches replay unchanged.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> generationConfig.maxOutputTokens
    thinking_level     explicit or omitted  -> generationConfig.thinkingConfig.thinkingLevel
    include_thoughts   explicit or omitted  -> generationConfig.thinkingConfig.includeThoughts
    temperature        explicit or omitted  -> generationConfig.temperature
adapter_options:
    api_version: v1 (default) | v1beta     base_url: https://generativelanguage.googleapis.com
    timeout_s: <float> (default 600)
    pricing: {input_usd_per_mtok, output_usd_per_mtok, cached_input_usd_per_mtok?,
              long_context?: {threshold_input_tokens, <same rate keys>}, source, verified_on}

Usage: cached tokens are part of promptTokenCount; billed output is candidates plus thoughts tokens
("output price including thinking tokens").
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

STATE_FORMAT = "gemini-generate-content-v1"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
PROBE_MAX_TOKENS = 2048
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "thinking_level", "include_thoughts", "temperature"})
# Finish reasons that mean the provider declined or blocked the output.
REFUSAL_REASONS = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
                             "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION"})
TransportFactory = Callable[[], httpx2.BaseTransport]


class GeminiAdapter:
    name = "gemini"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        base = str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        timeout = float(str(config.adapter_options.get("timeout_s", 600)))
        return httpx2.Client(base_url=base, transport=transport, timeout=timeout,
                             headers={"x-goog-api-key": self._api_key, "content-type": "application/json"})

    @staticmethod
    def _version(config: ModelConfiguration) -> str:
        return str(config.adapter_options.get("api_version", "v1"))

    def preflight(self, config: ModelConfiguration) -> str | None:
        return None if self._api_key else _NO_KEY

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
        unsupported = sorted(explicit_names - SUPPORTED_PARAMETERS)
        model_check, limits = self._model_check(config)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "google",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(
                name="parameter_support",
                passed=not unsupported and "max_output_tokens" in explicit_names,
                detail=f"unsupported: {unsupported}" if unsupported else
                ("max_output_tokens must be explicit" if "max_output_tokens" not in explicit_names else "ok"),
            ),
            model_check,
            context_check(config, requirements, limits.get("inputTokenLimit")),
        ]
        probes: list[dict[str, Any]] = []
        if requirements.requires_continuation:
            if model_check.passed is not True:
                checks.append(QualificationCheck(name="continuation", passed=None,
                                                 detail="not evaluated: model identity was not verified"))
            else:
                checks.append(self._continuation_check(config, probes))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _model_check(self, config: ModelConfiguration) -> tuple[QualificationCheck, dict[str, Any]]:
        """Verify model identity and limits against the first-party Models API."""
        transport = CapturingTransport(self._transport_factory())
        try:
            with self._client(config, transport) as client:
                response = client.get(f"/{self._version(config)}/models/{config.model}")
        except AdapterError as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"), {}
        except httpx2.HTTPError as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"), {}
        if response.status_code != 200:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"HTTP {response.status_code}: {_error_status(response)}"), {}
        info = response.json()
        problems = []
        if info.get("name") != f"models/{config.model}":
            problems.append(f"API reports name {info.get('name')}")
        max_input, max_output = info.get("inputTokenLimit"), info.get("outputTokenLimit")
        requested = explicit(config, "max_output_tokens")
        if isinstance(max_input, int) and config.context_limit is not None and config.context_limit > max_input:
            problems.append(f"context_limit {config.context_limit} > inputTokenLimit {max_input}")
        if isinstance(max_output, int) and isinstance(requested, int) and requested > max_output:
            problems.append(f"max_output_tokens {requested} > outputTokenLimit {max_output}")
        check = QualificationCheck(
            name="model_identity", passed=not problems,
            detail="; ".join(problems) or f"{info.get('name')} (version {info.get('version')}): "
                                           f"inputTokenLimit={max_input} outputTokenLimit={max_output}",
        )
        return check, {"inputTokenLimit": max_input if isinstance(max_input, int) else None}

    def _continuation_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Live probe: one baseline, then two sibling follow-ups forked from the same snapshot.

        Passes only if both follow-ups complete, each replays the baseline model content exactly
        (including thought signatures), neither sees its sibling, and the snapshot is unchanged.
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
            content = _candidate(execution.native_response or {}).get("content") or {}
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = json.dumps(snapshot.native_state, sort_keys=True)
            base_contents = prepared.native_request["body"]["contents"]
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = run(f"branch-{word}", follow, snapshot)
                sent = child.native_request["body"]["contents"]
                if sent[: len(base_contents) + 1] != base_contents + [content]:
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
        signed = any("thoughtSignature" in p for p in content.get("parts") or [])
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact stateless replay"
                   + (" (including thought signatures)" if signed else " (no thought signature returned)"),
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
        system = [m.content for m in messages if m.role is Role.SYSTEM]
        if any(m.role is Role.SYSTEM for m in messages[len(system):]):
            raise AdapterError(FailureKind.INVALID_STATE, "system messages must lead the conversation")
        if parent is not None:
            parent.verify_shape()
            state = parent.native_state
            if state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if state.get("model") != config.model:
                # Thought signatures are bound to the model that produced them.
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state belongs to a different model")
            if messages[: len(parent.messages)] != parent.messages:
                raise AdapterError(FailureKind.INVALID_STATE, "conversation does not extend the baseline snapshot")
            contents = copy.deepcopy(state["contents"])
            system_instruction = copy.deepcopy(state.get("systemInstruction"))
            new = messages[len(parent.messages):]
        else:
            contents = []
            system_instruction = {"parts": [{"text": "\n\n".join(system)}]} if system else None
            new = messages[len(system):]
        for m in new:
            if m.role is Role.ASSISTANT:
                raise AdapterError(FailureKind.INVALID_STATE,
                                   "model turns must come from a continuation snapshot, not plain text")
            contents.append({"role": "user", "parts": [{"text": m.content}]})

        generation: dict[str, Any] = {"maxOutputTokens": max_tokens}
        thinking = {k: v for k, v in (("thinkingLevel", explicit(config, "thinking_level")),
                                      ("includeThoughts", explicit(config, "include_thoughts"))) if v is not None}
        if thinking:
            generation["thinkingConfig"] = thinking
        if (temperature := explicit(config, "temperature")) is not None:
            generation["temperature"] = temperature
        body: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system_instruction is not None:
            body["systemInstruction"] = system_instruction
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": f"POST /{self._version(config)}/models/{config.model}:generateContent",
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
        path = prepared.native_request["endpoint"].split(" ", 1)[1]
        try:
            with self._client(config, transport) as client:
                response = client.post(path, content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:
            # The request may have been processed and billed.
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        if response.status_code != 200:
            status = _error_status(response)
            if response.status_code == 429:
                kind = FailureKind.RATE_LIMITED
            elif response.status_code >= 500:
                kind = FailureKind.TRANSIENT_TRANSPORT
            else:
                kind = FailureKind.PROVIDER_ERROR
            return failed(kind, status or f"http_{response.status_code}", _error_message(response), CostIncurred.NO,
                          status=response.status_code)

        sent = json.loads(transport.exchanges[-1]["request_body"])
        if sent != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data = response.json()
        request_id = data.get("responseId")
        observations = _observations(config, data.get("usageMetadata") or {})

        def failure(kind: FailureKind, error: dict[str, Any]) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=request_id,
                                   finish_reason=_candidate(data).get("finishReason"), cost_incurred=CostIncurred.YES,
                                   failure_kind=kind, observations=observations, error=error)

        if data.get("modelVersion") != config.model:
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "model_mismatch", "requested": config.model, "served": data.get("modelVersion")})
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            return failure(FailureKind.REFUSAL, {"type": "prompt_blocked", "reason": blocked})
        candidate = _candidate(data)
        if not candidate:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_candidates"})
        reason = candidate.get("finishReason")
        if reason in REFUSAL_REASONS:
            return failure(FailureKind.REFUSAL, {"type": "refusal", "reason": reason,
                                                 "safety_ratings": candidate.get("safetyRatings")})
        if reason not in ("STOP", "MAX_TOKENS"):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason, request_id=request_id,
                               cost_incurred=CostIncurred.YES, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        candidate = _candidate(data)
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        thoughts = [p["text"] for p in parts if p.get("thought") and p.get("text")]
        return ExtractedResponse(
            final_answer=text,
            truncated=candidate.get("finishReason") == "MAX_TOKENS",
            finish_reason=candidate.get("finishReason"),
            reasoning_text="\n\n".join(thoughts) if thoughts else None,
        )

    def fork_state(
        self, prepared: PreparedInvocation, execution: NativeExecution, extracted: ExtractedResponse
    ) -> ContinuationSnapshot:
        data = execution.native_response or {}
        body = prepared.native_request["body"]
        content = copy.deepcopy(_candidate(data).get("content") or {"role": "model", "parts": []})
        return ContinuationSnapshot(
            adapter=self.name,
            configuration_key=str(data.get("modelVersion")),
            messages=prepared.messages + (Message(role=Role.ASSISTANT, content=extracted.final_answer),),
            native_state={
                "format": STATE_FORMAT,
                "model": data.get("modelVersion"),
                "response_id": data.get("responseId"),
                "systemInstruction": copy.deepcopy(body.get("systemInstruction")),
                # The model's content is replayed exactly as returned, including thought signatures.
                "contents": copy.deepcopy(body["contents"]) + [content],
            },
        )


_NO_KEY = ("no Gemini credentials found: add google.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set GOOGLE_API_KEY or GEMINI_API_KEY")


def _candidate(data: dict[str, Any]) -> dict[str, Any]:
    candidates = data.get("candidates") or []
    first = candidates[0] if candidates else {}
    return first if isinstance(first, dict) else {}


def _error(response: httpx2.Response) -> dict[str, Any]:
    try:
        value = response.json().get("error")
    except (json.JSONDecodeError, AttributeError):
        return {}
    return value if isinstance(value, dict) else {}


def _error_status(response: httpx2.Response) -> str | None:
    status = _error(response).get("status")
    return str(status) if status else None


def _error_message(response: httpx2.Response) -> str:
    return str(_error(response).get("message") or response.text[:500])


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "gemini usageMetadata"
    values = {
        "input_tokens": usage.get("promptTokenCount"),
        "cached_input_tokens": usage.get("cachedContentTokenCount", 0 if "promptTokenCount" in usage else None),
        "candidates_tokens": usage.get("candidatesTokenCount"),
        "reasoning_tokens": usage.get("thoughtsTokenCount", 0 if "candidatesTokenCount" in usage else None),
        "tool_use_prompt_tokens": usage.get("toolUsePromptTokenCount", 0 if "promptTokenCount" in usage else None),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if v is not None
        else Observation.missing("tokens", source, "not reported in usageMetadata")
        for name, v in values.items()
    }
    candidates, thoughts = values["candidates_tokens"], values["reasoning_tokens"]
    obs["output_tokens"] = (
        Observation(value=int(candidates) + int(thoughts or 0), unit="tokens",
                    source=f"{source}: candidatesTokenCount + thoughtsTokenCount")
        if candidates is not None else Observation.missing("tokens", source, "candidatesTokenCount not reported")
    )
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")

    def count(name: str) -> int | None:
        value = obs[name].value
        return int(value) if isinstance(value, int | float) else None

    input_tokens, output_tokens = count("input_tokens"), count("output_tokens")
    if input_tokens is None or output_tokens is None:
        return Observation.missing("usd", "configuration pricing", "input or output tokens not reported")
    if count("tool_use_prompt_tokens"):
        return Observation.missing("usd", "configuration pricing", "tool-use prompt tokens are not priced")
    rates: dict[str, Any] = pricing
    long_context = pricing.get("long_context")
    if isinstance(long_context, dict) and isinstance(threshold := long_context.get("threshold_input_tokens"), int):
        if input_tokens > threshold:
            rates = long_context
    cached = count("cached_input_tokens") or 0
    total = 0.0
    for tokens, key in ((input_tokens - cached, "input_usd_per_mtok"), (cached, "cached_input_usd_per_mtok"),
                        (output_tokens, "output_usd_per_mtok")):
        if not tokens:
            continue
        rate = rates.get(key)
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", f"no price for {key}")
        total += tokens * rate / 1_000_000
    return Observation(value=round(total, 8), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')})")
