"""Local Ollama adapter (`POST /api/chat`), for open-weight models served by a local Ollama engine.

The Ollama server is the isolated inference worker: it owns model loading and generation in its own
process, and the harness keeps scheduling and persistence. Every exchange with it is captured at the
transport boundary like a cloud call.

Evidence and control:
- The checkpoint is pinned by digest and the engine by version. Tags such as `qwen3.5:9b-mlx` are
  mutable (a re-pull changes the weights), so qualification and every invocation check the digest; a
  mismatch is a failure, never a silent substitution.
- `num_ctx` must be explicit. Some runners do not enforce it (the MLX runner was observed processing a
  prompt far beyond num_ctx, 2026-11-12), so a call whose prompt plus output exceeds the configured window
  is recorded as context_exceeded: it did not run under the configuration that was declared.
- Only explicitly configured options are sent; omitted ones use the model's Modelfile defaults, which
  qualification records.
- Continuation replays the exact native messages, including the assistant `thinking` field as returned.
- No API billing: api_cost_usd is 0. Electricity and hardware amortisation are not estimated.
- Timing comes from the engine (load, prompt evaluation, generation); memory from `/api/ps` after the call.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> options.num_predict (includes thinking tokens)
    context_window     explicit, required   -> options.num_ctx
    think              explicit or omitted  -> think
    temperature, top_p, top_k, min_p, presence_penalty, seed
                       explicit or omitted  -> options.<same name>
adapter_options:
    digest: <sha256>            required; the model digest from `ollama list` / `/api/tags`
    engine_version: <version>   required; `ollama --version`
    base_url: http://127.0.0.1:11434 (default)    timeout_s: 1800 (default)    keep_alive: 10m (default)
"""

from __future__ import annotations

import copy
import json
import os
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
from ..cloud.common import CollectingSink, context_check, explicit, failed
from ..cloud.wire import CapturingTransport, wire_record

STATE_FORMAT = "ollama-chat-v1"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
PROBE_MAX_TOKENS = 2048
SAMPLING_OPTIONS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed")
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "context_window", "think", *SAMPLING_OPTIONS})
TransportFactory = Callable[[], httpx2.BaseTransport]


class OllamaAdapter:
    name = "ollama"

    def __init__(self, *, transport_factory: TransportFactory | None = None) -> None:
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        base = str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        timeout = float(str(config.adapter_options.get("timeout_s", 1800)))
        return httpx2.Client(base_url=base, transport=transport, timeout=timeout,
                             headers={"content-type": "application/json"})

    def preflight(self, config: ModelConfiguration) -> str | None:
        """An unreachable engine would fail every call. Makes one local request, no generation."""
        try:
            with self._client(config, CapturingTransport(self._transport_factory())) as client:
                client.get("/api/version")
        except httpx2.HTTPError as exc:
            return f"Ollama is not reachable at {config.adapter_options.get('base_url', DEFAULT_BASE_URL)}: {exc}"
        return None

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
        unsupported = sorted(explicit_names - SUPPORTED_PARAMETERS)
        missing = sorted({"max_output_tokens", "context_window"} - explicit_names)
        engine_check, model_check, model_context = self._identity_checks(config)
        window = explicit(config, "context_window")
        limit = min(v for v in (window, model_context) if isinstance(v, int)) if isinstance(window, int) else None
        checks = [
            QualificationCheck(
                name="adapter_identity", passed=config.adapter == self.name and config.track is Track.LOCAL,
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(
                name="parameter_support", passed=not unsupported and not missing,
                detail=(f"unsupported: {unsupported}" if unsupported else "")
                + (f" must be explicit: {missing}" if missing else "") or "ok",
            ),
            engine_check,
            model_check,
            context_check(config, requirements, limit),
        ]
        probes: list[dict[str, Any]] = []
        if requirements.requires_continuation:
            if model_check.passed is not True or engine_check.passed is not True:
                checks.append(QualificationCheck(name="continuation", passed=None,
                                                 detail="not evaluated: engine or model identity was not verified"))
            else:
                checks.append(self._continuation_check(config, probes))
                checks.append(self._memory_check(config))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _identity_checks(self, config: ModelConfiguration
                         ) -> tuple[QualificationCheck, QualificationCheck, int | None]:
        transport = CapturingTransport(self._transport_factory())
        try:
            with self._client(config, transport) as client:
                version = client.get("/api/version").json().get("version")
                tags = client.get("/api/tags").json().get("models") or []
                show_response = client.post("/api/show", content=json.dumps({"model": config.model}))
        except httpx2.HTTPError as exc:
            unreachable = QualificationCheck(name="engine_identity", passed=None, detail=f"not evaluated: {exc}")
            return unreachable, QualificationCheck(name="model_identity", passed=None,
                                                   detail="not evaluated: engine unreachable"), None
        wanted_engine = config.adapter_options.get("engine_version")
        engine = QualificationCheck(
            name="engine_identity", passed=wanted_engine is not None and version == wanted_engine,
            detail=f"ollama {version}" + ("" if version == wanted_engine else f"; configuration pins {wanted_engine}"),
        )
        wanted_digest = config.adapter_options.get("digest")
        local = next((m for m in tags if m.get("name") == config.model), None)
        problems = []
        if local is None:
            problems.append(f"{config.model} is not pulled")
        elif not wanted_digest or local.get("digest") != wanted_digest:
            problems.append(f"digest {str(local.get('digest'))[:12]} does not match pinned {str(wanted_digest)[:12]}")
        show = show_response.json() if show_response.status_code == 200 else {}
        info = show.get("model_info") or {}
        context = next((v for k, v in info.items() if k.endswith(".context_length") and isinstance(v, int)), None)
        window = explicit(config, "context_window")
        if isinstance(window, int) and isinstance(context, int) and window > context:
            problems.append(f"context_window {window} > model context_length {context}")
        think = explicit(config, "think")
        if think is not None and "thinking" not in (show.get("capabilities") or []):
            problems.append("think is set but the model has no thinking capability")
        levels = (show.get("thinking") or {}).get("values") if isinstance(show.get("thinking"), dict) else None
        if think is not None and isinstance(levels, list) and think not in levels:
            problems.append(f"think={think!r} is not one of the model's thinking values {levels}")
        details = show.get("details") or {}
        model = QualificationCheck(
            name="model_identity", passed=not problems,
            detail="; ".join(problems) or (
                f"{config.model} digest {str(wanted_digest)[:12]}, {details.get('parameter_size')} "
                f"{details.get('quantization_level')}, context_length {context}; Modelfile defaults: "
                f"{' '.join(str(show.get('parameters') or '').split())}"),
        )
        return engine, model, context

    def _continuation_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Local probe: a warm-up baseline, then two sibling follow-ups forked from the same snapshot."""
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
            probes.append({
                "step": step, "status": execution.status, "request_id": execution.request_id,
                "failure_kind": execution.failure_kind.value if execution.failure_kind else None,
                "cost_incurred": execution.cost_incurred.value, "cost_usd": 0.0,
                "max_output_tokens": probe_tokens,
                "observations": {k: v.value for k, v in execution.observations.items()},
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
            returned = (execution.native_response or {}).get("message") or {}
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = json.dumps(snapshot.native_state, sort_keys=True)
            base_messages_sent = prepared.native_request["body"]["messages"]
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = run(f"branch-{word}", follow, snapshot)
                sent = child.native_request["body"]["messages"]
                if sent[: len(base_messages_sent) + 1] != base_messages_sent + [returned]:
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
                   + (" (including thinking)" if returned.get("thinking") else ""),
        )

    def _memory_check(self, config: ModelConfiguration) -> QualificationCheck:
        """After the probes the model is loaded: it must be fully resident in accelerator memory."""
        loaded = self._loaded(config, CapturingTransport(self._transport_factory()))
        if loaded is None:
            return QualificationCheck(name="memory", passed=False, detail="model not reported as loaded by /api/ps")
        size, vram = loaded.get("size"), loaded.get("size_vram")
        host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return QualificationCheck(
            name="memory", passed=isinstance(size, int) and size == vram,
            detail=f"loaded {_gb(size)} GB, {_gb(vram)} GB in accelerator memory, host RAM {_gb(host)} GB, "
                   f"engine context {loaded.get('context_length')}",
        )

    def _loaded(self, config: ModelConfiguration, transport: CapturingTransport) -> dict[str, Any] | None:
        try:
            with self._client(config, transport) as client:
                models = client.get("/api/ps").json().get("models") or []
        except httpx2.HTTPError:
            return None
        return next((m for m in models if m.get("name") == config.model), None)

    # ----- invocation ------------------------------------------------------------------------------

    def prepare(
        self,
        config: ModelConfiguration,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        ctx: InvocationContext,
    ) -> PreparedInvocation:
        max_tokens, window = explicit(config, "max_output_tokens"), explicit(config, "context_window")
        if not isinstance(max_tokens, int) or not isinstance(window, int):
            raise AdapterError(FailureKind.INVALID_STATE, "max_output_tokens and context_window must be explicit integers")
        if parent is not None:
            parent.verify_shape()
            state = parent.native_state
            if state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if state.get("model") != config.model or state.get("digest") != config.adapter_options.get("digest"):
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state belongs to a different checkpoint")
            if messages[: len(parent.messages)] != parent.messages:
                raise AdapterError(FailureKind.INVALID_STATE, "conversation does not extend the baseline snapshot")
            native = copy.deepcopy(state["messages"])
            new = messages[len(parent.messages):]
            leading = 0
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

        options: dict[str, Any] = {"num_ctx": window, "num_predict": max_tokens}
        options.update({name: v for name in SAMPLING_OPTIONS if (v := explicit(config, name)) is not None})
        body: dict[str, Any] = {"model": config.model, "messages": native, "stream": False, "options": options,
                                "keep_alive": str(config.adapter_options.get("keep_alive", "10m"))}
        if (think := explicit(config, "think")) is not None:
            body["think"] = think
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": "POST /api/chat",
                "client": {"package": "httpx2", "version": httpx2.__version__, "retries": 0},
                "digest": config.adapter_options.get("digest"),  # pinned checkpoint, verified before each call
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
                tags = client.get("/api/tags").json().get("models") or []
                local = next((m for m in tags if m.get("name") == config.model), None)
                pinned = config.adapter_options.get("digest")
                if local is None or local.get("digest") != pinned:
                    return failed(FailureKind.PROVIDER_ERROR, "checkpoint_changed",
                                  f"{config.model} digest is {local.get('digest') if local else 'absent'}, "
                                  f"pinned {pinned}", CostIncurred.NO)
                response = client.post("/api/chat", content=json.dumps(body))
        except httpx2.TimeoutException as exc:
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.NO)
        except httpx2.TransportError as exc:
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), CostIncurred.NO)
        if response.status_code != 200:
            message = _error_message(response)
            return failed(FailureKind.PROVIDER_ERROR, f"http_{response.status_code}", message, CostIncurred.NO,
                          status=response.status_code)
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body")
        data = response.json()
        loaded = self._loaded(config, transport)
        observations = _observations(data, loaded)

        def failure(kind: FailureKind, error: dict[str, Any]) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, finish_reason=data.get("done_reason"),
                                   failure_kind=kind, observations=observations, error=error)

        if data.get("model") != config.model:
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "model_mismatch", "requested": config.model, "served": data.get("model")})
        if data.get("done") is not True or data.get("done_reason") not in ("stop", "length"):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "unfinished", "done_reason": data.get("done_reason")})
        used = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
        if used > body["options"]["num_ctx"]:
            return failure(FailureKind.CONTEXT_EXCEEDED,
                           {"type": "context_window_exceeded", "tokens": used, "num_ctx": body["options"]["num_ctx"]})
        return NativeExecution(status="completed", native_response=data, finish_reason=data.get("done_reason"),
                               cost_incurred=CostIncurred.NO, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        message = data.get("message") or {}
        return ExtractedResponse(
            final_answer=str(message.get("content") or ""),
            truncated=data.get("done_reason") == "length",
            finish_reason=data.get("done_reason"),
            reasoning_text=message.get("thinking") or None,
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
                "digest": prepared.native_request["digest"],
                # The assistant message is replayed exactly as returned, including `thinking`.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(data.get("message") or {})],
            },
        )


def _gb(value: Any) -> str:
    return f"{value / 1024**3:.1f}" if isinstance(value, int | float) else "?"


def _error_message(response: httpx2.Response) -> str:
    try:
        return str(response.json().get("error") or response.text[:500])
    except (json.JSONDecodeError, AttributeError):
        return response.text[:500]


def _observations(data: dict[str, Any], loaded: dict[str, Any] | None) -> dict[str, Observation]:
    source = "ollama response"
    obs: dict[str, Observation] = {}
    for name, key in (("input_tokens", "prompt_eval_count"), ("cached_input_tokens", "prompt_eval_cached_count"),
                      ("output_tokens", "eval_count")):
        value = data.get(key)
        obs[name] = (Observation(value=int(value), unit="tokens", source=f"{source}: {key}") if value is not None
                     else Observation.missing("tokens", source, f"{key} not reported"))
    obs["reasoning_tokens"] = Observation.missing("tokens", source, "included in eval_count; not reported separately")
    for name, key in (("engine_load_ms", "load_duration"), ("engine_prompt_eval_ms", "prompt_eval_duration"),
                      ("engine_eval_ms", "eval_duration"), ("engine_total_ms", "total_duration")):
        value = data.get(key)
        obs[name] = (Observation(value=round(int(value) / 1e6, 3), unit="ms", source=f"{source}: {key}")
                     if value is not None else Observation.missing("ms", source, f"{key} not reported"))
    tokens, eval_ns = data.get("eval_count"), data.get("eval_duration")
    obs["output_tokens_per_s"] = (
        Observation(value=round(tokens / (eval_ns / 1e9), 3), unit="tokens/s", source=f"{source}: eval_count / eval_duration")
        if isinstance(tokens, int) and isinstance(eval_ns, int) and eval_ns > 0
        else Observation.missing("tokens/s", source, "eval_count or eval_duration not reported")
    )
    for name, key in (("model_memory_bytes", "size"), ("model_accelerator_memory_bytes", "size_vram")):
        value = (loaded or {}).get(key)
        obs[name] = (Observation(value=int(value), unit="bytes", source=f"ollama /api/ps: {key}") if value is not None
                     else Observation.missing("bytes", "ollama /api/ps", "model not reported as loaded"))
    obs["api_cost_usd"] = Observation(value=0.0, unit="usd", source="local inference: no API billing")
    obs["energy"] = Observation.missing("kWh", "local inference", "electricity and hardware amortisation not estimated")
    return obs
