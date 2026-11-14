"""Local MLX-LM adapter: a pinned checkpoint run by `mlx-lm` in an isolated worker subprocess.

The worker (`archbench/workers/mlx_worker.py`) runs in its own pinned environment and holds one loaded
model; this adapter starts it, keeps it warm across calls with the same configuration, and replaces it
when the configuration changes, so only one benchmark model is loaded at a time.

Evidence and control (architecture.md sections 3 and 4, spec sections 5.3 and 6):
- Engine: the worker reports its package versions; every pinned version must match or nothing runs.
- Checkpoint: pinned by Hugging Face repository, commit, and the sha256 of every file in the snapshot.
  The worker hashes the snapshot before loading and refuses any difference. Hugging Face runs offline.
- Prompt: the checkpoint's own chat template is applied with exactly the configured template kwargs. The
  rendered prompt and the input token IDs passed to the engine are saved as artifacts.
- Context: `context_window` is enforced before generation: prompt tokens + max_output_tokens must fit.
- Sampling: mlx-lm defaults to greedy decoding, which is not the model developer's recommendation, so
  temperature must be explicit. Only explicit settings are sent; omitted ones are mlx-lm defaults (off).
  An omitted seed is drawn by the worker and recorded, so every sampled answer is reproducible.
- Continuation: follow-ups replay the native message list, with the assistant turn split into content and
  `reasoning_content` as parsed. The chat template decides what history shows (Qwen3.5's drops earlier
  thinking, as its model card recommends).
- Memory: MLX peak/active memory and KV-cache size from the worker; host memory pressure, swap traffic,
  and the worker's footprint from a controller-side sampler that survives a worker crash.
- No API billing: api_cost_usd is 0. Electricity and hardware amortisation are not estimated.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_tokens (includes thinking tokens and the stop token)
    context_window     explicit, required   -> enforced limit for prompt + max_tokens
    temperature        explicit, required   -> sampler temp
    top_p, top_k, min_p                     -> sampler
    presence_penalty, frequency_penalty, repetition_penalty and their *_context_size
                                            -> mlx-lm logits processors (window over the last N tokens)
    enable_thinking    explicit or omitted  -> chat template kwarg
    seed               explicit or omitted  -> mx.random.seed (omitted: drawn by the worker, recorded)
adapter_options:
    python: .venv-mlx/bin/python          required; the pinned worker environment
    engine: {package: version}            required; must include python, mlx, mlx-lm, transformers, tokenizers
    revision: <commit sha>                required; the model field is the repository
    files: {name: sha256}                 required; every file of the snapshot
    timeout_s: 1800   load_timeout_s: 600   prefill_step_size: 2048
    memory_preflight: {output_tokens: 64, max_swapout_mb: 64, max_pressure_level: 2}
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...domain.configuration import ModelConfiguration, ParameterState, Track
from ...domain.execution import CostIncurred, FailureKind, Observation
from ...domain.scenario import Message, Role
from ...workers.host_memory import HostProbe, MemorySampler
from ...workers.mlx_process import SubprocessWorker, WorkerExited, WorkerProcess, WorkerTimeout
from ...workers.mlx_worker import PROTOCOL
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

STATE_FORMAT = "mlx-lm-chat-v1"
PROBE_MAX_TOKENS = 4096
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "presence_context_size",
            "frequency_penalty", "frequency_context_size", "repetition_penalty", "repetition_context_size")
REQUIRED = frozenset({"max_output_tokens", "context_window", "temperature"})
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "context_window", "enable_thinking", "seed", *SAMPLING})
REQUIRED_PINS = ("python", "mlx", "mlx-lm", "transformers", "tokenizers")
HELLO_TIMEOUT_S = 120.0
WorkerFactory = Callable[[str], WorkerProcess]


class _WorkerFailure(Exception):
    def __init__(self, error_type: str, message: str, kind: FailureKind = FailureKind.PROVIDER_ERROR,
                 **extra: Any) -> None:
        super().__init__(message)
        self.error_type, self.kind, self.extra = error_type, kind, extra


class _Session:
    def __init__(self, key: str, worker: WorkerProcess, hello: dict[str, Any], load: dict[str, Any]) -> None:
        self.key, self.worker, self.hello, self.load = key, worker, hello, load
        self.load_reported = False


def manifest_sha256(files: Any) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class MlxLmAdapter:
    name = "mlx-lm"

    def __init__(self, *, worker_factory: WorkerFactory | None = None, probe: HostProbe | None = None,
                 sample_interval_s: float = 0.25) -> None:
        self._factory: WorkerFactory = worker_factory or SubprocessWorker
        self._probe = probe or HostProbe()
        self._interval = sample_interval_s
        self._session: _Session | None = None
        self._lock = threading.RLock()
        atexit.register(self.close)

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.worker.close()
                self._session = None

    def preflight(self, config: ModelConfiguration) -> str | None:
        python = config.adapter_options.get("python")
        if not isinstance(python, str):
            return "adapter_options.python is not set"
        return None if Path(python).expanduser().exists() else f"worker interpreter {python} does not exist"

    # ----- worker session --------------------------------------------------------------------------

    def _session_key(self, config: ModelConfiguration) -> str:
        opts = config.adapter_options
        return json.dumps([opts.get("python"), opts.get("engine"), config.model, opts.get("revision"),
                           manifest_sha256(opts.get("files"))], sort_keys=True)

    def _ensure(self, config: ModelConfiguration) -> tuple[_Session, bool]:
        """The loaded session for this configuration, starting (and loading) a new worker if needed."""
        key = self._session_key(config)
        if self._session is not None and self._session.key == key and self._session.worker.pid is not None:
            return self._session, False
        self.close()
        opts = config.adapter_options
        try:
            worker = self._factory(str(opts.get("python")))
        except (FileNotFoundError, OSError) as exc:
            raise _WorkerFailure("worker_start", str(exc)) from None
        try:
            hello = self._call(worker, {"op": "hello"}, HELLO_TIMEOUT_S)
            if hello.get("protocol") != PROTOCOL:
                raise _WorkerFailure("protocol_mismatch", f"worker speaks {hello.get('protocol')}, expected {PROTOCOL}")
            if problems := _version_problems(opts.get("engine"), hello.get("versions") or {}):
                raise _WorkerFailure("engine_mismatch", "; ".join(problems), versions=hello.get("versions"))
            load = self._call(worker, {"op": "load", "repository": config.model, "revision": opts.get("revision"),
                                       "files": opts.get("files")},
                              float(str(opts.get("load_timeout_s", 600))))
        except BaseException:
            worker.close()
            raise
        self._session = _Session(key, worker, hello, load)
        return self._session, True

    @staticmethod
    def _call(worker: WorkerProcess, message: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        try:
            reply = worker.request(message, timeout_s)
        except WorkerTimeout as exc:
            raise _WorkerFailure("timeout", str(exc), FailureKind.AMBIGUOUS_TIMEOUT) from None
        except WorkerExited as exc:
            raise _WorkerFailure("worker_exited", str(exc), **exc.status) from None
        if not reply.get("ok"):
            error = dict(reply.get("error") or {})
            kind = FailureKind.CONTEXT_EXCEEDED if error.get("type") == "context_exceeded" else FailureKind.PROVIDER_ERROR
            raise _WorkerFailure(str(error.pop("type", "worker_error")), str(error.pop("message", "")), kind, **error)
        result: dict[str, Any] = reply.get("result") or {}
        return result

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
        unsupported = sorted(explicit_names - SUPPORTED_PARAMETERS)
        missing = sorted(REQUIRED - explicit_names)
        pins = _pin_problems(config)
        checks = [
            QualificationCheck(
                name="adapter_identity", passed=config.adapter == self.name and config.track is Track.LOCAL,
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(
                name="parameter_support", passed=not unsupported and not missing,
                detail=(f"unsupported: {unsupported}" if unsupported else "")
                + (f" must be explicit: {missing} (mlx-lm defaults to greedy decoding)" if missing else "") or "ok",
            ),
            QualificationCheck(name="configuration_pins", passed=not pins, detail="; ".join(pins) or "ok"),
        ]
        probes: list[dict[str, Any]] = []
        if pins or missing:  # probes need the pins and the required parameters
            reason = "pins missing" if pins else f"required parameters missing: {missing}"
            checks.append(QualificationCheck(name="engine_identity", passed=None, detail=f"not evaluated: {reason}"))
            return QualificationReport.from_checks(config, self.name, requirements, checks, probes)
        with self._lock:
            session: _Session | None = None
            try:
                session, _ = self._ensure(config)
            except _WorkerFailure as exc:
                engine_ok = exc.error_type not in ("engine_mismatch", "protocol_mismatch", "worker_start")
                checks.append(QualificationCheck(
                    name="engine_identity", passed=None if engine_ok else False,
                    detail=f"{exc.error_type}: {exc}" if not engine_ok else "not evaluated: worker failed to load"))
                checks.append(QualificationCheck(name="model_identity", passed=False, detail=f"{exc.error_type}: {exc}"))
            if session is None:
                return QualificationReport.from_checks(config, self.name, requirements, checks, probes)
            checks.append(self._engine_check(session))
            model_check, max_position = self._model_check(config, session)
            checks.append(model_check)
            window = explicit(config, "context_window")
            limits = [v for v in (window, max_position) if isinstance(v, int)]
            checks.append(context_check(config, requirements, min(limits) if limits else None))
            if requirements.requires_continuation and model_check.passed:
                checks.extend(self._continuation_checks(config, session, probes))
                checks.append(self._memory_preflight(config, probes))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _engine_check(self, session: _Session) -> QualificationCheck:
        versions = session.hello.get("versions") or {}
        device = session.hello.get("device") or {}
        host = self._probe.host_info()
        return QualificationCheck(
            name="engine_identity", passed=True,
            detail=f"mlx-lm {versions.get('mlx-lm')}, mlx {versions.get('mlx')}, transformers "
                   f"{versions.get('transformers')}, python {versions.get('python')}; {device.get('device_name')} "
                   f"{_gb(device.get('memory_size'))} GB, recommended working set "
                   f"{_gb(device.get('max_recommended_working_set_size'))} GB; macOS {host.get('os_version')} "
                   f"({host.get('os_build')}), {host.get('model')}",
        )

    def _model_check(self, config: ModelConfiguration, session: _Session) -> tuple[QualificationCheck, int | None]:
        load = session.load
        model, tokenizer = load.get("model") or {}, load.get("tokenizer") or {}
        pinned = config.adapter_options.get("files")
        files: dict[str, Any] = pinned if isinstance(pinned, dict) else {}
        problems = []
        max_position = model.get("max_position_embeddings")
        window = explicit(config, "context_window")
        if isinstance(window, int) and isinstance(max_position, int) and window > max_position:
            problems.append(f"context_window {window} > max_position_embeddings {max_position}")
        template_file = files.get("chat_template.jinja")
        if template_file and tokenizer.get("chat_template_sha256") != template_file:
            problems.append("the tokenizer's chat template is not the snapshot's chat_template.jinja")
        if not tokenizer.get("chat_template_sha256"):
            problems.append("checkpoint has no chat template")
        if explicit(config, "enable_thinking") is True and not tokenizer.get("think_end"):
            problems.append("enable_thinking is true but the tokenizer has no thinking markers")
        quant = model.get("quantization") or {}
        return QualificationCheck(
            name="model_identity", passed=not problems,
            detail="; ".join(problems) or (
                f"{config.model}@{str(config.adapter_options.get('revision'))[:12]}, {len(files)} files match the "
                f"pinned sha256 manifest ({manifest_sha256(files)[:12]}); {model.get('class')}, "
                f"{model.get('parameters')} parameters, {quant.get('bits')}-bit {quant.get('mode')} group "
                f"{quant.get('group_size')} ({model.get('bits_per_weight')} bits/weight), weights "
                f"{_gb(load.get('weights_memory_bytes'))} GB; max_position_embeddings {max_position}; eos "
                f"{tokenizer.get('eos_token_ids')}; thinking markers {tokenizer.get('think_start')!r}/"
                f"{tokenizer.get('think_end')!r}; generation_config.json "
                f"{'present' if load.get('generation_config_present') else 'absent'}; load "
                f"{load.get('load_ms')} ms, warm-up {(load.get('warmup') or {}).get('ms')} ms"),
        ), max_position if isinstance(max_position, int) else None

    def _continuation_checks(self, config: ModelConfiguration, session: _Session, probes: list[dict[str, Any]]
                             ) -> list[QualificationCheck]:
        """A baseline and two sibling follow-ups forked from it (spec 6.3 items 3-6)."""
        budget = explicit(config, "max_output_tokens")
        probe_tokens = min(budget, PROBE_MAX_TOKENS) if isinstance(budget, int) else PROBE_MAX_TOKENS
        params = dict(config.parameters)
        params["max_output_tokens"] = params["max_output_tokens"].model_copy(update={"value": probe_tokens})
        probe = config.model_copy(update={"parameters": params})
        outputs: list[tuple[str, NativeExecution, ExtractedResponse | None]] = []

        def run(step: str, messages: tuple[Message, ...], parent: ContinuationSnapshot | None
                ) -> tuple[PreparedInvocation, NativeExecution]:
            ctx = InvocationContext(experiment_id="qualification", task_id=f"qualification:{config.key}",
                                    attempt_id=step, attempt_no=1, round=1 if parent is None else 2,
                                    purpose="qualification")
            sink = CollectingSink()
            prepared = self.prepare(probe, messages, parent, ctx)
            execution = self.execute(probe, prepared, ctx, sink)
            extracted = self.extract(execution) if execution.status == "completed" else None
            outputs.append((step, execution, extracted))
            probes.append({
                "step": step, "status": execution.status,
                "failure_kind": execution.failure_kind.value if execution.failure_kind else None,
                "error": execution.error, "cost_incurred": execution.cost_incurred.value, "cost_usd": 0.0,
                "max_output_tokens": probe_tokens, "finish_reason": execution.finish_reason,
                "final_answer": extracted.final_answer if extracted else None,
                "reasoning_chars": len(extracted.reasoning_text or "") if extracted else None,
                "observations": {k: v.value for k, v in execution.observations.items()},
                "rendered_prompt": sink.items.get("rendered-prompt.txt"),
            })
            return prepared, execution

        base = (Message(role=Role.SYSTEM, content="You are answering a harness qualification probe."),
                Message(role=Role.USER, content="Reply with exactly one word: baseline"))
        continuation: QualificationCheck
        try:
            prepared, execution = run("baseline", base, None)
            if execution.status != "completed":
                continuation = QualificationCheck(name="continuation", passed=False,
                                                  detail=f"baseline failed: {execution.failure_kind} {execution.error}")
            else:
                snapshot = self.fork_state(prepared, execution, self.extract(execution))
                before = json.dumps(snapshot.native_state, sort_keys=True)
                baseline_sent = prepared.native_request["body"]["messages"]
                replayed = snapshot.native_state["messages"]
                continuation = QualificationCheck(
                    name="continuation", passed=True,
                    detail="baseline forked into two independent branches with exact native message replay; the "
                           "chat template renders history")
                for word in ("alpha", "beta"):
                    follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                    child, result = run(f"branch-{word}", follow, snapshot)
                    sent = child.native_request["body"]["messages"]
                    problem = None
                    if sent[: len(replayed)] != replayed or replayed[: len(baseline_sent)] != baseline_sent:
                        problem = f"branch {word} did not replay the baseline messages exactly"
                    elif word == "beta" and "alpha" in json.dumps(sent):
                        problem = "sibling branch leaked"
                    elif result.status != "completed":
                        problem = f"branch {word} failed: {result.failure_kind} {result.error}"
                    if problem:
                        continuation = QualificationCheck(name="continuation", passed=False, detail=problem)
                        break
                if continuation.passed and json.dumps(snapshot.native_state, sort_keys=True) != before:
                    continuation = QualificationCheck(name="continuation", passed=False,
                                                      detail="forking mutated baseline state")
        except AdapterError as exc:
            continuation = QualificationCheck(name="continuation", passed=False, detail=str(exc))
        return [continuation, _decoding_check(config, session.load.get("tokenizer") or {}, outputs)]

    def _memory_preflight(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Fill the full context window (prompt + output) and require no severe pressure or swapping."""
        opts = config.adapter_options.get("memory_preflight")
        opts = opts if isinstance(opts, dict) else {}
        out_tokens = int(str(opts.get("output_tokens", 64)))
        max_swapout = int(str(opts.get("max_swapout_mb", 64))) * 1024**2
        max_level = int(str(opts.get("max_pressure_level", 2)))
        window = explicit(config, "context_window")
        if not isinstance(window, int):
            return QualificationCheck(name="memory_preflight", passed=False, detail="context_window is not explicit")
        prepared = self.prepare(config, (Message(role=Role.USER, content="memory preflight"),), None,
                                InvocationContext(experiment_id="qualification", task_id=f"qualification:{config.key}",
                                                  attempt_id="memory-preflight", attempt_no=1, round=1,
                                                  purpose="qualification"))
        body = prepared.native_request["body"]
        request = {"op": "stress", "target_prompt_tokens": window - out_tokens, "max_tokens": out_tokens,
                   "context_window": window, "chat_template_kwargs": body["chat_template_kwargs"],
                   "sampling": body["sampling"], "seed": body["seed"], "prefill_step_size": body["prefill_step_size"]}
        with self._lock:
            try:
                session, _ = self._ensure(config)
            except _WorkerFailure as exc:
                return QualificationCheck(name="memory_preflight", passed=False, detail=f"{exc.error_type}: {exc}")
            sampler = MemorySampler(session.worker.pid, interval_s=self._interval, probe=self._probe)
            result: dict[str, Any] | None = None
            failure: _WorkerFailure | None = None
            with sampler:
                try:
                    result = self._call(session.worker, request, _timeout(config))
                except _WorkerFailure as exc:
                    failure = exc
                    if exc.error_type in ("worker_exited", "timeout"):
                        self._session = None
        host_memory = sampler.summary()
        host = self._probe.host_info()
        probes.append({"step": "memory-preflight", "request": request, "result": result,
                       "error": {"type": failure.error_type, "message": str(failure), **failure.extra} if failure else None,
                       "host_memory": host_memory, "host": host})
        if failure is not None or result is None:
            return QualificationCheck(name="memory_preflight", passed=False,
                                      detail=f"{failure.error_type if failure else 'no result'}: {failure}; "
                                             f"host memory {host_memory}")
        used = int(result.get("prompt_tokens") or 0) + len(result.get("completion_token_ids") or []) + (
            1 if result.get("stop_token_id") is not None else 0)
        problems = []
        swapouts, level = host_memory.get("swapouts_bytes"), host_memory.get("pressure_level_max")
        if not host_memory.get("available"):
            problems.append(str(host_memory.get("reason")))
        else:  # a missing reading cannot show the host stayed within limits
            if not isinstance(swapouts, int):
                problems.append("swap-out counters unavailable")
            elif swapouts > max_swapout:
                problems.append(f"swapped out {_mb(swapouts)} MB (limit {_mb(max_swapout)} MB)")
            if not isinstance(level, int):
                problems.append("memory pressure level unavailable")
            elif level > max_level:
                problems.append(f"memory pressure reached {host_memory.get('pressure_max')}")
        memory, timing = result.get("memory") or {}, result.get("timing") or {}
        return QualificationCheck(
            name="memory_preflight", passed=not problems,
            detail=("; ".join(problems) + "; " if problems else "")
            + f"filled {used} of {window} context tokens ({result.get('prompt_tokens')} prompt + "
              f"{used - int(result.get('prompt_tokens') or 0)} generated); MLX peak {_gb(memory.get('peak_bytes'))} GB, "
              f"KV cache {_gb((result.get('kv_cache') or {}).get('bytes'))} GB, worker footprint max "
              f"{_gb(host_memory.get('worker_footprint_max_bytes'))} GB; host {host.get('chip')} "
              f"{_gb(host.get('memory_bytes'))} GB, pressure max {host_memory.get('pressure_max')}, free min "
              f"{host_memory.get('free_percent_min')}%, swapped out {_mb(swapouts)} MB, swapped in "
              f"{_mb(host_memory.get('swapins_bytes'))} MB; prefill {timing.get('prefill_ms')} ms "
              f"({timing.get('engine_prompt_tps')} tokens/s). Generation length beyond {out_tokens} tokens is "
              f"not exercised here",
        )

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
        opts = config.adapter_options
        manifest = manifest_sha256(opts.get("files"))
        if parent is not None:
            parent.verify_shape()
            state = parent.native_state
            if state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if (state.get("repository"), state.get("revision"), state.get("manifest_sha256")) != (
                    config.model, opts.get("revision"), manifest):
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
        kwargs = {} if (thinking := explicit(config, "enable_thinking")) is None else {"enable_thinking": thinking}
        body: dict[str, Any] = {
            "op": "generate", "messages": native, "chat_template_kwargs": kwargs, "max_tokens": max_tokens,
            "context_window": window,
            "sampling": {name: v for name in SAMPLING if (v := explicit(config, name)) is not None},
            "seed": explicit(config, "seed"),
            "prefill_step_size": int(str(opts.get("prefill_step_size", 2048))),
        }
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "engine": "mlx-lm",
                "worker": {"protocol": PROTOCOL, "python": opts.get("python"), "engine_pins": opts.get("engine")},
                "repository": config.model,
                "revision": opts.get("revision"),
                "manifest_sha256": manifest,
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
        body = prepared.native_request["body"]
        with self._lock:
            try:
                session, loaded_now = self._ensure(config)
            except _WorkerFailure as exc:
                return failed(exc.kind, exc.error_type, str(exc), CostIncurred.NO, **exc.extra)
            sampler = MemorySampler(session.worker.pid, interval_s=self._interval, probe=self._probe)
            result: dict[str, Any] | None = None
            failure: _WorkerFailure | None = None
            with sampler:
                try:
                    result = self._call(session.worker, body, _timeout(config))
                except _WorkerFailure as exc:
                    failure = exc
                    if exc.error_type in ("worker_exited", "timeout"):
                        self._session = None  # dead or killed; the next call starts a fresh worker
            load_info = None
            if loaded_now or not session.load_reported:
                session.load_reported = True
                load_info = session.load
        host_memory = sampler.summary()
        worker = {"pid": session.worker.pid, "versions": session.hello.get("versions"),
                  "device": session.hello.get("device"), "loaded_in_this_call": load_info is not None,
                  "load": load_info, "tokenizer": session.load.get("tokenizer"),
                  "engine_settings": session.load.get("engine_settings")}
        host = self._probe.host_info()
        if failure is not None:
            extra = dict(failure.extra)
            rendered, prompt_ids = extra.pop("rendered_prompt", None), extra.pop("prompt_token_ids", None)
            if isinstance(rendered, str):
                sink.write("rendered-prompt.txt", rendered)
                sink.write_json("input-token-ids.json", prompt_ids)
            return NativeExecution(
                status="failed", failure_kind=failure.kind, cost_incurred=CostIncurred.NO,
                error={"type": failure.error_type, "message": str(failure), **extra},
                native_response={"worker": worker, "host_memory": host_memory, "host": host},
                observations=_host_observations(host_memory),
            )
        assert result is not None
        rendered = str(result.pop("rendered_prompt"))
        sink.write("rendered-prompt.txt", rendered)
        sink.write_json("input-token-ids.json", result.pop("prompt_token_ids"))
        think_start = (session.load.get("tokenizer") or {}).get("think_start")
        # The generation prompt left a thinking block open: the completion starts inside it.
        result["thinking_opened"] = bool(think_start) and rendered.rstrip().endswith(str(think_start))
        data = {**result, "worker": worker, "host_memory": host_memory, "host": host}
        observations = _observations(data, session.load, load_info is not None)
        if data.get("finish_reason") not in ("stop", "length"):
            return NativeExecution(status="failed", native_response=data, finish_reason=data.get("finish_reason"),
                                   failure_kind=FailureKind.PROVIDER_ERROR, observations=observations,
                                   error={"type": "unfinished", "finish_reason": data.get("finish_reason")})
        return NativeExecution(status="completed", native_response=data, finish_reason=data.get("finish_reason"),
                               cost_incurred=CostIncurred.NO, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        tokenizer = (data.get("worker") or {}).get("tokenizer") or {}
        reasoning, answer = split_thinking(str(data.get("completion_text") or ""), tokenizer,
                                           thinking_block_opened(data, tokenizer))
        return ExtractedResponse(final_answer=answer, truncated=data.get("finish_reason") == "length",
                                 finish_reason=data.get("finish_reason"), reasoning_text=reasoning)

    def fork_state(
        self, prepared: PreparedInvocation, execution: NativeExecution, extracted: ExtractedResponse
    ) -> ContinuationSnapshot:
        request = prepared.native_request
        assistant: dict[str, Any] = {"role": "assistant", "content": extracted.final_answer}
        if extracted.reasoning_text is not None:
            assistant["reasoning_content"] = extracted.reasoning_text
        return ContinuationSnapshot(
            adapter=self.name,
            configuration_key=str(request["repository"]),
            messages=prepared.messages + (Message(role=Role.ASSISTANT, content=extracted.final_answer),),
            native_state={
                "format": STATE_FORMAT,
                "repository": request["repository"],
                "revision": request["revision"],
                "manifest_sha256": request["manifest_sha256"],
                "chat_template_sha256": ((execution.native_response or {}).get("worker") or {}).get(
                    "tokenizer", {}).get("chat_template_sha256"),
                "messages": copy.deepcopy(request["body"]["messages"]) + [assistant],
            },
        )


# ----- helpers ---------------------------------------------------------------------------------------


def split_thinking(text: str, tokenizer: dict[str, Any], opened: bool) -> tuple[str | None, str]:
    """Split a completion into (reasoning, answer) the way the chat template parses history.

    With a thinking block open (see `thinking_block_opened`) and no end marker, the whole completion is
    reasoning (generation stopped while thinking) and the answer is empty.
    """
    start, end = tokenizer.get("think_start"), tokenizer.get("think_end")
    if end and end in text:
        reasoning = text.split(end)[0].rstrip("\n")
        if start:
            reasoning = reasoning.split(start)[-1]
        return reasoning.lstrip("\n").strip(), text.split(end)[-1].lstrip("\n").strip()
    if opened:
        reasoning = text.strip()
        return (reasoning.removeprefix(start).strip() if start else reasoning), ""
    return None, text.strip()


def thinking_block_opened(data: dict[str, Any], tokenizer: dict[str, Any]) -> bool:
    """Whether the completion starts inside a thinking block.

    Either the generation prompt ended with the start marker (`thinking_opened`), or the template left it
    to the model (hybrid templates such as Qwen3) and the completion begins with it. A start marker later
    in the text is answer content, not a thinking block.
    """
    if data.get("thinking_opened"):
        return True
    start = tokenizer.get("think_start")
    return bool(start) and str(data.get("completion_text") or "").lstrip().startswith(str(start))


def _decoding_check(config: ModelConfiguration, tokenizer: dict[str, Any],
                    outputs: list[tuple[str, NativeExecution, ExtractedResponse | None]]) -> QualificationCheck:
    """Probe answers decode to the requested word, stop on an end-of-turn token, and parse thinking.

    Thinking is expected only when the tokenizer has thinking markers and the configuration does not disable it.
    """
    problems, notes = [], []
    thinking_expected = bool(tokenizer.get("think_end")) and explicit(config, "enable_thinking") is not False
    for step, execution, extracted in outputs:
        word = "baseline" if step == "baseline" else step.removeprefix("branch-")
        if extracted is None:
            problems.append(f"{step} did not complete")
            continue
        if execution.finish_reason != "stop":
            problems.append(f"{step} finished with {execution.finish_reason}, not an end-of-turn token")
        if not re.search(rf"\b{word}\b", extracted.final_answer, re.IGNORECASE):
            problems.append(f"{step} answered {extracted.final_answer[:80]!r}, expected {word!r}")
        opened = thinking_block_opened(execution.native_response or {}, tokenizer)
        if thinking_expected and not (opened and extracted.reasoning_text):
            problems.append(f"{step}: thinking expected but none was parsed")
        if not thinking_expected and extracted.reasoning_text:
            problems.append(f"{step}: thinking disabled but reasoning was produced")
        notes.append(f"{step}: {extracted.final_answer[:40]!r}, reasoning {len(extracted.reasoning_text or '')} chars")
    return QualificationCheck(name="output_decoding", passed=not problems and bool(outputs),
                              detail="; ".join(problems) or "; ".join(notes))


def _pin_problems(config: ModelConfiguration) -> list[str]:
    opts = config.adapter_options
    problems = []
    if not isinstance(opts.get("python"), str):
        problems.append("adapter_options.python is required")
    engine = opts.get("engine")
    if not isinstance(engine, dict) or any(not engine.get(p) for p in REQUIRED_PINS):
        problems.append(f"adapter_options.engine must pin {list(REQUIRED_PINS)}")
    if not isinstance(opts.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", str(opts.get("revision"))):
        problems.append("adapter_options.revision must be a 40-character commit sha")
    files = opts.get("files")
    if not isinstance(files, dict) or not files or any(
            not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in files.values()):
        problems.append("adapter_options.files must map every snapshot file to its sha256")
    return problems


def _version_problems(pins: Any, versions: dict[str, Any]) -> list[str]:
    if not isinstance(pins, dict):
        return ["no engine pins"]
    return [f"{name} {versions.get(name)} != pinned {want}" for name, want in sorted(pins.items())
            if versions.get(name) != want]


def _timeout(config: ModelConfiguration) -> float:
    return float(str(config.adapter_options.get("timeout_s", 1800)))


def _gb(value: Any) -> str:
    return f"{value / 1024**3:.1f}" if isinstance(value, int | float) else "?"


def _mb(value: Any) -> str:
    return f"{value / 1024**2:.0f}" if isinstance(value, int | float) else "?"


def _host_observations(host_memory: dict[str, Any]) -> dict[str, Observation]:
    source = "controller memory sampler"
    if not host_memory.get("available"):
        reason = str(host_memory.get("reason") or "unavailable")
        return {name: Observation.missing(unit, source, reason) for name, unit in (
            ("host_memory_pressure_max", "level"), ("host_swapout_bytes", "bytes"),
            ("host_swapin_bytes", "bytes"), ("worker_footprint_max_bytes", "bytes"),
            ("host_free_percent_min", "percent"))}
    obs: dict[str, Observation] = {}
    for name, key, unit in (("host_memory_pressure_max", "pressure_level_max", "level"),
                            ("host_swapout_bytes", "swapouts_bytes", "bytes"),
                            ("host_swapin_bytes", "swapins_bytes", "bytes"),
                            ("worker_footprint_max_bytes", "worker_footprint_max_bytes", "bytes"),
                            ("host_free_percent_min", "free_percent_min", "percent")):
        value = host_memory.get(key)
        obs[name] = (Observation(value=value, unit=unit, source=f"{source}: {key}") if isinstance(value, int)
                     else Observation.missing(unit, source, f"{key} not observed"))
    return obs


def _observations(data: dict[str, Any], load: dict[str, Any], loaded_now: bool) -> dict[str, Observation]:
    source = "mlx worker"
    obs = _host_observations(data.get("host_memory") or {})
    completion = data.get("completion_token_ids") or []
    generated = len(completion) + (1 if data.get("stop_token_id") is not None else 0)
    obs["input_tokens"] = Observation(value=int(data.get("prompt_tokens") or 0), unit="tokens",
                                      source=f"{source}: rendered prompt token ids")
    obs["cached_input_tokens"] = Observation(value=0, unit="tokens", source=f"{source}: fresh prompt cache per request")
    obs["output_tokens"] = Observation(value=generated, unit="tokens",
                                       source=f"{source}: generated token ids including the stop token")
    tokenizer = (data.get("worker") or {}).get("tokenizer") or {}
    end_ids = list(tokenizer.get("think_end_token_ids") or [])
    reasoning = None
    if end_ids:
        for i in range(len(completion) - len(end_ids) + 1):
            if completion[i: i + len(end_ids)] == end_ids:
                reasoning = i + len(end_ids)
                break
    if reasoning is None:
        reasoning = len(completion) if thinking_block_opened(data, tokenizer) else 0
    obs["reasoning_tokens"] = Observation(value=reasoning, unit="tokens",
                                          source=f"{source}: completion tokens through the thinking end marker")
    obs["engine_load_ms"] = Observation(
        value=float(load.get("load_ms") or 0.0) if loaded_now else 0.0, unit="ms",
        source=f"{source}: checkpoint load" + ("" if loaded_now else " (model already resident; no load in this call)"))
    obs["engine_warmup_ms"] = Observation(
        value=float((load.get("warmup") or {}).get("ms") or 0.0) if loaded_now else 0.0, unit="ms",
        source=f"{source}: fixed unscored warm-up" + ("" if loaded_now else " (done when the model was loaded)"))
    timing = data.get("timing") or {}
    for name, key in (("engine_prompt_eval_ms", "prefill_ms"), ("engine_eval_ms", "generation_ms"),
                      ("engine_total_ms", "total_ms")):
        value = timing.get(key)
        obs[name] = (Observation(value=float(value), unit="ms", source=f"{source}: {key}") if value is not None
                     else Observation.missing("ms", source, f"{key} not reported"))
    for name, key in (("output_tokens_per_s", "engine_generation_tps"), ("prompt_tokens_per_s", "engine_prompt_tps")):
        value = timing.get(key)
        obs[name] = (Observation(value=float(value), unit="tokens/s", source=f"{source}: mlx-lm {key}")
                     if value is not None else Observation.missing("tokens/s", source, f"{key} not reported"))
    memory = data.get("memory") or {}
    obs["model_memory_bytes"] = Observation(value=int(load.get("weights_memory_bytes") or 0), unit="bytes",
                                            source=f"{source}: MLX active memory after load")
    for name, value, key in (("peak_memory_bytes", memory.get("peak_bytes"), "MLX peak memory during the request"),
                             ("kv_cache_bytes", (data.get("kv_cache") or {}).get("bytes"), "prompt cache state size")):
        obs[name] = (Observation(value=int(value), unit="bytes", source=f"{source}: {key}") if value is not None
                     else Observation.missing("bytes", source, f"{key} not reported"))
    obs["api_cost_usd"] = Observation(value=0.0, unit="usd", source="local inference: no API billing")
    obs["energy"] = Observation.missing("kWh", "local inference", "electricity and hardware amortisation not estimated")
    return obs
