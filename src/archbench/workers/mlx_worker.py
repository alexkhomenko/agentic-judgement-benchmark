"""MLX-LM inference worker: one pinned checkpoint, sequential requests, JSON lines over stdin/stdout.

Runs in its own pinned environment (`.venv-mlx`, see `requirements-mlx-worker.txt`), started by the
`mlx-lm` adapter as `python -I mlx_worker.py`. It never imports archbench and never touches the network:
the adapter starts it with HF_HUB_OFFLINE=1 and the checkpoint must already be in the local Hugging Face
cache at the pinned revision.

Protocol: each request is one JSON object per line on stdin; each reply is one JSON object per line on
the original stdout. Library output that would go to stdout is redirected to stderr, which the adapter
keeps as a log. Replies carry `ok: true` or `ok: false` with an `error` object.

    {"op": "hello"}                       -> engine versions and hardware
    {"op": "load", ...}                   -> checkpoint manifest, load and warm-up timing, model facts
    {"op": "generate", ...}               -> rendered prompt, token ids, completion, timing, memory
    {"op": "stress", ...}                 -> long-context memory preflight at a target prompt length
    {"op": "shutdown"}

Cache policy: every request gets a fresh prompt (KV) cache, and the MLX buffer cache is cleared after
each request, so independent conversations share no state.

`python -I mlx_worker.py manifest <repository> <revision>` prints the file manifest of a cached snapshot,
for pinning a new checkpoint in a configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

PROTOCOL = "archbench-mlx-worker-v1"
PACKAGES = ("mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "huggingface_hub", "jinja2", "numpy")
FILLER_SENTENCE = "The archive lists every shipment, its route, and the clerk who signed for it. "
WARMUP_MESSAGES = [{"role": "user", "content": "Reply with exactly one word: ready"}]
WARMUP_TOKENS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(root: Path) -> dict[str, Path]:
    """Every file of a snapshot by relative name, resolving the cache's symlinks to blobs."""
    return {p.relative_to(root).as_posix(): p.resolve() for p in sorted(root.rglob("*")) if p.is_file()}


def manifest(root: Path) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in snapshot_files(root).items()}


def resolve_snapshot(repository: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found,unused-ignore]

    return Path(snapshot_download(repository, revision=revision, local_files_only=True))


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


class Engine:
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.info: dict[str, Any] | None = None

    # ----- operations ------------------------------------------------------------------------------

    def hello(self, _: dict[str, Any]) -> dict[str, Any]:
        import mlx.core as mx  # type: ignore[import-not-found,unused-ignore]

        mac = platform.mac_ver()
        return {"protocol": PROTOCOL, "pid": os.getpid(), "versions": versions(),
                "platform": {"system": platform.system(), "machine": platform.machine(), "macos": mac[0],
                             "python_implementation": platform.python_implementation()},
                "device": {k: v for k, v in mx.device_info().items()}}

    def load(self, req: dict[str, Any]) -> dict[str, Any]:
        import mlx.core as mx
        from mlx_lm.utils import compute_bits_per_weight, get_total_parameters, load  # type: ignore[import-not-found,unused-ignore]

        if self.model is not None:
            raise WorkerError("already_loaded", "this worker already holds a model; start a new worker")
        root = resolve_snapshot(req["repository"], req["revision"])
        files = snapshot_files(root)
        before = {name: (p.stat().st_size, p.stat().st_mtime_ns) for name, p in files.items()}
        started = time.perf_counter()
        hashes = {name: sha256_file(p) for name, p in files.items()}
        hash_ms = (time.perf_counter() - started) * 1000
        pinned: dict[str, str] = req["files"]
        if hashes != pinned:
            missing = sorted(set(pinned) - set(hashes))
            extra = sorted(set(hashes) - set(pinned))
            changed = sorted(n for n in set(pinned) & set(hashes) if pinned[n] != hashes[n])
            raise WorkerError("checkpoint_mismatch", f"missing {missing}, unexpected {extra}, changed {changed}",
                              manifest=hashes)
        mx.reset_peak_memory()
        started = time.perf_counter()
        model, tokenizer, config = load(str(root), return_config=True)
        load_ms = (time.perf_counter() - started) * 1000
        after = {name: (p.stat().st_size, p.stat().st_mtime_ns) for name, p in files.items()}
        if after != before:
            raise WorkerError("checkpoint_changed", "checkpoint files changed while loading")
        self.model, self.tokenizer = model, tokenizer
        load_peak = mx.get_peak_memory()
        weights_bytes = mx.get_active_memory()

        warm_started = time.perf_counter()
        warm = self._generate(WARMUP_MESSAGES, {}, WARMUP_TOKENS, None, {"temperature": 0.0}, 0, 2048)
        warmup_ms = (time.perf_counter() - warm_started) * 1000

        text_config = config.get("text_config") or {}
        template = tokenizer.chat_template if isinstance(tokenizer.chat_template, str) else None
        self.info = {
            "snapshot": str(root),
            "manifest": hashes,
            "hash_ms": round(hash_ms, 3),
            "load_ms": round(load_ms, 3),
            "load_peak_memory_bytes": load_peak,
            "weights_memory_bytes": weights_bytes,
            "warmup": {"ms": round(warmup_ms, 3), "prompt_tokens": warm["prompt_tokens"],
                       "generated_tokens": len(warm["completion_token_ids"]), "max_tokens": WARMUP_TOKENS,
                       "sampling": "greedy", "messages": WARMUP_MESSAGES},
            "model": {
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "class": f"{type(model).__module__}.{type(model).__name__}",
                "quantization": config.get("quantization"),
                "parameters": int(get_total_parameters(model)),
                "bits_per_weight": round(float(compute_bits_per_weight(model)), 4),
                "max_position_embeddings": text_config.get("max_position_embeddings",
                                                           config.get("max_position_embeddings")),
                "dtype": text_config.get("dtype", config.get("torch_dtype")),
            },
            "tokenizer": {
                "class": type(tokenizer._tokenizer).__name__,
                "eos_token_ids": sorted(int(t) for t in tokenizer.eos_token_ids),
                "bos_token": tokenizer.bos_token,
                "think_start": tokenizer.think_start,
                "think_end": tokenizer.think_end,
                "think_start_token_ids": list(tokenizer.think_start_tokens or ()),
                "think_end_token_ids": list(tokenizer.think_end_tokens or ()),
                "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest() if template else None,
            },
            "generation_config_present": "generation_config.json" in files,
            "engine_settings": {"prefill_step_size": 2048, "max_kv_size": None, "kv_bits": None,
                                "wired_limit": "mlx-lm default (recommended working set)",
                                "prompt_cache": "fresh per request", "buffer_cache": "cleared after each request",
                                "speculative_decoding": False, "batching": False},
        }
        mx.clear_cache()
        return self.info

    def generate(self, req: dict[str, Any]) -> dict[str, Any]:
        self._require_model()
        return self._generate(req["messages"], req.get("chat_template_kwargs") or {}, int(req["max_tokens"]),
                              int(req["context_window"]), req.get("sampling") or {}, req.get("seed"),
                              int(req.get("prefill_step_size", 2048)))

    def stress(self, req: dict[str, Any]) -> dict[str, Any]:
        """Long-context memory preflight: a filler prompt rendered to just under `target_prompt_tokens`."""
        self._require_model()
        target = int(req["target_prompt_tokens"])
        kwargs = req.get("chat_template_kwargs") or {}

        def build(reps: int) -> tuple[list[dict[str, Any]], int]:
            messages = [{"role": "user", "content": FILLER_SENTENCE * reps + "\nSummarise the archive in one line."}]
            return messages, len(self._render(messages, kwargs)[1])

        # Measure the template overhead and the per-sentence cost in context, then step down to fit.
        overhead = build(0)[1]
        probe = 64
        per_sentence = max((build(probe)[1] - overhead) / probe, 1.0)
        reps = max(0, int((target - overhead) / per_sentence))
        messages, length = build(reps)
        while length > target and reps > 0:
            reps -= 1
            messages, length = build(reps)
        result = self._generate(messages, kwargs, int(req["max_tokens"]), int(req["context_window"]),
                                req.get("sampling") or {}, req.get("seed"), int(req.get("prefill_step_size", 2048)))
        result.pop("rendered_prompt", None)
        result.pop("prompt_token_ids", None)
        result["filler_repetitions"] = reps
        return result

    # ----- generation ------------------------------------------------------------------------------

    def _require_model(self) -> None:
        if self.model is None:
            raise WorkerError("not_loaded", "load a checkpoint first")

    def _render(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> tuple[str, list[int]]:
        # The Hugging Face template is applied with exactly the configured kwargs (mlx-lm's wrapper would
        # inject enable_thinking); the rendered text is tokenized without adding special tokens again.
        text = self.tokenizer._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                             **kwargs)
        return text, list(self.tokenizer.encode(text, add_special_tokens=False))

    def _generate(self, messages: list[dict[str, Any]], kwargs: dict[str, Any], max_tokens: int,
                  context_window: int | None, sampling: dict[str, Any], seed: int | None,
                  prefill_step_size: int) -> dict[str, Any]:
        import mlx.core as mx
        from mlx_lm.generate import stream_generate  # type: ignore[import-not-found,unused-ignore]
        from mlx_lm.models.cache import make_prompt_cache  # type: ignore[import-not-found,unused-ignore]
        from mlx_lm.sample_utils import make_logits_processors, make_sampler  # type: ignore[import-not-found,unused-ignore]

        rendered, prompt_ids = self._render(messages, kwargs)
        if context_window is not None and len(prompt_ids) + max_tokens > context_window:
            raise WorkerError("context_exceeded",
                              f"prompt {len(prompt_ids)} tokens + max_tokens {max_tokens} > context_window "
                              f"{context_window}", prompt_tokens=len(prompt_ids), rendered_prompt=rendered,
                              prompt_token_ids=prompt_ids)
        seed_source = "configured"
        if seed is None:
            seed, seed_source = int.from_bytes(os.urandom(4), "little"), "drawn by worker (os.urandom)"
        mx.random.seed(seed)
        sampler = make_sampler(temp=float(sampling.get("temperature", 0.0)), top_p=float(sampling.get("top_p", 0.0)),
                               min_p=float(sampling.get("min_p", 0.0)), top_k=int(sampling.get("top_k", 0)))
        processors = make_logits_processors(
            repetition_penalty=sampling.get("repetition_penalty"),
            repetition_context_size=sampling.get("repetition_context_size", 20),
            presence_penalty=sampling.get("presence_penalty"),
            presence_context_size=sampling.get("presence_context_size", 20),
            frequency_penalty=sampling.get("frequency_penalty"),
            frequency_context_size=sampling.get("frequency_context_size", 20),
        )
        cache = make_prompt_cache(self.model)
        mx.reset_peak_memory()
        tokens: list[int] = []
        last: Any = None
        started = time.perf_counter()
        first_at: float | None = None
        for response in stream_generate(self.model, self.tokenizer, prompt=prompt_ids, max_tokens=max_tokens,
                                        sampler=sampler, logits_processors=processors or None,
                                        prompt_cache=cache, prefill_step_size=prefill_step_size):
            if first_at is None:
                first_at = time.perf_counter()
            tokens.append(int(response.token))
            last = response
        ended = time.perf_counter()
        finish = last.finish_reason if last is not None else None
        stop_token = tokens.pop() if finish == "stop" and tokens else None
        kv_bytes = 0
        kv_types: dict[str, int] = {}
        kv_dtypes: set[str] = set()
        for entry in cache:
            kv_types[type(entry).__name__] = kv_types.get(type(entry).__name__, 0) + 1
            for array in _arrays(entry.state):
                kv_bytes += int(array.nbytes)
                kv_dtypes.add(str(array.dtype))
        result = {
            "rendered_prompt": rendered,
            "prompt_token_ids": prompt_ids,
            "prompt_tokens": len(prompt_ids),
            "completion_token_ids": tokens,
            "completion_text": self.tokenizer._tokenizer.decode(tokens, skip_special_tokens=False),
            "stop_token_id": stop_token,
            "finish_reason": finish,
            "chat_template_kwargs": kwargs,
            "sampling": sampling,
            "seed": seed,
            "seed_source": seed_source,
            "max_tokens": max_tokens,
            "context_window": context_window,
            "prefill_step_size": prefill_step_size,
            "timing": {
                "prefill_ms": round(((first_at or ended) - started) * 1000, 3),
                "generation_ms": round((ended - (first_at or ended)) * 1000, 3),
                "total_ms": round((ended - started) * 1000, 3),
                "engine_prompt_tps": round(float(last.prompt_tps), 3) if last is not None else None,
                "engine_generation_tps": round(float(last.generation_tps), 3) if last is not None else None,
            },
            "memory": {"peak_bytes": mx.get_peak_memory(), "active_bytes": mx.get_active_memory(),
                       "buffer_cache_bytes": mx.get_cache_memory()},
            "kv_cache": {"bytes": kv_bytes, "types": kv_types, "dtypes": sorted(kv_dtypes)},
        }
        del cache
        mx.clear_cache()
        return result


def _arrays(state: Any) -> list[Any]:
    if isinstance(state, list | tuple):
        return [a for item in state for a in _arrays(item)]
    return [state] if hasattr(state, "nbytes") else []


class WorkerError(Exception):
    def __init__(self, kind: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.kind, self.extra = kind, extra


def serve() -> int:
    protocol_out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)  # anything a library prints goes to the log, never into the protocol stream
    sys.stdout = sys.stderr
    engine = Engine()
    handlers = {"hello": engine.hello, "load": engine.load, "generate": engine.generate, "stress": engine.stress}
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "shutdown":
                protocol_out.write(json.dumps({"ok": True, "result": {}}) + "\n")
                return 0
            if op not in handlers:
                raise WorkerError("unknown_op", f"unknown op {op!r}")
            reply: dict[str, Any] = {"ok": True, "result": handlers[op](request)}
        except WorkerError as exc:
            reply = {"ok": False, "error": {"type": exc.kind, "message": str(exc), **exc.extra}}
        except Exception as exc:  # noqa: BLE001 - reported to the adapter as evidence
            reply = {"ok": False, "error": {"type": "worker_exception", "message": f"{type(exc).__name__}: {exc}"}}
        protocol_out.write(json.dumps(reply) + "\n")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "manifest":
        print(json.dumps(manifest(resolve_snapshot(argv[2], argv[3])), indent=2))
        return 0
    if len(argv) == 1:
        return serve()
    print(f"usage: {argv[0]} [manifest <repository> <revision>]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
