"""Local MLX-LM adapter against a fake worker: engine and checkpoint pins, warm model reuse, rendered-prompt
and token-id evidence, thinking parsing, failure mapping (context, crash, timeout), continuation, the
controller-side memory sampler, qualification, the worker subprocess channel, and an offline P6 run."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.fake import FakeAdapter
from archbench.adapters.fake_judge import FakeJudgeAdapter
from archbench.adapters.local.mlx_lm import MlxLmAdapter, split_thinking
from archbench.adapters.registry import AdapterRegistry
from archbench.domain.configuration import ModelConfiguration
from archbench.domain.execution import CostIncurred, FailureKind
from archbench.domain.scenario import Message, Role
from archbench.evaluation.runner import resume_evaluation, start_evaluation
from archbench.execution.controller import Controller
from archbench.execution.invoke import invoke
from archbench.execution.qualification import qualify_configuration
from archbench.planning.compiler import compile_plan, load_experiment, persist_plan
from archbench.planning.loader import load_scenario
from archbench.reporting.report import publish_report
from archbench.storage.artifacts import ArtifactStore
from archbench.workers import mlx_worker
from archbench.workers.host_memory import HostProbe, MemorySampler
from archbench.workers.mlx_process import SubprocessWorker, WorkerExited, WorkerTimeout
from archbench.workers.mlx_worker import PROTOCOL

from .conftest import ROOT, Env
from .test_bias import _read_csv, bias_policy, recommender

CONFIG_PATH = ROOT / "configs/models/mlx/qwen3.5-9b-4bit.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
GB = 1024**3
THINK_END_ID = 248069
EOS = 248046
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)


class FakeProbe:
    """Host counters: swap-outs grow by `swapout_pages` per reading, pressure stays at `level`."""

    available = True

    def __init__(self, *, level: int = 1, swapout_pages: int = 0) -> None:
        self.level, self.swapout_pages, self._swapouts = level, swapout_pages, 0

    def pressure_level(self) -> int:
        return self.level

    def free_percent(self) -> int:
        return 60

    def swap_used(self) -> int:
        return GB

    def vm_counters(self) -> dict[str, int]:
        self._swapouts += self.swapout_pages
        return {"swapins": 0, "swapouts": self._swapouts, "pageouts": 0}

    def page_size(self) -> int:
        return 16384

    def footprint(self, pid: int) -> int:
        return 6 * GB

    def host_info(self) -> dict[str, Any]:
        return {"system": "Darwin", "chip": "Apple M4", "memory_bytes": 32 * GB, "os_version": "27.2"}


class FakeWorker:
    """Speaks the worker protocol in-process. Completions are '<reasoning>\\n</think>\\n\\n<answer>'."""

    instances: list[FakeWorker] = []

    def __init__(self, python: str, *, versions: dict[str, Any] | None = None, answer: str = "Use Qwen.",
                 finish: str = "stop", think_closed: bool = True, load_error: dict[str, Any] | None = None,
                 fail_generate: Exception | None = None) -> None:
        self.python, self.answer, self.finish, self.think_closed = python, answer, finish, think_closed
        self.versions = versions if versions is not None else dict(CONFIG.adapter_options["engine"])  # type: ignore[arg-type]
        self.load_error, self.fail_generate = load_error, fail_generate
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self._pid: int | None = 4242
        FakeWorker.instances.append(self)

    @property
    def pid(self) -> int | None:
        return self._pid

    def ops(self) -> list[str]:
        return [r["op"] for r in self.requests]

    def request(self, message: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        self.requests.append(json.loads(json.dumps(message)))
        op = message["op"]
        if op == "hello":
            return {"ok": True, "result": {"protocol": PROTOCOL, "versions": self.versions,
                                           "device": {"device_name": "Apple M4", "memory_size": 32 * GB}}}
        if op == "load":
            if self.load_error:
                return {"ok": False, "error": self.load_error}
            return {"ok": True, "result": {
                "load_ms": 1500.0, "warmup": {"ms": 400.0}, "weights_memory_bytes": 5 * GB,
                "generation_config_present": False,
                "model": {"class": "mlx_lm.models.qwen3_5.Model", "parameters": 8953801728, "bits_per_weight": 4.5,
                          "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                          "max_position_embeddings": 262144},
                "tokenizer": {"eos_token_ids": [EOS], "think_start": "<think>", "think_end": "</think>",
                              "think_start_token_ids": [248068], "think_end_token_ids": [THINK_END_ID],
                              "chat_template_sha256": CONFIG.adapter_options["files"]["chat_template.jinja"]},  # type: ignore[index,call-overload]
                "engine_settings": {"prefill_step_size": 2048}}}
        if self.fail_generate is not None:
            if isinstance(self.fail_generate, WorkerExited):
                self._pid = None
            raise self.fail_generate
        if op == "stress":
            return {"ok": True, "result": {
                "prompt_tokens": message["target_prompt_tokens"], "completion_token_ids": [1] * message["max_tokens"],
                "stop_token_id": None, "finish_reason": "length", "memory": {"peak_bytes": 9 * GB},
                "kv_cache": {"bytes": GB}, "timing": {"prefill_ms": 90000.0, "engine_prompt_tps": 320.0}}}
        thinking = message.get("chat_template_kwargs", {}).get("enable_thinking", True) is not False
        rendered = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in message["messages"])
        rendered += "<|im_start|>assistant\n" + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")
        prompt_ids = list(range(len(rendered) // 4))
        if len(prompt_ids) + message["max_tokens"] > message["context_window"]:
            return {"ok": False, "error": {"type": "context_exceeded", "message": "too long",
                                           "prompt_tokens": len(prompt_ids), "rendered_prompt": rendered,
                                           "prompt_token_ids": prompt_ids}}
        last = message["messages"][-1]["content"]
        answer = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        if not thinking:
            text, tokens = answer, [1, 2, 3]
        elif self.think_closed:
            text, tokens = f"private reasoning\n</think>\n\n{answer}", [5, 6, THINK_END_ID, 7, 8]
        else:
            text, tokens = "private reasoning that never ends", [5, 6, 7, 8]
        return {"ok": True, "result": {
            "rendered_prompt": rendered, "prompt_token_ids": prompt_ids, "prompt_tokens": len(prompt_ids),
            "completion_token_ids": tokens, "completion_text": text, "finish_reason": self.finish,
            "stop_token_id": EOS if self.finish == "stop" else None, "seed": 99, "seed_source": "drawn by worker",
            "timing": {"prefill_ms": 200.0, "generation_ms": 1000.0, "total_ms": 1200.0,
                       "engine_prompt_tps": 300.0, "engine_generation_tps": 45.0},
            "memory": {"peak_bytes": 6 * GB, "active_bytes": 5 * GB},
            "kv_cache": {"bytes": GB // 4, "types": {"KVCache": 8}, "dtypes": ["bfloat16"]}}}

    def close(self) -> None:
        self.closed = True
        self._pid = None


def adapter(probe: FakeProbe | None = None, **worker: Any) -> MlxLmAdapter:
    return MlxLmAdapter(worker_factory=lambda python: FakeWorker(python, **worker), probe=probe or FakeProbe(),
                        sample_interval_s=0.01)


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, mlx: MlxLmAdapter, config: ModelConfiguration = CONFIG,
         messages: tuple[Message, ...] = MESSAGES) -> tuple[Any, dict[str, str]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mlx, config, messages, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def with_param(name: str, setting: dict[str, Any] | None) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    if setting is None:
        del data["parameters"][name]
    else:
        data["parameters"][name] = setting
    return ModelConfiguration.model_validate(data)


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    FakeWorker.instances.clear()


def test_request_is_native_explicit_and_saves_prompt_evidence(tmp_path: Path) -> None:
    outcome, files = call(tmp_path, adapter())
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Qwen."
    (worker,) = FakeWorker.instances
    assert worker.python == ".venv-mlx/bin/python" and worker.ops() == ["hello", "load", "generate"]
    assert worker.requests[1]["files"] == CONFIG.adapter_options["files"]
    generate = worker.requests[2]
    assert generate == {
        "op": "generate", "chat_template_kwargs": {}, "max_tokens": 16000, "context_window": 32768, "seed": None,
        "prefill_step_size": 2048,
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    }  # enable_thinking, presence_context_size, repetition_penalty omitted: template / mlx-lm defaults
    assert files["rendered-prompt.txt"].endswith("<|im_start|>assistant\n<think>\n")
    assert json.loads(files["input-token-ids.json"])[:3] == [0, 1, 2]
    assert files["reasoning.txt"] == "private reasoning"
    response = json.loads(files["native-response.json"])
    assert "rendered_prompt" not in response and response["thinking_opened"] is True
    assert response["seed"] == 99 and response["worker"]["loaded_in_this_call"] is True


def test_observations_cover_tokens_timing_memory_and_zero_cost(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, adapter())
    obs = {k: v.value for k, v in outcome.observations.items()}
    assert obs["output_tokens"] == 6 and obs["reasoning_tokens"] == 3  # 5 tokens + stop; through </think>
    assert obs["engine_load_ms"] == 1500.0 and obs["engine_eval_ms"] == 1000.0
    assert obs["model_memory_bytes"] == 5 * GB and obs["peak_memory_bytes"] == 6 * GB
    assert obs["kv_cache_bytes"] == GB // 4 and obs["worker_footprint_max_bytes"] == 6 * GB
    assert obs["host_swapout_bytes"] == 0 and obs["host_memory_pressure_max"] == 1
    assert obs["api_cost_usd"] == 0.0 and outcome.cost_incurred is CostIncurred.NO
    assert obs["energy"] is None


def test_the_model_stays_loaded_until_the_configuration_changes(tmp_path: Path) -> None:
    mlx = adapter()
    first, _ = call(tmp_path, mlx)
    second, files = call(tmp_path, mlx)
    assert len(FakeWorker.instances) == 1 and FakeWorker.instances[0].ops() == ["hello", "load", "generate", "generate"]
    assert second.observations["engine_load_ms"].value == 0.0
    assert json.loads(files["native-response.json"])["worker"]["load"] is None
    other = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options, "revision": "f" * 40}})
    call(tmp_path, mlx, other)
    assert len(FakeWorker.instances) == 2 and FakeWorker.instances[0].closed  # one loaded model at a time
    mlx.close()
    assert FakeWorker.instances[1].closed


def test_an_engine_version_mismatch_never_generates(tmp_path: Path) -> None:
    versions = {**CONFIG.adapter_options["engine"], "mlx-lm": "0.33.0"}  # type: ignore[dict-item]
    outcome, files = call(tmp_path, adapter(versions=versions))
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and "engine_mismatch" in outcome.error
    assert FakeWorker.instances[0].ops() == ["hello"] and FakeWorker.instances[0].closed
    assert "error.json" in files


def test_a_changed_checkpoint_is_never_used(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, adapter(load_error={"type": "checkpoint_mismatch", "message": "changed ['config.json']"}))
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and "checkpoint_mismatch" in outcome.error
    assert "generate" not in FakeWorker.instances[0].ops()


def test_context_overflow_is_refused_before_generation_with_the_prompt_saved(tmp_path: Path) -> None:
    big = (MESSAGES[0], Message(role=Role.USER, content="x" * 80_000))
    outcome, files = call(tmp_path, adapter(), messages=big)
    assert outcome.failure_kind is FailureKind.CONTEXT_EXCEEDED
    assert "rendered-prompt.txt" in files and "input-token-ids.json" in files


@pytest.mark.parametrize(("error", "kind"), [
    (WorkerExited({"returncode": -9, "signal": "SIGKILL", "stderr_tail": "killed"}), FailureKind.PROVIDER_ERROR),
    (WorkerTimeout("no reply"), FailureKind.AMBIGUOUS_TIMEOUT),
])
def test_a_dead_or_stuck_worker_fails_with_memory_evidence_and_is_replaced(tmp_path: Path, error: Exception,
                                                                           kind: FailureKind) -> None:
    calls = iter([error, None])
    mlx = MlxLmAdapter(worker_factory=lambda python: FakeWorker(python, fail_generate=next(calls)),
                       probe=FakeProbe(swapout_pages=10), sample_interval_s=0.01)
    outcome, files = call(tmp_path, mlx)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, CostIncurred.NO)
    response = json.loads(files["native-response.json"])
    assert response["host_memory"]["swapouts_bytes"] > 0  # the sampler outlives the worker
    assert outcome.observations["host_swapout_bytes"].value > 0
    retry, _ = call(tmp_path, mlx)
    assert retry.succeeded and len(FakeWorker.instances) == 2


def test_running_out_of_tokens_while_thinking_leaves_an_empty_truncated_answer(tmp_path: Path) -> None:
    outcome, files = call(tmp_path, adapter(finish="length", think_closed=False))
    assert outcome.succeeded and outcome.extracted.truncated
    assert outcome.extracted.final_answer == "" and files["reasoning.txt"] == "private reasoning that never ends"
    assert outcome.observations["reasoning_tokens"].value == 4


def test_disabled_thinking_is_a_template_kwarg_and_yields_no_reasoning(tmp_path: Path) -> None:
    config = with_param("enable_thinking", {"state": "explicit", "value": False})
    outcome, files = call(tmp_path, adapter(), config)
    assert FakeWorker.instances[0].requests[2]["chat_template_kwargs"] == {"enable_thinking": False}
    assert outcome.extracted.final_answer == "Use Qwen." and "reasoning.txt" not in files
    assert outcome.observations["reasoning_tokens"].value == 0


@pytest.mark.parametrize(("text", "opened", "expected"), [
    ("plan\n</think>\n\nanswer\n", True, ("plan", "answer")),
    ("<think>\nplan</think>answer", False, ("plan", "answer")),
    ("still thinking", True, ("still thinking", "")),
    ("plain answer", False, (None, "plain answer")),
])
def test_split_thinking_follows_the_chat_template(text: str, opened: bool, expected: tuple[Any, str]) -> None:
    assert split_thinking(text, {"think_start": "<think>", "think_end": "</think>"}, opened) == expected


def test_continuation_replays_native_messages_with_reasoning_content(tmp_path: Path) -> None:
    mlx = adapter(answer="baseline answer")
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(mlx, CONFIG, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(mlx, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    (worker,) = FakeWorker.instances
    for sent, word in zip(worker.requests[3:], ("alpha", "beta")):
        assert sent["messages"][2] == {"role": "assistant", "content": "baseline answer",
                                       "reasoning_content": "private reasoning"}
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in json.dumps(worker.requests[4])
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    other = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options, "revision": "e" * 40}})
    with pytest.raises(AdapterError, match="different checkpoint"):
        mlx.prepare(other, follow, snapshot, ctx(2))
    with pytest.raises(AdapterError, match="lead the conversation"):
        mlx.prepare(CONFIG, (MESSAGES[1], MESSAGES[0]), None, ctx())


def test_qualification_covers_engine_checkpoint_decoding_continuation_and_memory() -> None:
    report = adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert list(checks) == ["adapter_identity", "parameter_support", "configuration_pins", "engine_identity",
                            "model_identity", "context_envelope", "continuation", "output_decoding", "memory_preflight"]
    assert "13 files match the pinned sha256 manifest" in checks["model_identity"].detail
    assert "filled 32768 of 32768 context tokens" in checks["memory_preflight"].detail
    stress = next(p for p in report.probes if p["step"] == "memory-preflight")
    assert stress["request"]["target_prompt_tokens"] == 32768 - 64
    assert {p["step"] for p in report.probes} == {"baseline", "branch-alpha", "branch-beta", "memory-preflight"}


@pytest.mark.parametrize(("mlx", "failing"), [
    (adapter(versions={**CONFIG.adapter_options["engine"], "mlx": "0.31.0"}), "engine_identity"),  # type: ignore[dict-item]
    (adapter(load_error={"type": "checkpoint_mismatch", "message": "x"}), "model_identity"),
    (adapter(probe=FakeProbe(swapout_pages=10_000)), "memory_preflight"),  # ~150 MB swapped out
    (adapter(probe=FakeProbe(level=4)), "memory_preflight"),  # critical pressure
    (adapter(answer="unused", think_closed=False, finish="length"), "output_decoding"),
])
def test_qualification_failures(mlx: MlxLmAdapter, failing: str) -> None:
    checks = {c.name: c.passed for c in mlx.qualify(CONFIG, REQUIREMENTS).checks}
    assert checks[failing] is False, checks


def test_temperature_must_be_explicit_because_mlx_lm_defaults_to_greedy() -> None:
    checks = {c.name: c for c in adapter().qualify(
        with_param("temperature", None), REQUIREMENTS.model_copy(update={"requires_continuation": False})).checks}
    assert checks["parameter_support"].passed is False and "greedy" in checks["parameter_support"].detail


def test_missing_pins_stop_qualification_before_starting_a_worker() -> None:
    config = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options, "files": {}}})
    checks = {c.name: c for c in adapter().qualify(config, REQUIREMENTS).checks}
    assert checks["configuration_pins"].passed is False and not FakeWorker.instances


def test_memory_sampler_reports_deltas_and_survives_without_a_process() -> None:
    with MemorySampler(None, interval_s=0.01, probe=FakeProbe(level=2, swapout_pages=3)) as sampler:  # type: ignore[arg-type]
        pass
    summary = sampler.summary()
    assert summary["available"] and summary["pressure_max"] == "warning"
    assert summary["swapouts_bytes"] == 3 * 16384 and summary["worker_footprint_max_bytes"] is None


@pytest.mark.skipif(sys.platform != "darwin", reason="kernel memory counters are read on macOS only")
def test_host_probe_reads_real_kernel_counters() -> None:
    probe = HostProbe()
    child = subprocess.Popen([sys.executable, "-c", "b = bytearray(64 * 1024 * 1024); import time; time.sleep(5)"])
    try:
        time.sleep(1)
        footprint = probe.footprint(child.pid)
    finally:
        child.kill()
        child.wait()
    assert footprint is not None and footprint > 60 * 1024**2  # struct layout matches the kernel's
    assert probe.pressure_level() in (1, 2, 4) and 0 <= (probe.free_percent() or -1) <= 100
    counters = probe.vm_counters()
    assert counters is not None and min(counters.values()) >= 0
    assert (probe.swap_used() or 0) >= 0 and probe.page_size() in (4096, 16384)
    info = probe.host_info()
    assert info["memory_bytes"] > 0 and info["os_version"]
    assert HostProbe().footprint(2**22) is None  # no such process


def test_host_probe_reports_unavailable_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    probe = HostProbe()
    assert probe.pressure_level() is None and probe.vm_counters() is None and probe.footprint(1) is None
    with MemorySampler(None, interval_s=0.01, probe=probe) as sampler:
        pass
    assert sampler.summary() == {"available": False, "reason": "host memory sampling is implemented for macOS only"}


def test_worker_manifest_hashes_resolved_snapshot_files(tmp_path: Path) -> None:
    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    blobs.mkdir()
    snap.mkdir()
    (blobs / "abc").write_text("{}")
    (snap / "config.json").symlink_to(blobs / "abc")
    assert mlx_worker.manifest(snap) == {
        "config.json": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"}


STUB_WORKER = textwrap.dedent("""
    import json, os, sys, time
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    for line in sys.stdin:
        op = json.loads(line)["op"]
        print("library noise on stdout")
        if op == "crash":
            print("fatal: out of memory", file=sys.stderr, flush=True)
            os._exit(3)
        if op == "hang":
            time.sleep(30)
        out.write(json.dumps({"ok": True, "result": {"op": op, "offline": os.environ.get("HF_HUB_OFFLINE"),
                                                     "secret": os.environ.get("ANTHROPIC_API_KEY")}}) + "\\n")
        if op == "shutdown":
            break
""")


def test_subprocess_channel_isolates_stdout_env_crashes_and_hangs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                                                  ) -> None:
    script = tmp_path / "stub_worker.py"
    script.write_text(STUB_WORKER)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    worker = SubprocessWorker(sys.executable, script=script)
    assert worker.request({"op": "hello"}, 10)["result"] == {"op": "hello", "offline": "1", "secret": None}
    with pytest.raises(WorkerExited) as crashed:
        worker.request({"op": "crash"}, 10)
    assert crashed.value.status["returncode"] == 3 and "out of memory" in crashed.value.status["stderr_tail"]
    assert worker.pid is None
    worker.close()
    hung = SubprocessWorker(sys.executable, script=script)
    with pytest.raises(WorkerTimeout):
        hung.request({"op": "hang"}, 0.5)
    assert hung.pid is None
    hung.close()
    with pytest.raises(FileNotFoundError):
        SubprocessWorker(str(tmp_path / "missing-python"), script=script)


def test_local_mlx_qwen_in_a_cross_vendor_p6_run(env: Env) -> None:
    mlx = adapter(answer="1. **Qwen3.5 9B**: runs locally and privately.")
    registry = AdapterRegistry({"mlx-lm": mlx, "fake": FakeAdapter(), "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-mlx", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
                          rubric="rubrics/agentic-design-bias/rubric.yaml", repetitions=2)
    loaded = load_experiment(path, env.base)
    scenario, _ = load_scenario(env.base / loaded.spec.scenario, env.base)
    for config, frozen in loaded.configurations:
        record = qualify_configuration(env.db, registry, config, frozen, scenario, "test")
        assert record.payload["report"]["passed"], record.payload["report"]["checks"]
    plan = compile_plan(path, base=env.base, db=env.db)
    persist_plan(env.db, env.store, plan)
    assert Controller(env.db, env.store, registry).run(plan).responses == 4
    run = start_evaluation(env.db, env.store, plan.experiment_id, bias_policy(env), base=env.base, registry=registry)
    resume_evaluation(env.db, env.store, registry, run.record_id)
    report = publish_report(env.db, env.store, plan.experiment_id, run.record_id)
    rows = {r["configuration"]: r for r in _read_csv(env, env.db.artifacts_for(report.record_id)["bias-summary.csv"])}
    qwen = rows["local-qwen35-9b-4bit-mlx@1"]
    assert qwen["own_vendor"] == "alibaba" and float(qwen["own_first_delta"]) == 1.0
