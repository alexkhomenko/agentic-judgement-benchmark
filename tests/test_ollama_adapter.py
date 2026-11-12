"""Local Ollama adapter against a mock engine: pinned digest and engine, explicit context window, failure
mapping, engine timing and memory, exact replay including thinking, qualification, and an offline P6 run
with a local Qwen candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.fake import FakeAdapter
from archbench.adapters.fake_judge import FakeJudgeAdapter
from archbench.adapters.local.ollama_chat import OllamaAdapter
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

from .conftest import ROOT, Env
from .test_bias import _read_csv, bias_policy, recommender

CONFIG_PATH = ROOT / "configs/models/ollama/qwen3.5-9b-mlx.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
DIGEST = str(CONFIG.adapter_options["digest"])
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
GB = 1024**3


def chat(text: str = "ok", *, thinking: str | None = "pondering", done_reason: str = "stop", model: str = MODEL,
         prompt: int = 100, output: int = 50) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if thinking:
        message["thinking"] = thinking
    return {"model": model, "created_at": "2026-11-12T00:00:00Z", "message": message, "done": True,
            "done_reason": done_reason, "total_duration": 3_000_000_000, "load_duration": 1_000_000_000,
            "prompt_eval_count": prompt, "prompt_eval_cached_count": 0, "prompt_eval_duration": 500_000_000,
            "eval_count": output, "eval_duration": 1_000_000_000}


class MockOllama:
    """Ollama engine: version, tags, show, ps, and chat (one-word probes or a scripted answer)."""

    def __init__(self, *, version: str = "0.40.14", digest: str = DIGEST, answer: Any = None,
                 vram: int = 9 * GB, chat_error: Exception | httpx2.Response | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.version, self.digest, self.answer, self.vram = version, digest, answer, vram
        self.chat_error = chat_error

    def chats(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == "/api/chat"]

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/version":
            return httpx2.Response(200, json={"version": self.version})
        if path == "/api/tags":
            return httpx2.Response(200, json={"models": [{"name": MODEL, "digest": self.digest}]})
        if path == "/api/show":
            return httpx2.Response(200, json={
                "thinking": {"values": [False, "low", "medium", "xhigh"], "default": "medium"},
                "capabilities": ["completion", "thinking"], "details": {"parameter_size": "9.4B",
                                                                        "quantization_level": "nvfp4"},
                "model_info": {"qwen3_5.context_length": 262144}, "parameters": "temperature 1\ntop_k 20"})
        if path == "/api/ps":
            return httpx2.Response(200, json={"models": [{"name": MODEL, "size": 9 * GB, "size_vram": self.vram,
                                                          "context_length": 32768}]})
        if self.chat_error is not None:
            if isinstance(self.chat_error, Exception):
                raise self.chat_error
            return self.chat_error
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        if last.startswith("Reply with exactly one word"):
            return httpx2.Response(200, json=chat(last.rsplit(" ", 1)[-1]))
        return httpx2.Response(200, json=self.answer if isinstance(self.answer, dict) else chat(self.answer or "ok"))

    def adapter(self) -> OllamaAdapter:
        return OllamaAdapter(transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, engine: MockOllama, config: ModelConfiguration = CONFIG) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(engine.adapter(), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_digest_checked_first(tmp_path: Path) -> None:
    engine = MockOllama(answer="Use Qwen.")
    outcome, files = call(tmp_path, engine)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Qwen."
    assert [r.url.path for r in engine.requests] == ["/api/tags", "/api/chat", "/api/ps"]
    (body,) = engine.chats()
    assert body == {
        "model": MODEL, "stream": False, "keep_alive": "10m",
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
        "options": {"num_ctx": 32768, "num_predict": 16000},
    }  # no think, temperature, or seed: Modelfile defaults apply
    wire = json.loads(files["wire-exchanges.json"])
    assert [w["url"].rsplit("/", 1)[-1] for w in wire] == ["tags", "chat", "ps"]
    assert files["reasoning.txt"] == "pondering"


def test_engine_timing_memory_and_zero_api_cost_are_observed(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockOllama())
    obs = {k: v.value for k, v in outcome.observations.items()}
    assert obs["input_tokens"] == 100 and obs["output_tokens"] == 50
    assert obs["engine_load_ms"] == 1000.0 and obs["engine_eval_ms"] == 1000.0
    assert obs["output_tokens_per_s"] == 50.0
    assert obs["model_memory_bytes"] == 9 * GB and obs["model_accelerator_memory_bytes"] == 9 * GB
    assert obs["api_cost_usd"] == 0.0 and outcome.cost_incurred is CostIncurred.NO
    assert obs["energy"] is None  # electricity is not estimated, never zero


def test_a_changed_checkpoint_is_never_used(tmp_path: Path) -> None:
    engine = MockOllama(digest="f" * 64)
    outcome, _ = call(tmp_path, engine)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR
    assert engine.chats() == []


def test_exceeding_the_configured_window_is_a_context_failure(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockOllama(answer=chat("ok", prompt=40_000, output=10)))
    assert outcome.failure_kind is FailureKind.CONTEXT_EXCEEDED


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockOllama(answer=chat("partial", done_reason="length")))
    assert outcome.succeeded and outcome.extracted.truncated and outcome.extracted.finish_reason == "length"


@pytest.mark.parametrize(("error", "kind"), [
    (httpx2.ConnectError("refused"), FailureKind.TRANSIENT_TRANSPORT),
    (httpx2.ReadTimeout("slow"), FailureKind.AMBIGUOUS_TIMEOUT),
    (httpx2.Response(404, json={"error": "model not found"}), FailureKind.PROVIDER_ERROR),
    (httpx2.Response(500, json={"error": "out of memory"}), FailureKind.PROVIDER_ERROR),
])
def test_failures_are_mapped_without_cost(tmp_path: Path, error: Any, kind: FailureKind) -> None:
    engine = MockOllama(chat_error=error)
    outcome, files = call(tmp_path, engine)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, CostIncurred.NO)
    assert sum(1 for r in engine.requests if r.url.path == "/api/chat") == 1
    assert "error.json" in files


@pytest.mark.parametrize("answer", [chat(model="qwen3.8:27b-mlx"), {**chat(), "done": False, "done_reason": "load"}])
def test_wrong_model_and_unfinished_answers_are_failures(tmp_path: Path, answer: dict[str, Any]) -> None:
    outcome, _ = call(tmp_path, MockOllama(answer=answer))
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR


def test_continuation_replays_the_thinking_message_exactly(tmp_path: Path) -> None:
    engine = MockOllama(answer=chat("baseline answer", thinking="private reasoning"))
    adapter = engine.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    for sent, word in zip(engine.chats()[1:], ("alpha", "beta")):
        assert sent["messages"][2] == {"role": "assistant", "content": "baseline answer",
                                       "thinking": "private reasoning"}
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in json.dumps(engine.chats()[2])
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    other = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options, "digest": "e" * 64}})
    with pytest.raises(AdapterError, match="different checkpoint"):
        adapter.prepare(other, follow, snapshot, ctx(2))
    with pytest.raises(AdapterError, match="lead the conversation"):
        adapter.prepare(CONFIG, (MESSAGES[1], MESSAGES[0]), None, ctx())


REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)


def test_qualification_pins_engine_digest_context_and_memory() -> None:
    report = MockOllama().adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "including thinking" in checks["continuation"].detail
    assert "9.0 GB in accelerator memory" in checks["memory"].detail
    assert "context_length 262144" in checks["model_identity"].detail


@pytest.mark.parametrize(("engine", "failing"), [
    (MockOllama(version="0.41.0"), "engine_identity"),
    (MockOllama(digest="a" * 64), "model_identity"),
    (MockOllama(vram=4 * GB), "memory"),  # partly offloaded to CPU
])
def test_qualification_failures(engine: MockOllama, failing: str) -> None:
    checks = {c.name: c.passed for c in engine.adapter().qualify(CONFIG, REQUIREMENTS).checks}
    assert checks[failing] is False


def test_context_window_must_be_explicit() -> None:
    data = CONFIG.model_dump(mode="json")
    del data["parameters"]["context_window"]
    checks = {c.name: c for c in MockOllama().adapter().qualify(
        ModelConfiguration.model_validate(data), REQUIREMENTS.model_copy(update={"requires_continuation": False})
    ).checks}
    assert checks["parameter_support"].passed is False and "context_window" in checks["parameter_support"].detail


def test_local_qwen_in_a_cross_vendor_p6_run(env: Env) -> None:
    engine = MockOllama(answer="1. **Qwen3.5 9B**: runs locally and privately.")
    registry = AdapterRegistry({"ollama": engine.adapter(), "fake": FakeAdapter(), "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-local", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    qwen = rows["ollama-qwen3-5-9b-mlx@1"]
    assert qwen["own_vendor"] == "alibaba" and qwen["own_first.k"] == "2"
    assert float(qwen["own_first_delta"]) == 1.0
    assert qwen["open_weights_unknown"] == "2"  # Qwen mixes open and closed releases


@pytest.mark.parametrize(("think", "ok"), [("medium", True), (False, True), ("high", False)])
def test_think_must_be_one_of_the_models_thinking_values(think: Any, ok: bool) -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["think"] = {"state": "explicit", "value": think}
    checks = {c.name: c for c in MockOllama().adapter().qualify(
        ModelConfiguration.model_validate(data), REQUIREMENTS.model_copy(update={"requires_continuation": False})
    ).checks}
    assert checks["model_identity"].passed is ok, checks["model_identity"].detail
