"""Gemini generateContent adapter against a mock transport: wire capture, redaction, no retries, failure
mapping, cost, stateless continuation with thought signatures, qualification, credentials, and an offline
P6 run with a Gemini candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.gemini_generate import GeminiAdapter
from archbench.adapters.fake import FakeAdapter
from archbench.adapters.fake_judge import FakeJudgeAdapter
from archbench.adapters.registry import AdapterRegistry, default_registry
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
from archbench.secrets import GUARD
from archbench.storage.artifacts import ArtifactStore

from .conftest import ROOT, Env
from .test_bias import _read_csv, bias_policy, recommender

API_KEY = "AIza" + "TestKey0123456789abcdefghijklmnopqrs"  # Google key shape: AIza + 35 characters
MODEL = "gemini-3.5-flash-lite"
CONFIG = ModelConfiguration.model_validate(
    yaml.safe_load((ROOT / "configs/models/google/gemini-3.5-flash-lite.yaml").read_text()))
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
Handler = Callable[[httpx2.Request], httpx2.Response]
SIGNATURE = "c2lnbmF0dXJlLWJ5dGVz+/=="


def generated(text: str = "ok", *, finish: str = "STOP", model: str = MODEL, thought: str | None = None,
              signature: str | None = SIGNATURE, usage: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if thought:
        parts.append({"text": thought, "thought": True})
    answer: dict[str, Any] = {"text": text}
    if signature:
        answer["thoughtSignature"] = signature
    parts.append(answer)
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish, "index": 0}],
        "usageMetadata": usage or {"promptTokenCount": 1000, "cachedContentTokenCount": 100,
                                   "candidatesTokenCount": 200, "thoughtsTokenCount": 300, "totalTokenCount": 1500},
        "modelVersion": model, "responseId": "resp-1", **extra,
    }


def ok(body: dict[str, Any]) -> httpx2.Response:
    return httpx2.Response(200, json=body)


def error(status: int, code: str) -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"code": status, "message": "x", "status": code}})


class Recorder:
    def __init__(self, *responses: httpx2.Response | Exception | Handler) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        nxt = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(nxt, Exception):
            raise nxt
        if callable(nxt) and not isinstance(nxt, httpx2.Response):
            return nxt(request)
        return nxt

    def adapter(self, api_key: str | None = API_KEY) -> GeminiAdapter:
        return GeminiAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, recorder: Recorder, *, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(recorder.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    rec = Recorder(ok(generated("Use Gemini.")))
    outcome, files = call(tmp_path, rec)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Gemini."
    (request,) = rec.requests
    assert str(request.url) == f"https://generativelanguage.googleapis.com/v1/models/{MODEL}:generateContent"
    assert request.headers["x-goog-api-key"] == API_KEY and "key=" not in str(request.url)
    body = json.loads(request.content)
    assert body == {
        "contents": [{"role": "user", "parts": [{"text": "Pick a model."}]}],
        "systemInstruction": {"parts": [{"text": "Be precise."}]},
        "generationConfig": {"maxOutputTokens": 16000},
    }  # no thinking level, temperature, safety settings, or tools
    wire = json.loads(files["wire-exchanges.json"])
    assert len(wire) == 1 and wire[0]["request_body"]["json"] == body
    assert wire[0]["request_headers"]["x-goog-api-key"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert outcome.observations["reasoning_tokens"].value == 300
    assert outcome.observations["output_tokens"].value == 500  # candidates + thoughts, as billed


def test_explicit_parameters_and_thought_summaries(tmp_path: Path) -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update({
        "thinking_level": {"state": "explicit", "value": "low"},
        "include_thoughts": {"state": "explicit", "value": True},
        "temperature": {"state": "explicit", "value": 0.2},
    })
    rec = Recorder(ok(generated("answer", thought="weighing options")))
    outcome, files = call(tmp_path, rec, config=ModelConfiguration.model_validate(data))
    assert json.loads(rec.requests[0].content)["generationConfig"] == {
        "maxOutputTokens": 16000, "thinkingConfig": {"thinkingLevel": "low", "includeThoughts": True},
        "temperature": 0.2,
    }
    assert outcome.extracted.final_answer == "answer"  # thought parts never leak into the answer
    assert files["reasoning.txt"] == "weighing options"


def test_cost_counts_cached_input_and_thinking_as_output(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(generated())))
    assert outcome.observations["api_cost_usd"].value == pytest.approx((900 * 0.30 + 100 * 0.03 + 500 * 2.50) / 1e6)
    outcome, _ = call(tmp_path, Recorder(ok(generated())), config=CONFIG.model_copy(update={"adapter_options": {}}))
    assert outcome.observations["api_cost_usd"].missing_reason == "no pricing in configuration"


@pytest.mark.parametrize(
    ("reply", "kind", "cost"),
    [
        (error(503, "UNAVAILABLE"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
        (error(429, "RESOURCE_EXHAUSTED"), FailureKind.RATE_LIMITED, CostIncurred.NO),
        (error(400, "INVALID_ARGUMENT"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
        (error(403, "PERMISSION_DENIED"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
        (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
        (httpx2.ConnectError("refused"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
    ],
)
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind,
                                               cost: CostIncurred) -> None:
    rec = Recorder(reply, ok(generated()))
    outcome, files = call(tmp_path, rec)
    assert not outcome.succeeded
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(rec.requests) == 1
    assert "error.json" in files


@pytest.mark.parametrize("body", [
    generated(finish="SAFETY"),
    generated(finish="PROHIBITED_CONTENT"),
    {"promptFeedback": {"blockReason": "SAFETY"}, "modelVersion": MODEL, "responseId": "r",
     "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 10}},
])
def test_blocks_are_recorded_as_refusals(tmp_path: Path, body: dict[str, Any]) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(body)))
    assert outcome.failure_kind is FailureKind.REFUSAL and outcome.cost_incurred is CostIncurred.YES


@pytest.mark.parametrize("body", [
    generated(model="gemini-3.8-flash"),
    generated(finish="OTHER"),
    {"candidates": [], "modelVersion": MODEL, "usageMetadata": {"promptTokenCount": 10}},
])
def test_wrong_model_and_unfinished_answers_are_failures(tmp_path: Path, body: dict[str, Any]) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(body)))
    assert not outcome.succeeded and outcome.failure_kind is FailureKind.PROVIDER_ERROR


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(generated("partial", finish="MAX_TOKENS"))))
    assert outcome.succeeded and outcome.extracted.truncated and outcome.extracted.finish_reason == "MAX_TOKENS"


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(ok(generated()))
    outcome, _ = call(tmp_path, rec, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and outcome.cost_incurred is CostIncurred.NO
    assert rec.requests == []
    assert "GEMINI_API_KEY" in str(rec.adapter(None).preflight(CONFIG))
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert default_registry().get("gemini").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("GEMINI_API_KEY", API_KEY)
    assert default_registry().get("gemini").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_thought_signatures_statelessly(tmp_path: Path) -> None:
    baseline = generated("baseline answer", thought="private reasoning")
    rec = Recorder(ok(baseline), ok(generated("branch a")), ok(generated("branch b")))
    adapter = rec.adapter()
    store = ArtifactStore(tmp_path / "results")
    base = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base"))
    snapshot = base.snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    for request, word in zip(rec.requests[1:], ("alpha", "beta")):
        sent = json.loads(request.content)
        assert sent["systemInstruction"] == {"parts": [{"text": "Be precise."}]}
        assert sent["contents"][1] == baseline["candidates"][0]["content"]  # exact, signature included
        assert sent["contents"][1]["parts"][-1]["thoughtSignature"] == SIGNATURE
        assert sent["contents"][2] == {"role": "user", "parts": [{"text": f"Follow-up {word}"}]}
    assert "alpha" not in rec.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "gemini-3.8-flash"}), follow, snapshot, ctx(2))
    plain = MESSAGES + (Message(role=Role.ASSISTANT, content="made up"), Message(role=Role.USER, content="?"))
    with pytest.raises(AdapterError, match="continuation snapshot"):
        adapter.prepare(CONFIG, plain, None, ctx(2))


class MockGemini:
    """Models API plus generateContent, answering one-word probes and P6 picks."""

    def __init__(self, *, input_limit: int = 1_048_576, answer: str = "1. **Gemini 3.8 Flash**: fits the job.") -> None:
        self.requests: list[httpx2.Request] = []
        self.input_limit = input_limit
        self.answer = answer

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.method == "GET":
            name = request.url.path.split("/models/", 1)[1]
            return httpx2.Response(200, json={"name": f"models/{name}", "version": "001",
                                              "inputTokenLimit": self.input_limit, "outputTokenLimit": 65536})
        body = json.loads(request.content)
        last = body["contents"][-1]["parts"][0]["text"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        model = request.url.path.split("/models/", 1)[1].split(":", 1)[0]
        return ok(generated(text, model=model))


REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)


def test_qualification_probes_identity_limits_and_branching() -> None:
    report = GeminiAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(MockGemini())).qualify(
        CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "thought signatures" in checks["continuation"].detail
    assert "inputTokenLimit=1048576" in checks["model_identity"].detail
    assert all(API_KEY not in json.dumps(p) for p in report.probes)


def test_qualification_rejects_a_context_limit_beyond_the_model() -> None:
    mock = MockGemini(input_limit=500_000)
    report = GeminiAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(mock)).qualify(
        CONFIG, REQUIREMENTS.model_copy(update={"requires_continuation": False}))
    assert {c.name: c.passed for c in report.checks}["model_identity"] is False


def test_gemini_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({
        "gemini": GeminiAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(MockGemini())),
        "fake": FakeAdapter(), "fake-judge": FakeJudgeAdapter(),
    })
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-gemini", [ROOT / "configs/models/google/gemini-3.5-flash-lite.yaml", other],
                          scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    gemini = rows["google-gemini-3-5-flash-lite@1"]
    assert gemini["own_vendor"] == "google" and gemini["own_first.k"] == "2"
    assert float(gemini["own_first_delta"]) == 1.0
