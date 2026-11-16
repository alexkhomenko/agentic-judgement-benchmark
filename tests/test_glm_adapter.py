"""GLM (Z.ai) chat-completions adapter against a mock transport: native request, cache-aware pricing,
business-code failure mapping, exact replay including reasoning_content, qualification by live probe, and
an offline P6 run with a GLM candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.glm_chat import GlmAdapter, _observations
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

API_KEY = "0123456789abcdef0123456789abcdef.AbCdEfGhIjKlMnOp"
CONFIG_PATH = ROOT / "configs/models/zhipu/glm-5.3.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)
NO_BRANCHES = REQUIREMENTS.model_copy(update={"requires_continuation": False})
EXPECTED_COST = (600 * 1.40 + 400 * 0.26 + 500 * 4.40) / 1e6


def completion(text: str = "ok", *, finish: str = "stop", model: str = MODEL, reasoning: str | None = "pondering",
               ) -> dict[str, Any]:
    return {
        "id": "2026111412000012345", "request_id": "req-1", "created": 1_794_000_000, "model": model,
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": text, "reasoning_content": reasoning}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
                  "prompt_tokens_details": {"cached_tokens": 400}},
    }


class MockGlm:
    def __init__(self, *, answer: str = "1. **GLM-5.3**: fits.", served: str | None = None,
                 reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer, self.served, self.reply = answer, served, reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        assert request.url.path == "/api/paas/v4/chat/completions", request.url
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return httpx2.Response(200, json=completion(text, model=self.served or body["model"]))

    def adapter(self, api_key: str | None = API_KEY) -> GlmAdapter:
        return GlmAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def with_parameters(**parameters: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update(parameters)
    return ModelConfiguration.model_validate(data)


def call(tmp_path: Path, mock: MockGlm, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockGlm(answer="Use GLM.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use GLM."
    (request,) = mock.requests
    assert str(request.url) == "https://api.z.ai/api/paas/v4/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "model": MODEL, "max_tokens": 16000,
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
    }  # no thinking object, effort, sampling parameters, or tools
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert files["reasoning.txt"] == "pondering"
    assert outcome.observations["uncached_input_tokens"].value == 600
    assert outcome.observations["reasoning_tokens"].value is None
    assert outcome.observations["api_cost_usd"].value == pytest.approx(EXPECTED_COST)


def test_explicit_parameters_map_to_native_fields(tmp_path: Path) -> None:
    mock = MockGlm()
    call(tmp_path, mock, with_parameters(
        thinking={"state": "explicit", "value": "enabled"}, clear_thinking={"state": "explicit", "value": False},
        reasoning_effort={"state": "explicit", "value": "high"}, temperature={"state": "explicit", "value": 0.6},
        top_p={"state": "explicit", "value": 0.9}))
    body = json.loads(mock.requests[0].content)
    assert body["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert (body["reasoning_effort"], body["temperature"], body["top_p"]) == ("high", 0.6, 0.9)


def test_mainland_base_url_keeps_its_path(tmp_path: Path) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=completion())

    adapter = GlmAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(handler))
    config = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options,
                                                           "base_url": "https://open.bigmodel.cn/api/paas/v4/"}})
    invoke(adapter, config, MESSAGES, None, ctx(), ArtifactStore(tmp_path / "r").sink("c"), fork=False)
    assert str(requests[0].url) == "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def test_cost_prices_cache_hits_and_misses_separately() -> None:
    assert _observations(CONFIG, completion()["usage"])["api_cost_usd"].value == pytest.approx(EXPECTED_COST)


def test_missing_cache_breakdown_leaves_cost_unknown() -> None:
    usage = {k: v for k, v in completion()["usage"].items() if k != "prompt_tokens_details"}
    obs = _observations(CONFIG, usage)
    assert obs["api_cost_usd"].value is None and obs["uncached_input_tokens"].value is None


def error(status: int, code: str, message: str = "nope") -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"code": code, "message": message}})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (error(400, "1301"), FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (error(400, "1261"), FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO),
    (error(400, "1211"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(401, "1000"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429, "1302"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(429, "1305"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(429, "1113"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429, "1310"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(500, "1234"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (httpx2.Response(504, text="gateway timeout"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockGlm(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


@pytest.mark.parametrize(("body", "kind"), [
    (completion(finish="sensitive"), FailureKind.REFUSAL),
    (completion(finish="model_context_window_exceeded"), FailureKind.CONTEXT_EXCEEDED),
    (completion(finish="network_error"), FailureKind.TRANSIENT_TRANSPORT),
    (completion(finish="tool_calls"), FailureKind.PROVIDER_ERROR),
    (completion(model="glm-5.2"), FailureKind.PROVIDER_ERROR),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind) -> None:
    outcome, _ = call(tmp_path, MockGlm(reply=httpx2.Response(200, json=body)))
    assert outcome.failure_kind is kind and outcome.cost_incurred is CostIncurred.YES


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockGlm(reply=httpx2.Response(200, json=completion("partial", finish="length"))))
    assert outcome.succeeded and outcome.extracted.truncated


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockGlm()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    assert default_registry().get("glm").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    assert default_registry().get("glm").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_the_message_exactly(tmp_path: Path) -> None:
    mock = MockGlm()
    adapter = mock.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    first = completion(mock.answer)["choices"][0]["message"]
    for request, word in zip(mock.requests[1:], ("alpha", "beta")):
        sent = json.loads(request.content)
        assert sent["messages"][2] == first  # content and reasoning_content, exactly as returned
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in mock.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "glm-5.2"}), follow, snapshot, ctx(2))


def test_qualification_proves_the_served_model_and_branching() -> None:
    mock = MockGlm()
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "served model glm-5.3" in checks["model_identity"].detail
    assert "including reasoning_content" in checks["continuation"].detail
    assert [p["step"] for p in report.probes] == ["baseline", "branch-alpha", "branch-beta"]
    assert all(json.loads(r.content)["max_tokens"] == 2048 for r in mock.requests)


def test_qualification_without_branches_makes_one_probe() -> None:
    mock = MockGlm()
    report = mock.adapter().qualify(CONFIG, NO_BRANCHES)
    assert report.passed and len(mock.requests) == 1 and "continuation" not in {c.name for c in report.checks}


def test_qualification_rejects_a_substituted_model() -> None:
    checks = {c.name: c for c in MockGlm(served="glm-5.2").adapter().qualify(CONFIG, REQUIREMENTS).checks}
    assert checks["model_identity"].passed is False and "model_mismatch" in checks["model_identity"].detail
    assert "continuation" not in checks


def test_qualification_without_credentials_is_not_evaluated() -> None:
    report = MockGlm().adapter(api_key=None).qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and checks["model_identity"].passed is None


def test_qualification_rejects_unsupported_parameters_without_spending() -> None:
    mock = MockGlm()
    config = with_parameters(frequency_penalty={"state": "explicit", "value": 0.5})
    checks = {c.name: c for c in mock.adapter().qualify(config, NO_BRANCHES).checks}
    assert checks["parameter_support"].passed is False and "frequency_penalty" in checks["parameter_support"].detail
    assert mock.requests == []


def test_qualification_rejects_an_oversized_output_budget() -> None:
    config = with_parameters(max_output_tokens={"state": "explicit", "value": 200_000})
    checks = {c.name: c for c in MockGlm().adapter().qualify(config, NO_BRANCHES).checks}
    assert checks["parameter_support"].passed is False and "131072" in checks["parameter_support"].detail


def test_qualification_rejects_the_coding_plan_endpoint() -> None:
    config = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options,
                                                           "base_url": "https://api.z.ai/api/coding/paas/v4"}})
    mock = MockGlm()
    checks = {c.name: c for c in mock.adapter().qualify(config, NO_BRANCHES).checks}
    assert checks["endpoint"].passed is False and "Coding Plan" in checks["endpoint"].detail
    assert mock.requests == []


def test_glm_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"glm": MockGlm().adapter(), "fake": FakeAdapter(),
                                "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-glm", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["glm-5.3@1"]
    assert row["own_vendor"] == "zhipu" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0
