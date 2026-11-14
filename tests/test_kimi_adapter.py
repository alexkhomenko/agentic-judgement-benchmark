"""Kimi (Moonshot AI) chat-completions adapter against a mock transport: native request, cache-aware
pricing, failure mapping, exact replay including reasoning_content, qualification against /v1/models, and
an offline P6 run with a Kimi candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.kimi_chat import KimiAdapter, _observations
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

API_KEY = "sk-moonshot-test-0123456789abcdefghij"
CONFIG_PATH = ROOT / "configs/models/moonshot/kimi-k3.1-pro.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)
WRITE_HEADERS = {"msh-usage-cache-write-tokens-5m": "256", "msh-usage-cache-write-tokens-1h": "0"}


def completion(text: str = "ok", *, finish: str = "stop", model: str = MODEL, reasoning: str | None = "pondering",
               ) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1_792_740_680, "model": model,
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": text, "reasoning_content": reasoning}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
                  "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 256},
                  "completion_tokens_details": {"reasoning_tokens": 300}},
    }


class MockKimi:
    def __init__(self, *, listed: dict[str, Any] | None = None, answer: str = "1. **Kimi K3.1 Pro**: fits.",
                 reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.listed = {"id": MODEL, "object": "model", "created": 1_792_000_000, "owned_by": "moonshot",
                       "context_length": 1048576, "supports_image_in": True, "supports_video_in": True,
                       "supports_reasoning": True, **(listed or {})}
        self.answer, self.reply = answer, reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path == "/v1/models":
            return httpx2.Response(200, json={"object": "list", "data": [self.listed]})
        assert request.url.path == "/v1/chat/completions", request.url
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return httpx2.Response(200, json=completion(text, model=body["model"]), headers=WRITE_HEADERS)

    def adapter(self, api_key: str | None = API_KEY) -> KimiAdapter:
        return KimiAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, mock: MockKimi, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockKimi(answer="Use Kimi.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Kimi."
    (request,) = mock.requests
    assert str(request.url) == "https://api.moonshot.ai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "model": MODEL, "max_completion_tokens": 16000,
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
    }  # no effort, thinking toggle, fixed sampling parameters, or tools
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert files["reasoning.txt"] == "pondering"
    assert outcome.observations["reasoning_tokens"].value == 300
    assert outcome.observations["uncached_input_tokens"].value == 1000 - 400 - 256
    assert outcome.observations["api_cost_usd"].value == pytest.approx((344 * 2.0 + 400 * 0.2 + 256 * 2.0 + 500 * 10.0) / 1e6)


def test_explicit_parameters_map_to_native_fields(tmp_path: Path) -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update({"reasoning_effort": {"state": "explicit", "value": "low"},
                               "thinking": {"state": "explicit", "value": "disabled"}})
    mock = MockKimi()
    call(tmp_path, mock, ModelConfiguration.model_validate(data))
    body = json.loads(mock.requests[0].content)
    assert body["reasoning_effort"] == "low" and body["thinking"] == {"type": "disabled"}


def test_custom_base_url_keeps_its_path(tmp_path: Path) -> None:
    mock = MockKimi()
    config = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options,
                                                           "base_url": "https://kimi.example.test/v1/"}})
    call(tmp_path, mock, config)
    assert str(mock.requests[0].url) == "https://kimi.example.test/v1/chat/completions"


def test_cost_prices_reads_writes_and_misses_separately() -> None:
    usage = completion()["usage"]
    obs = _observations(CONFIG, usage, WRITE_HEADERS)
    expected = (344 * 2.0 + 400 * 0.2 + 256 * 2.0 + 500 * 10.0) / 1e6
    assert obs["api_cost_usd"].value == pytest.approx(expected)
    one_hour = {"msh-usage-cache-write-tokens-5m": "0", "msh-usage-cache-write-tokens-1h": "256"}
    assert _observations(CONFIG, usage, one_hour)["api_cost_usd"].value == pytest.approx(expected + 256 * 2.0 / 1e6)


@pytest.mark.parametrize("headers", [{}, {"msh-usage-cache-write-tokens-5m": "100",
                                          "msh-usage-cache-write-tokens-1h": "0"}])
def test_cache_writes_without_a_matching_ttl_split_leave_cost_unknown(headers: dict[str, str]) -> None:
    obs = _observations(CONFIG, completion()["usage"], headers)
    assert obs["api_cost_usd"].value is None and "TTL" in str(obs["api_cost_usd"].missing_reason)


def test_no_cache_writes_need_no_ttl_split() -> None:
    usage = completion()["usage"]
    usage["prompt_tokens_details"] = {"cached_tokens": 0, "cache_write_tokens": 0}
    assert _observations(CONFIG, usage, {})["api_cost_usd"].value == pytest.approx((1000 * 2.0 + 500 * 10.0) / 1e6)


def error(status: int, kind: str, message: str = "nope") -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"type": kind, "message": message}})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (error(400, "content_filter"), FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (error(400, "invalid_request_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(401, "invalid_authentication_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(404, "resource_not_found_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429, "exceeded_current_quota_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429, "rate_limit_reached_error"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(429, "engine_overloaded_error"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(503, "server_unavailable"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (httpx2.Response(504, text="gateway timeout"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockKimi(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


@pytest.mark.parametrize(("body", "kind"), [
    (completion(finish="tool_calls"), FailureKind.PROVIDER_ERROR),
    (completion(finish="client_disconnected"), FailureKind.PROVIDER_ERROR),
    (completion(model="kimi-k3"), FailureKind.PROVIDER_ERROR),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind) -> None:
    outcome, _ = call(tmp_path, MockKimi(reply=httpx2.Response(200, json=body)))
    assert outcome.failure_kind is kind and outcome.cost_incurred is CostIncurred.YES


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockKimi(reply=httpx2.Response(200, json=completion("partial", finish="length"))))
    assert outcome.succeeded and outcome.extracted.truncated


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockKimi()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    assert default_registry().get("kimi").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("MOONSHOT_API_KEY", API_KEY)
    assert default_registry().get("kimi").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_the_message_exactly(tmp_path: Path) -> None:
    mock = MockKimi()
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
        adapter.prepare(CONFIG.model_copy(update={"model": "kimi-k3"}), follow, snapshot, ctx(2))


def test_qualification_records_the_model_card() -> None:
    report = MockKimi().adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "context_length=1048576, supports_reasoning=True" in checks["model_identity"].detail
    assert "including reasoning_content" in checks["continuation"].detail


def test_qualification_rejects_fixed_sampling_parameters() -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["temperature"] = {"state": "explicit", "value": 0.2}
    checks = {c.name: c for c in MockKimi().adapter().qualify(
        ModelConfiguration.model_validate(data), REQUIREMENTS.model_copy(update={"requires_continuation": False})).checks}
    assert checks["parameter_support"].passed is False and "temperature" in checks["parameter_support"].detail


@pytest.mark.parametrize(("listed", "expected"), [
    ({"id": "kimi-other"}, "not listed"),
    ({"context_length": 262144}, "context_limit"),
])
def test_qualification_rejects_identity_problems(listed: dict[str, Any], expected: str) -> None:
    checks = {c.name: c for c in MockKimi(listed=listed).adapter().qualify(
        CONFIG, REQUIREMENTS.model_copy(update={"requires_continuation": False})).checks}
    assert checks["model_identity"].passed is False and expected in checks["model_identity"].detail


def test_kimi_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"kimi": MockKimi().adapter(), "fake": FakeAdapter(),
                                "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-kimi", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["kimi-k3.1-pro@1"]
    assert row["own_vendor"] == "moonshot" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0
