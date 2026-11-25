"""xAI adapter (Responses API through the OpenAI SDK at api.x.ai) against a mock transport: credential
isolation from OpenAI environment variables, provider-reported cost, stateless replay, qualification with
build fingerprint and price checks, and an offline P6 run with a Grok candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.xai_responses import XAIAdapter
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
from .test_openai_adapter import response, search_call, searched

XAI_KEY = "xai-TestKey0123456789abcdefghij"
OPENAI_KEY = "sk-proj-should-never-reach-xai-0123456789"
CONFIG_PATH = ROOT / "configs/models/xai/grok-4.3.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)


def grok(text: str = "ok", ticks: int | None = 12_345_678, **kwargs: Any) -> dict[str, Any]:
    body = response(text, model=MODEL, **kwargs)
    if ticks is not None:
        body["usage"]["cost_in_usd_ticks"] = ticks
    return body


SEARCH_TOOL = {"type": "web_search", "filters": {"excluded_domains": ["example.org"]}}
CITED = "Use Grok.[[1]](https://example.com/)"


def searching_config(**pricing: float) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["web_search"] = {"state": "explicit", "value": SEARCH_TOOL}
    data["adapter_options"]["pricing"].update(pricing)
    return ModelConfiguration.model_validate(data)


def grok_searched(text: str = CITED, *, calls: int | None = 3, ticks: int = 30_000_000, **kwargs: Any
                  ) -> dict[str, Any]:
    """Two searches and one page browse (all billed web search calls), then the cited answer."""
    body = searched(searches=None, text=text, model=MODEL, **kwargs)
    body["usage"]["cost_in_usd_ticks"] = ticks  # tokens and tool calls together, as xAI documents
    body["usage"]["num_server_side_tools_used"] = calls or 0
    if calls is not None:
        body["usage"]["server_side_tool_usage_details"] = {
            "web_search_calls": calls, "x_search_calls": 0, "x_posts_fetched": 0, "code_interpreter_calls": 0,
            "file_search_calls": 0, "mcp_calls": 0, "document_search_calls": 0, "image_generation_calls": 0}
    return body


class MockXAI:
    """Models API (prices in cents per 100M tokens), language-models, and the Responses API."""

    def __init__(self, *, card: dict[str, Any] | None = None, answer: str = "1. **Grok 4.3**: fits.",
                 search: bool = False) -> None:
        self.requests: list[httpx2.Request] = []
        self.card = card or {}
        self.answer = answer
        self.search = search  # answer with web_search_call items before the message

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        if path.startswith("/v1/models/"):
            return httpx2.Response(200, json={
                "id": path.rsplit("/", 1)[-1], "object": "model", "aliases": ["grok-4.3-latest"],
                "context_length": 1_000_000, "capabilities": {"default_reasoning_effort": "low"},
                "prompt_text_token_price": 12500, "cached_prompt_text_token_price": 2000,
                "completion_text_token_price": 25000, **self.card})
        if path.startswith("/v1/language-models/"):
            return httpx2.Response(200, json={"id": MODEL, "version": "1.0.3", "fingerprint": "fp_grok43"})
        body = json.loads(request.content)
        last = body["input"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        reply = grok_searched(text) if self.search else grok(text)
        return httpx2.Response(200, json=reply, headers={"x-request-id": "req_x"})

    def adapter(self, api_key: str | None = XAI_KEY) -> XAIAdapter:
        return XAIAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def call(tmp_path: Path, mock: MockXAI, api_key: str | None = XAI_KEY, *, config: ModelConfiguration = CONFIG
         ) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    ctx = InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=1)
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx, sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


@pytest.fixture
def openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every OpenAI variable the SDK would pick up, so isolation is tested against the worst case."""
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_ORG_ID", "org-private-openai-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_private_openai_project")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-private-0123456789abcdef")


def test_requests_go_to_xai_with_only_the_xai_key(tmp_path: Path, openai_env: None) -> None:
    GUARD.register(XAI_KEY)
    mock = MockXAI(answer="Use Grok.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Grok."
    (request,) = mock.requests
    assert str(request.url) == "https://api.x.ai/v1/responses"
    assert request.headers["authorization"] == f"Bearer {XAI_KEY}"
    for header in ("openai-organization", "openai-project"):
        assert header not in request.headers
    sent = str(request.headers) + request.content.decode()
    for leaked in (OPENAI_KEY, "org-private-openai-org", "proj_private_openai_project", "sk-admin-private"):
        assert leaked not in sent
    assert json.loads(request.content) == {
        "model": MODEL, "instructions": "Be precise.", "input": [{"role": "user", "content": "Pick a model."}],
        "max_output_tokens": 16000, "store": False, "include": ["reasoning.encrypted_content"],
    }
    for text in files.values():
        assert XAI_KEY not in text


def test_without_an_xai_key_nothing_is_sent_even_if_openai_is_configured(tmp_path: Path, openai_env: None,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockXAI()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and outcome.cost_incurred is CostIncurred.NO
    assert mock.requests == []
    assert "XAI_API_KEY" in str(mock.adapter(None).preflight(CONFIG))
    checks = {c.name: c for c in mock.adapter(None).qualify(CONFIG, REQUIREMENTS).checks}
    assert checks["model_identity"].passed is None and mock.requests == []
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert default_registry().get("xai").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("XAI_API_KEY", XAI_KEY)
    assert default_registry().get("xai").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_provider_reported_cost_is_recorded_beside_the_configured_price(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockXAI())
    obs = outcome.observations
    assert obs["api_cost_usd"].value == pytest.approx(0.0012345678)
    assert obs["api_cost_usd"].source == "provider-reported usage.cost_in_usd_ticks"
    # 800 uncached * $1.25/M + 200 cached * $0.20/M + 500 output * $2.50/M
    assert obs["api_cost_usd_configured"].value == pytest.approx((800 * 1.25 + 200 * 0.20 + 500 * 2.50) / 1e6)


class Replies(MockXAI):
    """Answers every Responses API request with the given bodies, in order (the last one repeats)."""

    def __init__(self, *bodies: dict[str, Any]) -> None:
        super().__init__()
        self.bodies = list(bodies)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        return httpx2.Response(200, json=body, headers={"x-request-id": "req_x"})


def test_web_search_tool_is_sent_verbatim_and_searches_are_counted_without_double_charging(tmp_path: Path) -> None:
    base = (800 * 1.25 + 200 * 0.20 + 500 * 2.50) / 1e6
    mock = Replies(grok_searched())
    outcome, files = call(tmp_path, mock, config=searching_config(web_search_usd_per_request=0.005))
    assert json.loads(mock.requests[0].content)["tools"] == [SEARCH_TOOL]
    assert outcome.succeeded and outcome.extracted.final_answer == CITED
    assert json.loads(files["native-response.json"])["output"][1]["type"] == "web_search_call"
    obs = outcome.observations
    assert (obs["web_search_requests"].value, obs["web_search_requests"].source) == (
        3, "xai usage.server_side_tool_usage_details.web_search_calls")
    # The ticks already include the tool calls: reported as-is, never topped up with the search rate.
    assert obs["api_cost_usd"].value == pytest.approx(0.003)
    assert obs["api_cost_usd_configured"].value == pytest.approx(base + 3 * 0.005)
    # Missing rate: the configured cross-check is unknown; the provider-reported cost stands.
    outcome, _ = call(tmp_path, Replies(grok_searched()), config=searching_config())
    assert outcome.observations["api_cost_usd_configured"].missing_reason == "no price for web_search_requests"
    assert outcome.observations["api_cost_usd"].value == pytest.approx(0.003)
    # No usage breakdown: count successful web_search_call items (page browsing is a billed call on xAI).
    body = grok_searched(calls=None)
    body["output"].insert(4, search_call(4, status="failed"))
    outcome, _ = call(tmp_path, Replies(body), config=searching_config(web_search_usd_per_request=0.005))
    assert outcome.observations["web_search_requests"].value == 3
    assert "counted from web_search_call" in outcome.observations["web_search_requests"].source
    # Without ticks, api_cost_usd is the configured price including searches.
    del body["usage"]["cost_in_usd_ticks"]
    outcome, _ = call(tmp_path, Replies(body), config=searching_config(web_search_usd_per_request=0.005))
    assert outcome.observations["api_cost_usd"].value == pytest.approx(base + 3 * 0.005)
    assert "api_cost_usd_configured" not in outcome.observations
    # Without the parameter: no tools and no observation.
    mock = Replies(grok())
    outcome, _ = call(tmp_path, mock)
    assert "web_search_requests" not in outcome.observations and "tools" not in json.loads(mock.requests[0].content)


def test_unfinished_server_tool_loop_is_an_incomplete_failure(tmp_path: Path) -> None:
    body = grok_searched()
    body["output"] = body["output"][:4]  # completed, but no answer after the last search call
    outcome, _ = call(tmp_path, Replies(body), config=searching_config(web_search_usd_per_request=0.005))
    assert not outcome.succeeded and outcome.failure_kind is FailureKind.PROVIDER_ERROR
    assert outcome.cost_incurred is CostIncurred.YES and outcome.observations["api_cost_usd"].value is not None


@pytest.mark.parametrize("value", [{"type": "x_search"}, {"type": "web_search_2025_08_26"}, [SEARCH_TOOL]])
def test_web_search_value_must_be_xai_web_search(value: Any) -> None:
    data = searching_config().model_dump(mode="json")
    data["parameters"]["web_search"]["value"] = value
    ctx = InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=1)
    with pytest.raises(AdapterError, match="web_search"):
        MockXAI().adapter().prepare(ModelConfiguration.model_validate(data), MESSAGES, None, ctx)


def test_qualification_replays_search_items_statelessly() -> None:
    mock = MockXAI(search=True)
    report = mock.adapter().qualify(searching_config(web_search_usd_per_request=0.005), REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    posts = [json.loads(r.content) for r in mock.requests if r.method == "POST"]
    assert len(posts) == 3 and all(p["tools"] == [SEARCH_TOOL] for p in posts)
    assert [i["type"] for i in posts[1]["input"][1:6]] == [
        "reasoning", "web_search_call", "web_search_call", "web_search_call", "message"]


def test_missing_output_budget_skips_continuation_probes() -> None:
    data = CONFIG.model_dump(mode="json")
    del data["parameters"]["max_output_tokens"]
    mock = MockXAI()
    report = mock.adapter().qualify(ModelConfiguration.model_validate(data), REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed
    assert checks["parameter_support"].passed is False and "max_output_tokens" in checks["parameter_support"].detail
    assert checks["continuation"].passed is None and "parameters are invalid" in checks["continuation"].detail
    assert all(r.method == "GET" for r in mock.requests)


def test_qualification_records_fingerprint_and_replays_statelessly() -> None:
    report = MockXAI().adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "fingerprint=fp_grok43" in checks["model_identity"].detail
    assert "default_reasoning_effort=low" in checks["model_identity"].detail
    assert "encrypted reasoning" in checks["continuation"].detail


@pytest.mark.parametrize(("card", "expected"), [
    ({"completion_text_token_price": 30000}, "output_usd_per_mtok is 2.5 but xAI publishes 3.0"),
    ({"context_length": 256000}, "context_limit"),
    ({"id": "grok-4.3-0901"}, "API reports id"),
])
def test_qualification_rejects_price_drift_and_identity_problems(card: dict[str, Any], expected: str) -> None:
    checks = {c.name: c for c in MockXAI(card=card).adapter().qualify(
        CONFIG, REQUIREMENTS.model_copy(update={"requires_continuation": False})).checks}
    assert checks["model_identity"].passed is False and expected in checks["model_identity"].detail


def test_grok_candidate_in_a_cross_vendor_p6_run(env: Env, openai_env: None) -> None:
    registry = AdapterRegistry({"xai": MockXAI().adapter(), "fake": FakeAdapter(), "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-grok", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    grok_row = rows["xai-grok-4-3@1"]
    assert grok_row["own_vendor"] == "xai" and grok_row["own_first.k"] == "2"
    assert float(grok_row["own_first_delta"]) == 1.0
