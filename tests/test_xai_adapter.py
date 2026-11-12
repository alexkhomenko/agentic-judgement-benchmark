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

from archbench.adapters.base import InvocationContext, QualificationRequirements
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
from .test_openai_adapter import response

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


class MockXAI:
    """Models API (prices in cents per 100M tokens), language-models, and the Responses API."""

    def __init__(self, *, card: dict[str, Any] | None = None, answer: str = "1. **Grok 4.3**: fits.") -> None:
        self.requests: list[httpx2.Request] = []
        self.card = card or {}
        self.answer = answer

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
        return httpx2.Response(200, json=grok(text), headers={"x-request-id": "req_x"})

    def adapter(self, api_key: str | None = XAI_KEY) -> XAIAdapter:
        return XAIAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def call(tmp_path: Path, mock: MockXAI, api_key: str | None = XAI_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    ctx = InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=1)
    outcome = invoke(mock.adapter(api_key), CONFIG, MESSAGES, None, ctx, sink, fork=False)
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
