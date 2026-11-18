"""Meta adapter (Muse on the Meta Model API's Responses endpoint, through the OpenAI SDK) against a mock transport:
credential isolation from OpenAI environment variables, Standard-tier enforcement, Meta's error mapping, no
long-context guard for Spark, stateless replay of encrypted reasoning, and an offline P6 run with a Muse candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import InvocationContext, QualificationRequirements
from archbench.adapters.cloud.meta_responses import MetaAdapter
from archbench.adapters.fake import FakeAdapter
from archbench.adapters.fake_judge import FakeJudgeAdapter
from archbench.adapters.redaction import redact_text
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

META_KEY = "LLM_1607358788850350_TestSecret0123456789abcdef"
OPENAI_KEY = "sk-proj-should-never-reach-meta-0123456789"
CONFIG_PATH = ROOT / "configs/models/meta/muse-spark-1.3.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)


def muse(text: str = "ok", **kwargs: Any) -> dict[str, Any]:
    return response(text, model=MODEL, **kwargs)


def error(status: int, code: str | None, message: str = "nope", kind: str = "invalid_request_error") -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": message, "type": kind, "param": None, "code": code}})


class MockMeta:
    def __init__(self, *, answer: str = "1. **Muse Spark**: fits.", reply: httpx2.Response | None = None,
                 card: dict[str, Any] | httpx2.Response | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer, self.reply = answer, reply
        self.card = card if card is not None else {"id": MODEL, "object": "model", "created": 1785000000,
                                                   "owned_by": "meta"}

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        assert request.url.host == "api.meta.ai", request.url
        if request.method == "GET":
            assert request.url.path == f"/v1/models/{MODEL}", request.url
            return self.card if isinstance(self.card, httpx2.Response) else httpx2.Response(200, json=self.card)
        assert request.url.path == "/v1/responses", request.url
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["input"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return httpx2.Response(200, json=muse(text))

    @property
    def calls(self) -> list[httpx2.Request]:
        return [r for r in self.requests if r.method == "POST"]

    def adapter(self, api_key: str | None = META_KEY) -> MetaAdapter:
        return MetaAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, mock: MockMeta, config: ModelConfiguration = CONFIG,
         api_key: str | None = META_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_stateless_and_never_carries_openai_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in (("OPENAI_API_KEY", OPENAI_KEY), ("OPENAI_ORG_ID", "org-leak"),
                        ("OPENAI_PROJECT_ID", "proj-leak")):
        monkeypatch.setenv(name, value)
    GUARD.register(META_KEY)
    mock = MockMeta(answer="Use Muse.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Muse."
    (request,) = mock.requests
    assert request.headers["authorization"] == f"Bearer {META_KEY}"
    assert "openai-organization" not in request.headers and "openai-project" not in request.headers
    assert json.loads(request.content) == {
        "model": MODEL, "instructions": "Be precise.", "input": [{"role": "user", "content": "Pick a model."}],
        "max_output_tokens": 16000, "store": False, "include": ["reasoning.encrypted_content"],
    }  # no reasoning effort, temperature, tools, or truncation
    for text in files.values():
        assert META_KEY not in text and OPENAI_KEY not in text
    assert outcome.observations["api_cost_usd"].value == pytest.approx((800 * 1.25 + 200 * 0.15 + 500 * 4.25) / 1e6)


def test_long_prompts_are_priced_without_a_long_context_guard(tmp_path: Path) -> None:
    usage = {"input_tokens": 500_000, "input_tokens_details": {"cached_tokens": 0},
             "output_tokens": 1000, "output_tokens_details": {"reasoning_tokens": 600}, "total_tokens": 501_000}
    outcome, _ = call(tmp_path, MockMeta(reply=httpx2.Response(200, json=muse(usage=usage))))
    assert outcome.observations["api_cost_usd"].value == pytest.approx((500_000 * 1.25 + 1000 * 4.25) / 1e6)


def test_missing_meta_key_fails_without_falling_back_to_openai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    mock = MockMeta()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert default_registry().get("meta").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("MODEL_API_KEY", META_KEY)
    assert default_registry().get("meta").preflight(CONFIG) is None  # type: ignore[attr-defined]


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (error(400, "content_policy_violation", "The request was blocked by our content policy."),
     FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (error(400, None, "This model's maximum context length is 1048576 tokens. However, you requested 1100000 "
                      "tokens (1084000 in the input, 16000 for the output)."), FailureKind.CONTEXT_EXCEEDED,
     CostIncurred.NO),
    (error(400, None, "reasoning.effort 'none' is not supported"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(401, "invalid_api_key", kind="authentication_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(402, None, kind="billing_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(404, "model_not_found"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(413, "payload_too_large"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429, "rate_limit_exceeded", kind="rate_limit_error"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(500, None, kind="server_error"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (error(503, "service_overloaded", kind="server_error"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (error(504, "gateway_timeout", kind="server_error"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_meta_errors_are_mapped_and_never_retried(tmp_path: Path, reply: httpx2.Response, kind: FailureKind,
                                                  cost: CostIncurred) -> None:
    mock = MockMeta(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


def test_refusal_parts_and_other_served_models_are_failures(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockMeta(reply=httpx2.Response(200, json=muse(refusal="I can't help with that."))))
    assert outcome.failure_kind is FailureKind.REFUSAL
    outcome, _ = call(tmp_path, MockMeta(reply=httpx2.Response(200, json=response(model="muse-spark-1.2"))))
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR


def test_verbosity_is_rejected_without_spending() -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["verbosity"] = {"state": "explicit", "value": "low"}
    mock = MockMeta()
    checks = {c.name: c for c in mock.adapter().qualify(ModelConfiguration.model_validate(data), REQUIREMENTS).checks}
    assert checks["parameter_support"].passed is False and "verbosity" in checks["parameter_support"].detail
    assert mock.calls == []


def test_qualification_checks_the_models_endpoint_and_replays_encrypted_reasoning() -> None:
    mock = MockMeta()
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "owned_by=meta" in checks["model_identity"].detail and "no dated snapshots" in checks["model_identity"].detail
    assert "including encrypted reasoning" in checks["continuation"].detail
    first = muse("baseline")["output"]
    for request in mock.calls[1:]:
        assert json.loads(request.content)["input"][1:3] == first  # reasoning and message items, exactly


@pytest.mark.parametrize(("model", "message"), [
    ("muse-spark-1.3-contributor", "Contributor tier"),
    ("llama-4-maverick", "not a Muse model"),
])
def test_qualification_rejects_contributor_and_non_muse_models_without_calls(model: str, message: str) -> None:
    mock = MockMeta()
    report = mock.adapter().qualify(CONFIG.model_copy(update={"model": model}), REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert checks["model_identity"].passed is False and message in checks["model_identity"].detail
    assert mock.requests == []


def test_qualification_fails_for_a_model_the_team_cannot_use() -> None:
    mock = MockMeta(card=error(404, "model_not_found"))
    checks = {c.name: c for c in mock.adapter().qualify(CONFIG, REQUIREMENTS).checks}
    assert checks["model_identity"].passed is False and mock.calls == []


def test_meta_keys_are_redacted_and_guarded() -> None:
    assert META_KEY not in redact_text(f"bad key {META_KEY}")
    GUARD.clear()
    assert GUARD.find(f"leak {META_KEY}") == "an API-key-shaped string"


def test_muse_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"meta": MockMeta().adapter(), "fake": FakeAdapter(), "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-muse", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["muse-spark-1.3@1"]
    assert row["own_vendor"] == "meta" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0
