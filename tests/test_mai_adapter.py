"""Microsoft MAI adapter (MAI-Thinking-1 on Foundry) against a mock transport: native request, served-model checks,
cached-input pricing, failure mapping (HTTP errors, content-safety blocks, finish reasons, tool calls reported as
"stop"), exact replay of encrypted reasoning, qualification by live probe, and an offline P6 run with an MAI
candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.mai_chat import MaiAdapter, _observations
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

API_KEY = "0123456789abcdef0123456789abcdef"
CONFIG_PATH = ROOT / "configs/models/microsoft/mai-thinking-1.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
DEPLOYMENT = CONFIG.model
URL = "https://archbench-test.services.ai.azure.com/mai/v1/chat/completions"
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)
NO_BRANCHES = REQUIREMENTS.model_copy(update={"requires_continuation": False})
USAGE = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
         "prompt_tokens_details": {"cached_tokens": 0}}
EXPECTED_COST = (1000 * 2.00 + 500 * 8.00) / 1e6
REASONING = {"encrypted_content": "gAAAAB-opaque-state"}


def completion(text: str | None = "ok", *, finish: str = "stop", model: str = "MAI-Thinking-1",
               usage: dict[str, Any] | None = USAGE, reasoning: dict[str, Any] | None = REASONING,
               **message: Any) -> dict[str, Any]:
    return {"id": "mai-api-1", "object": "chat.completion", "created": 1785277080, "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": {
                "role": "assistant", "content": text, "refusal": None, "reasoning": reasoning, **message}}],
            "usage": usage}


class MockMai:
    def __init__(self, *, answer: str = "1. **MAI-Thinking**: fits.", model: str = "MAI-Thinking-1",
                 reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer, self.model, self.reply = answer, model, reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        assert request.method == "POST" and str(request.url) == URL, request.url
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        reasoning = REASONING if body.get("reasoning_display") == "encrypted" else None
        return httpx2.Response(200, json=completion(text, model=self.model, reasoning=reasoning),
                               headers={"apim-request-id": "apim-1"})

    def adapter(self, api_key: str | None = API_KEY) -> MaiAdapter:
        return MaiAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def with_parameters(**parameters: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update(parameters)
    return ModelConfiguration.model_validate(data)


def with_options(**options: Any) -> ModelConfiguration:
    return CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options, **options}})


def call(tmp_path: Path, mock: MockMai, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockMai(answer="Use MAI.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use MAI."
    (request,) = mock.requests
    assert request.headers["api-key"] == API_KEY and "authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": DEPLOYMENT, "max_completion_tokens": 16000, "reasoning_display": "encrypted",
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
    }  # no max_tokens, sampling parameters, tools, or stream
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["api-key"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert "reasoning.txt" not in files  # encrypted reasoning is opaque
    assert outcome.observations["api_cost_usd"].value == pytest.approx(EXPECTED_COST)


def test_replay_reasoning_can_be_switched_off(tmp_path: Path) -> None:
    mock = MockMai()
    outcome, _ = call(tmp_path, mock, with_options(replay_reasoning=False))
    assert outcome.succeeded and "reasoning_display" not in json.loads(mock.requests[0].content)


def test_request_id_prefers_the_gateway_header(tmp_path: Path) -> None:
    adapter = MockMai().adapter()
    prepared = adapter.prepare(CONFIG, MESSAGES, None, ctx())
    assert adapter.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "a").sink("c")).request_id == "apim-1"
    adapter = MockMai(reply=httpx2.Response(200, json=completion())).adapter()
    assert adapter.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "b").sink("c")).request_id == "mai-api-1"


@pytest.mark.parametrize("served", ["MAI-Thinking-1", DEPLOYMENT])
def test_responses_may_name_the_model_or_the_deployment(tmp_path: Path, served: str) -> None:
    outcome, _ = call(tmp_path, MockMai(model=served))
    assert outcome.succeeded


def test_a_response_from_another_model_is_a_failure(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockMai(model="gpt-6-luna"))
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and not outcome.succeeded


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockMai(reply=httpx2.Response(200, json=completion("part", finish="length"))))
    assert outcome.succeeded and outcome.extracted.truncated


def test_cost_prices_cached_tokens_only_when_a_price_is_configured() -> None:
    cached = {**USAGE, "prompt_tokens_details": {"cached_tokens": 400}}
    obs = _observations(CONFIG, cached)
    assert obs["cached_input_tokens"].value == 400 and obs["api_cost_usd"].value is None
    assert "cached_input_usd_per_mtok" in str(obs["api_cost_usd"].missing_reason)
    priced = with_options(pricing={**CONFIG.adapter_options["pricing"], "cached_input_usd_per_mtok": 0.5})  # type: ignore[dict-item]
    assert _observations(priced, cached)["api_cost_usd"].value == pytest.approx((600 * 2 + 400 * 0.5 + 500 * 8) / 1e6)
    assert _observations(CONFIG, {})["api_cost_usd"].value is None


def error(status: int, body: Any = None) -> httpx2.Response:
    return httpx2.Response(status, json=body if body is not None else {"error": {"code": "x", "message": "nope"}})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (error(400, {"error": {"code": "invalid_parameter", "message": "Unsupported parameter: 'max_tokens'"}}),
     FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(400, {"error": {"type": "SafetyBlockedError", "message": "The request was blocked."}}),
     FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (error(400, {"error": {"code": "content_filter", "message": "The response was filtered."}}),
     FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (error(400, {"error": {"message": "This request exceeds the model's context window."}}),
     FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO),
    (error(401), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(403), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(404), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(422), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(500), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (error(502), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (error(503), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (httpx2.Response(504, text="gateway timeout"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockMai(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


@pytest.mark.parametrize(("body", "kind"), [
    (completion(finish="content filter"), FailureKind.REFUSAL),
    (completion(finish="content_filter"), FailureKind.REFUSAL),
    (completion(None, refusal="I can't help with that."), FailureKind.REFUSAL),
    # Microsoft reports tool calls with finish_reason "stop"; no tools are sent, so any tool call is a failure.
    (completion(None, tool_calls=[{"id": "call_1", "type": "function"}]), FailureKind.PROVIDER_ERROR),
    (completion(finish="tool_calls"), FailureKind.PROVIDER_ERROR),
    (completion(None), FailureKind.PROVIDER_ERROR),
    ({**completion(), "choices": []}, FailureKind.PROVIDER_ERROR),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind) -> None:
    outcome, _ = call(tmp_path, MockMai(reply=httpx2.Response(200, json=body)))
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, CostIncurred.YES)


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockMai()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    assert default_registry().get("mai").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("AZURE_API_KEY", API_KEY)
    assert default_registry().get("mai").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_the_message_with_encrypted_reasoning_exactly(tmp_path: Path) -> None:
    mock = MockMai()
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
        assert sent["messages"][2] == first  # content, refusal and encrypted reasoning, exactly as returned
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in mock.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different deployment"):
        adapter.prepare(CONFIG.model_copy(update={"model": "other-deployment"}), follow, snapshot, ctx(2))
    with pytest.raises(AdapterError, match="different deployment"):
        adapter.prepare(with_options(base_url="https://other.services.ai.azure.com"), follow, snapshot, ctx(2))


def test_qualification_probes_identity_and_branching() -> None:
    mock = MockMai()
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "'MAI-Thinking-1'" in checks["model_identity"].detail and "2026-06-01" in checks["model_identity"].detail
    assert "including encrypted reasoning" in checks["continuation"].detail
    assert [p["step"] for p in report.probes] == ["baseline", "branch-alpha", "branch-beta"]
    assert all(json.loads(r.content)["max_completion_tokens"] == 2048 for r in mock.requests)


def test_qualification_without_branches_makes_one_probe() -> None:
    mock = MockMai()
    report = mock.adapter().qualify(CONFIG, NO_BRANCHES)
    checks = {c.name: c for c in report.checks}
    assert report.passed and len(mock.requests) == 1 and "continuation" not in checks


def test_qualification_fails_on_another_served_model() -> None:
    report = MockMai(model="gpt-6-luna").adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert checks["model_identity"].passed is False and "gpt-6-luna" in checks["model_identity"].detail


def test_qualification_explains_the_zero_default_quota() -> None:
    report = MockMai(reply=error(429)).adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and "0 tokens per minute" in checks["continuation"].detail


def test_qualification_fails_when_encrypted_reasoning_is_missing() -> None:
    mock = MockMai(reply=httpx2.Response(200, json=completion("baseline", reasoning=None)))
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert "no reasoning.encrypted_content" in checks["continuation"].detail and len(mock.requests) == 1


def test_qualification_without_credentials_fails_without_calls() -> None:
    mock = MockMai()
    report = mock.adapter(api_key=None).qualify(CONFIG, REQUIREMENTS)
    assert not report.passed and mock.requests == []


@pytest.mark.parametrize(("parameters", "options", "message"), [
    ({"temperature": {"state": "explicit", "value": 0.5}}, {}, "temperature"),
    ({"reasoning_effort": {"state": "explicit", "value": "high"}}, {}, "reasoning_effort"),
    ({"max_output_tokens": {"state": "explicit", "value": 64001}}, {}, "1..64000"),
    ({}, {"max_output_tokens_limit": None}, "max_output_tokens_limit"),
    ({}, {"replay_reasoning": "yes"}, "replay_reasoning"),
])
def test_qualification_rejects_bad_parameters_without_spending(parameters: dict[str, Any], options: dict[str, Any],
                                                              message: str) -> None:
    mock = MockMai()
    config = with_options(**options) if options else CONFIG
    if parameters:
        data = config.model_dump(mode="json")
        data["parameters"].update(parameters)
        config = ModelConfiguration.model_validate(data)
    checks = {c.name: c for c in mock.adapter().qualify(config, REQUIREMENTS).checks}
    assert checks["parameter_support"].passed is False and message in checks["parameter_support"].detail
    assert mock.requests == []


@pytest.mark.parametrize("base_url", [
    "https://<your-foundry-resource>.services.ai.azure.com",  # the shipped placeholder
    "https://archbench-test.openai.azure.com", "http://archbench-test.services.ai.azure.com",
    "https://archbench-test.services.ai.azure.com/mai/v1", "https://evil.example.com", None,
])
def test_qualification_rejects_other_endpoints_without_spending(base_url: str | None) -> None:
    mock = MockMai()
    checks = {c.name: c for c in mock.adapter().qualify(with_options(base_url=base_url), REQUIREMENTS).checks}
    assert checks["endpoint"].passed is False and mock.requests == []


@pytest.mark.parametrize("served_model", ["Phi-4", "gpt-6-luna", None])
def test_qualification_rejects_non_mai_models_without_spending(served_model: str | None) -> None:
    mock = MockMai()
    checks = {c.name: c for c in mock.adapter().qualify(with_options(served_model=served_model), REQUIREMENTS).checks}
    assert checks["first_party_model"].passed is False and mock.requests == []


def test_mai_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"mai": MockMai().adapter(), "fake": FakeAdapter(), "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-mai", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["mai-thinking-1@1"]
    assert row["own_vendor"] == "microsoft" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0
