"""Microsoft MAI chat-completions adapter against a mock transport shaped like live responses (2026-11-18): native
request on the deployment, case-insensitive model echo, encrypted-reasoning replay, cost with unpriced cached tokens,
failure mapping, qualification by identity probe (no /models route), and an offline P6 run next to another vendor."""

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

API_KEY = "foundry-test-key-0123456789abcdefghijklmnop"
CONFIG_PATH = ROOT / "configs/models/microsoft/mai-thinking-1.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
ENDPOINT = "https://example-resource.services.ai.azure.com/mai/v1/chat/completions"
BLOB = "o2d2ZXJzaW9uAW5lbmNyeXB0ZWRfZGVrc4GiZmtla19pZHhAYjU4ZGRl"
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)


def with_options(**options: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["adapter_options"].update(options)
    return ModelConfiguration.model_validate(data)


def completion(text: str = "ok", *, finish: str = "stop", model: str = "mai-thinking-1", blob: str | None = BLOB,
               cached: int = 2, reasoning_tokens: int = 83, refusal: str | None = None) -> dict[str, Any]:
    reasoning = {"encrypted_content": blob, "content": None, "summary": None} if blob else None
    return {
        "id": "mai-api-7ae43dd1", "object": "chat.completion", "created": 1_795_006_225, "model": model,
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": text, "tool_calls": [], "refusal": refusal,
                                 "thinking": None, "reasoning": reasoning, "encrypted_content": None,
                                 "annotations": None}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
                  "prompt_tokens_details": {"cached_tokens": cached},
                  "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
        "prompt_filter_results": None, "debug": None,
    }


class MockMai:
    def __init__(self, *, answer: str = "1. **MAI-Thinking-1**: fits.", served: str = "mai-thinking-1",
                 reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer, self.served, self.reply = answer, served, reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path != "/mai/v1/chat/completions":
            return httpx2.Response(404, json={"error": {"code": "404", "message": "Resource not found"}})
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        blob = BLOB if body.get("reasoning_display") == "encrypted" else None
        return httpx2.Response(200, json=completion(text, model=self.served, blob=blob),
                               headers={"apim-request-id": "apim-1"})

    def adapter(self, api_key: str | None = API_KEY) -> MaiAdapter:
        return MaiAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


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
    assert str(request.url) == ENDPOINT
    assert request.headers["api-key"] == API_KEY and "authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": "mai-thinking-1", "max_completion_tokens": 16000, "reasoning_display": "encrypted",
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
    }  # no temperature, top_p, effort, or tools
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["api-key"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert outcome.observations["reasoning_tokens"].value == 83
    assert outcome.extracted.reasoning_text is None  # encrypted only


def test_reasoning_display_is_sent_only_when_replaying(tmp_path: Path) -> None:
    mock = MockMai()
    call(tmp_path, mock, with_options(replay_reasoning=False))
    assert "reasoning_display" not in json.loads(mock.requests[0].content)


@pytest.mark.parametrize("served", ["mai-thinking-1", "MAI-Thinking-1"])
def test_model_echo_is_compared_case_insensitively(tmp_path: Path, served: str) -> None:
    outcome, _ = call(tmp_path, MockMai(served=served))
    assert outcome.succeeded


def test_cost_with_and_without_a_cached_input_price() -> None:
    usage = completion()["usage"]
    unknown = _observations(CONFIG, usage)
    assert unknown["api_cost_usd"].value is None and "2 cached" in str(unknown["api_cost_usd"].missing_reason)
    assert unknown["api_cost_usd_if_uncached"].value == pytest.approx((1000 * 2.0 + 500 * 8.0) / 1e6)
    assert unknown["uncached_input_tokens"].value == 998
    no_cache = _observations(CONFIG, completion(cached=0)["usage"])
    assert no_cache["api_cost_usd"].value == pytest.approx(0.006) and "api_cost_usd_if_uncached" not in no_cache
    pricing = {**CONFIG.adapter_options["pricing"], "cached_input_usd_per_mtok": 0.2}  # type: ignore[dict-item]
    priced = _observations(with_options(pricing=pricing), usage)
    assert priced["api_cost_usd"].value == pytest.approx((998 * 2.0 + 2 * 0.2 + 500 * 8.0) / 1e6)


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (httpx2.Response(429, json={"error": {"code": "429", "message": "Rate limit"}}), FailureKind.RATE_LIMITED,
     CostIncurred.NO),
    (httpx2.Response(503, json={"error": {"message": "Service unavailable"}}), FailureKind.TRANSIENT_TRANSPORT,
     CostIncurred.NO),
    (httpx2.Response(401, json={"error": {"code": "401", "message": "Access denied"}}), FailureKind.PROVIDER_ERROR,
     CostIncurred.NO),
    (httpx2.Response(422, json={"error": {"message": "Request validation failed for field(s): reasoning_effort.",
                                          "type": "invalid_request_error", "code": "request_validation_error"}}),
     FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (httpx2.Response(400, json={"error": {"message": "Invalid request"}}), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (httpx2.Response(400, json={"error": {"code": "content_filter", "message": "The prompt was filtered"}}),
     FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (httpx2.Response(400, json={"error": {"message": "This model's maximum context length is 256000 tokens"}}),
     FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO),
    (httpx2.Response(504, json={"error": {"message": "Gateway timeout"}}), FailureKind.AMBIGUOUS_TIMEOUT,
     CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"),
     FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockMai(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


@pytest.mark.parametrize(("body", "kind"), [
    (completion(finish="content_filter"), FailureKind.REFUSAL),
    (completion(refusal="I can't help with that."), FailureKind.REFUSAL),
    (completion(finish="tool_calls"), FailureKind.PROVIDER_ERROR),
    (completion(model="mai-code-2-flash"), FailureKind.PROVIDER_ERROR),
    (completion(blob=None), FailureKind.PROVIDER_ERROR),  # reasoned, but nothing to replay
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind) -> None:
    outcome, _ = call(tmp_path, MockMai(reply=httpx2.Response(200, json=body)))
    assert outcome.failure_kind is kind and outcome.cost_incurred is CostIncurred.YES


def test_truncation_is_extracted(tmp_path: Path) -> None:
    body = completion("", finish="length", blob=None, reasoning_tokens=0)  # observed with a 5-token budget
    outcome, _ = call(tmp_path, MockMai(reply=httpx2.Response(200, json=body)))
    assert outcome.succeeded and outcome.extracted.truncated


def test_missing_credentials_or_endpoint_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockMai()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    placeholder = with_options(base_url="https://<your-foundry-resource>.services.ai.azure.com")
    outcome, _ = call(tmp_path, mock, placeholder)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    assert mock.adapter().preflight(placeholder) is not None
    monkeypatch.delenv("AZURE_INFERENCE_CREDENTIAL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", API_KEY)  # a different service's key is never used
    assert default_registry().get("mai").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("AZURE_INFERENCE_CREDENTIAL", API_KEY)
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
        assert sent["messages"][2] == first  # content and reasoning.encrypted_content, exactly as returned
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in mock.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "other-deployment"}), follow, snapshot, ctx(2))


def test_qualification_verifies_identity_with_one_probe() -> None:
    mock = MockMai()
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "serves mai-thinking-1" in checks["model_identity"].detail
    assert "2026-06-01 is declared" in checks["model_identity"].detail
    assert "encrypted reasoning" in checks["continuation"].detail
    identity = json.loads(mock.requests[0].content)
    assert identity["max_completion_tokens"] == 16 and "reasoning_display" not in identity
    assert [p["step"] for p in report.probes] == ["identity", "baseline", "branch-alpha", "branch-beta"]


@pytest.mark.parametrize(("config", "mock", "expected"), [
    (CONFIG, MockMai(served="mai-code-2-flash"), "serves mai-code-2-flash"),
    (CONFIG, MockMai(reply=httpx2.Response(404, json={"error": {"code": "DeploymentNotFound",
                                                                 "message": "The deployment does not exist"}})),
     "HTTP 404"),
    (with_options(served_model=None), MockMai(), "served_model is required"),
    (with_options(max_output_tokens_limit=8000), MockMai(), "max_output_tokens 16000 > model max_output_tokens 8000"),
])
def test_qualification_rejects_identity_problems(config: ModelConfiguration, mock: MockMai, expected: str) -> None:
    checks = {c.name: c for c in mock.adapter().qualify(
        config, REQUIREMENTS.model_copy(update={"requires_continuation": False})).checks}
    assert checks["model_identity"].passed is False and expected in checks["model_identity"].detail


def test_undocumented_parameters_are_rejected_without_probing() -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["reasoning_effort"] = {"state": "explicit", "value": "high"}
    mock = MockMai()
    report = mock.adapter().qualify(ModelConfiguration.model_validate(data), REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and "reasoning_effort" in checks["parameter_support"].detail
    assert checks["continuation"].passed is None
    assert len(mock.requests) == 1  # the identity probe only


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
