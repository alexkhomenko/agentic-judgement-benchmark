"""Amazon Bedrock Converse adapter against a mock transport: native request, Region-derived endpoints, cache-aware
pricing, failure mapping (HTTP errors, Nova content filters, stop reasons), exact replay of reasoning blocks,
qualification against the control plane (inference profile and foundation model, Amazon models only), and an
offline P6 run with a Nova candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.bedrock_converse import BedrockAdapter, _observations
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

API_KEY = "ABSKQmVkcm9ja0FQSUtleS1leGFtcGxlLWtleS1mb3ItdGVzdHM="
CONFIG_PATH = ROOT / "configs/models/amazon/nova-2-lite-us.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
PROFILE = CONFIG.model
BASE = "amazon.nova-2-lite-v1:0"
RUNTIME = "https://bedrock-runtime.us-east-1.amazonaws.com"
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)
NO_BRANCHES = REQUIREMENTS.model_copy(update={"requires_continuation": False})
USAGE = {"inputTokens": 1000, "outputTokens": 500, "cacheReadInputTokens": 200, "totalTokens": 1700}
EXPECTED_COST = (1000 * 0.33 + 200 * 0.0825 + 500 * 2.75) / 1e6
REASONING = {"reasoningContent": {"reasoningText": {"text": "[REDACTED]"}}}
PROFILE_CARD = {
    "inferenceProfileId": PROFILE, "status": "ACTIVE", "type": "SYSTEM_DEFINED",
    "models": [{"modelArn": f"arn:aws:bedrock:{r}::foundation-model/{BASE}"}
               for r in ("us-east-1", "us-east-2", "us-west-2")],
}
MODEL_CARD = {"modelDetails": {
    "modelId": BASE, "modelName": "Nova 2 Lite", "providerName": "Amazon", "outputModalities": ["TEXT"],
    "inferenceTypesSupported": ["INFERENCE_PROFILE"], "modelLifecycle": {"status": "ACTIVE"},
}}


def converse(text: str = "ok", *, reasoning: bool = False, stop: str = "end_turn",
             usage: dict[str, Any] | None = USAGE, blocks: list[dict[str, Any]] | None = None,
             **extra: Any) -> dict[str, Any]:
    content = ([REASONING] if reasoning else []) + [{"text": text}] + (blocks or [])
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": stop,
            "usage": usage, "metrics": {"latencyMs": 1234}, **extra}


class MockBedrock:
    def __init__(self, *, answer: str = "1. **Amazon Nova**: fits.", reasoning: bool = False,
                 profile: dict[str, Any] | httpx2.Response = PROFILE_CARD,
                 model: dict[str, Any] | httpx2.Response = MODEL_CARD,
                 reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer, self.reasoning, self.profile, self.model, self.reply = answer, reasoning, profile, model, reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.method == "GET":
            assert request.url.host == "bedrock.us-east-1.amazonaws.com", request.url
            path = request.url.path
            card = self.profile if path == f"/inference-profiles/{PROFILE}" else self.model
            assert path in (f"/inference-profiles/{PROFILE}", f"/foundation-models/{BASE}"), path
            return card if isinstance(card, httpx2.Response) else httpx2.Response(200, json=card)
        assert str(request.url) == f"{RUNTIME}/model/{PROFILE}/converse", request.url
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        last = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return httpx2.Response(200, json=converse(text, reasoning=self.reasoning),
                               headers={"x-amzn-requestid": "req-1"})

    @property
    def chats(self) -> list[httpx2.Request]:
        return [r for r in self.requests if r.method == "POST"]

    def adapter(self, api_key: str | None = API_KEY) -> BedrockAdapter:
        return BedrockAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def with_parameters(**parameters: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update(parameters)
    return ModelConfiguration.model_validate(data)


def with_options(**options: Any) -> ModelConfiguration:
    return CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options, **options}})


def call(tmp_path: Path, mock: MockBedrock, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockBedrock(answer="Use Nova.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Nova."
    (request,) = mock.requests
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "system": [{"text": "Be precise."}],
        "messages": [{"role": "user", "content": [{"text": "Pick a model."}]}],
        "inferenceConfig": {"maxTokens": 16000},
    }  # no sampling parameters, reasoningConfig, guardrail, tools, service tier, or model id in the body
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    adapter = mock.adapter()
    prepared = adapter.prepare(CONFIG, MESSAGES, None, ctx())
    assert adapter.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "b").sink("c")).request_id == "req-1"
    obs = outcome.observations
    assert (obs["input_tokens"].value, obs["cached_input_tokens"].value, obs["cache_write_input_tokens"].value) == (
        1000, 200, 0)
    assert obs["provider_latency_ms"].value == 1234
    assert obs["api_cost_usd"].value == pytest.approx(EXPECTED_COST)


def test_explicit_parameters_map_to_native_fields(tmp_path: Path) -> None:
    mock = MockBedrock()
    call(tmp_path, mock, with_parameters(
        temperature={"state": "explicit", "value": 0.3}, top_p={"state": "explicit", "value": 0.8},
        top_k={"state": "explicit", "value": 40}, reasoning={"state": "explicit", "value": "enabled"},
        reasoning_effort={"state": "explicit", "value": "medium"}))
    body = json.loads(mock.requests[0].content)
    assert body["inferenceConfig"] == {"maxTokens": 16000, "temperature": 0.3, "topP": 0.8}
    assert body["additionalModelRequestFields"] == {
        "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}, "inferenceConfig": {"topK": 40}}


def test_high_reasoning_effort_sends_no_inference_config(tmp_path: Path) -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["max_output_tokens"] = {"state": "omitted", "source": "AWS requires it unset at high effort"}
    data["parameters"]["reasoning"] = {"state": "explicit", "value": "enabled"}
    data["parameters"]["reasoning_effort"] = {"state": "explicit", "value": "high"}
    mock = MockBedrock()
    outcome, _ = call(tmp_path, mock, ModelConfiguration.model_validate(data))
    body = json.loads(mock.requests[0].content)
    assert outcome.succeeded and "inferenceConfig" not in body
    assert body["additionalModelRequestFields"] == {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}}


def _high_effort(**options: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["max_output_tokens"] = {"state": "omitted", "source": "AWS requires it unset at high effort"}
    data["parameters"]["reasoning"] = {"state": "explicit", "value": "enabled"}
    data["parameters"]["reasoning_effort"] = {"state": "explicit", "value": "high"}
    data["adapter_options"] = {**data["adapter_options"], **options}
    return ModelConfiguration.model_validate(data)


def test_high_reasoning_effort_qualifies_reserving_the_model_maximum_output() -> None:
    mock = MockBedrock()
    report = mock.adapter().qualify(_high_effort(), REQUIREMENTS.model_copy(update={"output_budget_tokens": None}))
    checks = {c.name: c for c in report.checks}
    assert report.passed and len(mock.chats) == 3, checks
    assert "reserves max_output_tokens_limit (64000)" in checks["context_envelope"].detail
    assert all("inferenceConfig" not in json.loads(r.content) for r in mock.chats)


def test_high_reasoning_effort_fails_the_envelope_when_the_reserved_output_does_not_fit() -> None:
    report = MockBedrock().adapter().qualify(_high_effort(max_output_tokens_limit=600_000),
                                             REQUIREMENTS.model_copy(update={"output_budget_tokens": None}))
    assert {c.name: c for c in report.checks}["context_envelope"].passed is False


def test_high_reasoning_effort_still_requires_the_output_limit() -> None:
    report = MockBedrock().adapter().qualify(_high_effort(max_output_tokens_limit=None),
                                             REQUIREMENTS.model_copy(update={"output_budget_tokens": None}))
    assert "max_output_tokens_limit" in {c.name: c for c in report.checks}["parameter_support"].detail


def test_text_blocks_are_joined_redacted_reasoning_is_dropped_and_truncation_is_extracted(tmp_path: Path) -> None:
    reply = converse("part one, ", reasoning=True, stop="max_tokens", blocks=[{"text": "part two"}])
    outcome, files = call(tmp_path, MockBedrock(reply=httpx2.Response(200, json=reply)))
    assert outcome.succeeded and outcome.extracted.truncated
    assert outcome.extracted.final_answer == "part one, part two"
    assert outcome.extracted.reasoning_text is None and "reasoning.txt" not in files


def test_visible_reasoning_text_is_extracted(tmp_path: Path) -> None:
    thought = {"reasoningContent": {"reasoningText": {"text": "pondering", "signature": "sig"}}}
    reply = converse("answer")
    reply["output"]["message"]["content"].insert(0, thought)
    outcome, _ = call(tmp_path, MockBedrock(reply=httpx2.Response(200, json=reply)))
    assert outcome.extracted.reasoning_text == "pondering"


def test_cost_counts_cache_reads_and_writes_separately_and_checks_the_total() -> None:
    assert _observations(CONFIG, {"usage": USAGE})["api_cost_usd"].value == pytest.approx(EXPECTED_COST)
    no_cache = {"inputTokens": 1000, "outputTokens": 500, "totalTokens": 1500}
    assert _observations(CONFIG, {"usage": no_cache})["api_cost_usd"].value == pytest.approx(
        (1000 * 0.33 + 500 * 2.75) / 1e6)
    # If inputTokens already included the cache read, the total would not add up: never guess the cost.
    inclusive = {**USAGE, "totalTokens": 1500}
    assert _observations(CONFIG, {"usage": inclusive})["api_cost_usd"].value is None
    obs = _observations(CONFIG, {})
    assert obs["api_cost_usd"].value is None and obs["cached_input_tokens"].value is None


def error(status: int, error_type: str | None = None, message: str = "nope") -> httpx2.Response:
    headers = {"x-amzn-errortype": f"{error_type}:http://internal.amazon.com/coral/com.amazon.bedrock/"} \
        if error_type else {}
    return httpx2.Response(status, json={"message": message}, headers={**headers, "x-amzn-requestid": "err-1"})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (error(400, "ValidationException"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(400, "ValidationException", "Input is too long for requested model."),
     FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO),
    (error(400, "ValidationException", "This request has been blocked by our content filters. Please adjust your "
                                       "text prompt to submit a new request."), FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (error(400, "ThrottlingException"), FailureKind.RATE_LIMITED, CostIncurred.NO),  # CommonErrors says 400
    (error(403, "AccessDeniedException"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(403, "ExpiredTokenException"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(404, "ResourceNotFoundException"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(408, "ModelTimeoutException"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (error(413), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(424, "ModelErrorException"), FailureKind.PROVIDER_ERROR, CostIncurred.UNKNOWN),
    (error(429, "ThrottlingException"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(429, "ModelNotReadyException"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (error(500, "InternalServerException"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (error(503, "ServiceUnavailableException"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (error(529), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (httpx2.Response(504, text="gateway timeout"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockBedrock(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


@pytest.mark.parametrize(("body", "kind"), [
    (converse(stop="content_filtered"), FailureKind.REFUSAL),
    (converse(stop="guardrail_intervened"), FailureKind.REFUSAL),
    ({**converse(), "error": "The generated text has been blocked by our content filters."}, FailureKind.REFUSAL),
    (converse(stop="model_context_window_exceeded"), FailureKind.CONTEXT_EXCEEDED),
    (converse(stop="tool_use"), FailureKind.PROVIDER_ERROR),
    (converse(stop="stop_sequence"), FailureKind.PROVIDER_ERROR),
    (converse(stop="malformed_model_output"), FailureKind.PROVIDER_ERROR),
    (converse(blocks=[{"toolUse": {"toolUseId": "t1"}}]), FailureKind.PROVIDER_ERROR),
    ({"stopReason": "end_turn", "usage": USAGE}, FailureKind.PROVIDER_ERROR),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind) -> None:
    outcome, _ = call(tmp_path, MockBedrock(reply=httpx2.Response(200, json=body)))
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, CostIncurred.YES)


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockBedrock()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    assert default_registry().get("bedrock").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", API_KEY)
    assert default_registry().get("bedrock").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_the_message_exactly(tmp_path: Path) -> None:
    mock = MockBedrock(reasoning=True)
    adapter = mock.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    first = converse(mock.answer, reasoning=True)["output"]["message"]
    for request, word in zip(mock.requests[1:], ("alpha", "beta")):
        sent = json.loads(request.content)
        assert sent["system"] == [{"text": "Be precise."}]
        assert sent["messages"][1] == first  # reasoning and text blocks, exactly as returned
        assert sent["messages"][2] == {"role": "user", "content": [{"text": f"Follow-up {word}"}]}
    assert "alpha" not in mock.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "us.amazon.nova-pro-v1:0"}), follow, snapshot, ctx(2))
    with pytest.raises(AdapterError, match="system messages"):
        adapter.prepare(CONFIG, snapshot.messages + (Message(role=Role.SYSTEM, content="x"),), snapshot, ctx(2))


def test_qualification_checks_the_profile_the_model_and_branching() -> None:
    mock = MockBedrock(reasoning=True)
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "['us-east-1', 'us-east-2', 'us-west-2']" in checks["model_identity"].detail
    assert "including reasoning blocks" in checks["continuation"].detail
    assert [p["step"] for p in report.probes] == ["inference_profile", "model_lookup", "foundation_model",
                                                  "baseline", "branch-alpha", "branch-beta"]
    assert all(json.loads(r.content)["inferenceConfig"]["maxTokens"] == 2048 for r in mock.chats)


def test_qualification_without_branches_makes_one_chat_probe() -> None:
    mock = MockBedrock()
    report = mock.adapter().qualify(CONFIG, NO_BRANCHES)
    checks = {c.name: c for c in report.checks}
    assert report.passed and len(mock.chats) == 1 and "continuation" not in checks


def test_an_in_region_model_id_needs_on_demand_support() -> None:
    mock = MockBedrock()
    report = mock.adapter().qualify(CONFIG.model_copy(update={"model": BASE}), REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert "does not support ON_DEMAND" in checks["model_identity"].detail and mock.chats == []


@pytest.mark.parametrize("types", [["ON_DEMAND"], ["ON_DEMAND", "PROVISIONED"], []])
def test_a_profile_id_does_not_depend_on_the_undocumented_inference_type(types: list[str]) -> None:
    mock = MockBedrock(model=_model(inferenceTypesSupported=types))
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed and checks["model_identity"].passed is True and len(mock.chats) == 3
    assert f"inference_types={types}" in checks["model_identity"].detail


def _model(**details: Any) -> dict[str, Any]:
    return {"modelDetails": {**MODEL_CARD["modelDetails"], **details}}


@pytest.mark.parametrize(("profile", "model", "message"), [
    ({**PROFILE_CARD, "status": "INACTIVE"}, MODEL_CARD, "inference profile status"),
    ({**PROFILE_CARD, "models": [{"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"}]},
     MODEL_CARD, "routes to"),
    (PROFILE_CARD, _model(providerName="Anthropic"), "not Amazon"),
    (PROFILE_CARD, _model(modelLifecycle={"status": "LEGACY"}), "lifecycle is 'LEGACY'"),
    (PROFILE_CARD, _model(outputModalities=["IMAGE"]), "does not output text"),
    (PROFILE_CARD, _model(modelId="amazon.nova-lite-v1:0"), "control plane names"),
    (error(404, "ResourceNotFoundException", "profile not found"), MODEL_CARD, "HTTP 404"),
    (PROFILE_CARD, error(403, "AccessDeniedException", "not allowed"), "HTTP 403"),
])
def test_qualification_rejects_a_bad_model_without_chat_calls(profile: Any, model: Any, message: str) -> None:
    mock = MockBedrock(profile=profile, model=model)
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert checks["model_identity"].passed is False and message in checks["model_identity"].detail
    assert checks["continuation"].passed is None and mock.chats == []


def test_qualification_without_credentials_is_not_evaluated() -> None:
    mock = MockBedrock()
    report = mock.adapter(api_key=None).qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and checks["model_identity"].passed is None and mock.requests == []


@pytest.mark.parametrize("model", [
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "meta.llama4-maverick-17b-instruct-v1:0",
    "global.mistral.mistral-large-3-v1:0",
])
def test_qualification_rejects_other_vendors_models_without_spending(model: str) -> None:
    mock = MockBedrock()
    checks = {c.name: c for c in mock.adapter().qualify(CONFIG.model_copy(update={"model": model}),
                                                        REQUIREMENTS).checks}
    assert checks["first_party_model"].passed is False and mock.requests == []


@pytest.mark.parametrize(("parameters", "options", "message"), [
    ({"thinking": {"state": "explicit", "value": "enabled"}}, {}, "thinking"),
    ({"max_output_tokens": {"state": "explicit", "value": 64001}}, {}, "1..64000"),
    ({}, {"max_output_tokens_limit": None}, "max_output_tokens_limit"),
    ({"reasoning": {"state": "explicit", "value": "on"}}, {}, "'enabled' or 'disabled'"),
    ({"reasoning": {"state": "explicit", "value": "enabled"}}, {}, "requires reasoning_effort"),
    ({"reasoning_effort": {"state": "explicit", "value": "low"}}, {}, "requires reasoning: enabled"),
    ({"reasoning": {"state": "explicit", "value": "enabled"},
      "reasoning_effort": {"state": "explicit", "value": "max"}}, {}, "'low', 'medium' or 'high'"),
    ({"reasoning": {"state": "explicit", "value": "enabled"},
      "reasoning_effort": {"state": "explicit", "value": "high"}}, {}, "['max_output_tokens']"),
])
def test_qualification_rejects_bad_parameters_without_spending(parameters: dict[str, Any], options: dict[str, Any],
                                                              message: str) -> None:
    mock = MockBedrock()
    config = with_options(**options) if options else CONFIG
    if parameters:
        data = config.model_dump(mode="json")
        data["parameters"].update(parameters)
        config = ModelConfiguration.model_validate(data)
    checks = {c.name: c for c in mock.adapter().qualify(config, REQUIREMENTS).checks}
    assert checks["parameter_support"].passed is False and message in checks["parameter_support"].detail
    assert mock.requests == []


@pytest.mark.parametrize("region", ["us-east-2", "us-west-1", None, "us-east-1.example.com"])
def test_qualification_rejects_regions_without_api_key_support(region: str | None) -> None:
    mock = MockBedrock()
    checks = {c.name: c for c in mock.adapter().qualify(with_options(region=region), REQUIREMENTS).checks}
    assert checks["endpoint"].passed is False and mock.requests == []


def test_bedrock_api_keys_are_redacted_from_text() -> None:
    short_term = "bedrock-api-key-YmVkcm9jay5hbWF6b25hd3MuY29tLz9BY3Rpb249Q2FsbFdpdGhCZWFyZXJUb2tlbg=="
    for key in (API_KEY, short_term):
        assert key not in redact_text(f"error: invalid key {key}")
        assert GUARD.find(f"leak {key}") is not None


def test_nova_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"bedrock": MockBedrock().adapter(), "fake": FakeAdapter(),
                                "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-nova", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["nova-2-lite-us@1"]
    assert row["own_vendor"] == "aws" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0
