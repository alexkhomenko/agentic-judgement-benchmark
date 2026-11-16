"""Cohere Chat API v2 adapter against a mock transport: native request, billed-unit pricing, failure mapping
(HTTP and finish reasons), exact replay of thinking and text blocks, qualification against the first-party
model card, and an offline P6 run with a Cohere candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.cohere_chat import CohereAdapter, _observations
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

API_KEY = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcd"
CONFIG_PATH = ROOT / "configs/models/cohere/command-a-plus-05-2026.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)
NO_BRANCHES = REQUIREMENTS.model_copy(update={"requires_continuation": False})
USAGE = {"billed_units": {"input_tokens": 800, "output_tokens": 500},
         "tokens": {"input_tokens": 1000, "output_tokens": 500, "reasoning_tokens": 300}, "cached_tokens": 200}
EXPECTED_COST = (800 * 2.50 + 500 * 10.00) / 1e6
CARD = {"name": MODEL, "endpoints": ["chat"], "default_endpoints": [], "finetuned": False, "is_deprecated": False,
        "context_length": 128000, "features": ["reasoning", "safety_modes"],
        "sampling_defaults": {"temperature": 0.6, "p": 0.95}}


def chat(text: str = "ok", *, thinking: str | None = "pondering", finish: str = "COMPLETE",
         usage: dict[str, Any] | None = USAGE, blocks: list[dict[str, Any]] | None = None,
         **message: Any) -> dict[str, Any]:
    content = ([{"type": "thinking", "thinking": thinking}] if thinking else []) + [{"type": "text", "text": text}]
    return {"id": "gen-1", "finish_reason": finish,
            "message": {"role": "assistant", "content": content + (blocks or []), **message}, "usage": usage}


class MockCohere:
    def __init__(self, *, answer: str = "1. **Command A+**: fits.", card: dict[str, Any] | httpx2.Response = CARD,
                 reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer, self.card, self.reply = answer, card, reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.method == "GET":
            assert request.url.path == f"/v1/models/{MODEL}", request.url
            return self.card if isinstance(self.card, httpx2.Response) else httpx2.Response(200, json=self.card)
        assert request.url.path == "/v2/chat", request.url
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        last = json.loads(request.content)["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return httpx2.Response(200, json=chat(text))

    @property
    def chats(self) -> list[httpx2.Request]:
        return [r for r in self.requests if r.method == "POST"]

    def adapter(self, api_key: str | None = API_KEY) -> CohereAdapter:
        return CohereAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def with_parameters(**parameters: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update(parameters)
    return ModelConfiguration.model_validate(data)


def with_options(**options: Any) -> ModelConfiguration:
    return CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options, **options}})


def call(tmp_path: Path, mock: MockCohere, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockCohere(answer="Use Command A+.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Command A+."
    (request,) = mock.requests
    assert str(request.url) == "https://api.cohere.com/v2/chat"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "model": MODEL, "max_tokens": 16000,
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
    }  # no thinking, safety_mode, sampling parameters, documents, or tools
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert files["reasoning.txt"] == "pondering"
    obs = outcome.observations
    assert (obs["input_tokens"].value, obs["billed_input_tokens"].value, obs["cached_input_tokens"].value) == (
        1000, 800, 200)
    assert obs["reasoning_tokens"].value == 300
    assert obs["api_cost_usd"].value == pytest.approx(EXPECTED_COST)


def test_explicit_parameters_map_to_native_fields(tmp_path: Path) -> None:
    mock = MockCohere()
    call(tmp_path, mock, with_parameters(
        thinking={"state": "explicit", "value": "enabled"},
        thinking_token_budget={"state": "explicit", "value": 8000},
        safety_mode={"state": "explicit", "value": "STRICT"},
        temperature={"state": "explicit", "value": 0.6}, top_p={"state": "explicit", "value": 0.9},
        top_k={"state": "explicit", "value": 40}, frequency_penalty={"state": "explicit", "value": 0.1},
        presence_penalty={"state": "explicit", "value": 0.2}, seed={"state": "explicit", "value": 7}))
    body = json.loads(mock.requests[0].content)
    assert body["thinking"] == {"type": "enabled", "token_budget": 8000}
    assert {k: body[k] for k in ("safety_mode", "temperature", "p", "k", "frequency_penalty", "presence_penalty",
                                 "seed")} == {"safety_mode": "STRICT", "temperature": 0.6, "p": 0.9, "k": 40,
                                              "frequency_penalty": 0.1, "presence_penalty": 0.2, "seed": 7}
    assert "top_p" not in body and "top_k" not in body


def test_text_blocks_are_joined_and_truncation_is_extracted(tmp_path: Path) -> None:
    reply = chat("part one, ", finish="MAX_TOKENS", blocks=[{"type": "text", "text": "part two"}])
    outcome, _ = call(tmp_path, MockCohere(reply=httpx2.Response(200, json=reply)))
    assert outcome.succeeded and outcome.extracted.truncated
    assert outcome.extracted.final_answer == "part one, part two"


def test_cost_uses_billed_units_not_counted_tokens() -> None:
    assert _observations(CONFIG, USAGE)["api_cost_usd"].value == pytest.approx(EXPECTED_COST)
    obs = _observations(CONFIG, {"tokens": USAGE["tokens"]})
    assert obs["api_cost_usd"].value is None and obs["input_tokens"].value == 1000


def error(status: int, message: str = "nope") -> httpx2.Response:
    return httpx2.Response(status, json={"id": "err-1", "message": message})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (error(400), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(401), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(402), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(404), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(422), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(498), FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (error(499), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (error(500), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (error(503), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (httpx2.Response(504, text="gateway timeout"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockCohere(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


@pytest.mark.parametrize(("body", "kind", "cost"), [
    (chat(finish="TIMEOUT"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (chat(finish="ERROR"), FailureKind.PROVIDER_ERROR, CostIncurred.UNKNOWN),
    (chat(finish="TOOL_CALL"), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (chat(finish="STOP_SEQUENCE"), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (chat(blocks=[{"type": "image_url", "image_url": {}}]), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (chat(tool_calls=[{"id": "t1"}]), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    ({"id": "gen-1", "finish_reason": "COMPLETE", "usage": USAGE}, FailureKind.PROVIDER_ERROR, CostIncurred.YES),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind,
                                         cost: CostIncurred) -> None:
    outcome, _ = call(tmp_path, MockCohere(reply=httpx2.Response(200, json=body)))
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockCohere()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("CO_API_KEY", raising=False)
    assert default_registry().get("cohere").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("CO_API_KEY", API_KEY)
    assert default_registry().get("cohere").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_the_message_exactly(tmp_path: Path) -> None:
    mock = MockCohere()
    adapter = mock.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    first = chat(mock.answer)["message"]
    for request, word in zip(mock.requests[1:], ("alpha", "beta")):
        sent = json.loads(request.content)
        assert sent["messages"][2] == first  # thinking and text blocks, exactly as returned
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in mock.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "command-a-03-2025"}), follow, snapshot, ctx(2))


def test_qualification_checks_the_model_card_and_branching() -> None:
    mock = MockCohere()
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "context_length=128000" in checks["model_identity"].detail
    assert "sampling_defaults" in checks["model_identity"].detail
    assert "including thinking blocks" in checks["continuation"].detail
    assert [p["step"] for p in report.probes] == ["model_card", "baseline", "branch-alpha", "branch-beta"]
    assert all(json.loads(r.content)["max_tokens"] == 2048 for r in mock.chats)


def test_qualification_without_branches_makes_one_chat_probe() -> None:
    mock = MockCohere()
    report = mock.adapter().qualify(CONFIG, NO_BRANCHES)
    checks = {c.name: c for c in report.checks}
    assert report.passed and len(mock.chats) == 1 and "continuation" not in checks
    assert "with thinking blocks" in checks["chat_probe"].detail


def test_probe_thinking_budget_fits_the_probe_output_budget() -> None:
    mock = MockCohere()
    config = with_parameters(thinking={"state": "explicit", "value": "enabled"},
                             thinking_token_budget={"state": "explicit", "value": 12000})
    assert mock.adapter().qualify(config, REQUIREMENTS).passed
    assert all(json.loads(r.content)["thinking"] == {"type": "enabled", "token_budget": 1024} for r in mock.chats)


@pytest.mark.parametrize(("card", "message"), [
    ({**CARD, "name": "command-a-03-2025"}, "model card names"),
    ({**CARD, "endpoints": ["embed"]}, "does not serve chat"),
    ({**CARD, "is_deprecated": True}, "deprecated"),
    ({**CARD, "context_length": 64000}, "context_limit 128000 > context_length 64000"),
    (error(404, "model not found"), "HTTP 404"),
])
def test_qualification_rejects_a_bad_model_card_without_chat_calls(card: Any, message: str) -> None:
    mock = MockCohere(card=card)
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert checks["model_identity"].passed is False and message in checks["model_identity"].detail
    assert checks["continuation"].passed is None and mock.chats == []


def test_qualification_without_credentials_is_not_evaluated() -> None:
    mock = MockCohere()
    report = mock.adapter(api_key=None).qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and checks["model_identity"].passed is None and mock.requests == []


@pytest.mark.parametrize(("parameters", "options", "message"), [
    ({"reasoning_effort": {"state": "explicit", "value": "high"}}, {}, "reasoning_effort"),
    ({"max_output_tokens": {"state": "explicit", "value": 64001}}, {}, "1..64000"),
    ({}, {"max_output_tokens_limit": None}, "max_output_tokens_limit"),
    ({"thinking": {"state": "explicit", "value": "on"}}, {}, "'enabled' or 'disabled'"),
    ({"thinking_token_budget": {"state": "explicit", "value": 4000}}, {}, "requires thinking: enabled"),
    ({"thinking": {"state": "explicit", "value": "enabled"},
      "thinking_token_budget": {"state": "explicit", "value": 16000}}, {}, "below max_output_tokens"),
])
def test_qualification_rejects_bad_parameters_without_spending(parameters: dict[str, Any], options: dict[str, Any],
                                                              message: str) -> None:
    mock = MockCohere()
    config = with_options(**options) if options else CONFIG
    if parameters:
        data = config.model_dump(mode="json")
        data["parameters"].update(parameters)
        config = ModelConfiguration.model_validate(data)
    checks = {c.name: c for c in mock.adapter().qualify(config, REQUIREMENTS).checks}
    assert checks["parameter_support"].passed is False and message in checks["parameter_support"].detail
    assert mock.requests == []


@pytest.mark.parametrize("base_url", [
    "https://api.cohere.ai/compatibility/v1", "https://bedrock-runtime.us-east-1.amazonaws.com",
    "http://api.cohere.com",
])
def test_qualification_rejects_other_endpoints_without_spending(base_url: str) -> None:
    mock = MockCohere()
    checks = {c.name: c for c in mock.adapter().qualify(with_options(base_url=base_url), REQUIREMENTS).checks}
    assert checks["endpoint"].passed is False and mock.requests == []


def test_cohere_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"cohere": MockCohere().adapter(), "fake": FakeAdapter(),
                                "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-cohere", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["command-a-plus-05-2026@1"]
    assert row["own_vendor"] == "cohere" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0


def test_trace_header_is_the_request_id_with_the_response_id_as_fallback(tmp_path: Path) -> None:
    adapter = MockCohere(reply=httpx2.Response(200, json=chat(), headers={"x-debug-trace-id": "trace-1"})).adapter()
    prepared = adapter.prepare(CONFIG, MESSAGES, None, ctx())
    assert adapter.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "a").sink("c")).request_id == "trace-1"
    adapter = MockCohere().adapter()
    assert adapter.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "b").sink("c")).request_id == "gen-1"
