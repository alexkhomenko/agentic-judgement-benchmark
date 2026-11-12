"""OpenAI Responses adapter against a mock transport: wire capture, redaction, no hidden retries, failure
mapping, cost, stateless continuation with encrypted reasoning, qualification, and an offline P6 run with
an OpenAI candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.openai_responses import OpenAIAdapter
from archbench.adapters.fake import FakeAdapter
from archbench.adapters.fake_judge import FakeJudgeAdapter
from archbench.adapters.registry import AdapterRegistry
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
from archbench.storage.artifacts import ArtifactStore

from .conftest import ROOT, Env
from .test_bias import _read_csv, bias_policy, recommender

API_KEY = "sk-proj-test-0123456789abcdefghij"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load((ROOT / "configs/models/openai/gpt-6-luna.yaml").read_text()))
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
Handler = Callable[[httpx2.Request], httpx2.Response]


def response(text: str = "ok", *, status: str = "completed", model: str = "gpt-6-luna", reason: str | None = None,
             refusal: str | None = None, encrypted: str | None = "enc-abc", usage: dict[str, Any] | None = None
             ) -> dict[str, Any]:
    reasoning: dict[str, Any] = {"id": "rs_1", "type": "reasoning",
                                 "summary": [{"type": "summary_text", "text": "thought about it"}]}
    if encrypted is not None:
        reasoning["encrypted_content"] = encrypted
    content = ([{"type": "refusal", "refusal": refusal}] if refusal else
               [{"type": "output_text", "text": text, "annotations": []}])
    return {
        "id": "resp_1", "object": "response", "created_at": 1_800_000_000, "model": model, "status": status,
        "incomplete_details": {"reason": reason} if reason else None, "error": None,
        "output": [reasoning, {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                               "content": content}],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": usage or {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 0},
                           "output_tokens": 500, "output_tokens_details": {"reasoning_tokens": 300},
                           "total_tokens": 1500},
    }


def ok(body: dict[str, Any], status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, json=body, headers={"x-request-id": "req_123"})


def error(status: int, code: str, kind: str = "invalid_request_error") -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": "x", "type": kind, "param": None, "code": code}},
                           headers={"x-request-id": "req_err"})


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

    def adapter(self, api_key: str | None = API_KEY) -> OpenAIAdapter:
        return OpenAIAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, recorder: Recorder, *, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(recorder.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_stateless_and_captured(tmp_path: Path) -> None:
    rec = Recorder(ok(response("Use Claude.")))
    outcome, files = call(tmp_path, rec)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Claude."
    (request,) = rec.requests
    assert request.url.path == "/v1/responses"
    body = json.loads(request.content)
    assert body == {
        "model": "gpt-6-luna", "instructions": "Be precise.",
        "input": [{"role": "user", "content": "Pick a model."}],
        "max_output_tokens": 16000, "store": False, "include": ["reasoning.encrypted_content"],
    }  # no reasoning effort, temperature, service tier, or previous_response_id
    wire = json.loads(files["wire-exchanges.json"])
    assert len(wire) == 1 and wire[0]["request_body"]["json"] == body
    assert wire[0]["response_headers"]["x-request-id"] == "req_123"
    for text in files.values():
        assert API_KEY not in text
    assert files["reasoning.txt"] == "thought about it"
    assert outcome.observations["reasoning_tokens"].value == 300


def test_explicit_parameters_map_to_native_fields(tmp_path: Path) -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update({
        "reasoning_effort": {"state": "explicit", "value": "low"},
        "reasoning_summary": {"state": "explicit", "value": "concise"},
        "verbosity": {"state": "explicit", "value": "low"},
    })
    data["adapter_options"]["replay_reasoning"] = False
    rec = Recorder(ok(response()))
    call(tmp_path, rec, config=ModelConfiguration.model_validate(data))
    body = json.loads(rec.requests[0].content)
    assert body["reasoning"] == {"effort": "low", "summary": "concise"}
    assert body["text"] == {"verbosity": "low"}
    assert "include" not in body


def test_cost_uses_cached_and_long_context_rates(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(response())))
    # 800 uncached * $0.10/M + 200 cached * $0.01/M + 500 output * $0.50/M
    assert outcome.observations["api_cost_usd"].value == pytest.approx((800 * 0.10 + 200 * 0.01 + 500 * 0.50) / 1e6)
    long = {"input_tokens": 300_000, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 300_100}
    outcome, _ = call(tmp_path, Recorder(ok(response(usage=long))))
    assert outcome.observations["api_cost_usd"].value == pytest.approx((300_000 * 0.20 + 100 * 0.75) / 1e6)
    unpriced = CONFIG.model_copy(update={"adapter_options": {}})
    outcome, _ = call(tmp_path, Recorder(ok(response())), config=unpriced)
    assert outcome.observations["api_cost_usd"].missing_reason == "no pricing in configuration"


def test_unattributed_cache_writes_make_cost_unknown(tmp_path: Path) -> None:
    usage = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 400},
             "output_tokens": 10, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 1010}
    outcome, _ = call(tmp_path, Recorder(ok(response(usage=usage))))
    assert outcome.observations["api_cost_usd"].value is None


@pytest.mark.parametrize(
    ("reply", "kind", "cost"),
    [
        (error(500, "server_error", "server_error"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
        (error(429, "rate_limit_exceeded", "requests"), FailureKind.RATE_LIMITED, CostIncurred.NO),
        (error(429, "insufficient_quota", "insufficient_quota"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
        (error(400, "context_length_exceeded"), FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO),
        (error(400, "invalid_value"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
        (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
        (httpx2.ConnectError("refused"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
    ],
)
def test_failures_are_mapped_and_never_retried_by_the_sdk(
    tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred
) -> None:
    rec = Recorder(reply, ok(response()))
    outcome, files = call(tmp_path, rec)
    assert not outcome.succeeded
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(rec.requests) == 1  # max_retries=0: the harness, not the SDK, decides on retries
    assert "error.json" in files


@pytest.mark.parametrize("body", [
    response(refusal="I can't help with that."),
    response("", status="incomplete", reason="content_filter"),
])
def test_refusals_are_recorded_not_rerouted(tmp_path: Path, body: dict[str, Any]) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(body)))
    assert outcome.failure_kind is FailureKind.REFUSAL and outcome.cost_incurred is CostIncurred.YES


@pytest.mark.parametrize("body", [
    response(model="gpt-6-sol"),
    {**response(), "status": "failed", "error": {"code": "server_error", "message": "x"}},
    response(status="incomplete", reason="max_messages"),
])
def test_wrong_model_and_failed_responses_are_failures(tmp_path: Path, body: dict[str, Any]) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(body)))
    assert not outcome.succeeded and outcome.failure_kind is FailureKind.PROVIDER_ERROR


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(response("partial", status="incomplete", reason="max_output_tokens"))))
    assert outcome.succeeded and outcome.extracted.truncated
    assert outcome.extracted.finish_reason == "incomplete:max_output_tokens"


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rec = Recorder(ok(response()))
    outcome, _ = call(tmp_path, rec, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and outcome.cost_incurred is CostIncurred.NO
    assert rec.requests == []
    assert "OPENAI_API_KEY" in str(rec.adapter(None).preflight(CONFIG))
    assert rec.adapter().preflight(CONFIG) is None


def test_continuation_replays_encrypted_reasoning_statelessly(tmp_path: Path) -> None:
    baseline = response("baseline answer")
    rec = Recorder(ok(baseline), ok(response("branch a")), ok(response("branch b")))
    adapter = rec.adapter()
    store = ArtifactStore(tmp_path / "results")
    base = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base"))
    snapshot = base.snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    for sent, word in ((json.loads(r.content), w) for r, w in zip(rec.requests[1:], ("alpha", "beta"))):
        assert sent["instructions"] == "Be precise."
        assert "previous_response_id" not in sent and sent["store"] is False
        assert sent["input"][0] == {"role": "user", "content": "Pick a model."}
        assert sent["input"][1:3] == baseline["output"]  # reasoning (encrypted) and message, unchanged
        assert sent["input"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in rec.requests[2].content.decode()  # siblings never see each other
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "gpt-6-sol"}), follow, snapshot, ctx(2))
    plain = MESSAGES + (Message(role=Role.ASSISTANT, content="made up"), Message(role=Role.USER, content="?"))
    with pytest.raises(AdapterError, match="continuation snapshot"):
        adapter.prepare(CONFIG, plain, None, ctx(2))


class MockOpenAI:
    """Models API plus Responses API, answering one-word probes and P6 picks."""

    def __init__(self, *, encrypted: bool = True, answer: str = "1. **GPT-6 Luna**: fits the job.") -> None:
        self.requests: list[httpx2.Request] = []
        self.encrypted = encrypted
        self.answer = answer

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.method == "GET":
            model = request.url.path.rsplit("/", 1)[-1]
            return httpx2.Response(200, json={"id": model, "object": "model", "created": 1, "owned_by": "openai"})
        body = json.loads(request.content)
        last = body["input"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return ok(response(text, model=body["model"], encrypted="enc-xyz" if self.encrypted else None))


REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)


def test_qualification_probes_identity_and_stateless_branching() -> None:
    mock = MockOpenAI()
    report = OpenAIAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(mock)).qualify(
        CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "encrypted reasoning" in checks["continuation"].detail
    assert len(report.probes) == 3
    assert all(API_KEY not in json.dumps(p) for p in report.probes)


def test_qualification_fails_without_replayable_reasoning() -> None:
    mock = MockOpenAI(encrypted=False)
    report = OpenAIAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(mock)).qualify(
        CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert checks["continuation"].passed is False and "encrypted_content" in checks["continuation"].detail


def test_qualification_rejects_unsupported_parameters() -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["top_p"] = {"state": "explicit", "value": 0.5}
    report = OpenAIAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(MockOpenAI())).qualify(
        ModelConfiguration.model_validate(data), REQUIREMENTS.model_copy(update={"requires_continuation": False}))
    assert {c.name: c.passed for c in report.checks}["parameter_support"] is False


def test_openai_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    mock = MockOpenAI()
    registry = AdapterRegistry({
        "openai": OpenAIAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(mock)),
        "fake": FakeAdapter(), "fake-judge": FakeJudgeAdapter(),
    })
    env.registry = registry
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-cross", [ROOT / "configs/models/openai/gpt-6-luna.yaml", other],
                          scenario="scenarios/agentic-design-bias/p6-pick-model",
                          rubric="rubrics/agentic-design-bias/rubric.yaml", repetitions=2)
    loaded = load_experiment(path, env.base)
    scenario, _ = load_scenario(env.base / loaded.spec.scenario, env.base)
    for config, frozen in loaded.configurations:
        record = qualify_configuration(env.db, registry, config, frozen, scenario, "test")
        assert record.payload["report"]["passed"], record.payload["report"]["checks"]
    plan = compile_plan(path, base=env.base, db=env.db)
    persist_plan(env.db, env.store, plan)
    summary = Controller(env.db, env.store, registry).run(plan)
    assert summary.responses == 4

    run = start_evaluation(env.db, env.store, plan.experiment_id, bias_policy(env), base=env.base, registry=registry)
    resume_evaluation(env.db, env.store, registry, run.record_id)
    report = publish_report(env.db, env.store, plan.experiment_id, run.record_id)
    rows = {r["configuration"]: r for r in _read_csv(env, env.db.artifacts_for(report.record_id)["bias-summary.csv"])}
    luna = rows["openai-gpt-6-luna@1"]
    assert luna["own_vendor"] == "openai" and luna["own_first.k"] == "2"
    assert float(luna["own_first_delta"]) == 1.0  # the anthropic-provider candidate never picks GPT
    assert float(rows["anthropic-picker@1"]["own_first_delta"]) == 1.0
