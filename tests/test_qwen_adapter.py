"""Qwen (Alibaba Cloud Model Studio) chat-completions adapter against a mock transport: native streamed request,
stream reassembly, cache-aware pricing, failure mapping (HTTP and in-stream), served-model checks for an alias,
exact replay including reasoning_content, endpoint and region checks, qualification by live probe, and an
offline P6 run with a Qwen candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.qwen_chat import QwenAdapter, _observations
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

API_KEY = "sk-0123456789abcdef0123456789abcdef"
CONFIG_PATH = ROOT / "configs/models/alibaba/qwen4-max.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)
NO_BRANCHES = REQUIREMENTS.model_copy(update={"requires_continuation": False})
USAGE = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
         "prompt_tokens_details": {"cached_tokens": 400},
         "completion_tokens_details": {"reasoning_tokens": 300, "text_tokens": 500}}
EXPECTED_COST = (600 * 1.50 + 400 * 0.075 + 500 * 4.50) / 1e6
WORKSPACE = "https://ws-test.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"


def chunk(delta: dict[str, Any] | None = None, *, finish: str | None = None, model: str = MODEL,
          usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1_794_000_000, "model": model,
            "system_fingerprint": None, "service_tier": None,
            "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
            "usage": usage}


def stream(text: str = "ok", *, reasoning: str | None = "pondering", finish: str = "stop", model: str = MODEL,
           usage: dict[str, Any] | None = USAGE, done: bool = True, extra: list[dict[str, Any]] | None = None,
           ) -> httpx2.Response:
    chunks = [chunk({"role": "assistant", "content": "", "reasoning_content": None}, model=model)]
    if reasoning:
        half = len(reasoning) // 2
        chunks += [chunk({"reasoning_content": reasoning[:half]}, model=model),
                   chunk({"reasoning_content": reasoning[half:]}, model=model)]
    half = len(text) // 2
    chunks += [chunk({"content": text[:half]}, model=model), chunk({"content": text[half:]}, model=model)]
    chunks += extra or []
    chunks.append(chunk({}, finish=finish, model=model))
    if usage is not None:
        chunks.append(chunk(None, model=model, usage=usage))
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + ("data: [DONE]\n\n" if done else "")
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream", "x-request-id": "req-1"})


class MockQwen:
    def __init__(self, *, answer: str = "1. **Qwen4-Max**: fits.", served: str | None = None,
                 reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer, self.served, self.reply = answer, served, reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        assert request.url.path == "/compatible-mode/v1/chat/completions", request.url
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return stream(text, model=self.served or body["model"])

    def adapter(self, api_key: str | None = API_KEY) -> QwenAdapter:
        return QwenAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def with_parameters(**parameters: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update(parameters)
    return ModelConfiguration.model_validate(data)


def with_options(**options: Any) -> ModelConfiguration:
    return CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options, **options}})


def call(tmp_path: Path, mock: MockQwen, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_streamed_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockQwen(answer="Use Qwen.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Qwen."
    (request,) = mock.requests
    assert str(request.url) == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "model": MODEL, "max_tokens": 16000, "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
    }  # no thinking switches, sampling parameters, search, or tools
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    assert "data: [DONE]" in wire[0]["response_body"]["text"]  # the raw event stream is the captured body
    for text in files.values():
        assert API_KEY not in text
    assert files["reasoning.txt"] == "pondering"
    assert outcome.observations["uncached_input_tokens"].value == 600
    assert outcome.observations["reasoning_tokens"].value == 300
    assert outcome.observations["api_cost_usd"].value == pytest.approx(EXPECTED_COST)


def test_stream_is_reassembled_into_a_completion(tmp_path: Path) -> None:
    adapter = MockQwen(reply=stream("Hello world", reasoning="think hard")).adapter()
    prepared = adapter.prepare(CONFIG, MESSAGES, None, ctx())
    execution = adapter.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "r").sink("c"))
    assert execution.status == "completed" and execution.request_id == "req-1"
    data = execution.native_response or {}
    assert data["choices"] == [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "Hello world", "reasoning_content": "think hard"}}]
    assert (data["id"], data["model"], data["usage"]) == ("chatcmpl-1", MODEL, USAGE)
    assert data["stream"] == {"chunks": 7, "done": True, "models": [MODEL]}


def test_non_thinking_reply_has_no_reasoning_field(tmp_path: Path) -> None:
    adapter = MockQwen(reply=stream("plain", reasoning=None)).adapter()
    prepared = adapter.prepare(CONFIG, MESSAGES, None, ctx())
    execution = adapter.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "r").sink("c"))
    assert (execution.native_response or {})["choices"][0]["message"] == {"role": "assistant", "content": "plain"}


def test_explicit_parameters_map_to_native_fields(tmp_path: Path) -> None:
    mock = MockQwen()
    call(tmp_path, mock, with_parameters(
        enable_thinking={"state": "explicit", "value": True},
        thinking_budget={"state": "explicit", "value": 8192},
        preserve_thinking={"state": "explicit", "value": False},
        temperature={"state": "explicit", "value": 0.6}, top_p={"state": "explicit", "value": 0.9},
        top_k={"state": "explicit", "value": 40}, presence_penalty={"state": "explicit", "value": 0.5},
        repetition_penalty={"state": "explicit", "value": 1.05}, seed={"state": "explicit", "value": 7}))
    body = json.loads(mock.requests[0].content)
    assert {k: body[k] for k in ("enable_thinking", "thinking_budget", "preserve_thinking", "temperature", "top_p",
                                 "top_k", "presence_penalty", "repetition_penalty", "seed")} == {
        "enable_thinking": True, "thinking_budget": 8192, "preserve_thinking": False, "temperature": 0.6,
        "top_p": 0.9, "top_k": 40, "presence_penalty": 0.5, "repetition_penalty": 1.05, "seed": 7}
    assert "reasoning_effort" not in body


def test_workspace_base_url_keeps_its_path(tmp_path: Path) -> None:
    mock = MockQwen()
    call(tmp_path, mock, with_options(base_url=WORKSPACE + "/"))
    assert str(mock.requests[0].url) == WORKSPACE + "/chat/completions"


def test_cost_prices_cache_hits_and_misses_separately() -> None:
    assert _observations(CONFIG, USAGE)["api_cost_usd"].value == pytest.approx(EXPECTED_COST)


def test_cache_writes_are_priced_by_ttl() -> None:
    usage = {**USAGE, "prompt_tokens_details": {
        "cached_tokens": 400, "cache_write_tokens": 300, "cache_creation_input_tokens": 300,
        "cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 100}}}
    obs = _observations(CONFIG, usage)
    assert obs["uncached_input_tokens"].value == 300  # cache_creation_input_tokens is not subtracted twice
    expected = (300 * 1.50 + 400 * 0.075 + 200 * 1.875 + 100 * 3.00 + 500 * 4.50) / 1e6
    assert obs["api_cost_usd"].value == pytest.approx(expected)


def test_cache_writes_without_a_ttl_breakdown_leave_cost_unknown() -> None:
    usage = {**USAGE, "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 300}}
    obs = _observations(CONFIG, usage)
    assert obs["api_cost_usd"].value is None and obs["uncached_input_tokens"].value == 300


def test_omitted_cache_fields_count_as_zero_but_missing_details_leave_cost_unknown() -> None:
    obs = _observations(CONFIG, {**USAGE, "prompt_tokens_details": {}})
    assert obs["cached_input_tokens"].value == 0 and obs["uncached_input_tokens"].value == 1000
    assert obs["api_cost_usd"].value == pytest.approx((1000 * 1.50 + 500 * 4.50) / 1e6)
    obs = _observations(CONFIG, {k: v for k, v in USAGE.items() if k != "prompt_tokens_details"})
    assert obs["api_cost_usd"].value is None and obs["uncached_input_tokens"].value is None


def openai_error(status: int, code: str, message: str = "nope") -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"code": code, "message": message, "type": "invalid_request_error"}})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (openai_error(400, "data_inspection_failed"), FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (httpx2.Response(400, json={"code": "DataInspectionFailed", "message": "x", "request_id": "r"}),
     FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (openai_error(400, "CustomRoleBlocked"), FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (openai_error(400, "invalid_parameter_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (openai_error(401, "invalid_api_key"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (openai_error(404, "model_not_found"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (openai_error(429, "Throttling.RateQuota"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (openai_error(500, "internal_error"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (httpx2.Response(504, text="gateway timeout"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockQwen(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


def _raw(*events: str) -> httpx2.Response:
    return httpx2.Response(200, text="".join(f"data: {e}\n\n" for e in events),
                           headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (_raw(json.dumps(chunk({"content": "par"})), json.dumps({"error": {"code": "data_inspection_failed"}})),
     FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (_raw(json.dumps(chunk({"content": "par"})), json.dumps({"code": "InternalError", "message": "x"})),
     FailureKind.PROVIDER_ERROR, CostIncurred.UNKNOWN),
    (stream(done=False), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
    (_raw(json.dumps(chunk({"content": "par"})), "[DONE]"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
    (_raw("{not json"), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (stream(extra=[chunk({"tool_calls": [{"index": 0}]})]), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (stream(extra=[chunk({"encrypted_content": "opaque"})]), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (stream(finish="tool_calls"), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (stream(finish="content_filter"), FailureKind.REFUSAL, CostIncurred.YES),
    (stream(model="qwen3.8-max"), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (stream(extra=[chunk({"content": "!"}, model="qwen3.8-max")]), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (stream(usage={**USAGE, "completion_tokens": 16001}), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
])
def test_unusable_streams_are_failures(tmp_path: Path, reply: httpx2.Response, kind: FailureKind,
                                       cost: CostIncurred) -> None:
    outcome, _ = call(tmp_path, MockQwen(reply=reply))
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)


def test_model_mismatch_names_requested_expected_and_served(tmp_path: Path) -> None:
    adapter = MockQwen(served="qwen3.8-max").adapter()
    prepared = adapter.prepare(CONFIG, MESSAGES, None, ctx())
    execution = adapter.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "r").sink("c"))
    assert execution.error == {"type": "model_mismatch", "requested": MODEL, "expected": MODEL,
                               "served": ["qwen3.8-max"]}


def test_configured_served_model_is_accepted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockQwen(served="qwen4-max-2026-11-16"),
                      with_options(served_model="qwen4-max-2026-11-16"))
    assert outcome.succeeded


def test_missing_usage_completes_with_unknown_cost(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockQwen(reply=stream(usage=None)))
    assert outcome.succeeded and outcome.observations["api_cost_usd"].value is None


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockQwen(reply=stream("partial", finish="length")))
    assert outcome.succeeded and outcome.extracted.truncated


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockQwen()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    assert default_registry().get("qwen").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("DASHSCOPE_API_KEY", API_KEY)
    assert default_registry().get("qwen").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_the_message_exactly(tmp_path: Path) -> None:
    mock = MockQwen()
    adapter = mock.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    first = {"role": "assistant", "content": mock.answer, "reasoning_content": "pondering"}
    for request, word in zip(mock.requests[1:], ("alpha", "beta")):
        sent = json.loads(request.content)
        assert sent["messages"][2] == first  # content and reasoning_content, exactly as reassembled
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in mock.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "qwen3.8-max"}), follow, snapshot, ctx(2))


def test_qualification_proves_the_served_model_and_branching() -> None:
    mock = MockQwen()
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "served model qwen4-max" in checks["model_identity"].detail
    assert "ap-southeast-1" in checks["endpoint"].detail
    assert "including reasoning_content" in checks["continuation"].detail
    assert [p["step"] for p in report.probes] == ["baseline", "branch-alpha", "branch-beta"]
    assert all(json.loads(r.content)["max_tokens"] == 2048 for r in mock.requests)


def test_qualification_without_branches_makes_one_probe() -> None:
    mock = MockQwen()
    report = mock.adapter().qualify(CONFIG, NO_BRANCHES)
    assert report.passed and len(mock.requests) == 1 and "continuation" not in {c.name for c in report.checks}


def test_qualification_rejects_a_substituted_model() -> None:
    checks = {c.name: c for c in MockQwen(served="qwen3.8-max").adapter().qualify(CONFIG, REQUIREMENTS).checks}
    assert checks["model_identity"].passed is False and "model_mismatch" in checks["model_identity"].detail
    assert "continuation" not in checks


def test_qualification_without_credentials_is_not_evaluated() -> None:
    report = MockQwen().adapter(api_key=None).qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and checks["model_identity"].passed is None


@pytest.mark.parametrize(("parameters", "message"), [
    ({"enable_search": {"state": "explicit", "value": True}}, "enable_search"),
    ({"reasoning_effort": {"state": "explicit", "value": "high"},
      "thinking_budget": {"state": "explicit", "value": 4096}}, "cannot both be set"),
    ({"thinking_budget": {"state": "explicit", "value": 0}}, "positive integer"),
    ({"max_output_tokens": {"state": "explicit", "value": 300_000}}, "262144"),
    ({"enable_thinking": {"state": "explicit", "value": False},
      "max_output_tokens": {"state": "explicit", "value": 200_000}}, "131072"),
])
def test_qualification_rejects_bad_parameters_without_spending(parameters: dict[str, Any], message: str) -> None:
    mock = MockQwen()
    checks = {c.name: c for c in mock.adapter().qualify(with_parameters(**parameters), NO_BRANCHES).checks}
    assert checks["parameter_support"].passed is False and message in checks["parameter_support"].detail
    assert mock.requests == []


@pytest.mark.parametrize(("base_url", "message"), [
    ("https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "token-plan endpoint"),
    ("https://trial.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "trial endpoint"),
    ("https://coding-intl.dashscope.aliyuncs.com/v1", "not the OpenAI-compatible"),
    ("https://coding-intl.dashscope.aliyuncs.com/compatible-mode/v1", "not a pay-as-you-go"),
    ("https://dashscope-intl.aliyuncs.com/api/v1", "not the OpenAI-compatible"),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "pricing is for ap-southeast-1"),
    ("https://ws-test.us-east-1.maas.aliyuncs.com/compatible-mode/v1", "serves us-east-1"),
    ("http://dashscope-intl.aliyuncs.com/compatible-mode/v1", "not https"),
])
def test_qualification_rejects_other_endpoints_without_spending(base_url: str, message: str) -> None:
    mock = MockQwen()
    checks = {c.name: c for c in mock.adapter().qualify(with_options(base_url=base_url), NO_BRANCHES).checks}
    assert checks["endpoint"].passed is False and message in checks["endpoint"].detail
    assert mock.requests == []


def test_qualification_accepts_a_workspace_endpoint_in_the_priced_region() -> None:
    checks = {c.name: c for c in MockQwen().adapter().qualify(with_options(base_url=WORKSPACE), NO_BRANCHES).checks}
    assert checks["endpoint"].passed is True


def test_qwen_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"qwen": MockQwen().adapter(), "fake": FakeAdapter(),
                                "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-qwen", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["qwen4-max@1"]
    assert row["own_vendor"] == "alibaba" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0
