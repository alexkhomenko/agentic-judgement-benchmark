"""MiniMax (Anthropic-compatible Messages API) adapter against a mock transport: native request without an
Anthropic credential, cache-aware pricing, failure mapping (HTTP, base_resp in a 200 body, stop reasons),
served-model checks for an alias, exact replay of signed thinking blocks, endpoint and region checks,
qualification by live probe, and an offline P6 run next to a candidate from another vendor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.minimax_messages import MiniMaxAdapter, _observations
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

API_KEY = "sk-api-0123456789abcdefghijklmnopqrstuvwxyz"
CONFIG_PATH = ROOT / "configs/models/minimax/minimax-m3.1-flash.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)
NO_BRANCHES = REQUIREMENTS.model_copy(update={"requires_continuation": False})
USAGE = {"input_tokens": 600, "output_tokens": 500, "cache_read_input_tokens": 400, "cache_creation_input_tokens": 0}
EXPECTED_COST = (600 * 0.15 + 400 * 0.015 + 500 * 0.60) / 1e6
THINKING = {"type": "thinking", "thinking": "pondering", "signature": "a1b2c3d4"}


def message(text: str = "ok", *, thinking: dict[str, Any] | None = THINKING, stop: str = "end_turn",
            model: str = MODEL, usage: dict[str, Any] = USAGE, **extra: Any) -> dict[str, Any]:
    content = ([thinking] if thinking else []) + [{"type": "text", "text": text}]
    return {"id": "msg-1", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop, "usage": usage, **extra}


class MockMiniMax:
    def __init__(self, *, answer: str = "1. **MiniMax-M3.1-Flash**: fits.", served: str | None = None,
                 reply: httpx2.Response | Exception | None = None,
                 replies: list[httpx2.Response | Exception] | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer, self.served, self.reply = answer, served, reply
        self.replies = list(replies or [])  # served in order before falling back to reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        assert request.url.path == "/anthropic/v1/messages", request.url
        if self.replies:
            nxt = self.replies.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return httpx2.Response(200, json=message(text, model=self.served or body["model"]))

    def adapter(self, api_key: str | None = API_KEY) -> MiniMaxAdapter:
        return MiniMaxAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def with_parameters(**parameters: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update(parameters)
    return ModelConfiguration.model_validate(data)


def with_options(**options: Any) -> ModelConfiguration:
    return CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options, **options}})


def call(tmp_path: Path, mock: MockMiniMax, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockMiniMax(answer="Use MiniMax.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use MiniMax."
    (request,) = mock.requests
    assert str(request.url) == "https://api.minimax.io/anthropic/v1/messages"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert "x-api-key" not in request.headers and "anthropic-version" not in request.headers
    assert json.loads(request.content) == {
        "model": MODEL, "max_tokens": 16000, "system": "Be precise.",
        "messages": [{"role": "user", "content": "Pick a model."}],
    }  # no effort, thinking, sampling parameters, service tier, or tools
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert files["reasoning.txt"] == "pondering"
    assert outcome.observations["api_cost_usd"].value == pytest.approx(EXPECTED_COST)


def test_no_anthropic_credential_is_ever_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-leave-0123456789")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    adapter = default_registry().get("minimax")
    assert adapter._api_key is None  # type: ignore[attr-defined]
    assert adapter.preflight(CONFIG) is not None
    mock = MockMiniMax()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []


def test_explicit_parameters_map_to_native_fields(tmp_path: Path) -> None:
    mock = MockMiniMax()
    call(tmp_path, mock, with_parameters(effort={"state": "explicit", "value": "high"},
                                         thinking={"state": "explicit", "value": "adaptive"}))
    body = json.loads(mock.requests[0].content)
    assert body["output_config"] == {"effort": "high"} and body["thinking"] == {"type": "adaptive"}


def test_mainland_base_url_keeps_its_path(tmp_path: Path) -> None:
    mock = MockMiniMax()
    call(tmp_path, mock, with_options(base_url="https://api.minimax.cn/anthropic/"))
    assert str(mock.requests[0].url) == "https://api.minimax.cn/anthropic/v1/messages"


def test_cost_prices_new_input_cache_reads_and_output() -> None:
    assert _observations(CONFIG, USAGE)["api_cost_usd"].value == pytest.approx(EXPECTED_COST)


def test_thinking_tokens_are_recorded_when_reported() -> None:
    assert _observations(CONFIG, USAGE)["reasoning_tokens"].value is None
    usage = {**USAGE, "output_tokens_details": {"thinking_tokens": 300}}
    assert _observations(CONFIG, usage)["reasoning_tokens"].value == 300


def test_cache_writes_without_a_price_leave_cost_unknown() -> None:
    obs = _observations(CONFIG, {**USAGE, "cache_creation_input_tokens": 100})
    assert obs["api_cost_usd"].value is None
    priced = with_options(pricing={**CONFIG.adapter_options["pricing"], "cache_write_usd_per_mtok": 0.3})
    assert _observations(priced, {**USAGE, "cache_creation_input_tokens": 100})["api_cost_usd"].value == \
        pytest.approx(EXPECTED_COST + 100 * 0.3 / 1e6)


def error(status: int, kind: str, message_text: str = "nope") -> httpx2.Response:
    return httpx2.Response(status, json={"type": "error", "request_id": "req_1",
                                         "error": {"type": kind, "message": message_text}})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (error(400, "invalid_request_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(401, "authentication_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(402, "insufficient_balance_error", "insufficient balance (1008)"), FailureKind.PROVIDER_ERROR,
     CostIncurred.NO),
    (error(404, "not_found_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(413, "request_too_large"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429, "rate_limit_error"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(500, "api_error"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (error(529, "overloaded_error"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (httpx2.Response(504, text="gateway timeout"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockMiniMax(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


def test_http_error_records_the_body_request_id(tmp_path: Path) -> None:
    adapter = MockMiniMax(reply=error(400, "invalid_request_error")).adapter()
    prepared = adapter.prepare(CONFIG, MESSAGES, None, ctx())
    execution = adapter.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "r").sink("c"))
    assert execution.request_id == "req_1" and (execution.error or {})["provider_type"] == "invalid_request_error"
    traced = MockMiniMax(reply=httpx2.Response(200, json=message(), headers={"trace-id": "trace-1"})).adapter()
    assert traced.execute(CONFIG, prepared, ctx(), ArtifactStore(tmp_path / "t").sink("c")).request_id == "trace-1"


@pytest.mark.parametrize(("body", "kind", "cost"), [
    (message(base_resp={"status_code": 1027, "status_msg": "output sensitive"}), FailureKind.REFUSAL,
     CostIncurred.UNKNOWN),
    (message(base_resp={"status_code": 1026, "status_msg": "input sensitive"}), FailureKind.REFUSAL,
     CostIncurred.UNKNOWN),
    (message(base_resp={"status_code": 1002, "status_msg": "rate limit"}), FailureKind.RATE_LIMITED,
     CostIncurred.UNKNOWN),
    (message(base_resp={"status_code": 1008, "status_msg": "insufficient balance"}), FailureKind.PROVIDER_ERROR,
     CostIncurred.UNKNOWN),
    (message(model="MiniMax-M3"), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (message(stop="tool_use"), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (message(stop="pause_turn"), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
    (message(thinking={"type": "redacted_thinking", "data": "x"}), FailureKind.PROVIDER_ERROR, CostIncurred.YES),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind,
                                         cost: CostIncurred) -> None:
    outcome, _ = call(tmp_path, MockMiniMax(reply=httpx2.Response(200, json=body)))
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)


def test_zero_base_resp_and_missing_thinking_are_fine(tmp_path: Path) -> None:
    body = message("plain", thinking=None, base_resp={"status_code": 0, "status_msg": "success"})
    outcome, _ = call(tmp_path, MockMiniMax(reply=httpx2.Response(200, json=body)))
    assert outcome.succeeded and outcome.extracted.reasoning_text is None


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockMiniMax(reply=httpx2.Response(200, json=message("", stop="max_tokens"))))
    assert outcome.succeeded and outcome.extracted.truncated and outcome.extracted.final_answer == ""


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockMiniMax()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.setenv("MINIMAX_API_KEY", API_KEY)
    assert default_registry().get("minimax").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_signed_thinking_exactly(tmp_path: Path) -> None:
    mock = MockMiniMax()
    adapter = mock.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    first = {"role": "assistant", "content": message(mock.answer)["content"]}
    for request, word in zip(mock.requests[1:], ("alpha", "beta")):
        sent = json.loads(request.content)
        assert sent["system"] == "Be precise."
        assert sent["messages"][1] == first  # signed thinking block and text, exactly as returned
        assert sent["messages"][2] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in mock.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "MiniMax-M3"}), follow, snapshot, ctx(2))


def test_system_messages_must_lead(tmp_path: Path) -> None:
    late = MESSAGES + (Message(role=Role.SYSTEM, content="late"),)
    with pytest.raises(AdapterError, match="must lead"):
        MockMiniMax().adapter().prepare(CONFIG, late, None, ctx())


def test_qualification_proves_the_served_model_and_branching() -> None:
    mock = MockMiniMax()
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "served model MiniMax-M3.1-Flash" in checks["model_identity"].detail
    assert "(global)" in checks["endpoint"].detail
    assert "signed thinking blocks" in checks["continuation"].detail
    assert [p["step"] for p in report.probes] == ["baseline", "branch-alpha", "branch-beta"]
    assert all(json.loads(r.content)["max_tokens"] == 2048 for r in mock.requests)


def test_qualification_without_branches_makes_one_probe() -> None:
    mock = MockMiniMax()
    report = mock.adapter().qualify(CONFIG, NO_BRANCHES)
    assert report.passed and len(mock.requests) == 1 and "continuation" not in {c.name for c in report.checks}


def test_qualification_rejects_a_rerouted_model() -> None:
    checks = {c.name: c for c in MockMiniMax(served="MiniMax-M3").adapter().qualify(CONFIG, REQUIREMENTS).checks}
    assert checks["model_identity"].passed is False and "model_mismatch" in checks["model_identity"].detail
    assert "continuation" not in checks


def test_qualification_without_credentials_is_not_evaluated() -> None:
    report = MockMiniMax().adapter(api_key=None).qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and checks["model_identity"].passed is None


@pytest.mark.parametrize(("parameters", "message_text"), [
    ({"temperature": {"state": "explicit", "value": 0.7}}, "ignores sampling parameters"),
    ({"top_p": {"state": "explicit", "value": 0.9}}, "unsupported"),
    ({"effort": {"state": "explicit", "value": "none"}}, "effort must be one of"),
    ({"thinking": {"state": "explicit", "value": "enabled"}}, "'adaptive' or 'disabled'"),
    ({"max_output_tokens": {"state": "explicit", "value": 600_000}}, "524288"),
])
def test_qualification_rejects_bad_parameters_without_spending(parameters: dict[str, Any], message_text: str) -> None:
    mock = MockMiniMax()
    checks = {c.name: c for c in mock.adapter().qualify(with_parameters(**parameters), NO_BRANCHES).checks}
    assert checks["parameter_support"].passed is False and message_text in checks["parameter_support"].detail
    assert mock.requests == []


@pytest.mark.parametrize(("base_url", "message_text"), [
    ("https://api.minimax.io/v1", "not MiniMax's Anthropic-compatible API"),
    ("https://api.minimaxi.com/anthropic", "not MiniMax's Anthropic-compatible API"),
    ("https://api.anthropic.com", "not MiniMax's Anthropic-compatible API"),
    ("http://api.minimax.io/anthropic", "not MiniMax's Anthropic-compatible API"),
    ("https://api.minimax.cn/anthropic", "mainland platform but pricing is for global"),
])
def test_qualification_rejects_other_endpoints_without_spending(base_url: str, message_text: str) -> None:
    mock = MockMiniMax()
    checks = {c.name: c for c in mock.adapter().qualify(with_options(base_url=base_url), NO_BRANCHES).checks}
    assert checks["endpoint"].passed is False and message_text in checks["endpoint"].detail
    assert mock.requests == []


def test_minimax_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"minimax": MockMiniMax().adapter(), "fake": FakeAdapter(),
                                "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-minimax", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["minimax-m3.1-flash@1"]
    assert row["own_vendor"] == "minimax" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0


# ----- web search (server tool on the Anthropic-compatible Messages API) -------------------------------------------

SEARCH_TOOL: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search"}


def searching_config(**pricing: float) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["web_search"] = {"state": "explicit", "value": SEARCH_TOOL}
    data["adapter_options"]["pricing"].update(pricing)
    return ModelConfiguration.model_validate(data)


def searched(*, searches: int = 2, stop: str = "end_turn", reported: int | None = None) -> dict[str, Any]:
    """Shaped like the documented server-tools response: lead-in text, searches, thinking, then the answer."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": "I'll search. "}]
    for i in range(searches):
        blocks += [
            {"type": "server_tool_use", "id": f"call_function_x_{i}", "name": "web_search", "input": {"query": f"q{i}"}},
            {"type": "web_search_tool_result", "tool_use_id": f"call_function_x_{i}",
             "content": [{"type": "web_search_result", "title": "T", "url": "https://example.com",
                          "page_age": "2026/11/20 07:08:47", "content": "result text"}]},
        ]
    blocks += [THINKING, {"type": "text", "text": "Answer."}]
    usage = dict(USAGE)
    if reported is not None:
        usage["server_tool_use"] = {"web_search_requests": reported}
    return message(stop=stop, usage=usage, base_resp={"status_code": 0, "status_msg": ""}) | {"content": blocks}


def test_web_search_tool_is_sent_verbatim_and_searches_are_counted_and_priced(tmp_path: Path) -> None:
    mock = MockMiniMax(reply=httpx2.Response(200, json=searched()))
    outcome, files = call(tmp_path, mock, searching_config(web_search_usd_per_request=0.01))
    assert json.loads(mock.requests[0].content)["tools"] == [SEARCH_TOOL] and len(mock.requests) == 1
    assert outcome.succeeded and outcome.extracted.final_answer == "I'll search. Answer."  # text blocks only
    assert "result text" not in outcome.extracted.final_answer and files["reasoning.txt"] == "pondering"
    assert outcome.observations["web_search_requests"].value == 2
    assert "server_tool_use web_search blocks" in outcome.observations["web_search_requests"].source
    assert outcome.observations["api_cost_usd"].value == pytest.approx(EXPECTED_COST + 0.02)
    outcome, _ = call(tmp_path, MockMiniMax(reply=httpx2.Response(200, json=searched(reported=3))),
                      searching_config(web_search_usd_per_request=0.01))
    assert outcome.observations["web_search_requests"].value == 3  # a reported usage count wins
    outcome, _ = call(tmp_path, MockMiniMax(reply=httpx2.Response(200, json=searched())), searching_config())
    assert outcome.observations["api_cost_usd"].missing_reason == "no price for web_search_requests"
    outcome, _ = call(tmp_path, MockMiniMax(), searching_config())
    assert outcome.observations["web_search_requests"].value == 0  # offered but unused: priced as no searches
    assert outcome.observations["api_cost_usd"].value == pytest.approx(EXPECTED_COST)
    plain = MockMiniMax()
    outcome, _ = call(tmp_path, plain)
    assert "web_search_requests" not in outcome.observations and "tools" not in json.loads(plain.requests[0].content)


def test_search_blocks_without_the_parameter_are_still_unexpected(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockMiniMax(reply=httpx2.Response(200, json=searched())))
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and "unexpected_content" in str(outcome.error)


def test_paused_server_tool_loop_still_paused_after_the_resumes_is_an_incomplete_failure(tmp_path: Path) -> None:
    mock = MockMiniMax(reply=httpx2.Response(200, json=searched(stop="pause_turn")))
    outcome, files = call(tmp_path, mock, searching_config(web_search_usd_per_request=0.01))
    assert not outcome.succeeded and outcome.failure_kind is FailureKind.PROVIDER_ERROR
    assert outcome.cost_incurred is CostIncurred.YES and '"incomplete"' in str(outcome.error)
    assert len(mock.requests) == 4 and len(json.loads(files["wire-exchanges.json"])) == 4  # 3 resumes by default
    assert outcome.observations["web_search_requests"].value == 8
    assert outcome.observations["api_cost_usd"].value == pytest.approx(4 * (EXPECTED_COST + 0.02))


def test_paused_turn_is_resumed_with_its_content_and_searches_are_summed(tmp_path: Path) -> None:
    # Seen live 2026-11-25: MiniMax paused after 12 searches on a framework question.
    paused, final = searched(searches=2, stop="pause_turn"), searched(searches=1)
    mock = MockMiniMax(replies=[httpx2.Response(200, json=paused), httpx2.Response(200, json=final)])
    outcome, files = call(tmp_path, mock, searching_config(web_search_usd_per_request=0.01))
    assert outcome.succeeded and len(mock.requests) == 2
    first, second = (json.loads(r.content) for r in mock.requests)
    assert second["messages"][:-1] == first["messages"] and second["tools"] == first["tools"]
    assert second["messages"][-1] == {"role": "assistant", "content": paused["content"]}  # no extra user turn
    assert outcome.extracted.final_answer == "I'll search. Answer.I'll search. Answer."
    assert outcome.observations["web_search_requests"].value == 3
    assert outcome.observations["api_cost_usd"].value == pytest.approx(2 * EXPECTED_COST + 0.03)
    native = json.loads(files["native-response.json"])
    assert native["content"] == paused["content"] + final["content"] and native["harness_pause_turns"] == [paused]


def test_a_failed_resume_keeps_the_billed_paused_call(tmp_path: Path) -> None:
    mock = MockMiniMax(replies=[httpx2.Response(200, json=searched(stop="pause_turn")),
                                httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error",
                                                                                      "message": "busy"}})])
    outcome, _ = call(tmp_path, mock, searching_config(web_search_usd_per_request=0.01))
    assert not outcome.succeeded and outcome.cost_incurred is CostIncurred.YES
    assert outcome.observations["web_search_requests"].value == 2
    # The overloaded resume was not billed: the paused call's cost is the whole cost.
    assert outcome.observations["api_cost_usd"].value == pytest.approx(EXPECTED_COST + 0.02)
    assert "api_cost_usd_lower_bound" not in outcome.observations


def test_a_timed_out_resume_marks_the_total_cost_unknown_and_keeps_the_billed_subtotal(tmp_path: Path) -> None:
    mock = MockMiniMax(replies=[httpx2.Response(200, json=searched(stop="pause_turn")),
                                httpx2.ReadTimeout("read timed out")])
    outcome, _ = call(tmp_path, mock, searching_config(web_search_usd_per_request=0.01))
    assert outcome.failure_kind is FailureKind.AMBIGUOUS_TIMEOUT and outcome.cost_incurred is CostIncurred.YES
    total, bound = outcome.observations["api_cost_usd"], outcome.observations["api_cost_usd_lower_bound"]
    assert total.value is None and "may have been billed" in str(total.missing_reason)
    assert bound.value == pytest.approx(EXPECTED_COST + 0.02)


@pytest.mark.parametrize("value", [
    {"type": "custom", "name": "web_search"},
    {"type": "web_search_20250305", "name": "search"},
    {"type": "web_search"},  # the Responses API form
])
def test_web_search_value_must_be_the_server_search_tool(value: dict[str, Any]) -> None:
    data = searching_config().model_dump(mode="json")
    data["parameters"]["web_search"]["value"] = value
    config = ModelConfiguration.model_validate(data)
    with pytest.raises(AdapterError) as raised:
        MockMiniMax().adapter().prepare(config, MESSAGES, None, ctx())
    assert raised.value.kind is FailureKind.INVALID_STATE
    mock = MockMiniMax()
    checks = {c.name: c for c in mock.adapter().qualify(config, NO_BRANCHES).checks}
    assert checks["parameter_support"].passed is False and mock.requests == []


def test_continuation_replays_search_blocks_with_signed_thinking(tmp_path: Path) -> None:
    replies = [searched(), message("Follow-up answer.")]
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=replies.pop(0))

    adapter = MiniMaxAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(handler))
    config = searching_config(web_search_usd_per_request=0.01)
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, config, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    follow = snapshot.messages + (Message(role=Role.USER, content="Follow-up"),)
    assert invoke(adapter, config, follow, snapshot, ctx(2), store.sink("child")).succeeded
    sent = json.loads(requests[1].content)
    assert sent["messages"][1] == {"role": "assistant", "content": searched()["content"]}  # exactly as returned
    assert sent["messages"][2] == {"role": "user", "content": "Follow-up"} and sent["tools"] == [SEARCH_TOOL]
