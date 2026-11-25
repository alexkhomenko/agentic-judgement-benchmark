"""Kimi (Moonshot AI) chat-completions adapter against a mock transport: native request, cache-aware
pricing, failure mapping, exact replay including reasoning_content, qualification against /v1/models, and
an offline P6 run with a Kimi candidate next to a candidate from another vendor."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.kimi_chat import KimiAdapter, _observations
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

API_KEY = "sk-moonshot-test-0123456789abcdefghij"
CONFIG_PATH = ROOT / "configs/models/moonshot/kimi-k3.1-pro.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)
WRITE_HEADERS = {"msh-usage-cache-write-tokens-5m": "256", "msh-usage-cache-write-tokens-1h": "0"}


def completion(text: str = "ok", *, finish: str = "stop", model: str = MODEL, reasoning: str | None = "pondering",
               ) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1_792_740_680, "model": model,
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": text, "reasoning_content": reasoning}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
                  "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 256},
                  "completion_tokens_details": {"reasoning_tokens": 300}},
    }


class MockKimi:
    def __init__(self, *, listed: dict[str, Any] | None = None, answer: str = "1. **Kimi K3.1 Pro**: fits.",
                 reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.listed = {"id": MODEL, "object": "model", "created": 1_792_000_000, "owned_by": "moonshot",
                       "context_length": 1048576, "supports_image_in": True, "supports_video_in": True,
                       "supports_reasoning": True, **(listed or {})}
        self.answer, self.reply = answer, reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path == "/v1/models":
            return httpx2.Response(200, json={"object": "list", "data": [self.listed]})
        assert request.url.path == "/v1/chat/completions", request.url
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return httpx2.Response(200, json=completion(text, model=body["model"]), headers=WRITE_HEADERS)

    def adapter(self, api_key: str | None = API_KEY) -> KimiAdapter:
        return KimiAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, mock: MockKimi, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockKimi(answer="Use Kimi.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Kimi."
    (request,) = mock.requests
    assert str(request.url) == "https://api.moonshot.ai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "model": MODEL, "max_completion_tokens": 16000,
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
    }  # no effort, thinking toggle, fixed sampling parameters, or tools
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert files["reasoning.txt"] == "pondering"
    assert outcome.observations["reasoning_tokens"].value == 300
    assert outcome.observations["uncached_input_tokens"].value == 1000 - 400 - 256
    assert outcome.observations["api_cost_usd"].value == pytest.approx((344 * 2.0 + 400 * 0.2 + 256 * 2.0 + 500 * 10.0) / 1e6)


def test_explicit_parameters_map_to_native_fields(tmp_path: Path) -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update({"reasoning_effort": {"state": "explicit", "value": "low"},
                               "thinking": {"state": "explicit", "value": "disabled"}})
    mock = MockKimi()
    call(tmp_path, mock, ModelConfiguration.model_validate(data))
    body = json.loads(mock.requests[0].content)
    assert body["reasoning_effort"] == "low" and body["thinking"] == {"type": "disabled"}


def test_custom_base_url_keeps_its_path(tmp_path: Path) -> None:
    mock = MockKimi()
    config = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options,
                                                           "base_url": "https://kimi.example.test/v1/"}})
    call(tmp_path, mock, config)
    assert str(mock.requests[0].url) == "https://kimi.example.test/v1/chat/completions"


def test_cost_prices_reads_writes_and_misses_separately() -> None:
    usage = completion()["usage"]
    obs = _observations(CONFIG, usage, WRITE_HEADERS)
    expected = (344 * 2.0 + 400 * 0.2 + 256 * 2.0 + 500 * 10.0) / 1e6
    assert obs["api_cost_usd"].value == pytest.approx(expected)
    one_hour = {"msh-usage-cache-write-tokens-5m": "0", "msh-usage-cache-write-tokens-1h": "256"}
    assert _observations(CONFIG, usage, one_hour)["api_cost_usd"].value == pytest.approx(expected + 256 * 2.0 / 1e6)


@pytest.mark.parametrize("headers", [{}, {"msh-usage-cache-write-tokens-5m": "100",
                                          "msh-usage-cache-write-tokens-1h": "0"}])
def test_cache_writes_without_a_matching_ttl_split_leave_cost_unknown(headers: dict[str, str]) -> None:
    obs = _observations(CONFIG, completion()["usage"], headers)
    assert obs["api_cost_usd"].value is None and "TTL" in str(obs["api_cost_usd"].missing_reason)


# The non-streaming example from the Chat Completions reference (platform.kimi.ai/docs/api/chat): a cold
# cache reports only cache_write_tokens; cached_tokens is absent both top-level and in prompt_tokens_details.
DOCUMENTED_COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-6adb0d430e590c70159c6ca1", "object": "chat.completion", "created": 1_792_740_680,
    "model": MODEL,
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "1 + 1 = 2", "reasoning_content": "..."}}],
    "usage": {"completion_tokens": 112, "prompt_tokens": 373, "total_tokens": 485,
              "completion_tokens_details": {"reasoning_tokens": 90},
              "prompt_tokens_details": {"cache_write_tokens": 256}},
}
DOCUMENTED_COST = (117 * 2.0 + 256 * 2.0 + 112 * 10.0) / 1e6  # 373 = 256 written + 117 uncached; no reads


def test_documented_cold_cache_response_is_priced() -> None:
    obs = _observations(CONFIG, copy.deepcopy(DOCUMENTED_COMPLETION["usage"]), WRITE_HEADERS)
    assert obs["cached_input_tokens"].value == 0 and "omitted" in obs["cached_input_tokens"].source
    assert obs["uncached_input_tokens"].value == 117
    assert obs["api_cost_usd"].value == pytest.approx(DOCUMENTED_COST)


def test_documented_cold_cache_response_replays_with_cost(tmp_path: Path) -> None:
    reply = httpx2.Response(200, json=DOCUMENTED_COMPLETION, headers=WRITE_HEADERS)
    outcome, _ = call(tmp_path, MockKimi(reply=reply))
    assert outcome.cost_incurred is CostIncurred.YES
    assert outcome.observations["cached_input_tokens"].value == 0
    assert outcome.observations["cache_write_5m_tokens"].value == 256
    assert outcome.observations["api_cost_usd"].value == pytest.approx(DOCUMENTED_COST)


def test_usage_without_a_cache_breakdown_leaves_cost_unknown() -> None:
    usage = copy.deepcopy(DOCUMENTED_COMPLETION["usage"])
    del usage["prompt_tokens_details"]
    obs = _observations(CONFIG, usage, WRITE_HEADERS)
    assert obs["cached_input_tokens"].value is None and obs["api_cost_usd"].value is None


def test_no_cache_writes_need_no_ttl_split() -> None:
    usage = completion()["usage"]
    usage["prompt_tokens_details"] = {"cached_tokens": 0, "cache_write_tokens": 0}
    assert _observations(CONFIG, usage, {})["api_cost_usd"].value == pytest.approx((1000 * 2.0 + 500 * 10.0) / 1e6)



def test_warm_turn_without_write_fields_or_headers_is_priced_as_no_writes() -> None:
    # Observed 2026-11-25 (third round of a search loop): cached_tokens only, no cache-write headers.
    usage = {"cached_tokens": 2688, "completion_tokens": 4235, "prompt_tokens": 9719, "total_tokens": 13954,
             "completion_tokens_details": {"reasoning_tokens": 3608}, "prompt_tokens_details": {"cached_tokens": 2688}}
    obs = _observations(CONFIG, usage, {"msh-context-cache-token-saved": "2688"})
    assert obs["cache_write_tokens"].value == 0 and obs["uncached_input_tokens"].value == 9719 - 2688
    assert obs["api_cost_usd"].value == pytest.approx(((9719 - 2688) * 2.0 + 2688 * 0.2 + 4235 * 10.0) / 1e6)
    # A write header without the usage field is still an unexplained write: cost stays unknown.
    assert _observations(CONFIG, usage, WRITE_HEADERS)["api_cost_usd"].value is None

def error(status: int, kind: str, message: str = "nope") -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"type": kind, "message": message}})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (error(400, "content_filter"), FailureKind.REFUSAL, CostIncurred.UNKNOWN),
    (error(400, "invalid_request_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(401, "invalid_authentication_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(404, "resource_not_found_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429, "exceeded_current_quota_error"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (error(429, "rate_limit_reached_error"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(429, "engine_overloaded_error"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (error(499, "client_closed_request"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
    (httpx2.Response(499, text="client closed request"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
    (error(503, "server_unavailable"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (httpx2.Response(504, text="gateway timeout"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockKimi(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


@pytest.mark.parametrize(("body", "kind"), [
    (completion(finish="tool_calls"), FailureKind.PROVIDER_ERROR),
    (completion(finish="client_disconnected"), FailureKind.PROVIDER_ERROR),
    (completion(model="kimi-k3"), FailureKind.PROVIDER_ERROR),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind) -> None:
    outcome, _ = call(tmp_path, MockKimi(reply=httpx2.Response(200, json=body)))
    assert outcome.failure_kind is kind and outcome.cost_incurred is CostIncurred.YES


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockKimi(reply=httpx2.Response(200, json=completion("partial", finish="length"))))
    assert outcome.succeeded and outcome.extracted.truncated


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockKimi()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    assert default_registry().get("kimi").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("MOONSHOT_API_KEY", API_KEY)
    assert default_registry().get("kimi").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_the_message_exactly(tmp_path: Path) -> None:
    mock = MockKimi()
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
        assert sent["messages"][2] == first  # content and reasoning_content, exactly as returned
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in mock.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "kimi-k3"}), follow, snapshot, ctx(2))


def test_qualification_records_the_model_card() -> None:
    report = MockKimi().adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "context_length=1048576, supports_reasoning=True" in checks["model_identity"].detail
    assert "including reasoning_content" in checks["continuation"].detail


def test_qualification_rejects_fixed_sampling_parameters() -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["temperature"] = {"state": "explicit", "value": 0.2}
    checks = {c.name: c for c in MockKimi().adapter().qualify(
        ModelConfiguration.model_validate(data), REQUIREMENTS.model_copy(update={"requires_continuation": False})).checks}
    assert checks["parameter_support"].passed is False and "temperature" in checks["parameter_support"].detail


@pytest.mark.parametrize(("name", "state", "expected"), [
    ("max_output_tokens", None, "max_output_tokens must be explicit"),
    ("max_output_tokens", {"state": "omitted"}, "max_output_tokens must be explicit"),
    ("temperature", {"state": "explicit", "value": 0.2}, "temperature"),
])
def test_invalid_parameters_skip_continuation_probes(name: str, state: dict[str, Any] | None, expected: str) -> None:
    data = CONFIG.model_dump(mode="json")
    if state is None:
        del data["parameters"][name]
    else:
        data["parameters"][name] = state
    mock = MockKimi()
    report = mock.adapter().qualify(ModelConfiguration.model_validate(data), REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed
    assert checks["parameter_support"].passed is False and expected in checks["parameter_support"].detail
    assert checks["continuation"].passed is None and "parameters are invalid" in checks["continuation"].detail
    assert [r.url.path for r in mock.requests] == ["/v1/models"]


@pytest.mark.parametrize(("listed", "expected"), [
    ({"id": "kimi-other"}, "not listed"),
    ({"context_length": 262144}, "context_limit"),
])
def test_qualification_rejects_identity_problems(listed: dict[str, Any], expected: str) -> None:
    checks = {c.name: c for c in MockKimi(listed=listed).adapter().qualify(
        CONFIG, REQUIREMENTS.model_copy(update={"requires_continuation": False})).checks}
    assert checks["model_identity"].passed is False and expected in checks["model_identity"].detail


def test_kimi_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"kimi": MockKimi().adapter(), "fake": FakeAdapter(),
                                "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-kimi", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["kimi-k3.1-pro@1"]
    assert row["own_vendor"] == "moonshot" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0


# ----- web search (Formula official web-search tool, client-side round trip) ------------------------------------

SEARCH_TOOL: dict[str, Any] = {  # GET /v1/formulas/moonshot/web-search:latest/tools, as documented
    "type": "function",
    "function": {"name": "web_search", "description": "Search the web for information",
                 "parameters": {"type": "object", "properties": {"query": {"description": "What to search for",
                                                                           "type": "string"}},
                                "required": ["query"]}},
}
FIBERS_PATH = "/v1/formulas/moonshot/web-search:latest/fibers"
PER_CALL_COST = (344 * 2.0 + 400 * 0.2 + 256 * 2.0 + 500 * 10.0) / 1e6


def searching_config(*, pricing: dict[str, Any] | None = None, **options: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["web_search"] = {"state": "explicit", "value": SEARCH_TOOL}
    data["adapter_options"]["pricing"].update(pricing or {})
    data["adapter_options"].update(options)
    return ModelConfiguration.model_validate(data)


def tool_calls(round_number: int = 1, *, calls: int = 1, name: str = "web_search") -> dict[str, Any]:
    body = completion("", finish="tool_calls", reasoning=f"search first ({round_number})")
    body["id"] = f"chatcmpl-tools-{round_number}"
    body["choices"][0]["message"]["tool_calls"] = [
        {"id": f"web_search:{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps({"query": f"q{round_number}.{i}",
                                                            "usage": {"total_tokens": 900}})}}
        for i in range(calls)]
    return body


def fiber_for(request: httpx2.Request) -> dict[str, Any]:
    sent = json.loads(request.content)
    return {"id": "fiber-1", "object": "fiber", "status": "succeeded", "formula": "moonshot/web-search:latest",
            "context": {"input": json.dumps(sent),
                        "encrypted_output": f"----MOONSHOT ENCRYPTED BEGIN----{sent['arguments']}----MOONSHOT "
                                            "ENCRYPTED END----"}}


class SearchingKimi(MockKimi):
    """Chat replies follow a script, then answer like MockKimi; fibers succeed unless told otherwise."""

    def __init__(self, script: list[dict[str, Any] | httpx2.Response],
                 fiber: dict[str, Any] | httpx2.Response | None = None) -> None:
        super().__init__(answer="Final answer.")
        self.script, self.fiber = list(script), fiber

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path.startswith("/v1/formulas/"):
            self.requests.append(request)
            assert request.url.path == FIBERS_PATH, request.url
            if isinstance(self.fiber, httpx2.Response):
                return self.fiber
            return httpx2.Response(200, json=self.fiber if self.fiber is not None else fiber_for(request))
        if request.url.path == "/v1/chat/completions" and self.script:
            self.requests.append(request)
            reply = self.script.pop(0)
            return reply if isinstance(reply, httpx2.Response) else httpx2.Response(200, json=reply,
                                                                                     headers=WRITE_HEADERS)
        return super().__call__(request)


def test_web_search_loop_is_sent_verbatim_captured_summed_and_priced(tmp_path: Path) -> None:
    mock = SearchingKimi([tool_calls(calls=2)])
    outcome, files = call(tmp_path, mock, searching_config(pricing={"web_search_usd_per_request": 0.005}))
    assert outcome.succeeded and outcome.extracted.final_answer == "Final answer."
    paths = [r.url.path for r in mock.requests]
    assert paths == ["/v1/chat/completions", FIBERS_PATH, FIBERS_PATH, "/v1/chat/completions"]
    first, last = json.loads(mock.requests[0].content), json.loads(mock.requests[3].content)
    assert first["tools"] == [SEARCH_TOOL] and last["tools"] == [SEARCH_TOOL]
    assert {k: v for k, v in first.items() if k != "tools"} == {
        "model": MODEL, "max_completion_tokens": 16000,
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}]}
    returned = tool_calls(calls=2)["choices"][0]["message"]
    for i, request in enumerate(mock.requests[1:3]):  # the tool call's name and arguments, verbatim
        assert json.loads(request.content) == {"name": "web_search",
                                               "arguments": returned["tool_calls"][i]["function"]["arguments"]}
    results = [fiber_for(r)["context"]["encrypted_output"] for r in mock.requests[1:3]]
    assert last["messages"] == first["messages"] + [returned] + [
        {"role": "tool", "tool_call_id": f"web_search:{i}", "content": results[i]} for i in range(2)]
    assert len(json.loads(files["wire-exchanges.json"])) == 4
    native = json.loads(files["native-response.json"])
    (record,) = native["harness_web_search_rounds"]
    assert record["chat_completion"]["id"] == "chatcmpl-tools-1" and len(record["fibers"]) == 2
    assert native["choices"][0]["message"]["content"] == "Final answer."
    assert files["reasoning.txt"] == "search first (1)\n\npondering"
    obs = outcome.observations
    assert obs["web_search_requests"].value == 2 and obs["web_search_requests"].unit == "requests"
    assert (obs["input_tokens"].value, obs["output_tokens"].value, obs["cache_write_5m_tokens"].value) == (2000, 1000, 512)
    assert obs["api_cost_usd"].value == pytest.approx(2 * PER_CALL_COST + 2 * 0.005)


def test_web_search_without_a_rate_leaves_cost_unknown_and_unused_search_is_free(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, SearchingKimi([tool_calls()]), searching_config())
    assert outcome.observations["web_search_requests"].value == 1
    assert outcome.observations["api_cost_usd"].missing_reason == "no price for web_search_requests"
    mock = SearchingKimi([])  # answers without calling the tool
    outcome, _ = call(tmp_path, mock, searching_config())
    assert len(mock.requests) == 1 and outcome.observations["web_search_requests"].value == 0
    assert outcome.observations["api_cost_usd"].value == pytest.approx(PER_CALL_COST)
    plain = MockKimi()
    outcome, _ = call(tmp_path, plain)
    assert "web_search_requests" not in outcome.observations and "tools" not in json.loads(plain.requests[0].content)


def test_web_search_loop_beyond_the_bound_is_an_incomplete_failure(tmp_path: Path) -> None:
    mock = SearchingKimi([tool_calls(1), tool_calls(2)])
    outcome, files = call(tmp_path, mock, searching_config(web_search_max_rounds=1))
    assert not outcome.succeeded and outcome.failure_kind is FailureKind.PROVIDER_ERROR
    assert outcome.cost_incurred is CostIncurred.YES and '"incomplete"' in str(outcome.error)
    assert [r.url.path for r in mock.requests] == ["/v1/chat/completions", FIBERS_PATH, "/v1/chat/completions"]
    assert len(json.loads(files["wire-exchanges.json"])) == 3
    assert outcome.observations["web_search_requests"].value == 1
    assert outcome.observations["input_tokens"].value == 2000
    assert len(json.loads(files["native-response.json"])["harness_web_search_rounds"]) == 1


@pytest.mark.parametrize(("script", "fiber", "kind", "error_type", "priced"), [
    ([tool_calls(name="other_tool")], None, FailureKind.PROVIDER_ERROR, "unexpected_tool_call", True),
    ([tool_calls()], {"id": "fiber-1", "status": "failed", "context": {}}, FailureKind.PROVIDER_ERROR,
     "tool_error", True),
    ([tool_calls()], error(500, "server_error"), FailureKind.TRANSIENT_TRANSPORT, "tool_http_500", True),
    ([tool_calls(), httpx2.Response(504, text="gateway timeout")], None, FailureKind.AMBIGUOUS_TIMEOUT,
     "http_504", False),
])
def test_web_search_loop_failures_are_recorded_with_the_billed_rounds(
        tmp_path: Path, script: list[Any], fiber: Any, kind: FailureKind, error_type: str, priced: bool) -> None:
    outcome, files = call(tmp_path, SearchingKimi(script, fiber),
                          searching_config(pricing={"web_search_usd_per_request": 0.005}))
    assert outcome.failure_kind is kind and outcome.cost_incurred is CostIncurred.YES
    assert f'"type": "{error_type}"' in str(outcome.error) and "error.json" in files
    assert (outcome.observations["api_cost_usd"].value is not None) is priced


@pytest.mark.parametrize(("value", "options"), [
    ({"type": "builtin_function", "function": {"name": "$web_search"}}, {}),  # retired 2026-10-20
    ({"type": "web_search"}, {}),
    ({"type": "function", "function": {"name": "web_search"}}, {}),
    (SEARCH_TOOL, {"web_search_formula": "moonshot/date:latest"}),
    (SEARCH_TOOL, {"web_search_max_rounds": 0}),
])
def test_web_search_value_must_be_the_formula_tool_declaration(value: dict[str, Any], options: dict[str, Any]) -> None:
    data = searching_config(**options).model_dump(mode="json")
    data["parameters"]["web_search"]["value"] = value
    with pytest.raises(AdapterError) as raised:
        MockKimi().adapter().prepare(ModelConfiguration.model_validate(data), MESSAGES, None, ctx())
    assert raised.value.kind is FailureKind.INVALID_STATE


def test_continuation_after_a_search_replays_the_tool_rounds(tmp_path: Path) -> None:
    config = searching_config()
    mock = SearchingKimi([tool_calls()])
    adapter = mock.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, config, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    final_sent = json.loads(mock.requests[2].content)["messages"]
    follow = snapshot.messages + (Message(role=Role.USER, content="Follow-up"),)
    assert invoke(adapter, config, follow, snapshot, ctx(2), store.sink("child")).succeeded
    sent = json.loads(mock.requests[3].content)
    final = completion("Final answer.")["choices"][0]["message"]
    assert sent["messages"] == final_sent + [final, {"role": "user", "content": "Follow-up"}]
    assert sent["messages"][3]["role"] == "tool" and sent["tools"] == [SEARCH_TOOL]


def test_qualification_with_web_search_accepts_a_searching_baseline() -> None:
    mock = SearchingKimi([tool_calls()])
    report = mock.adapter().qualify(searching_config(), REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert report.probes[0]["cost_usd"] is None  # a search without a configured rate is never priced as free
