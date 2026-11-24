"""OpenRouter adapter pinned to Alibaba for Qwen4-Max, against a mock transport shaped like OpenRouter's documented
responses (2026-11-20): pinned routing in every request, served-provider and model verification from routing
metadata, exact reasoning replay, errors in 200 bodies, cost, and qualification by the endpoints list plus a
streamed debug echo of the body forwarded to Alibaba. Not yet checked against live traffic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.openrouter_chat import OpenRouterAdapter, _observations
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
from archbench.reporting.report import CostTally, publish_report
from archbench.secrets import GUARD
from archbench.storage.artifacts import ArtifactStore

from .conftest import ROOT, Env
from .test_bias import _read_csv, bias_policy, recommender

API_KEY = "sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef"
CONFIG_PATH = ROOT / "configs/models/alibaba/qwen4-max-openrouter.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
CHAT = "https://openrouter.ai/api/v1/chat/completions"
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)
THOUGHT = "The user wants one model. Qwen fits."


def configured(**changes: Any) -> ModelConfiguration:
    data = CONFIG.model_dump(mode="json")
    for key, value in changes.items():
        if key in ("upstream_model", "endpoint_model", "upstream_expect", "pricing"):
            data["adapter_options"][key] = value
        elif key == "parameters":
            data["parameters"].update(value)
        else:
            data[key] = value
    return ModelConfiguration.model_validate(data)


# The usage chunk OpenRouter sends just before [DONE] (shape observed live 2026-11-20): $0.0035625 at list price.
USAGE_CHUNK = {"id": "gen-2", "object": "chat.completion.chunk", "model": "qwen/qwen4-max", "provider": "Alibaba",
               "choices": [{"index": 0, "delta": {"content": "", "role": "assistant"}, "finish_reason": "stop",
                            "native_finish_reason": "stop"}],
               "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500, "cost": 0.0035625,
                         "is_byok": False,
                         "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                         "completion_tokens_details": {"reasoning_tokens": 300}}}
HIDDEN = "<included to upstream but not in debug info>"  # how OpenRouter shows enable_thinking (observed live)


def metadata(provider: str = "Alibaba", endpoint: str = "qwen/qwen4-max-20261115") -> dict[str, Any]:
    return {"requested": "qwen/qwen4-max", "strategy": "direct", "summary": f"available=1, selected={provider}",
            "attempt": 1, "is_byok": False,
            "endpoints": {"total": 1, "available": [{"provider": provider, "model": endpoint,
                                                     "selected": True}]}}


def completion(text: str = "ok", *, finish: str = "stop", model: str = "qwen/qwen4-max", thought: str | None = THOUGHT,
               reasoning_tokens: int | None = 40, meta: dict[str, Any] | None = None, cached: int = 0,
               cost: float | None = 0.0021) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text, "refusal": None, "reasoning": thought}
    if thought:
        message["reasoning_details"] = [{"type": "reasoning.text", "text": thought, "format": "unknown", "index": 0}]
    usage: dict[str, Any] = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
                             "prompt_tokens_details": {"cached_tokens": cached},
                             "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-1795000000-abcdef", "object": "chat.completion", "created": 1_795_000_000, "model": model,
        "choices": [{"index": 0, "finish_reason": finish, "native_finish_reason": finish, "message": message}],
        "usage": usage, "openrouter_metadata": metadata() if meta is None else meta,
    }


def endpoints(**changes: Any) -> dict[str, Any]:
    endpoint = {
        "name": "Alibaba | qwen/qwen4-max-20261115", "provider_name": "Alibaba", "tag": "alibaba",
        "context_length": 1_000_000, "max_completion_tokens": 131_072, "max_prompt_tokens": None,
        "quantization": "unknown", "status": 0, "supports_implicit_caching": False,
        "pricing": {"prompt": "0.000001425", "completion": "0.000004275", "input_cache_read": "0.00000007125",
                    "input_cache_write": "0.00000178125", "discount": 0.05},
        "supported_parameters": ["reasoning", "include_reasoning", "max_tokens", "temperature", "top_p", "seed",
                                 "presence_penalty", "response_format", "tools", "tool_choice", "structured_outputs",
                                 "logprobs", "top_logprobs", "top_k", "frequency_penalty", "stop",
                                 "reasoning_effort"],
    }
    endpoint.update(changes)
    return {"data": {"id": "qwen/qwen4-max", "name": "Qwen: Qwen4 Max", "endpoints": [endpoint]}}


def default_echo(body: dict[str, Any]) -> dict[str, Any]:
    """Alibaba's OpenAI-compatible body as OpenRouter might forward it (the mapping is what qualification shows)."""
    messages = []
    for m in body["messages"]:
        forwarded = {"role": m["role"], "content": m["content"]}
        if m.get("reasoning"):
            forwarded["reasoning_content"] = m["reasoning"]
        messages.append(forwarded)
    return {"model": "qwen4-max", "messages": messages, "max_tokens": body["max_tokens"],
            "enable_thinking": HIDDEN, "reasoning_effort": body.get("reasoning", {}).get("effort"), "stream": True,
            "stream_options": {"include_usage": True}}


class MockOpenRouter:
    def __init__(self, *, answer: str = "1. **Qwen**: fits.", reply: httpx2.Response | Exception | None = None,
                 listing: dict[str, Any] | None = None, echo: Any = default_echo,
                 thought: str | None = THOUGHT, stream_tail: list[dict[str, Any]] | None = None,
                 stream_reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.answer, self.reply, self.echo, self.thought = answer, reply, echo, thought
        self.stream_tail = [USAGE_CHUNK] if stream_tail is None else stream_tail
        self.stream_reply = stream_reply
        self.listing = listing or endpoints()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.method == "GET" and request.url.path == "/api/v1/models/qwen/qwen4-max/endpoints":
            return httpx2.Response(200, json=self.listing)
        if request.url.path != "/api/v1/chat/completions":
            return httpx2.Response(404, json={"error": {"code": 404, "message": "Not found"}})
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        if body.get("stream") and self.stream_reply is not None:
            if isinstance(self.stream_reply, Exception):
                raise self.stream_reply
            return self.stream_reply
        if body.get("stream"):
            chunks = [{"id": "gen-2", "provider": "Alibaba", "model": "qwen/qwen4-max", "object": "chat.completion.chunk",
                       "choices": [], "debug": {"echo_upstream_body": self.echo(body)}},
                      {"id": "gen-2", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}]},
                      *self.stream_tail]
            stream = ": OPENROUTER PROCESSING\n\n" + "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
            return httpx2.Response(200, text=stream + "data: [DONE]\n\n",
                                   headers={"content-type": "text/event-stream"})
        body_json = completion(text, thought=self.thought, reasoning_tokens=40 if self.thought else 0)
        return httpx2.Response(200, json=body_json, headers={"x-generation-id": "gen-1795000000-abcdef"})

    def adapter(self, api_key: str | None = API_KEY) -> OpenRouterAdapter:
        return OpenRouterAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, mock: MockOpenRouter, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_pinned_to_alibaba_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockOpenRouter(answer="Use Qwen.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Qwen."
    (request,) = mock.requests
    assert str(request.url) == CHAT
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert request.headers["x-openrouter-metadata"] == "enabled"
    assert "http-referer" not in request.headers and "x-openrouter-title" not in request.headers
    assert json.loads(request.content) == {
        "model": "qwen/qwen4-max", "max_tokens": 16000, "reasoning": {"effort": "high"},
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
        "provider": {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True},
    }  # no sampling parameters, no models/route fallbacks, no stream
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert outcome.extracted.reasoning_text == THOUGHT
    assert outcome.observations["reasoning_tokens"].value == 40


def test_explicit_sampling_parameters_are_sent_under_their_own_names(tmp_path: Path) -> None:
    mock = MockOpenRouter()
    config = configured(parameters={"temperature": {"state": "explicit", "value": 0.6},
                                    "reasoning_effort": {"state": "omitted", "documented_default": "high"}})
    call(tmp_path, mock, config)
    sent = json.loads(mock.requests[0].content)
    assert sent["temperature"] == 0.6 and "reasoning" not in sent


@pytest.mark.parametrize(("body", "kind", "error_type"), [
    (completion(meta=metadata("Novita")), FailureKind.PROVIDER_ERROR, "provider_mismatch"),
    (completion(meta={}), FailureKind.PROVIDER_ERROR, "provider_mismatch"),  # nobody named: unverified
    (completion(meta=metadata(endpoint="qwen/qwen4-max-20261201")), FailureKind.PROVIDER_ERROR, "endpoint_mismatch"),
    (completion(model="qwen/qwen3-max"), FailureKind.PROVIDER_ERROR, "model_mismatch"),
    (completion(finish="content_filter"), FailureKind.REFUSAL, "content_filter"),
    (completion(finish="tool_calls"), FailureKind.PROVIDER_ERROR, "finish_reason"),
    (completion(thought=None), FailureKind.PROVIDER_ERROR, "missing_reasoning"),  # reasoned, nothing to replay
    (completion(thought=None, reasoning_tokens=None), FailureKind.PROVIDER_ERROR, "missing_reasoning"),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind,
                                         error_type: str) -> None:
    outcome, _ = call(tmp_path, MockOpenRouter(reply=httpx2.Response(200, json=body)))
    assert outcome.failure_kind is kind and outcome.cost_incurred is CostIncurred.YES
    assert error_type in str(outcome.error)


def test_errors_inside_a_200_body_are_failures_with_uncertain_cost(tmp_path: Path) -> None:
    bare = {"error": {"code": 502, "message": "Provider returned error", "metadata": {"provider_name": "Alibaba"}}}
    outcome, _ = call(tmp_path, MockOpenRouter(reply=httpx2.Response(200, json=bare)))
    assert (outcome.failure_kind, outcome.cost_incurred) == (FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN)
    mid = {**completion("partial", finish="error"), "error": {"code": 429, "message": "Upstream rate limit"}}
    outcome, _ = call(tmp_path, MockOpenRouter(reply=httpx2.Response(200, json=mid)))
    assert (outcome.failure_kind, outcome.cost_incurred) == (FailureKind.RATE_LIMITED, CostIncurred.YES)


def test_truncation_and_zero_reasoning_are_accepted(tmp_path: Path) -> None:
    body = completion("", finish="length", thought=None, reasoning_tokens=0)
    outcome, _ = call(tmp_path, MockOpenRouter(reply=httpx2.Response(200, json=body)))
    assert outcome.succeeded and outcome.extracted.truncated and outcome.extracted.reasoning_text is None


def test_reasoning_text_falls_back_to_reasoning_details(tmp_path: Path) -> None:
    body = completion()
    body["choices"][0]["message"]["reasoning"] = None
    outcome, _ = call(tmp_path, MockOpenRouter(reply=httpx2.Response(200, json=body)))
    assert outcome.succeeded and outcome.extracted.reasoning_text == THOUGHT


def err(status: int, message: str, **metadata: Any) -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"code": status, "message": message, "metadata": metadata}})


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (err(429, "Rate limit exceeded", error_type="rate_limit_exceeded"), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (err(402, "Insufficient credits"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (err(401, "No auth credentials found"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (err(400, "Provider returned error", raw='{"code":"data_inspection_failed"}'), FailureKind.REFUSAL,
     CostIncurred.UNKNOWN),
    (err(403, "Input flagged", reasons=["violence"], flagged_input="..."), FailureKind.REFUSAL, CostIncurred.NO),
    (err(400, "This endpoint's maximum context length is 1000000 tokens"), FailureKind.CONTEXT_EXCEEDED,
     CostIncurred.NO),
    (err(400, "Invalid request"), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (err(502, "Provider returned error"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (err(503, "No allowed providers are available for the selected model"), FailureKind.TRANSIENT_TRANSPORT,
     CostIncurred.NO),
    (err(500, "Internal Server Error"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
    (err(408, "Request timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
    (httpx2.ConnectError("connection refused"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind,
                                              cost: CostIncurred) -> None:
    mock = MockOpenRouter(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


def test_cost_from_configured_prices_with_openrouter_charge_beside_it() -> None:
    obs = _observations(CONFIG, completion(cached=200)["usage"])
    assert obs["uncached_input_tokens"].value == 800 and obs["cache_write_tokens"].value == 0
    assert obs["api_cost_usd"].value == pytest.approx((800 * 1.425 + 200 * 0.07125 + 500 * 4.275) / 1e6)
    assert obs["openrouter_charge"].value == 0.0021 and obs["openrouter_charge"].unit == "credits"
    writes = completion()["usage"]
    writes["prompt_tokens_details"] = {"cached_tokens": 0, "cache_write_tokens": 300}
    assert _observations(CONFIG, writes)["api_cost_usd"].value == pytest.approx(
        (700 * 1.425 + 300 * 1.78125 + 500 * 4.275) / 1e6)
    unknown = completion(cost=None)["usage"]
    del unknown["prompt_tokens_details"]
    obs = _observations(CONFIG, unknown)
    assert obs["api_cost_usd"].value is None and obs["openrouter_charge"].value is None


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockOpenRouter()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert default_registry().get("openrouter").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    assert default_registry().get("openrouter").preflight(CONFIG) is None  # type: ignore[attr-defined]


@pytest.mark.parametrize("config", [configured(provider="openai", model="openai/gpt-6-luna"),
                                    configured(model="qwen/qwen4-max:nitro")])
def test_only_vendor_routes_without_variants_are_prepared(config: ModelConfiguration) -> None:
    with pytest.raises(AdapterError, match="no OpenRouter route"):
        MockOpenRouter().adapter().prepare(config, MESSAGES, None, ctx())


def test_continuation_replays_the_message_with_reasoning_exactly(tmp_path: Path) -> None:
    mock = MockOpenRouter()
    adapter = mock.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None and snapshot.native_state["served_by"] == "Alibaba"
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    first = completion(mock.answer)["choices"][0]["message"]
    for request, word in zip(mock.requests[1:], ("alpha", "beta")):
        sent = json.loads(request.content)
        assert sent["messages"][2] == first  # content, reasoning and reasoning_details, exactly as returned
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in mock.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(configured(model="qwen/qwen4-plus"), follow, snapshot, ctx(2))


def test_qualification_checks_endpoint_and_forwarded_body() -> None:
    mock = MockOpenRouter()
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "Alibaba | qwen/qwen4-max-20261115" in checks["model_identity"].detail
    assert '"reasoning_effort": "high"' in checks["upstream_request"].detail
    assert "reasoning and reasoning_details" in checks["continuation"].detail
    assert [p["step"] for p in report.probes] == ["endpoints", "upstream-baseline", "upstream-echo",
                                                  "baseline", "branch-alpha", "branch-beta"]
    listing = report.probes[0]
    assert listing["cost_incurred"] == "no" and listing["endpoint"]["tag"] == "alibaba"
    streamed_probe = report.probes[2]
    assert streamed_probe["status"] == "completed" and streamed_probe["cost_incurred"] == "yes"
    assert streamed_probe["cost_usd"] == pytest.approx(0.0035625) and streamed_probe["openrouter_charge"] == 0.0035625
    assert streamed_probe["usage"]["reasoning_tokens"] == 300 and streamed_probe["request_id"] == "gen-2"
    tally = CostTally()
    for probe in report.probes:
        tally.add(probe.get("cost_incurred", "unknown"), probe.get("cost_usd"))
    assert tally.unknown == 0 and tally.known > 0.0035625
    echo_probe = next(p for p in report.probes if p["step"] == "upstream-echo")
    assert echo_probe["upstream_body"]["model"] == "qwen4-max"
    assert "enable_thinking is hidden in the echo; the baseline returned reasoning" in checks["upstream_request"].detail
    streamed = json.loads(mock.requests[2].content)
    assert streamed["stream"] is True and streamed["debug"] == {"echo_upstream_body": True}
    assert streamed["max_tokens"] == 16000 and streamed["messages"][2]["reasoning"] == THOUGHT
    assert json.loads(mock.requests[1].content)["max_tokens"] == 2048  # the baseline probe is capped


@pytest.mark.parametrize(("listing", "config", "expected"), [
    (endpoints(name="Alibaba | qwen/qwen4-max-20261201"), CONFIG, "serves 'qwen4-max-20261201'"),
    (endpoints(pricing={"prompt": "0.0000015", "completion": "0.000004275", "input_cache_read": "0.00000007125",
                        "input_cache_write": "0.00000178125"}), CONFIG, "input_usd_per_mtok: configured 1.425"),
    (endpoints(supported_parameters=["max_tokens"]), CONFIG, "does not support ['reasoning']"),
    (endpoints(max_completion_tokens=8192), CONFIG, "16000 > endpoint max_completion_tokens 8192"),
    (endpoints(provider_name="Alibaba Cloud", tag="alibaba"), CONFIG, "not 'Alibaba'"),
    (endpoints(tag="novita"), CONFIG, "found 0"),
    (endpoints(), configured(upstream_model=""), "upstream_model is required"),
    (endpoints(), configured(endpoint_model=""), "endpoint_model is required"),
])
def test_qualification_rejects_endpoint_problems(listing: dict[str, Any], config: ModelConfiguration,
                                                 expected: str) -> None:
    mock = MockOpenRouter(listing=listing)
    report = mock.adapter().qualify(config, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert checks["model_identity"].passed is False and expected in checks["model_identity"].detail, checks
    assert checks["upstream_request"].passed is None and checks["continuation"].passed is None
    assert all(r.method == "GET" for r in mock.requests)  # no paid probe


def test_qualification_rejects_a_second_alibaba_endpoint() -> None:
    listing = endpoints()
    listing["data"]["endpoints"].append({**listing["data"]["endpoints"][0], "tag": "alibaba/sg",
                                         "name": "Alibaba | qwen/qwen4-max-20261115 (sg)"})
    checks = {c.name: c for c in MockOpenRouter(listing=listing).adapter().qualify(CONFIG, REQUIREMENTS).checks}
    assert checks["model_identity"].passed is False and "found 2" in checks["model_identity"].detail


def test_endpoint_pin_selects_one_of_multiple_vendor_routes() -> None:
    config = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options,
                                                          "provider_endpoint": "alibaba/us-east-1"}})
    listing = endpoints()
    listing["data"]["endpoints"].append({**listing["data"]["endpoints"][0], "tag": "alibaba/us-east-1"})
    mock = MockOpenRouter(listing=listing)
    report = mock.adapter().qualify(config, REQUIREMENTS)
    assert report.passed, report.checks
    assert report.probes[0]["endpoint"]["tag"] == "alibaba/us-east-1"
    for request in mock.requests:
        if request.method == "POST":
            assert json.loads(request.content)["provider"] == {
                "only": ["alibaba/us-east-1"], "allow_fallbacks": False, "require_parameters": True}
    missing = MockOpenRouter(listing=endpoints())
    check, _ = missing.adapter()._model_check(config, [])
    assert check.passed is False and "found 0" in check.detail
    assert all(request.method == "GET" for request in missing.requests)


@pytest.mark.parametrize("endpoint", ["novita/us-east-1", "alibaba", "alibaba/", "alibaba/us:fast", 123])
def test_endpoint_pin_cannot_escape_the_vendor(endpoint: Any) -> None:
    config = CONFIG.model_copy(update={"adapter_options": {**CONFIG.adapter_options,
                                                          "provider_endpoint": endpoint}})
    mock = MockOpenRouter()
    check, _ = mock.adapter()._model_check(config, [])
    assert check.passed is False and "provider_endpoint" in check.detail
    assert not mock.requests
    with pytest.raises(AdapterError, match="provider_endpoint"):
        mock.adapter().prepare(config, MESSAGES, None, ctx())


def edit_assistant(**fields: Any) -> Any:
    """An echo whose forwarded assistant message has `fields` replaced (None removes the field)."""
    def echo(body: dict[str, Any]) -> dict[str, Any]:
        forwarded = default_echo(body)
        for m in forwarded["messages"]:
            if m["role"] == "assistant":
                for key, value in fields.items():
                    if value is None:
                        m.pop(key, None)
                    else:
                        m[key] = value
        return forwarded
    return echo


strip_reasoning = edit_assistant(reasoning_content=None)


@pytest.mark.parametrize(("echo", "config", "expected"), [
    (lambda b: {**default_echo(b), "temperature": 0.7}, CONFIG, "temperature omitted in configuration"),
    (lambda b: {**default_echo(b), "max_tokens": 65536}, CONFIG, "max_tokens forwarded as 65536"),
    (lambda b: {k: v for k, v in default_echo(b).items() if k != "max_tokens"}, CONFIG, "no output cap forwarded"),
    (lambda b: {**default_echo(b), "max_completion_tokens": 131072}, CONFIG,
     "max_completion_tokens forwarded as 131072"),
    (lambda b: {**default_echo(b), "model": "qwen4-max-20261115"}, CONFIG, "vendor model 'qwen4-max-20261115'"),
    (lambda b: {**default_echo(b), "enable_search": True}, CONFIG, "enable_search is on"),
    (strip_reasoning, CONFIG, "earlier reasoning is not forwarded"),
    # The reviewer's reproduction: reasoning is present but is not the baseline's.
    (edit_assistant(reasoning_content="Something unrelated."), CONFIG, "differs from the baseline's"),
    (edit_assistant(reasoning_content=THOUGHT[:20]), CONFIG, "differs from the baseline's"),  # truncated
    (edit_assistant(content="different"), CONFIG, "replayed answer forwarded as 'different'"),
    (lambda b: {**default_echo(b), "messages": [m for m in default_echo(b)["messages"] if m["role"] != "assistant"]},
     CONFIG, "found 0"),
    (default_echo, configured(upstream_expect={"thinking_budget": 131072}), "upstream_expect thinking_budget=131072"),
])
def test_qualification_rejects_a_forwarded_body_that_departs_from_the_configuration(
        echo: Any, config: ModelConfiguration, expected: str) -> None:
    report = MockOpenRouter(echo=echo).adapter().qualify(config, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and checks["upstream_request"].passed is False
    assert expected in checks["upstream_request"].detail


OMITTED_EFFORT = {"reasoning_effort": {"state": "omitted", "documented_default": "high"}}
LOW_EFFORT = {"reasoning_effort": {"state": "explicit", "value": "low"}}


def without_effort(body: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in default_echo(body).items() if k != "reasoning_effort"}


@pytest.mark.parametrize(("echo", "config", "expected"), [
    # The reviewer's reproduction: effort and thinking switch both contradict the configured effort "high".
    (lambda b: {**default_echo(b), "reasoning_effort": "low", "enable_thinking": False}, CONFIG,
     "reasoning_effort forwarded as 'low', configuration implies 'high'"),
    (lambda b: {**default_echo(b), "enable_thinking": False}, CONFIG,
     "enable_thinking forwarded as False, configuration implies True"),
    (lambda b: {k: v for k, v in default_echo(b).items() if k not in ("reasoning_effort", "enable_thinking")}, CONFIG,
     "no control carrying the effort level was forwarded (only [])"),
    # The reviewer's reproduction: effort "low" dropped, only the thinking switch survives.
    (lambda b: {**without_effort(b), "enable_thinking": True}, configured(parameters=LOW_EFFORT),
     "reasoning_effort 'low' configured but no control carrying the effort level was forwarded "
     "(only ['enable_thinking'])"),
    (without_effort, CONFIG, "(only ['enable_thinking'])"),  # hidden switch, effort "high" dropped
    (lambda b: {**default_echo(b), "preserve_thinking": False}, CONFIG,
     "preserve_thinking forwarded as False, configuration implies True"),  # its documented default
    (lambda b: {**default_echo(b), "reasoning_effort": "low"}, configured(parameters=OMITTED_EFFORT),
     "reasoning_effort forwarded as 'low', configuration implies 'high'"),  # omitted: the documented default
])
def test_qualification_rejects_forwarded_reasoning_controls_that_contradict_the_configuration(
        echo: Any, config: ModelConfiguration, expected: str) -> None:
    report = MockOpenRouter(echo=echo).adapter().qualify(config, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and checks["upstream_request"].passed is False
    assert expected in checks["upstream_request"].detail


@pytest.mark.parametrize("extra", [{"thinking_budget": 12800}, {"seed": 1234}])
def test_forwarded_values_the_configuration_does_not_determine_leave_qualification_incomplete(
        extra: dict[str, Any]) -> None:
    report = MockOpenRouter(echo=lambda b: {**default_echo(b), **extra}).adapter().qualify(CONFIG, REQUIREMENTS)
    check = next(c for c in report.checks if c.name == "upstream_request")
    assert not report.passed and check.passed is None
    assert "not verified" in check.detail and json.dumps(extra) in check.detail
    pinned = configured(upstream_expect=extra)  # after checking the value against the vendor's docs
    assert MockOpenRouter(echo=lambda b: {**default_echo(b), **extra}).adapter().qualify(pinned, REQUIREMENTS).passed


def test_omitted_effort_qualifies_when_the_forwarded_controls_match_documented_defaults() -> None:
    config = configured(parameters=OMITTED_EFFORT)
    echo = lambda b: {**default_echo(b), "reasoning_effort": "high"}  # noqa: E731
    report = MockOpenRouter(echo=echo).adapter().qualify(config, REQUIREMENTS)
    assert report.passed, report.checks


@pytest.mark.parametrize("forwarded", [
    {"reasoning_content": f"  {THOUGHT}\n"},  # surrounding whitespace only
    {"reasoning_content": None, "reasoning": [{"type": "thinking", "thinking": THOUGHT}]},  # as thinking blocks
])
def test_faithfully_forwarded_reasoning_qualifies_in_other_shapes(forwarded: dict[str, Any]) -> None:
    report = MockOpenRouter(echo=edit_assistant(**forwarded)).adapter().qualify(CONFIG, REQUIREMENTS)
    assert report.passed, report.checks


def test_a_hidden_enable_thinking_contradicted_by_behaviour_fails() -> None:
    # The configuration implies thinking on, the echo hides the switch, and the baseline returned no reasoning.
    report = MockOpenRouter(thought=None).adapter().qualify(CONFIG, REQUIREMENTS)
    check = next(c for c in report.checks if c.name == "upstream_request")
    assert not report.passed and check.passed is False
    assert "enable_thinking is hidden in the echo and the baseline returned no reasoning" in check.detail


def test_other_hidden_values_leave_qualification_incomplete() -> None:
    echo = lambda b: {**default_echo(b), "preserve_thinking": HIDDEN}  # noqa: E731
    report = MockOpenRouter(echo=echo).adapter().qualify(CONFIG, REQUIREMENTS)
    check = next(c for c in report.checks if c.name == "upstream_request")
    assert not report.passed and check.passed is None and "preserve_thinking" in check.detail
    # A pin cannot verify a hidden value: one equal to some real setting leaves it unverified...
    report = MockOpenRouter(echo=echo).adapter().qualify(configured(upstream_expect={"preserve_thinking": True}),
                                                          REQUIREMENTS)
    check = next(c for c in report.checks if c.name == "upstream_request")
    assert not report.passed and check.passed is None and "preserve_thinking" in check.detail


def test_pinning_the_echo_placeholder_does_not_verify_it() -> None:
    # The reviewer's reproduction: echo and pin both carry the placeholder.
    echo = lambda b: {**default_echo(b), "preserve_thinking": HIDDEN}  # noqa: E731
    report = MockOpenRouter(echo=echo).adapter().qualify(configured(upstream_expect={"preserve_thinking": HIDDEN}),
                                                          REQUIREMENTS)
    check = next(c for c in report.checks if c.name == "upstream_request")
    assert not report.passed and check.passed is False
    assert "upstream_expect preserve_thinking pins an echo placeholder" in check.detail


def test_a_pinned_hidden_enable_thinking_is_still_checked_by_behaviour() -> None:
    pinned = configured(upstream_expect={"enable_thinking": True})
    report = MockOpenRouter(thought=None).adapter().qualify(pinned, REQUIREMENTS)
    check = next(c for c in report.checks if c.name == "upstream_request")
    assert not report.passed and check.passed is False and "returned no reasoning" in check.detail
    assert MockOpenRouter().adapter().qualify(pinned, REQUIREMENTS).passed


def test_effort_none_may_be_carried_by_the_thinking_switch_alone() -> None:
    config = configured(parameters={"reasoning_effort": {"state": "explicit", "value": "none"}})
    echo = lambda b: {**without_effort(b), "enable_thinking": False}  # noqa: E731
    report = MockOpenRouter(echo=echo, thought=None).adapter().qualify(config, REQUIREMENTS)
    assert report.passed, report.checks


def test_a_thinking_budget_carries_the_effort_once_pinned() -> None:
    echo = lambda b: {**without_effort(b), "thinking_budget": 12800}  # noqa: E731
    report = MockOpenRouter(echo=echo).adapter().qualify(CONFIG, REQUIREMENTS)
    assert next(c for c in report.checks if c.name == "upstream_request").passed is None  # unverified, not failed
    pinned = configured(upstream_expect={"thinking_budget": 12800})
    assert MockOpenRouter(echo=echo).adapter().qualify(pinned, REQUIREMENTS).passed


def test_the_output_cap_may_be_forwarded_under_max_completion_tokens() -> None:
    def renamed(body: dict[str, Any]) -> dict[str, Any]:
        echo = default_echo(body)
        echo["max_completion_tokens"] = echo.pop("max_tokens")
        return echo

    report = MockOpenRouter(echo=renamed).adapter().qualify(CONFIG, REQUIREMENTS)
    assert report.passed, report.checks


@pytest.mark.parametrize(("mock", "status", "cost", "kind"), [
    (MockOpenRouter(stream_tail=[]), "completed", "unknown", None),  # no usage chunk: billed, amount unknown
    (MockOpenRouter(stream_tail=[{"id": "gen-2", "provider": "Alibaba", "error": {"code": 502, "message": "Upstream"},
                                  "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]},
                                 USAGE_CHUNK]), "failed", "yes", "transient_transport"),
    (MockOpenRouter(stream_reply=err(502, "Provider returned error")), "failed", "no", "transient_transport"),
    (MockOpenRouter(stream_reply=err(408, "Request timed out")), "failed", "unknown", "ambiguous_timeout"),
    (MockOpenRouter(stream_reply=httpx2.ReadTimeout("read timed out")), "failed", "unknown", "transient_transport"),
])
def test_the_streamed_probe_is_always_accounted(mock: MockOpenRouter, status: str, cost: str,
                                               kind: str | None) -> None:
    report = mock.adapter().qualify(CONFIG, REQUIREMENTS)
    probe = next(p for p in report.probes if p["step"] == "upstream-echo")
    assert (probe["status"], probe["cost_incurred"], probe["failure_kind"]) == (status, cost, kind)
    assert (probe["cost_usd"] is not None) == (cost == "yes")


def test_qualification_without_an_echo_is_not_evaluated() -> None:
    report = MockOpenRouter(echo=lambda b: None).adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert not report.passed and checks["upstream_request"].passed is None


def test_unsupported_parameters_are_rejected_without_probing() -> None:
    mock = MockOpenRouter()
    config = configured(parameters={"seed": {"state": "explicit", "value": 7},
                                    "reasoning_effort": {"state": "explicit", "value": "extreme"}})
    report = mock.adapter().qualify(config, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert "seed" in checks["parameter_support"].detail and "extreme" in checks["parameter_support"].detail
    assert checks["upstream_request"].passed is None and checks["continuation"].passed is None
    assert all(r.method == "GET" for r in mock.requests)


def test_qwen_through_openrouter_is_attributed_to_alibaba_in_a_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"openrouter": MockOpenRouter().adapter(), "fake": FakeAdapter(),
                                "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-qwen-or", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["qwen4-max-openrouter@2"]
    assert row["own_vendor"] == "alibaba" and row["own_first.k"] == "2"
