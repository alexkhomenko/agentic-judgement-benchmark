"""OpenRouter adapter pinned to Alibaba for Qwen4-Max, against a mock transport shaped like OpenRouter's documented
responses (2026-11-20): pinned routing in every request, served-provider and model verification from routing
metadata, exact reasoning replay, errors in 200 bodies, cost, and qualification by the endpoints list plus a
streamed debug echo of the body forwarded to Alibaba. Not yet checked against live traffic."""

from __future__ import annotations

import json
import re
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


# ----- Google (Gemini on Google AI Studio) ----------------------------------------------------------------------
# Mock shaped like the public endpoints listing (read 2026-11-25: one standard Google AI Studio endpoint plus its flex
# and priority tiers, and Vertex reported as "Google") and like OpenRouter's documented Gemini responses: reasoning as
# reasoning_details (text, plus thought signatures as reasoning.encrypted, format google-gemini-v1), search usage in
# usage.server_tool_use_details, citations in message.annotations. The generateContent echo is an assumption that
# live qualification confirms.

SEARCH_TOOL = {"type": "openrouter:web_search", "parameters": {"engine": "native"}}
# Inline (tests must not read experiment data): the shape of
# configs/models/agentic-design-bias-v2{,-search}/google-gemini-4-hydrogen-openrouter.yaml.
GOOGLE_DATA: dict[str, Any] = {
    "configuration_id": "v2-google-gemini-4-hydrogen-openrouter", "revision": 1, "track": "cloud",
    "adapter": "openrouter", "provider": "google", "model": "google/gemini-4-hydrogen",
    "profile": "native_recommended", "context_limit": 1_048_576,
    "parameters": {
        "max_output_tokens": {"state": "explicit", "value": 16000},
        "reasoning_effort": {"state": "explicit", "value": "high", "documented_default": "high"},
        "include_thoughts": {"state": "omitted"},
        "temperature": {"state": "omitted"},
    },
    "adapter_options": {
        "endpoint_model": "gemini-4-hydrogen-20261112", "upstream_model": "gemini-4-hydrogen", "upstream_expect": {},
        "timeout_s": 900,
        "pricing": {"input_usd_per_mtok": 1.5, "cached_input_usd_per_mtok": 0.075,
                    "cache_write_usd_per_mtok": 0.0833333333333333, "output_usd_per_mtok": 7.5,
                    "source": "openrouter endpoints listing", "verified_on": "2026-11-25"},
    },
}
GOOGLE = ModelConfiguration.model_validate(GOOGLE_DATA)
GOOGLE_SEARCH = ModelConfiguration.model_validate({
    **GOOGLE_DATA, "configuration_id": "v2-search-google-gemini-4-hydrogen-openrouter",
    "parameters": {**GOOGLE_DATA["parameters"], "web_search": {"state": "explicit", "value": SEARCH_TOOL}},
    "adapter_options": {**GOOGLE_DATA["adapter_options"],
                        "pricing": {**GOOGLE_DATA["adapter_options"]["pricing"], "web_search_usd_per_request": 0.014}},
})
GTHOUGHT = "**Choosing a model** The user wants one pick."
SIGNATURE = "CsUBAY89a1+thought-signature=="
CITATION = {"type": "url_citation", "url_citation": {
    "url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc", "title": "example.com",
    "content": "x", "start_index": 0, "end_index": 4}}


def google_configured(base: ModelConfiguration = GOOGLE, **changes: Any) -> ModelConfiguration:
    data = base.model_dump(mode="json")
    for key, value in changes.items():
        if key in ("upstream_model", "endpoint_model", "upstream_expect", "provider_endpoint"):
            data["adapter_options"][key] = value
        elif key == "pricing":
            data["adapter_options"]["pricing"].update(value)
        elif key == "parameters":
            data["parameters"].update(value)
        else:
            data[key] = value
    return ModelConfiguration.model_validate(data)


def google_endpoint(tag: str = "google-ai-studio", tier: str | None = None, **changes: Any) -> dict[str, Any]:
    endpoint = {
        "name": "Google AI Studio | google/gemini-4-hydrogen-20261112", "provider_name": "Google AI Studio",
        "tag": tag, "service_tier": tier, "context_length": 1_048_576, "max_completion_tokens": 65_536,
        "quantization": "unknown", "status": 0, "supports_implicit_caching": False,
        "pricing": {"prompt": "0.0000015", "completion": "0.0000075", "internal_reasoning": "0.0000075",
                    "input_cache_read": "0.000000075", "input_cache_write": "0.0000000833333333333333",
                    "web_search": "0.014", "discount": 0},
        "supported_parameters": ["reasoning", "include_reasoning", "max_tokens", "temperature", "top_p", "seed",
                                 "response_format", "tools", "tool_choice", "structured_outputs", "reasoning_effort"],
        "native_tools": {"openrouter:web_search": {"type": "google_search"}},
    }
    endpoint.update(changes)
    return endpoint


def google_endpoints(*extra: dict[str, Any], **changes: Any) -> dict[str, Any]:
    cheap = {"prompt": "0.00000075", "completion": "0.00000375", "web_search": "0.014"}
    listed = [google_endpoint(tag="google-ai-studio/flex", tier="flex", pricing=cheap),
              google_endpoint(**changes),
              google_endpoint(tag="google-ai-studio/priority", tier="priority"),
              google_endpoint(tag="google-vertex/global", provider_name="Google",
                              name="Google | google/gemini-4-hydrogen-20261112"),
              *extra]
    return {"data": {"id": "google/gemini-4-hydrogen", "name": "Google: Gemini 4 Hydrogen", "endpoints": listed}}


def google_metadata(provider: str = "Google AI Studio", pipeline: list[dict[str, Any]] | None = None
                    ) -> dict[str, Any]:
    meta = metadata(provider, "google/gemini-4-hydrogen-20261112")
    meta["requested"] = "google/gemini-4-hydrogen"
    if pipeline is not None:
        meta["pipeline"] = pipeline
    return meta


def gcompletion(text: str = "ok", *, thought: str | None = GTHOUGHT, signature: str | None = SIGNATURE,
                reasoning_tokens: int | None = 40, searches: int | None = None, annotations: bool = False,
                finish: str = "stop", native: str = "STOP", tier: str | None = "default",
                meta: dict[str, Any] | None = None, orchestrated: int | None = None,
                cost: float | None = 0.0048) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text, "refusal": None, "reasoning": thought}
    details = []
    if thought:
        details.append({"type": "reasoning.text", "text": thought, "format": "google-gemini-v1", "index": 0})
    if signature:
        details.append({"type": "reasoning.encrypted", "data": signature, "id": "tool_0",
                        "format": "google-gemini-v1", "index": 1})
    if details:
        message["reasoning_details"] = details
    if annotations:
        message["annotations"] = [CITATION]
    usage: dict[str, Any] = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
                             "prompt_tokens_details": {"cached_tokens": 0},
                             "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}
    if cost is not None:
        usage["cost"] = cost
    if searches is not None or orchestrated is not None:
        usage["server_tool_use_details"] = {
            k: v for k, v in (("web_search_requests", searches), ("tool_calls_requested", orchestrated))
            if v is not None}
    body = {"id": "gen-1795000000-google", "object": "chat.completion", "created": 1_795_000_000,
            "model": "google/gemini-4-hydrogen", "service_tier": tier,
            "choices": [{"index": 0, "finish_reason": finish, "native_finish_reason": native, "message": message}],
            "usage": usage, "openrouter_metadata": google_metadata() if meta is None else meta}
    if tier is None:
        del body["service_tier"]
    return body


def gemini_echo(body: dict[str, Any]) -> dict[str, Any]:
    """The generateContent body OpenRouter would forward to Google AI Studio for this request."""
    contents: list[dict[str, Any]] = []
    system = None
    for m in body["messages"]:
        if m["role"] == "system":
            system = {"parts": [{"text": m["content"]}]}
        elif m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        else:
            parts: list[dict[str, Any]] = []
            if m.get("reasoning"):
                parts.append({"text": m["reasoning"], "thought": True})
            answer: dict[str, Any] = {"text": m["content"]}
            signatures = [d["data"] for d in m.get("reasoning_details") or [] if d["type"] == "reasoning.encrypted"]
            if signatures:
                answer["thoughtSignature"] = signatures[0]
            parts.append(answer)
            contents.append({"role": "model", "parts": parts})
    generation: dict[str, Any] = {"maxOutputTokens": body["max_tokens"]}
    if "reasoning" in body:
        generation["thinkingConfig"] = {"thinkingLevel": body["reasoning"]["effort"]}
    echo: dict[str, Any] = {"model": "gemini-4-hydrogen", "contents": contents, "generationConfig": generation}
    if system:
        echo["systemInstruction"] = system
    if body.get("tools"):
        echo["tools"] = [{"googleSearch": {}}]
    return echo


class MockGoogle(MockOpenRouter):
    def __init__(self, *, searches: int | None = None, annotations: bool = False, signature: str | None = SIGNATURE,
                 **kwargs: Any) -> None:
        kwargs.setdefault("echo", gemini_echo)
        kwargs.setdefault("listing", google_endpoints())
        kwargs.setdefault("thought", GTHOUGHT)
        super().__init__(**kwargs)
        self.searches, self.annotations, self.signature = searches, annotations, signature

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            self.requests.append(request)
            if request.url.path == "/api/v1/models/google/gemini-4-hydrogen/endpoints":
                return httpx2.Response(200, json=self.listing)
            return httpx2.Response(404, json={"error": {"code": 404, "message": "Not found"}})
        body = json.loads(request.content)
        if body.get("stream") or self.reply is not None:
            response = super().__call__(request)
            if body.get("stream") and self.reply is None and self.stream_reply is None:  # name Google, not Alibaba
                response = httpx2.Response(200, text=response.text.replace('"Alibaba"', '"Google AI Studio"'),
                                           headers={"content-type": "text/event-stream"})
            return response
        self.requests.append(request)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if "reply with exactly one word" in last.lower() else self.answer
        searching = "tools" in body
        return httpx2.Response(200, json=gcompletion(
            text, thought=self.thought, signature=self.signature, reasoning_tokens=40 if self.thought else 0,
            searches=self.searches if searching else None, annotations=self.annotations and searching),
            headers={"x-generation-id": "gen-1795000000-google"})


def test_google_request_is_pinned_to_google_ai_studio_with_the_native_default_effort(tmp_path: Path) -> None:
    mock = MockGoogle(answer="Use Gemini.")
    outcome, _ = call(tmp_path, mock, GOOGLE)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Gemini."
    assert json.loads(mock.requests[0].content) == {
        "model": "google/gemini-4-hydrogen", "max_tokens": 16000, "reasoning": {"effort": "high"},
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
        "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False, "require_parameters": True},
    }  # no tools, no sampling, no service tier: the standard AI Studio endpoint only
    assert "web_search_requests" not in outcome.observations
    assert outcome.extracted.reasoning_text == GTHOUGHT


def test_alibaba_route_is_unchanged_and_cannot_search() -> None:
    from archbench.adapters.cloud.openrouter_chat import ROUTES
    assert (ROUTES["alibaba"].slug, ROUTES["alibaba"].name) == ("alibaba", "Alibaba")
    assert (ROUTES["google"].slug, ROUTES["google"].name) == ("google-ai-studio", "Google AI Studio")
    searching = configured(parameters={"web_search": {"state": "explicit", "value": SEARCH_TOOL}})
    with pytest.raises(AdapterError, match="no native web search") as raised:
        MockOpenRouter().adapter().prepare(searching, MESSAGES, None, ctx())
    assert raised.value.kind is FailureKind.INVALID_STATE
    mock = MockOpenRouter()
    checks = {c.name: c for c in mock.adapter().qualify(searching, REQUIREMENTS).checks}
    assert checks["parameter_support"].passed is False and all(r.method == "GET" for r in mock.requests)


@pytest.mark.parametrize(("body", "error_type"), [
    (gcompletion(meta=google_metadata("Google")), "provider_mismatch"),  # Vertex
    (gcompletion(tier="flex"), "tier_mismatch"),
    (gcompletion(tier="priority"), "tier_mismatch"),
])
def test_google_responses_from_another_service_or_tier_are_failures(tmp_path: Path, body: dict[str, Any],
                                                                    error_type: str) -> None:
    outcome, _ = call(tmp_path, MockGoogle(reply=httpx2.Response(200, json=body)), GOOGLE)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and error_type in str(outcome.error)


def test_google_tier_is_accepted_when_unreported_and_signatures_alone_are_reasoning(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockGoogle(reply=httpx2.Response(200, json=gcompletion(tier=None))), GOOGLE)
    assert outcome.succeeded
    signed_only = gcompletion(thought=None)  # reasoning tokens, no text, but a replayable thought signature
    outcome, _ = call(tmp_path, MockGoogle(reply=httpx2.Response(200, json=signed_only)), GOOGLE)
    assert outcome.succeeded and outcome.extracted.reasoning_text is None
    neither = gcompletion(thought=None, signature=None)
    outcome, _ = call(tmp_path, MockGoogle(reply=httpx2.Response(200, json=neither)), GOOGLE)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and "missing_reasoning" in str(outcome.error)


@pytest.mark.parametrize("provider_endpoint", ["google-ai-studio/flex", "google-ai-studio/priority",
                                               "google-vertex/global"])
def test_google_cannot_be_pinned_to_a_tier_or_another_service(provider_endpoint: str) -> None:
    with pytest.raises(AdapterError, match="provider_endpoint"):
        MockGoogle().adapter().prepare(google_configured(provider_endpoint=provider_endpoint), MESSAGES, None, ctx())


def test_google_web_search_tool_is_sent_verbatim_and_searches_are_counted_and_priced(tmp_path: Path) -> None:
    mock = MockGoogle(answer="Gemini, per [1].", searches=2, annotations=True)
    outcome, files = call(tmp_path, mock, GOOGLE_SEARCH)
    assert json.loads(mock.requests[0].content)["tools"] == [SEARCH_TOOL]
    assert outcome.succeeded and outcome.extracted.final_answer == "Gemini, per [1]."  # annotations stay native
    assert json.loads(files["native-response.json"])["choices"][0]["message"]["annotations"] == [CITATION]
    obs = outcome.observations
    assert obs["web_search_requests"].value == 2 and obs["web_search_requests"].unit == "requests"
    assert "server_tool_use_details.web_search_requests" in obs["web_search_requests"].source
    assert obs["api_cost_usd"].value == pytest.approx((1000 * 1.5 + 500 * 7.5) / 1e6 + 2 * 0.014)
    assert obs["openrouter_charge"].value == 0.0048  # recorded beside, includes server tool charges
    unpriced = google_configured(GOOGLE_SEARCH, pricing={"web_search_usd_per_request": None})
    outcome, _ = call(tmp_path, MockGoogle(searches=2), unpriced)
    assert outcome.observations["api_cost_usd"].missing_reason == "no price for web_search_requests"
    outcome, _ = call(tmp_path, MockGoogle(searches=None), GOOGLE_SEARCH)  # offered but unused
    assert outcome.observations["web_search_requests"].value == 0
    assert outcome.observations["api_cost_usd"].value == pytest.approx((1000 * 1.5 + 500 * 7.5) / 1e6)
    outcome, _ = call(tmp_path, MockGoogle(searches=None, annotations=True), GOOGLE_SEARCH)  # cited, not counted
    assert outcome.observations["web_search_requests"].value is None
    assert outcome.observations["api_cost_usd"].missing_reason == "web_search_requests not reported"


@pytest.mark.parametrize("body", [
    gcompletion("", finish="stop", searches=3),  # loop ended without answer text
    gcompletion("partial", finish="stop", native="TOO_MANY_TOOL_CALLS", searches=3),
    gcompletion("partial", finish="tool_calls", native="STOP", searches=3),
    gcompletion("partial", finish="error", native="MALFORMED_FUNCTION_CALL", searches=3),
])
def test_google_incomplete_server_tool_loop_is_a_failure(tmp_path: Path, body: dict[str, Any]) -> None:
    outcome, _ = call(tmp_path, MockGoogle(reply=httpx2.Response(200, json=body)), GOOGLE_SEARCH)
    if body["choices"][0]["finish_reason"] != "error":
        assert json.loads(str(outcome.error))["type"] == "incomplete"
    assert not outcome.succeeded and outcome.cost_incurred is CostIncurred.YES


@pytest.mark.parametrize("body", [
    gcompletion("Gemini.", searches=2, orchestrated=2),  # OpenRouter ran the search itself (Exa or another engine)
    gcompletion("Gemini.", searches=1, meta=google_metadata(pipeline=[
        {"type": "server_tools", "name": "server-tools", "data": {"mode": "sdk", "tools": ["openrouter:web_search"]}}])),
])
def test_google_search_not_run_natively_is_a_failure(tmp_path: Path, body: dict[str, Any]) -> None:
    outcome, _ = call(tmp_path, MockGoogle(reply=httpx2.Response(200, json=body)), GOOGLE_SEARCH)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and "search_engine_mismatch" in str(outcome.error)
    native = gcompletion("Gemini.", searches=1, meta=google_metadata(pipeline=[
        {"type": "server_tools", "name": "server-tools", "data": {"mode": "native"}}]))
    assert call(tmp_path, MockGoogle(reply=httpx2.Response(200, json=native)), GOOGLE_SEARCH)[0].succeeded


@pytest.mark.parametrize("value", [
    {"type": "openrouter:web_search"},  # engine-less: auto, falls back to Exa
    {"type": "openrouter:web_search", "parameters": {}},
    {"type": "openrouter:web_search", "parameters": {"engine": "auto"}},
    {"type": "openrouter:web_search", "parameters": {"engine": "exa"}},
    {"type": "openrouter:web_search", "parameters": {"engine": "native", "allowed_domains": ["example.com"]}},
    {"type": "openrouter:web_search", "parameters": {"engine": "native", "max_results": 5}},
    {"type": "web_search_preview"},
    {"google_search": {}},  # the direct adapter's tool is not an OpenRouter tool
    "openrouter:web_search",
])
def test_google_web_search_must_pin_the_native_engine(value: Any) -> None:
    config = google_configured(GOOGLE_SEARCH, parameters={"web_search": {"state": "explicit", "value": value}})
    with pytest.raises(AdapterError) as raised:
        MockGoogle().adapter().prepare(config, MESSAGES, None, ctx())
    assert raised.value.kind is FailureKind.INVALID_STATE
    mock = MockGoogle()
    checks = {c.name: c for c in mock.adapter().qualify(config, REQUIREMENTS).checks}
    assert checks["parameter_support"].passed is False and "web_search" in checks["parameter_support"].detail
    assert checks["upstream_request"].passed is None and all(r.method == "GET" for r in mock.requests)


@pytest.mark.parametrize("effort", ["none", "max"])
def test_google_efforts_that_are_not_thinking_levels_are_rejected(effort: str) -> None:
    config = google_configured(parameters={"reasoning_effort": {"state": "explicit", "value": effort}})
    check = next(c for c in MockGoogle().adapter().qualify(config, REQUIREMENTS).checks
                 if c.name == "parameter_support")
    assert check.passed is False and effort in check.detail


def test_google_continuation_replays_signatures_and_search_annotations_exactly(tmp_path: Path) -> None:
    mock = MockGoogle(searches=1, annotations=True)
    adapter = mock.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, GOOGLE_SEARCH, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None and snapshot.native_state["served_by"] == "Google AI Studio"
    follow = snapshot.messages + (Message(role=Role.USER, content="Follow-up"),)
    assert invoke(adapter, GOOGLE_SEARCH, follow, snapshot, ctx(2), store.sink("child")).succeeded
    first = gcompletion(mock.answer, annotations=True)["choices"][0]["message"]
    assert json.loads(mock.requests[1].content)["messages"][2] == first  # signatures and annotations included


@pytest.mark.parametrize("config", [GOOGLE, GOOGLE_SEARCH])
def test_google_qualification_checks_the_standard_ai_studio_endpoint_and_the_forwarded_gemini_body(
        config: ModelConfiguration) -> None:
    mock = MockGoogle(searches=1, annotations=True)
    report = mock.adapter().qualify(config, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert report.probes[0]["endpoint"]["tag"] == "google-ai-studio"  # not flex, priority or Vertex
    detail = checks["upstream_request"].detail
    assert "forwarded to Google AI Studio" in detail and '"thinkingLevel": "high"' in detail
    assert "reasoning and reasoning_details" in checks["continuation"].detail
    echo_probe = next(p for p in report.probes if p["step"] == "upstream-echo")
    assert echo_probe["upstream_body"]["generationConfig"]["maxOutputTokens"] == 16000
    baseline = json.loads(mock.requests[1].content)
    if config is GOOGLE_SEARCH:
        assert "runs natively as google_search" in checks["model_identity"].detail
        assert "Google Search grounding forwarded" in detail and "replayed message carried 1 annotations" in detail
        assert baseline["messages"][-1]["content"].startswith("Search the web")
    else:
        assert "googleSearch" not in detail and baseline["messages"][-1]["content"].startswith("Reply with")


def without(path: str) -> Any:
    def echo(body: dict[str, Any]) -> dict[str, Any]:
        forwarded = gemini_echo(body)
        *parents, last = path.split(".")
        target = forwarded
        for key in parents:
            target = target[key]
        target.pop(last, None)
        return forwarded
    return echo


def with_values(**values: Any) -> Any:
    """gemini_echo with dotted paths (written with __ for dots) set to values."""
    def echo(body: dict[str, Any]) -> dict[str, Any]:
        forwarded = gemini_echo(body)
        for dotted, value in values.items():
            *parents, last = dotted.split("__")
            target = forwarded
            for key in parents:
                target = target.setdefault(key, {})
            target[last] = value
        return forwarded
    return echo


def model_parts(**changes: Any) -> Any:
    def echo(body: dict[str, Any]) -> dict[str, Any]:
        forwarded = gemini_echo(body)
        for content in forwarded["contents"]:
            if content["role"] == "model":
                for part in content["parts"]:
                    for key, value in changes.items():
                        if value is None:
                            part.pop(key, None)
                        elif key in part:
                            part[key] = value
        return forwarded
    return echo


@pytest.mark.parametrize(("echo", "config", "expected"), [
    (without("tools"), GOOGLE_SEARCH, "web_search configured but 0 Google Search tools"),
    (with_values(tools=[{"googleSearch": {}}]), GOOGLE, "Google Search tool forwarded although web_search is not"),
    (with_values(tools=[{"functionDeclarations": [{"name": "web_search"}]}]), GOOGLE_SEARCH,
     "tools forwarded that the configuration does not request"),  # an OpenRouter-run search offered as a function
    (with_values(tools=[{"googleSearchRetrieval": {}}]), GOOGLE_SEARCH, "0 Google Search tools"),
    (with_values(generationConfig__maxOutputTokens=65536), GOOGLE, "maxOutputTokens forwarded as 65536"),
    (without("generationConfig.maxOutputTokens"), GOOGLE, "no output cap forwarded"),
    (with_values(generationConfig__temperature=1.0), GOOGLE, "temperature omitted in configuration"),
    (with_values(generationConfig__topP=0.95), GOOGLE, "top_p omitted in configuration"),
    (with_values(generationConfig__seed=0), GOOGLE, "generationConfig.seed=0 forwarded"),
    (with_values(generationConfig__candidateCount=2), GOOGLE, "generationConfig.candidateCount=2 forwarded"),
    (with_values(generationConfig__thinkingConfig={"thinkingLevel": "medium"}), GOOGLE,
     "thinkingLevel forwarded as 'medium', configuration implies 'high'"),
    (with_values(generationConfig__thinkingConfig={"thinkingBudget": 24576}), GOOGLE,
     "reasoning_effort 'high' configured but no generationConfig.thinkingConfig.thinkingLevel"),
    (with_values(model="gemini-4-hydrogen-preview"), GOOGLE, "vendor model 'gemini-4-hydrogen-preview'"),
    (model_parts(thoughtSignature=None), GOOGLE, "thought signatures the baseline returned are not forwarded"),
    (model_parts(thoughtSignature="CsUBAY89a1+other=="), GOOGLE, "1 of 1 baseline thought signatures"),
    (model_parts(text="different"), GOOGLE, "replayed answer forwarded as"),
    (with_values(systemInstruction={"parts": [{"text": "Be brief."}]}), GOOGLE, "system instruction forwarded"),
])
def test_google_qualification_rejects_a_forwarded_body_that_departs_from_the_configuration(
        echo: Any, config: ModelConfiguration, expected: str) -> None:
    report = MockGoogle(echo=echo).adapter().qualify(config, REQUIREMENTS)
    check = next(c for c in report.checks if c.name == "upstream_request")
    assert not report.passed and check.passed is False and expected in check.detail, check.detail


def test_google_default_candidate_count_and_openrouter_labels_need_no_pin() -> None:
    # Seen on every live forwarded body 2026-11-25: candidateCount 1 (Google's default) and a safety_identifier label.
    echo = with_values(generationConfig__candidateCount=1, labels={"safety_identifier": "abc123"})
    report = MockGoogle(echo=echo).adapter().qualify(GOOGLE, REQUIREMENTS)
    assert report.passed, [c for c in report.checks if not c.passed]


@pytest.mark.parametrize(("extra", "pin"), [
    ({"generationConfig__thinkingConfig": {"thinkingLevel": "high", "includeThoughts": True}},
     {"generationConfig.thinkingConfig.includeThoughts": True}),
    ({"safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}]},
     {"safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}]}),
    ({"tools": [{"googleSearch": {"timeRangeFilter": {}}}]}, {"tools": [{"googleSearch": {"timeRangeFilter": {}}}]}),
])
def test_google_forwarded_values_the_configuration_does_not_determine_leave_qualification_incomplete(
        extra: dict[str, Any], pin: dict[str, Any]) -> None:
    config = GOOGLE_SEARCH
    report = MockGoogle(echo=with_values(**extra)).adapter().qualify(config, REQUIREMENTS)
    check = next(c for c in report.checks if c.name == "upstream_request")
    assert not report.passed and check.passed is None and "not verified" in check.detail
    assert next(iter(pin)) in check.detail
    pinned = google_configured(config, upstream_expect=pin)
    report = MockGoogle(echo=with_values(**extra)).adapter().qualify(pinned, REQUIREMENTS)
    assert report.passed, report.checks
    wrong = google_configured(config, upstream_expect={next(iter(pin)): "something else"})
    check = next(c for c in MockGoogle(echo=with_values(**extra)).adapter().qualify(wrong, REQUIREMENTS).checks
                 if c.name == "upstream_request")
    assert check.passed is False and "upstream_expect" in check.detail


def test_google_echo_without_a_model_id_or_with_redacted_signatures_still_verifies() -> None:
    # Live echoes never carry the model (it is in Google's URL); the pinned dated endpoint identifies it.
    check = next(c for c in MockGoogle(echo=without("model")).adapter().qualify(GOOGLE, REQUIREMENTS).checks
                 if c.name == "upstream_request")
    assert check.passed is True and "identified by the pinned endpoint" in check.detail
    check = next(c for c in MockGoogle(echo=model_parts(thoughtSignature=HIDDEN)).adapter()
                 .qualify(GOOGLE, REQUIREMENTS).checks if c.name == "upstream_request")
    assert check.passed is True and "redacted in OpenRouter's debug echo" in check.detail
    doubled = model_parts(thoughtSignature=HIDDEN)

    def extra_signature(body: dict[str, Any]) -> dict[str, Any]:
        forwarded = doubled(body)
        model = next(c for c in forwarded["contents"] if c["role"] == "model")
        model["parts"].append({"text": "", "thoughtSignature": HIDDEN})
        return forwarded
    check = next(c for c in MockGoogle(echo=extra_signature).adapter()
                 .qualify(GOOGLE, REQUIREMENTS).checks if c.name == "upstream_request")
    assert check.passed is False and "2 thought signatures forwarded (values redacted)" in check.detail


def test_google_omitted_effort_qualifies_with_no_level_or_the_documented_default() -> None:
    omitted = google_configured(parameters={"reasoning_effort": {"state": "omitted", "documented_default": "high"}})
    assert MockGoogle().adapter().qualify(omitted, REQUIREMENTS).passed  # no thinkingConfig: Google's default
    report = MockGoogle(echo=with_values(generationConfig__thinkingConfig={"thinkingLevel": "high"})).adapter() \
        .qualify(omitted, REQUIREMENTS)
    assert report.passed, report.checks
    report = MockGoogle(echo=with_values(generationConfig__thinkingConfig={"thinkingLevel": "medium"})).adapter() \
        .qualify(omitted, REQUIREMENTS)  # OpenRouter's catalog default_effort for Hydrogen, not Google's
    assert next(c for c in report.checks if c.name == "upstream_request").passed is False


def test_google_echo_that_is_not_a_generate_content_body_is_not_evaluated() -> None:
    report = MockGoogle(echo=default_echo).adapter().qualify(GOOGLE, REQUIREMENTS)  # an OpenAI-style body
    check = next(c for c in report.checks if c.name == "upstream_request")
    assert check.passed is None and "not a Gemini generateContent body" in check.detail


@pytest.mark.parametrize(("listing", "config", "expected"), [
    (google_endpoints(google_endpoint(tag="google-ai-studio/us")), GOOGLE, "found 2"),
    ({"data": {"id": "google/gemini-4-hydrogen", "endpoints": [google_endpoint(tag="google-ai-studio/flex",
                                                                               tier="flex")]}},
     GOOGLE, "found 0"),  # only a tier endpoint: never served by a base-slug request
    (google_endpoints(native_tools={}), GOOGLE_SEARCH, "searches would use another engine"),
    (google_endpoints(native_tools={"openrouter:web_search": {"type": "exa"}}), GOOGLE_SEARCH, "'exa'"),
    (google_endpoints(pricing={**google_endpoint()["pricing"], "web_search": "0.035"}), GOOGLE_SEARCH,
     "web_search_usd_per_request: configured 0.014, OpenRouter 0.035"),
    (google_endpoints(pricing={**google_endpoint()["pricing"], "internal_reasoning": "0.00001"}), GOOGLE,
     "reasoning is costed as output"),
    (google_endpoints(supported_parameters=["reasoning", "max_tokens"]), GOOGLE_SEARCH,
     "does not support ['tools']"),
])
def test_google_qualification_rejects_endpoint_problems(listing: dict[str, Any], config: ModelConfiguration,
                                                        expected: str) -> None:
    mock = MockGoogle(listing=listing)
    checks = {c.name: c for c in mock.adapter().qualify(config, REQUIREMENTS).checks}
    assert checks["model_identity"].passed is False and expected in checks["model_identity"].detail, checks
    assert all(r.method == "GET" for r in mock.requests)  # no paid probe



def snake_case(body: dict[str, Any]) -> dict[str, Any]:
    """gemini_echo spelled the other way Gemini's REST API accepts."""
    def snake(value: Any) -> Any:
        if isinstance(value, dict):
            return {re.sub(r"([A-Z])", lambda m: "_" + m.group(1).lower(), k): snake(v) for k, v in value.items()}
        return [snake(v) for v in value] if isinstance(value, list) else value
    return snake(gemini_echo(body))  # type: ignore[no-any-return]


def test_google_echo_in_snake_case_is_checked_the_same_way() -> None:
    report = MockGoogle(echo=snake_case, searches=1).adapter().qualify(GOOGLE_SEARCH, REQUIREMENTS)
    assert report.passed, report.checks
    echo_probe = next(p for p in report.probes if p["step"] == "upstream-echo")
    assert "generation_config" in echo_probe["upstream_body"]  # recorded as received


def test_google_forwarded_thought_text_must_be_the_baselines() -> None:
    def split_thought(body: dict[str, Any]) -> dict[str, Any]:
        forwarded = gemini_echo(body)
        for content in forwarded["contents"]:
            if content["role"] == "model":
                head, tail = GTHOUGHT[:10], GTHOUGHT[10:]
                content["parts"][0:1] = [{"text": head, "thought": True}, {"text": tail + "\n", "thought": True}]
        return forwarded

    assert MockGoogle(echo=split_thought).adapter().qualify(GOOGLE, REQUIREMENTS).passed

    def other_thought(body: dict[str, Any]) -> dict[str, Any]:
        forwarded = gemini_echo(body)
        for content in forwarded["contents"]:
            if content["role"] == "model":
                content["parts"][0]["text"] = "Something unrelated."
        return forwarded

    check = next(c for c in MockGoogle(echo=other_thought).adapter().qualify(GOOGLE, REQUIREMENTS).checks
                 if c.name == "upstream_request")
    assert check.passed is False and "forwarded thought text" in check.detail

    def no_thought_text(body: dict[str, Any]) -> dict[str, Any]:  # signatures alone carry Gemini's reasoning
        forwarded = gemini_echo(body)
        for content in forwarded["contents"]:
            if content["role"] == "model":
                content["parts"] = [p for p in content["parts"] if not p.get("thought")]
        return forwarded

    assert MockGoogle(echo=no_thought_text).adapter().qualify(GOOGLE, REQUIREMENTS).passed
