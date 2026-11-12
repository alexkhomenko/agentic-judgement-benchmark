"""DeepSeek chat-completions adapter against a mock transport: native request, time-of-day pricing,
failure mapping, exact replay, qualification against /models, and an offline P6 run with a DeepSeek
candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.deepseek_chat import DeepSeekAdapter, _observations, pricing_period
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

API_KEY = "sk-deepseek-test-0123456789abcdefghij"
CONFIG_PATH = ROOT / "configs/models/deepseek/deepseek-flash.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
PRICING: dict[str, Any] = dict(CONFIG.adapter_options["pricing"])  # type: ignore[arg-type]
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)


def completion(text: str = "ok", *, finish: str = "stop", model: str = MODEL, reasoning: str | None = "pondering",
               ) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1_800_000_000, "model": model,
        "system_fingerprint": "fp_v41_flash_1017",
        "choices": [{"index": 0, "finish_reason": finish, "logprobs": None,
                     "message": {"role": "assistant", "content": text, "reasoning_content": reasoning}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
                  "prompt_cache_hit_tokens": 400, "prompt_cache_miss_tokens": 600,
                  "completion_tokens_details": {"reasoning_tokens": 300}},
    }


class MockDeepSeek:
    def __init__(self, *, listed: dict[str, Any] | None = None, answer: str = "1. **DeepSeek V4.1 Flash**: fits.",
                 reply: httpx2.Response | Exception | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.listed = {"id": MODEL, "object": "model", "name": "DeepSeek-V4.1-Flash", "context_window": 1048576,
                       "max_output_tokens": 393216, "effort": {"supported_levels": ["low", "high", "max"],
                                                               "default_level": "high"},
                       **(listed or {})}
        self.answer, self.reply = answer, reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path == "/models":
            return httpx2.Response(200, json={"object": "list", "data": [self.listed]})
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        # DeepSeek pads non-streaming responses with keep-alive blank lines before the JSON.
        return httpx2.Response(200, content=b"\n\n" + json.dumps(completion(text, model=body["model"])).encode(),
                               headers={"content-type": "application/json"})

    def adapter(self, api_key: str | None = API_KEY) -> DeepSeekAdapter:
        return DeepSeekAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, mock: MockDeepSeek, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(mock.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    mock = MockDeepSeek(answer="Use DeepSeek.")
    outcome, files = call(tmp_path, mock)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use DeepSeek."
    (request,) = mock.requests
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "model": MODEL, "max_tokens": 16000,
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
    }  # no thinking toggle, effort, temperature, or tools
    wire = json.loads(files["wire-exchanges.json"])
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert files["reasoning.txt"] == "pondering"
    assert outcome.observations["reasoning_tokens"].value == 300


def test_explicit_parameters_map_to_native_fields(tmp_path: Path) -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update({"thinking": {"state": "explicit", "value": "disabled"},
                               "reasoning_effort": {"state": "explicit", "value": "low"}})
    mock = MockDeepSeek()
    call(tmp_path, mock, ModelConfiguration.model_validate(data))
    body = json.loads(mock.requests[0].content)
    assert body["thinking"] == {"type": "disabled"} and body["reasoning_effort"] == "low"


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


@pytest.mark.parametrize(("when", "pricing", "period"), [
    ("2026-11-10T02:30", {}, "unknown"),  # Tuesday peak hour, holiday list not configured
    ("2026-11-10T02:30", {"holidays_cover_through": "2026-12-31"}, "peak"),
    ("2026-11-10T02:30", {"holidays": ["2026-11-10"], "holidays_cover_through": "2026-12-31"}, "off_peak"),
    ("2026-11-14T02:30", {}, "off_peak"),  # Saturday
    ("2026-11-10T05:00", {}, "off_peak"),  # between the two peak windows
    ("2026-11-10T10:00", {}, "off_peak"),  # windows are end-exclusive
    ("2026-11-10T09:59", {"holidays_cover_through": "2026-12-31"}, "peak"),
])
def test_pricing_period(when: str, pricing: dict[str, Any], period: str) -> None:
    assert pricing_period({**PRICING, **pricing}, at(when)) == period


def test_cost_follows_the_pricing_period() -> None:
    usage = completion()["usage"]
    peak = (400 * 0.006 + 600 * 0.30 + 500 * 1.20) / 1e6
    off = _observations(CONFIG, usage, at("2026-11-14T02:30"))
    assert off["api_cost_usd"].value == pytest.approx(peak * 0.5)
    unknown = _observations(CONFIG, usage, at("2026-11-10T02:30"))
    assert unknown["api_cost_usd"].value is None
    assert unknown["api_cost_usd_if_peak"].value == pytest.approx(peak)


@pytest.mark.parametrize(("reply", "kind", "cost"), [
    (httpx2.Response(402, json={"error": {"message": "Insufficient Balance"}}), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (httpx2.Response(429, json={"error": {"message": "Rate Limit Reached"}}), FailureKind.RATE_LIMITED, CostIncurred.NO),
    (httpx2.Response(503, json={"error": {"message": "Server Overloaded"}}), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
    (httpx2.Response(401, json={"error": {"message": "Authentication Fails"}}), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
    (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
])
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind, cost: CostIncurred) -> None:
    mock = MockDeepSeek(reply=reply)
    outcome, files = call(tmp_path, mock)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(mock.requests) == 1 and "error.json" in files


@pytest.mark.parametrize(("body", "kind"), [
    (completion(finish="content_filter"), FailureKind.REFUSAL),
    (completion(finish="insufficient_system_resource"), FailureKind.TRANSIENT_TRANSPORT),
    (completion(finish="aborted"), FailureKind.PROVIDER_ERROR),
    (completion(model="deepseek-v4-pro"), FailureKind.PROVIDER_ERROR),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind) -> None:
    outcome, _ = call(tmp_path, MockDeepSeek(reply=httpx2.Response(200, json=body)))
    assert outcome.failure_kind is kind and outcome.cost_incurred is CostIncurred.YES


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, MockDeepSeek(reply=httpx2.Response(200, json=completion("partial", finish="length"))))
    assert outcome.succeeded and outcome.extracted.truncated


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MockDeepSeek()
    outcome, _ = call(tmp_path, mock, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and mock.requests == []
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert default_registry().get("deepseek").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("DEEPSEEK_API_KEY", API_KEY)
    assert default_registry().get("deepseek").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_the_message_exactly(tmp_path: Path) -> None:
    mock = MockDeepSeek()
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
    assert snapshot.native_state["system_fingerprint"] == "fp_v41_flash_1017"
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "deepseek-v4-pro"}), follow, snapshot, ctx(2))


def test_qualification_records_the_served_build() -> None:
    report = MockDeepSeek().adapter().qualify(CONFIG, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "served by DeepSeek-V4.1-Flash" in checks["model_identity"].detail
    assert "system_fingerprint fp_v41_flash_1017" in checks["continuation"].detail


@pytest.mark.parametrize(("listed", "expected"), [
    ({"id": "deepseek-other"}, "not listed"),
    ({"context_window": 128000}, "context_limit"),
    ({"max_output_tokens": 8192}, "max_output_tokens"),
])
def test_qualification_rejects_identity_problems(listed: dict[str, Any], expected: str) -> None:
    checks = {c.name: c for c in MockDeepSeek(listed=listed).adapter().qualify(
        CONFIG, REQUIREMENTS.model_copy(update={"requires_continuation": False})).checks}
    assert checks["model_identity"].passed is False and expected in checks["model_identity"].detail


def test_deepseek_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({"deepseek": MockDeepSeek().adapter(), "fake": FakeAdapter(),
                                "fake-judge": FakeJudgeAdapter()})
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-deepseek", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    row = rows["deepseek-flash@1"]
    assert row["own_vendor"] == "deepseek" and row["own_first.k"] == "2"
    assert float(row["own_first_delta"]) == 1.0
