"""Mistral chat-completions adapter against a mock transport: wire capture, redaction, no retries, failure
mapping, cost, stateless continuation with signed thinking chunks, qualification, credentials, and an
offline P6 run with a Mistral candidate next to a candidate from another vendor."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.mistral_chat import MistralAdapter
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

API_KEY = "mistral-test-key-0123456789abcdef"
CONFIG_PATH = ROOT / "configs/models/mistral/mistral-small-2603.yaml"
CONFIG = ModelConfiguration.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
MODEL = CONFIG.model
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Pick a model."))
Handler = Callable[[httpx2.Request], httpx2.Response]


def completion(text: str = "ok", *, finish: str = "stop", model: str = MODEL, thinking: str | None = "pondering",
               signature: str | None = "sig-xyz", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    if thinking is None:
        content: Any = text  # reasoning off: content is a plain string
    else:
        chunk: dict[str, Any] = {"type": "thinking", "thinking": [{"type": "text", "text": thinking}]}
        if signature:
            chunk["signature"] = signature
        content = [chunk, {"type": "text", "text": text}]
    return {
        "id": "cmpl-1", "object": "chat.completion", "model": model, "created": 1_800_000_000,
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": content, "tool_calls": None, "prefix": False}}],
        "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    }


def ok(body: dict[str, Any]) -> httpx2.Response:
    return httpx2.Response(200, json=body)


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

    def adapter(self, api_key: str | None = API_KEY) -> MistralAdapter:
        return MistralAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ctx(round_number: int = 1) -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number)


def call(tmp_path: Path, recorder: Recorder, *, config: ModelConfiguration = CONFIG,
         api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(recorder.adapter(api_key), config, MESSAGES, None, ctx(), sink, fork=False)
    return outcome, {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}


def test_request_is_native_explicit_and_captured_without_the_key(tmp_path: Path) -> None:
    GUARD.register(API_KEY)
    rec = Recorder(ok(completion("Use Mistral.")))
    outcome, files = call(tmp_path, rec)
    assert outcome.succeeded and outcome.extracted.final_answer == "Use Mistral."
    (request,) = rec.requests
    assert str(request.url) == "https://api.mistral.ai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    body = json.loads(request.content)
    assert body == {
        "model": MODEL, "max_tokens": 16000,
        "messages": [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Pick a model."}],
    }  # no reasoning effort, temperature, safe_prompt, stream, or prompt_cache_key
    wire = json.loads(files["wire-exchanges.json"])
    assert len(wire) == 1 and wire[0]["request_body"]["json"] == body
    assert wire[0]["request_headers"]["authorization"] == "[REDACTED]"
    for text in files.values():
        assert API_KEY not in text
    assert files["reasoning.txt"] == "pondering"


def test_explicit_parameters_map_to_native_names(tmp_path: Path) -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"].update({
        "reasoning_effort": {"state": "explicit", "value": "high"},
        "temperature": {"state": "explicit", "value": 0.3},
        "top_p": {"state": "explicit", "value": 0.9},
        "seed": {"state": "explicit", "value": 7},
    })
    rec = Recorder(ok(completion()))
    call(tmp_path, rec, config=ModelConfiguration.model_validate(data))
    body = json.loads(rec.requests[0].content)
    assert {k: body[k] for k in ("reasoning_effort", "temperature", "top_p", "random_seed")} == {
        "reasoning_effort": "high", "temperature": 0.3, "top_p": 0.9, "random_seed": 7}


def test_plain_string_content_when_reasoning_is_off(tmp_path: Path) -> None:
    outcome, files = call(tmp_path, Recorder(ok(completion("plain answer", thinking=None))))
    assert outcome.extracted.final_answer == "plain answer" and "reasoning.txt" not in files


def test_cost_uses_configured_rates_and_reported_cache(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(completion())))
    assert outcome.observations["api_cost_usd"].value == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1e6)
    cached = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
              "prompt_tokens_details": {"cached_tokens": 400}}
    outcome, _ = call(tmp_path, Recorder(ok(completion(usage=cached))))
    assert outcome.observations["api_cost_usd"].value == pytest.approx((600 * 0.15 + 400 * 0.015 + 500 * 0.60) / 1e6)
    outcome, _ = call(tmp_path, Recorder(ok(completion())), config=CONFIG.model_copy(update={"adapter_options": {}}))
    assert outcome.observations["api_cost_usd"].missing_reason == "no pricing in configuration"


@pytest.mark.parametrize(
    ("reply", "kind", "cost"),
    [
        (httpx2.Response(429, json={"message": "Requests rate limit exceeded"}), FailureKind.RATE_LIMITED, CostIncurred.NO),
        (httpx2.Response(503, json={"message": "unavailable"}), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
        (httpx2.Response(401, json={"message": "Unauthorized"}), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
        (httpx2.Response(422, json={"detail": [{"msg": "bad field"}]}), FailureKind.PROVIDER_ERROR, CostIncurred.NO),
        (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
        (httpx2.ConnectError("refused"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
    ],
)
def test_failures_are_mapped_and_never_retried(tmp_path: Path, reply: Any, kind: FailureKind,
                                               cost: CostIncurred) -> None:
    rec = Recorder(reply, ok(completion()))
    outcome, files = call(tmp_path, rec)
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(rec.requests) == 1
    assert "error.json" in files


@pytest.mark.parametrize(("body", "kind"), [
    (completion(finish="model_length"), FailureKind.CONTEXT_EXCEEDED),
    (completion(finish="error"), FailureKind.PROVIDER_ERROR),
    (completion(finish="tool_calls"), FailureKind.PROVIDER_ERROR),
    (completion(model="mistral-small-latest"), FailureKind.PROVIDER_ERROR),
    ({**completion(), "choices": []}, FailureKind.PROVIDER_ERROR),
])
def test_unusable_responses_are_failures(tmp_path: Path, body: dict[str, Any], kind: FailureKind) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(body)))
    assert outcome.failure_kind is kind and outcome.cost_incurred is CostIncurred.YES


def test_truncation_is_extracted(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(completion("partial", finish="length"))))
    assert outcome.succeeded and outcome.extracted.truncated and outcome.extracted.finish_reason == "length"


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(ok(completion()))
    outcome, _ = call(tmp_path, rec, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and outcome.cost_incurred is CostIncurred.NO
    assert rec.requests == []
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert default_registry().get("mistral").preflight(CONFIG) is not None  # type: ignore[attr-defined]
    monkeypatch.setenv("MISTRAL_API_KEY", API_KEY)
    assert default_registry().get("mistral").preflight(CONFIG) is None  # type: ignore[attr-defined]


def test_continuation_replays_signed_thinking_chunks_exactly(tmp_path: Path) -> None:
    baseline = completion("baseline answer", thinking="private reasoning", signature="sig-123")
    rec = Recorder(ok(baseline), ok(completion("branch a")), ok(completion("branch b")))
    adapter = rec.adapter()
    store = ArtifactStore(tmp_path / "results")
    snapshot = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base")).snapshot
    assert snapshot is not None
    before = json.dumps(snapshot.native_state, sort_keys=True)
    for word in ("alpha", "beta"):
        follow = snapshot.messages + (Message(role=Role.USER, content=f"Follow-up {word}"),)
        assert invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink(f"child-{word}")).succeeded
    for request, word in zip(rec.requests[1:], ("alpha", "beta")):
        sent = json.loads(request.content)
        assert sent["messages"][2] == baseline["choices"][0]["message"]  # exact, thinking and signature included
        assert sent["messages"][2]["content"][0]["signature"] == "sig-123"
        assert sent["messages"][3] == {"role": "user", "content": f"Follow-up {word}"}
    assert "alpha" not in rec.requests[2].content.decode()
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(CONFIG.model_copy(update={"model": "mistral-large-4-0"}), follow, snapshot, ctx(2))
    plain = MESSAGES + (Message(role=Role.ASSISTANT, content="made up"), Message(role=Role.USER, content="?"))
    with pytest.raises(AdapterError, match="continuation snapshot"):
        adapter.prepare(CONFIG, plain, None, ctx(2))


class MockMistral:
    """Models API plus chat completions, answering one-word probes and P6 picks."""

    def __init__(self, *, card: dict[str, Any] | None = None, answer: str = "1. **Mistral Large 4**: fits.") -> None:
        self.requests: list[httpx2.Request] = []
        self.card = card or {}
        self.answer = answer

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.method == "GET":
            model = request.url.path.rsplit("/", 1)[-1]
            return httpx2.Response(200, json={
                "id": model, "object": "model", "type": "base", "max_context_length": 262144,
                "capabilities": {"completion_chat": True}, "aliases": ["mistral-small-latest"],
                "deprecation": None, "default_model_temperature": 0.3, **self.card})
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        text = last.rsplit(" ", 1)[-1] if last.startswith("Reply with exactly one word") else self.answer
        return ok(completion(text, model=body["model"]))


REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=100, longest_follow_up_chars=0,
                                         output_budget_tokens=16000)


def qualify(mock: MockMistral, requirements: QualificationRequirements = REQUIREMENTS) -> dict[str, Any]:
    report = MistralAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(mock)).qualify(
        CONFIG, requirements)
    return {"report": report, **{c.name: c for c in report.checks}}


def test_qualification_probes_identity_limits_and_signed_replay() -> None:
    result = qualify(MockMistral())
    assert result["report"].passed, result
    assert "signed thinking chunks" in result["continuation"].detail
    assert "default_model_temperature=0.3" in result["model_identity"].detail


@pytest.mark.parametrize(("card", "expected"), [
    ({"id": "mistral-small-2606"}, "pin that id"),  # the configured id resolved to another model
    ({"max_context_length": 128000}, "context_limit"),
    ({"capabilities": {"completion_chat": False}}, "chat completion"),
])
def test_qualification_rejects_identity_problems(card: dict[str, Any], expected: str) -> None:
    result = qualify(MockMistral(card=card), REQUIREMENTS.model_copy(update={"requires_continuation": False}))
    assert result["model_identity"].passed is False and expected in result["model_identity"].detail


def test_deprecation_is_reported() -> None:
    result = qualify(MockMistral(card={"deprecation": "2027-03-01T00:00:00Z",
                                       "deprecation_replacement_model": "mistral-small-2609"}),
                     REQUIREMENTS.model_copy(update={"requires_continuation": False}))
    assert "DEPRECATED 2027-03-01" in result["model_identity"].detail


def test_mistral_candidate_in_a_cross_vendor_p6_run(env: Env) -> None:
    registry = AdapterRegistry({
        "mistral": MistralAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(MockMistral())),
        "fake": FakeAdapter(), "fake-judge": FakeJudgeAdapter(),
    })
    other = recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1)
    path = env.experiment("p6-mistral", [CONFIG_PATH, other], scenario="scenarios/agentic-design-bias/p6-pick-model",
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
    mistral = rows["mistral-small-2603@1"]
    assert mistral["own_vendor"] == "mistral" and mistral["own_first.k"] == "2"
    assert float(mistral["own_first_delta"]) == 1.0
