"""Anthropic adapter against a mock transport: wire capture, redaction, no hidden retries, failure
mapping, continuation state, and an offline end-to-end judge run."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import AdapterError, InvocationContext, QualificationRequirements
from archbench.adapters.cloud.anthropic_messages import AnthropicAdapter
from archbench.adapters.cloud.typesafe_jev import TypeSafeAdapter
from archbench.adapters.fake import FakeAdapter
from archbench.adapters.registry import AdapterRegistry
from archbench.domain.configuration import ModelConfiguration
from archbench.domain.execution import CostIncurred, FailureKind
from archbench.domain.records import RecordKind
from archbench.domain.scenario import Message, Role
from archbench.evaluation.runner import resume_evaluation, start_evaluation
from archbench.execution.controller import Controller
from archbench.execution.invoke import invoke
from archbench.storage.artifacts import ArtifactStore

from .conftest import ROOT, Env

API_KEY = "sk-ant-test-0123456789abcdef"
CONFIG = ModelConfiguration.model_validate(
    yaml.safe_load((ROOT / "configs/judges/anthropic/claude-opus-5-5.yaml").read_text())
)
MESSAGES = (Message(role=Role.SYSTEM, content="Be precise."), Message(role=Role.USER, content="Evaluate this."))
Handler = Callable[[httpx2.Request], httpx2.Response]


def message(text: str = "ok", *, stop: str = "end_turn", model: str = "claude-opus-5-5",
            thinking: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "id": "msg_01", "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "thinking", "thinking": thinking, "signature": "sig-abc"},
                    {"type": "text", "text": text}],
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0},
        **extra,
    }


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

    def adapter(self, api_key: str | None = API_KEY) -> AnthropicAdapter:
        return AnthropicAdapter(api_key=api_key, transport_factory=lambda: httpx2.MockTransport(self))


def ok(body: dict[str, Any], status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, json=body, headers={"request-id": "req_123"})


def ctx(round_number: int = 1, purpose: str = "judge") -> InvocationContext:
    return InvocationContext(experiment_id="e", task_id="t", attempt_id="a", attempt_no=1, round=round_number,
                             purpose=purpose)  # type: ignore[arg-type]


def call(tmp_path: Path, recorder: Recorder, *, config: ModelConfiguration = CONFIG,
         messages: tuple[Message, ...] = MESSAGES, api_key: str | None = API_KEY) -> tuple[Any, dict[str, Any]]:
    store = ArtifactStore(tmp_path / "results")
    sink = store.sink(f"calls/{len(list((tmp_path / 'results').glob('calls/*')))}")
    outcome = invoke(recorder.adapter(api_key), config, messages, None, ctx(), sink, fork=False)
    files = {ref.name: store.path(ref.relpath).read_text() for ref in sink.refs}
    return outcome, files


def test_request_is_native_explicit_and_captured_on_the_wire(tmp_path: Path) -> None:
    rec = Recorder(ok(message('{"a": 1}')))
    outcome, files = call(tmp_path, rec)
    assert outcome.succeeded and outcome.extracted.final_answer == '{"a": 1}'
    (request,) = rec.requests
    body = json.loads(request.content)
    assert body == {
        "model": "claude-opus-5-5", "max_tokens": 16000, "system": "Be precise.",
        "messages": [{"role": "user", "content": "Evaluate this."}],
        "output_config": {"effort": "high"},
    }  # no thinking, temperature, fallbacks, or other unconfigured fields
    wire = json.loads(files["wire-exchanges.json"])
    assert len(wire) == 1
    assert wire[0]["request_body"]["json"] == body and wire[0]["status"] == 200
    assert wire[0]["response_headers"]["request-id"] == "req_123"
    for text in files.values():
        assert API_KEY not in text
    assert outcome.cost_incurred is CostIncurred.YES


def test_cost_is_computed_from_configured_pricing(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(message())))
    # 1000 input * $4/M + 500 output * $20/M
    assert outcome.observations["api_cost_usd"].value == pytest.approx(0.004 + 0.01)
    unpriced = CONFIG.model_copy(update={"adapter_options": {}})
    outcome, _ = call(tmp_path, Recorder(ok(message())), config=unpriced)
    assert outcome.observations["api_cost_usd"].value is None
    assert outcome.observations["api_cost_usd"].missing_reason == "no pricing in configuration"


@pytest.mark.parametrize(
    ("response", "kind", "cost"),
    [
        (ok({"type": "error", "error": {"type": "overloaded_error", "message": "x"}}, 529),
         FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO),
        (ok({"type": "error", "error": {"type": "rate_limit_error", "message": "x"}}, 429),
         FailureKind.RATE_LIMITED, CostIncurred.NO),
        (ok({"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}, 400),
         FailureKind.PROVIDER_ERROR, CostIncurred.NO),
        (httpx2.ReadTimeout("read timed out"), FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN),
        (httpx2.ConnectError("refused"), FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN),
    ],
)
def test_failures_are_mapped_and_never_retried_by_the_sdk(
    tmp_path: Path, response: Any, kind: FailureKind, cost: CostIncurred
) -> None:
    rec = Recorder(response, ok(message()))
    outcome, files = call(tmp_path, rec)
    assert not outcome.succeeded
    assert (outcome.failure_kind, outcome.cost_incurred) == (kind, cost)
    assert len(rec.requests) == 1  # max_retries=0: the harness, not the SDK, decides on retries
    assert len(json.loads(files["wire-exchanges.json"])) == 1
    assert "error.json" in files


def test_refusal_is_recorded_not_rerouted(tmp_path: Path) -> None:
    body = message("", stop="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": "x"})
    outcome, files = call(tmp_path, Recorder(ok(body)))
    assert outcome.failure_kind is FailureKind.REFUSAL and outcome.cost_incurred is CostIncurred.YES
    assert json.loads(files["native-response.json"])["stop_details"]["category"] == "cyber"


def test_response_from_another_model_is_a_failure(tmp_path: Path) -> None:
    outcome, _ = call(tmp_path, Recorder(ok(message(model="claude-opus-5"))))
    assert not outcome.succeeded and outcome.failure_kind is FailureKind.PROVIDER_ERROR


def test_truncation_and_reasoning_summary_are_extracted(tmp_path: Path) -> None:
    outcome, files = call(tmp_path, Recorder(ok(message("partial", stop="max_tokens", thinking="summary"))))
    assert outcome.extracted.truncated and outcome.extracted.finish_reason == "max_tokens"
    assert files["reasoning.txt"] == "summary"


def test_missing_credentials_fail_before_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "no-profiles"))
    rec = Recorder(ok(message()))
    outcome, _ = call(tmp_path, rec, api_key=None)
    assert outcome.failure_kind is FailureKind.PROVIDER_ERROR and outcome.cost_incurred is CostIncurred.NO
    assert rec.requests == []


def test_evaluation_refuses_to_start_without_credentials(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from archbench.evaluation.runner import EvaluationPreflightError

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(env.tmp / "no-profiles"))
    plan = env.plan(env.experiment("exp-nokey", [env.config("alpha")], repetitions=1))
    registry = AdapterRegistry({"anthropic": Recorder(ok(message())).adapter(api_key=None),
                                "typesafe": TypeSafeAdapter(api_key=None)})
    before = len(env.db.list_records())
    with pytest.raises(EvaluationPreflightError, match="credentials"):
        start_evaluation(env.db, env.store, plan.experiment_id, ROOT / "configs/evaluation/anthropic.yaml",
                         base=env.base, registry=registry)
    assert len(env.db.list_records()) == before  # nothing frozen


def test_unsupported_parameters_fail_qualification(tmp_path: Path) -> None:
    data = CONFIG.model_dump(mode="json")
    data["parameters"]["temperature"] = {"state": "explicit", "value": 0.2}
    report = Recorder(ok({"id": "claude-opus-5-5", "type": "model", "display_name": "x",
                          "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1000000,
                          "max_tokens": 128000})).adapter().qualify(
        ModelConfiguration.model_validate(data),
        QualificationRequirements(baseline_prompt_chars=1, longest_follow_up_chars=1, output_budget_tokens=1,
                                  requires_continuation=False),
    )
    checks = {c.name: c for c in report.checks}
    assert checks["parameter_support"].passed is False
    assert checks["model_identity"].passed is True
    assert not report.passed


def test_continuation_replays_signed_thinking_blocks_unchanged(tmp_path: Path) -> None:
    rec = Recorder(ok(message("baseline answer")), ok(message("branch answer")))
    adapter = rec.adapter()
    store = ArtifactStore(tmp_path / "results")
    base = invoke(adapter, CONFIG, MESSAGES, None, ctx(), store.sink("base"))
    snapshot = base.snapshot
    assert snapshot is not None
    follow = snapshot.messages + (Message(role=Role.USER, content="Follow-up"),)
    before = json.dumps(snapshot.native_state, sort_keys=True)
    child = invoke(adapter, CONFIG, follow, snapshot, ctx(2), store.sink("child"))
    assert child.succeeded
    sent = json.loads(rec.requests[1].content)
    assert sent["system"] == "Be precise."
    assert sent["messages"][1] == {"role": "assistant", "content": message("baseline answer")["content"]}
    assert sent["messages"][1]["content"][0]["signature"] == "sig-abc"
    assert sent["messages"][2] == {"role": "user", "content": "Follow-up"}
    assert json.dumps(snapshot.native_state, sort_keys=True) == before
    other = CONFIG.model_copy(update={"model": "claude-sonnet-5-5"})
    with pytest.raises(AdapterError, match="different model"):
        adapter.prepare(other, follow, snapshot, ctx(2))
