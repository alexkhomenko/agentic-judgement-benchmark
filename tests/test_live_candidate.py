"""Real-candidate path with a mock Anthropic API: live continuation qualification, branching with
native replay, and the full Haiku debug experiment from qualification to report."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from archbench.adapters.base import QualificationRequirements
from archbench.adapters.cloud.anthropic_messages import AnthropicAdapter
from archbench.adapters.cloud.typesafe_jev import TypeSafeAdapter
from archbench.adapters.fake import FakeAdapter
from archbench.adapters.registry import AdapterRegistry
from archbench.domain.configuration import ModelConfiguration
from archbench.domain.records import RecordKind
from archbench.evaluation.adjudication import adjudicate
from archbench.evaluation.runner import load_run, resume_evaluation, start_evaluation
from archbench.execution.controller import Controller, task_id_for
from archbench.planning.compiler import compile_plan, persist_plan
from archbench.reporting.report import publish_report

from .conftest import ROOT, Env
from .test_evaluation import review_all
from .test_typesafe_adapter import TS_KEY, jev_response

API_KEY = "sk-ant-api03-live-candidate-test-0000"
HAIKU = ModelConfiguration.model_validate(
    yaml.safe_load((ROOT / "configs/models/anthropic/claude-haiku-5-5.yaml").read_text()))
SECTIONS = ["Recommended architecture", "Important assumptions", "Alternatives and trade-offs", "Component boundaries",
            "LLM and agent responsibilities", "Execution and orchestration", "Persistence and state", "Security",
            "Reliability", "Technology choices", "Operating cost", "Risks and architectural evolution"]


class MockAnthropic:
    """Routes the Models API, qualification probes, candidate turns, and judge calls."""

    def __init__(self, *, fail_post: int | None = None, max_input_tokens: int = 1_000_000) -> None:
        self.requests: list[httpx2.Request] = []
        self.fail_post = fail_post
        self.max_input_tokens = max_input_tokens
        self.posts = 0

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.method == "GET":
            model = request.url.path.rsplit("/", 1)[-1]
            return httpx2.Response(200, json={"id": model, "type": "model", "display_name": model,
                                              "created_at": "2026-01-01T00:00:00Z",
                                              "max_input_tokens": self.max_input_tokens, "max_tokens": 64000})
        self.posts += 1
        if self.fail_post == self.posts:
            return httpx2.Response(400, json={"type": "error", "error": {"type": "invalid_request_error",
                                                                          "message": "bad thinking block"}})
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        if "<evaluation_packet>" in user:
            text = self._judge(user)
        elif "qualification probe" in body.get("system", ""):
            text = user.rsplit(" ", 1)[-1]
        elif len(body["messages"]) == 1:
            text = "\n\n".join(f"## {i}. {s}\nUse shared infrastructure with a durable inbox."
                               for i, s in enumerate(SECTIONS, 1))
        else:
            text = f"## Response\nThe baseline holds; for this challenge add idempotency keys. ({len(body['messages'])})"
        return httpx2.Response(200, headers={"request-id": f"req_{self.posts}"}, json={
            "id": f"msg_{self.posts}", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "thinking", "thinking": "", "signature": f"sig-{self.posts}"},
                        {"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1500, "output_tokens": 800},
        })

    @staticmethod
    def _judge(user: str) -> str:
        match = re.search(r"^<evaluation_packet>\n(.*?)\n</evaluation_packet>$", user, re.DOTALL | re.MULTILINE)
        assert match
        packet = json.loads(match.group(1))
        quote = packet["candidate_answer"].strip().splitlines()[0]
        return json.dumps({"ratings": {d["id"]: {"rating": 3, "evidence": [quote]}
                                       for d in packet["rubric"]["dimensions"]},
                           "critical_failure_candidates": []})

    def adapter(self) -> AnthropicAdapter:
        return AnthropicAdapter(api_key=API_KEY, transport_factory=lambda: httpx2.MockTransport(self))

    def posted(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]


REQUIREMENTS = QualificationRequirements(baseline_prompt_chars=4000, longest_follow_up_chars=700,
                                         output_budget_tokens=16000, requires_continuation=True)


def test_live_continuation_check_forks_two_independent_branches() -> None:
    api = MockAnthropic()
    report = api.adapter().qualify(HAIKU, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert report.passed, checks
    assert "thinking blocks" in checks["continuation"].detail
    baseline, alpha, beta = api.posted()
    assert baseline["max_tokens"] == 2048 and "output_config" not in baseline and "thinking" not in baseline
    for branch, word in ((alpha, "alpha"), (beta, "beta")):
        assert branch["messages"][1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
        assert branch["messages"][2]["content"].endswith(word)
    assert "alpha" not in json.dumps(beta)
    assert [p["step"] for p in report.probes] == ["baseline", "branch-alpha", "branch-beta"]
    assert all(p["cost_usd"] > 0 and p["wire"] for p in report.probes)
    assert API_KEY not in json.dumps(report.model_dump(mode="json"))


def test_failed_branch_fails_qualification() -> None:
    report = MockAnthropic(fail_post=2).adapter().qualify(HAIKU, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert checks["continuation"].passed is False and "branch alpha failed" in checks["continuation"].detail
    assert not report.passed


def test_context_envelope_uses_the_api_reported_limit() -> None:
    report = MockAnthropic(max_input_tokens=20_000).adapter().qualify(HAIKU, REQUIREMENTS)
    checks = {c.name: c for c in report.checks}
    assert checks["context_envelope"].passed is False and "vs limit 20000" in checks["context_envelope"].detail


def test_haiku_debug_experiment_end_to_end(env: Env, tmp_path: Path) -> None:
    api = MockAnthropic()
    jev = MockTypeSafe()
    env.registry = AdapterRegistry({"fake": FakeAdapter(), "anthropic": api.adapter(),
                                    "typesafe": TypeSafeAdapter(api_key=TS_KEY,
                                                                transport_factory=lambda: httpx2.MockTransport(jev))})
    experiment = ROOT / "configs/experiments/haiku-smoke.yaml"
    (qualification,) = env.qualify(experiment)
    assert qualification.payload["report"]["passed"] is True
    plan = compile_plan(experiment, base=env.base, db=env.db)
    persist_plan(env.db, env.store, plan)
    summary = Controller(env.db, env.store, env.registry).run(plan)
    assert summary.task_status == {"succeeded": 4}

    key = plan.configurations[0].configuration.key
    candidate_posts = [b for b in api.posted() if "qualification probe" not in b.get("system", "")
                       and "<evaluation_packet>" not in b["messages"][-1]["content"]]
    baseline_post, *branches = candidate_posts
    assert len(branches) == 3 and baseline_post["max_tokens"] == 16000
    baseline_response = env.db.response_for_task(task_id_for(plan.experiment_id, key, 1, 1))
    assert baseline_response is not None
    native = env.store.read_json(env.db.artifacts_for(baseline_response.attempt_id)["native-response.json"])
    for branch in branches:
        assert len(branch["messages"]) == 3  # baseline user, baseline assistant (native), own follow-up
        assert branch["messages"][1] == {"role": "assistant", "content": native["content"]}

    run = start_evaluation(env.db, env.store, plan.experiment_id, ROOT / "configs/evaluation/anthropic-debug.yaml",
                           base=env.base, registry=env.registry)
    status = resume_evaluation(env.db, env.store, env.registry, run.record_id)
    assert status.judge_evaluations.done == 8 and status.classifications.done == 2
    judge_models = {b["model"] for b in api.posted() if "<evaluation_packet>" in b["messages"][-1]["content"]}
    assert judge_models == {"claude-haiku-5-5"}
    review_all(env, run)
    ctx = load_run(env.db, run.record_id)
    adjudicate(env.db, run, ctx.policy, ctx.rubric)
    report = publish_report(env.db, env.store, plan.experiment_id, run.record_id)
    assert report.payload["evaluation_complete"] is True
    text = env.store.read_verified(env.db.artifacts_for(report.record_id)["report.md"]).decode()
    probe_row = next(line for line in text.splitlines() if line.startswith("| qualification probes |"))
    assert probe_row.split("|")[2].strip() == "3" and float(probe_row.split("|")[3]) > 0
    for record in env.db.list_records(kind=RecordKind.JUDGE_EVALUATION, experiment_id=plan.experiment_id):
        assert record.payload["status"] == "completed"
    for path in env.store.root.rglob("*"):
        if path.is_file() and path.suffix != ".sqlite":
            data = path.read_bytes()
            assert API_KEY.encode() not in data and TS_KEY.encode() not in data, path
    dump = env.db.connection.iterdump()
    assert not any(API_KEY in line or TS_KEY in line for line in dump)


class MockTypeSafe:
    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=jev_response("shared_infrastructure"))


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
