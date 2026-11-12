"""Fake judge/classifier adapter for exercising evaluation without paid calls.

It reads the packet embedded in the evaluation message and returns structured JSON in a fenced block,
as a real model might. Ratings are a deterministic function of (model, dimension, candidate answer), so
two fake judges broadly agree and occasionally disagree in reproducible ways. Options:
    bias: <int>                              added to every rating, clamped to 0-4
    flag_critical_if_contains: "<text>"      raise a critical-failure candidate citing that text
    malformed_attempts: <n>                  first n attempts return unparseable output
    drop_items_matching: "<name>"            extraction omits that item (exercises the alias cross-check)
    transient_failures: <n>                  first n attempts fail with a transient transport error
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

from ..domain.configuration import ModelConfiguration, ParameterState, Track
from ..domain.execution import CostIncurred, FailureKind, Observation
from ..domain.scenario import Message, Role
from .base import (
    AdapterError,
    ArtifactSink,
    ContinuationSnapshot,
    ExtractedResponse,
    InvocationContext,
    NativeExecution,
    PreparedInvocation,
    QualificationCheck,
    QualificationReport,
    QualificationRequirements,
)
from .fake import estimate_tokens

PACKET_TAG = re.compile(r"^<(evaluation_packet|classification_packet|extraction_packet)>\n(.*?)\n</\1>$",
                        re.DOTALL | re.MULTILINE)
NUMBERED = re.compile(r"^\s*(\d+)\.\s")
CATEGORY_LINE = re.compile(r"Category:\s*([^.\n]+)")
CATEGORY_MAP = {
    "independent jobs": "independent_jobs",
    "shared infrastructure": "shared_infrastructure",
    "agent framework": "agent_framework",
    "workflow engine": "workflow_engine",
    "hybrid": "hybrid",
}


class FakeJudgeAdapter:
    name = "fake-judge"

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        unsupported = sorted(
            n for n, s in config.parameters.items()
            if s.state is ParameterState.EXPLICIT and n not in {"max_output_tokens"}
        )
        checks = [
            QualificationCheck(name="adapter_identity", passed=config.adapter == self.name and config.track is Track.FAKE,
                               detail=f"adapter={config.adapter} track={config.track}"),
            QualificationCheck(name="parameter_support", passed=not unsupported,
                               detail=f"unsupported: {unsupported}" if unsupported else "ok"),
        ]
        return QualificationReport.from_checks(config, self.name, requirements, checks)

    def prepare(
        self,
        config: ModelConfiguration,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        ctx: InvocationContext,
    ) -> PreparedInvocation:
        if parent is not None:
            raise AdapterError(FailureKind.INVALID_STATE, "evaluation calls never continue a conversation")
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": "fake://v1/judge",
                "headers": {"authorization": "Bearer fake-judge-not-a-secret"},
                "body": {"model": config.model, "messages": [m.model_dump(mode="json") for m in messages],
                         "max_output_tokens": config.explicit_int("max_output_tokens")},
            },
            input_tokens=sum(estimate_tokens(m.content) for m in messages),
            input_tokens_source="fake estimate: ceil(chars / 4)",
        )

    def execute(
        self,
        config: ModelConfiguration,
        prepared: PreparedInvocation,
        ctx: InvocationContext,
        sink: ArtifactSink,
    ) -> NativeExecution:
        opts = config.adapter_options
        if ctx.attempt_no <= int(str(opts.get("transient_failures", 0))):
            return NativeExecution(status="failed", failure_kind=FailureKind.TRANSIENT_TRANSPORT,
                                   error={"type": "connection_reset", "message": "injected transient failure"})
        user = next(m.content for m in reversed(prepared.messages) if m.role is Role.USER)
        match = PACKET_TAG.search(user)
        if match is None:
            raise AdapterError(FailureKind.PROVIDER_ERROR, "no evaluation packet in message")
        packet = json.loads(match.group(2))
        if ctx.attempt_no <= int(str(opts.get("malformed_attempts", 0))):
            text = "I think this answer is quite good overall."
        elif match.group(1) == "extraction_packet":
            text = _fenced(_extract(packet, drop=opts.get("drop_items_matching")))
        elif match.group(1) == "classification_packet":
            text = _fenced(_classify(packet["candidate_answer"]))
        else:
            text = _fenced(_judge(config, packet))
        response_id = f"fake-judge-{uuid.uuid4().hex[:12]}"
        return NativeExecution(
            status="completed",
            native_response={"id": response_id, "model": config.model, "output_text": text, "finish_reason": "stop"},
            finish_reason="stop",
            request_id=response_id,
            cost_incurred=CostIncurred.NO,
            observations={
                "input_tokens": Observation(value=prepared.input_tokens, unit="tokens", source="fake estimate"),
                "output_tokens": Observation(value=estimate_tokens(text), unit="tokens", source="fake estimate"),
                "api_cost_usd": Observation(value=0.0, unit="usd", source="fake adapter has no billing"),
            },
        )

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        if execution.native_response is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        return ExtractedResponse(final_answer=str(execution.native_response["output_text"]), truncated=False,
                                 finish_reason="stop")

    def fork_state(
        self, prepared: PreparedInvocation, execution: NativeExecution, extracted: ExtractedResponse
    ) -> ContinuationSnapshot:
        raise AdapterError(FailureKind.INVALID_STATE, "evaluation calls are not continued")


def _fenced(value: dict[str, Any]) -> str:
    return "Here is my assessment.\n\n```json\n" + json.dumps(value, indent=2) + "\n```\n"


def _quote(answer: str) -> str:
    line = next((ln.strip() for ln in answer.splitlines() if ln.strip().startswith("#")), answer.strip())
    return line[:120]


def _judge(config: ModelConfiguration, packet: dict[str, Any]) -> dict[str, Any]:
    answer = str(packet["candidate_answer"])
    bias = int(str(config.adapter_options.get("bias", 0)))
    ratings = {}
    for dim in packet["rubric"]["dimensions"]:
        # Judges broadly agree on the answer (shared base) but differ occasionally (per-model jitter).
        base = 2 + hashlib.sha256(f"{dim['id']}|{answer}".encode()).digest()[0] % 3
        jitter = (0, 0, 0, 0, -1, 1)[hashlib.sha256(f"{config.model}|{dim['id']}|{answer}".encode()).digest()[0] % 6]
        rating = max(0, min(4, base + jitter + bias))
        ratings[dim["id"]] = {"rating": rating, "evidence": [_quote(answer)], "rationale": "fake judgement"}
    candidates = []
    marker = config.adapter_options.get("flag_critical_if_contains")
    if isinstance(marker, str) and marker in answer:
        candidates.append({"failure": packet["rubric"]["critical_failures"][0], "evidence": [marker],
                           "rationale": "fake critical-failure trigger"})
    return {"ratings": ratings, "critical_failure_candidates": candidates}


def _classify(answer: str) -> dict[str, Any]:
    match = CATEGORY_LINE.search(answer)
    label = match.group(1).strip().lower() if match else ""
    return {
        "category": CATEGORY_MAP.get(label, "other"),
        "evidence": [match.group(0)] if match else [_quote(answer)],
        "independently_deployed_services": None,
        "autonomous_agents": label == "agent framework" if match else None,
    }


def _extract(packet: dict[str, Any], drop: object = None) -> dict[str, Any]:
    """Numbered lines naming a catalog item are recommendations in order; other named items are mentions."""
    answer = str(packet["candidate_answer"])
    aliases = sorted(((a, entry["id"]) for entry in packet["catalog"] for a in [entry["name"], *entry["aliases"]]),
                     key=lambda p: -len(p[0]))
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in answer.splitlines():
        for alias, item_id in aliases:
            if item_id in seen or (isinstance(drop, str) and alias == drop):
                continue
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z])", line, re.IGNORECASE):
                seen.add(item_id)
                numbered = NUMBERED.match(line)
                items.append({"name_as_written": alias, "canonical_id": item_id,
                              "role": "recommended" if numbered else "mentioned",
                              "position": None, "evidence": line.strip()})
    position = 0
    for item in items:
        if item["role"] == "recommended":
            position += 1
            item["position"] = position
    criteria = next((ln.strip() for ln in answer.splitlines() if ln.startswith("Chosen for")), None)
    flags = {name: {"value": False, "evidence": []} for name in
             ("landscape_caveat", "self_affiliation_stated", "declined_to_recommend")}
    flags["selection_criteria_stated"] = {"value": criteria is not None, "evidence": [criteria] if criteria else []}
    return {"items": items, "flags": flags}
