"""Deterministic-shape fake adapter for exercising the harness without paid calls.

It behaves like a native adapter in the ways the harness must handle:
- continuation requires a signed native state token, not just the assistant text;
- sampling is random unless an explicit seed is configured, and the seed is recorded;
- output beyond the explicit output budget is truncated with a native finish reason;
- failures can be injected through `adapter_options` for scored invocations only:
    transient_failures: {"<round>": <count>}   first N attempts fail with a transient transport error
    permanent_failure_rounds: [<round>, ...]   provider error, not retryable
    truncate_rounds: [<round>, ...]             force truncation
- `style: recommendations` answers any request with a numbered shortlist instead of an architecture:
    recommendation_pool: [<name>, ...]          candidates, earlier names more likely (weights n, n-1, ..., 1)
    recommendation_count: <n>                   items per answer (default 3)
    also_mentions: [<name>, ...]                named in a closing sentence without being recommended
"""

from __future__ import annotations

import hashlib
import math
import random
import secrets
import uuid
from typing import Any

from ..domain.common import canonical_json
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

SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "seed"})
STATE_FORMAT = "fake-transcript-v1"
_SIGNING_KEY = b"fake-adapter-continuation-signing-key"

CATEGORIES = (
    ("Independent jobs", "independent scheduled jobs with targeted LLM calls"),
    ("Shared infrastructure", "independent workflows on shared persistence, scheduling, and integrations"),
    ("Agent framework", "agent-oriented execution with tool use"),
    ("Workflow engine", "a durable workflow engine with explicit state"),
    ("Hybrid", "shared infrastructure with a durable workflow engine for approval flows"),
)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def sign_state(messages: tuple[Message, ...]) -> str:
    body = canonical_json([m.model_dump(mode="json") for m in messages]).encode("utf-8")
    return hashlib.sha256(_SIGNING_KEY + body).hexdigest()


class FakeAdapter:
    name = "fake"

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.FAKE,
                detail=f"adapter={config.adapter} track={config.track}",
            ),
            self._parameter_check(config),
            self._continuation_check(config),
            self._context_check(config, requirements),
        ]
        return QualificationReport.from_checks(config, self.name, requirements, checks)

    def _parameter_check(self, config: ModelConfiguration) -> QualificationCheck:
        unsupported = sorted(
            n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT and n not in SUPPORTED_PARAMETERS
        )
        return QualificationCheck(
            name="parameter_support",
            passed=not unsupported,
            detail="all explicit parameters supported" if not unsupported else f"unsupported: {', '.join(unsupported)}",
        )

    def _continuation_check(self, config: ModelConfiguration) -> QualificationCheck:
        base = (Message(role=Role.SYSTEM, content="qualification"), Message(role=Role.USER, content="baseline"))
        ctx = InvocationContext(
            experiment_id="qualification", task_id="qualification", attempt_id="q", attempt_no=1, round=1,
            purpose="qualification",
        )
        try:
            prepared = self.prepare(config, base, None, ctx)
            execution = self.execute(config, prepared, ctx, _NullSink())
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = canonical_json(snapshot)
            for follow_up in ("branch-a", "branch-b"):
                child = snapshot.messages + (Message(role=Role.USER, content=follow_up),)
                self.prepare(config, child, snapshot, ctx.model_copy(update={"round": 2}))
            if canonical_json(snapshot) != before:
                return QualificationCheck(name="continuation", passed=False, detail="forking mutated baseline state")
        except AdapterError as exc:
            return QualificationCheck(name="continuation", passed=False, detail=str(exc))
        return QualificationCheck(name="continuation", passed=True, detail="baseline forked twice without mutation")

    def _context_check(self, config: ModelConfiguration, req: QualificationRequirements) -> QualificationCheck:
        if config.context_limit is None:
            return QualificationCheck(name="context_envelope", passed=False, detail="context limit unknown")
        if req.output_budget_tokens is None:
            return QualificationCheck(name="context_envelope", passed=False, detail="output budget unknown")
        needed = (
            math.ceil(req.baseline_prompt_chars / 4)
            + req.output_budget_tokens  # permitted baseline answer length
            + math.ceil(req.longest_follow_up_chars / 4)
            + req.output_budget_tokens  # reserved follow-up generation
        )
        return QualificationCheck(
            name="context_envelope",
            passed=needed <= config.context_limit,
            detail=f"required {needed} tokens (estimated) vs limit {config.context_limit}",
        )

    # ----- invocation ------------------------------------------------------------------------------

    def prepare(
        self,
        config: ModelConfiguration,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        ctx: InvocationContext,
    ) -> PreparedInvocation:
        continuation: str | None = None
        if parent is not None:
            parent.verify_shape()
            if parent.native_state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if parent.native_state.get("state_token") != sign_state(parent.messages):
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state token does not match transcript")
            if messages[: len(parent.messages)] != parent.messages:
                raise AdapterError(FailureKind.INVALID_STATE, "conversation does not extend the baseline snapshot")
            continuation = str(parent.native_state["state_token"])
        request: dict[str, Any] = {
            "endpoint": "fake://v1/generate",
            "headers": {"authorization": "Bearer fake-not-a-secret", "content-type": "application/json"},
            "body": {
                "model": config.model,
                "messages": [m.model_dump(mode="json") for m in messages],
                "max_output_tokens": config.explicit_int("max_output_tokens"),
                "seed": config.explicit_int("seed"),
                "continuation_token": continuation,
            },
        }
        tokens = sum(estimate_tokens(m.content) for m in messages)
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request=request,
            input_tokens=tokens,
            input_tokens_source="fake estimate: ceil(chars / 4)",
        )

    def execute(
        self,
        config: ModelConfiguration,
        prepared: PreparedInvocation,
        ctx: InvocationContext,
        sink: ArtifactSink,
    ) -> NativeExecution:
        opts = config.adapter_options if ctx.purpose == "scored" else {}
        transient = opts.get("transient_failures") or {}
        if isinstance(transient, dict) and ctx.attempt_no <= int(str(transient.get(str(ctx.round), 0))):
            return NativeExecution(
                status="failed",
                failure_kind=FailureKind.TRANSIENT_TRANSPORT,
                error={"type": "connection_reset", "message": "injected transient failure"},
                cost_incurred=CostIncurred.NO,
            )
        if ctx.round in _int_list(opts.get("permanent_failure_rounds")):
            return NativeExecution(
                status="failed",
                failure_kind=FailureKind.PROVIDER_ERROR,
                error={"type": "invalid_request", "message": "injected permanent failure"},
                cost_incurred=CostIncurred.NO,
            )

        seed = config.explicit_int("seed")
        seed = seed if seed is not None else secrets.randbits(32)
        text = _compose(config, prepared.messages, random.Random(seed))
        budget = config.explicit_int("max_output_tokens")
        truncated = ctx.round in _int_list(opts.get("truncate_rounds"))
        if budget is not None and estimate_tokens(text) > budget:
            text, truncated = text[: budget * 4], True
        elif truncated:
            text = text[: len(text) // 2]
        finish = "max_output_tokens" if truncated else "stop"
        response_id = f"fake-{uuid.uuid4().hex[:12]}"
        output_tokens = estimate_tokens(text)
        return NativeExecution(
            status="completed",
            native_response={
                "id": response_id,
                "model": config.model,
                "output_text": text,
                "finish_reason": finish,
                "usage": {"input_tokens": prepared.input_tokens, "output_tokens": output_tokens},
                "sampling_seed": seed,
            },
            finish_reason=finish,
            request_id=response_id,
            cost_incurred=CostIncurred.NO,
            observations={
                "input_tokens": Observation(value=prepared.input_tokens, unit="tokens", source="fake estimate"),
                "output_tokens": Observation(value=output_tokens, unit="tokens", source="fake estimate"),
                "reasoning_tokens": Observation.missing("tokens", "fake", "fake adapter exposes no reasoning"),
                "api_cost_usd": Observation(value=0.0, unit="usd", source="fake adapter has no billing"),
            },
        )

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        if execution.status != "completed" or execution.native_response is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        finish = execution.native_response.get("finish_reason")
        return ExtractedResponse(
            final_answer=str(execution.native_response["output_text"]),
            truncated=finish == "max_output_tokens",
            finish_reason=str(finish) if finish is not None else None,
        )

    def fork_state(
        self,
        prepared: PreparedInvocation,
        execution: NativeExecution,
        extracted: ExtractedResponse,
    ) -> ContinuationSnapshot:
        messages = prepared.messages + (Message(role=Role.ASSISTANT, content=extracted.final_answer),)
        model = (execution.native_response or {}).get("model", "")
        return ContinuationSnapshot(
            adapter=self.name,
            configuration_key=str(model),
            messages=messages,
            native_state={
                "format": STATE_FORMAT,
                "response_id": execution.request_id,
                "state_token": sign_state(messages),
            },
        )


class _NullSink:
    def write(self, name: str, data: bytes | str) -> Any:
        raise RuntimeError("qualification does not persist adapter artifacts")

    def write_json(self, name: str, value: Any) -> Any:
        raise RuntimeError("qualification does not persist adapter artifacts")


def _str_list(value: object) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def _int_list(value: object) -> list[int]:
    return [int(v) for v in value] if isinstance(value, list) else []


def _compose(config: ModelConfiguration, messages: tuple[Message, ...], rng: random.Random) -> str:
    style = str(config.adapter_options.get("style", "concise"))
    if style == "recommendations":
        return _recommend(config, rng)
    paragraphs = 3 if style == "thorough" else 1
    user_turns = [m for m in messages if m.role is Role.USER]
    signature = f"_Prepared by FakeArchitect ({config.model})._"
    filler = (
        "This section states a position, the reasoning behind it, and the main trade-off accepted. "
        "The mechanism is described concretely enough to be checked."
    )
    if len(user_turns) == 1:
        title, summary = rng.choice(CATEGORIES)
        sections = [
            "Recommended architecture", "Important assumptions", "Alternatives and trade-offs",
            "Component boundaries", "LLM and agent responsibilities", "Execution and orchestration",
            "Persistence and state", "Security", "Reliability", "Technology choices", "Operating cost",
            "Risks and architectural evolution",
        ]
        body = [signature, ""]
        for i, name in enumerate(sections, start=1):
            body.append(f"## {i}. {name}")
            if name == "Recommended architecture":
                body.append(f"Category: {title}. Use {summary}.")
            body.extend([filler] * paragraphs)
            body.append("")
        return "\n".join(body).strip() + "\n"

    baseline = next((m.content for m in messages if m.role is Role.ASSISTANT), "")
    category = next((ln for ln in baseline.splitlines() if ln.startswith("Category:")), "Category: unknown.")
    challenge = user_turns[-1].content.strip().splitlines()[0].lstrip("# ").strip()
    return "\n".join(
        [
            signature,
            "",
            f"## Response to: {challenge}",
            f"My baseline recommendation was {category[len('Category: '):].rstrip('.')}.",
            *([filler] * paragraphs),
            "",
            "## Changes to the design",
            rng.choice(["The architecture remains suitable with targeted changes.", "Parts of the design change."]),
            *([filler] * paragraphs),
        ]
    ).strip() + "\n"


def _recommend(config: ModelConfiguration, rng: random.Random) -> str:
    opts = config.adapter_options
    pool = _str_list(opts.get("recommendation_pool"))
    count = min(int(str(opts.get("recommendation_count", 3))), len(pool))
    weights = list(range(len(pool), 0, -1))
    chosen: list[str] = []
    while len(chosen) < count:
        pick = rng.choices(pool, weights=weights)[0]
        if pick not in chosen:
            chosen.append(pick)
    lines = [f"_Prepared by FakeArchitect ({config.model})._", "",
             "Chosen for maturity, model portability, and maintenance effort:", ""]
    lines += [f"{i}. **{name}**: a solid option for personal agents." for i, name in enumerate(chosen, start=1)]
    mentions = _str_list(opts.get("also_mentions"))
    if mentions:
        lines += ["", f"Also worth knowing about: {', '.join(mentions)}."]
    return "\n".join(lines) + "\n"
