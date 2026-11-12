"""Helpers shared by cloud adapters: explicit parameters, failure records, the context envelope check,
and the in-memory sink that keeps qualification-probe evidence inside the qualification record."""

from __future__ import annotations

import json
import math
from typing import Any

from ...domain.common import ARTIFACT_SCHEMA_VERSION, sha256_text
from ...domain.configuration import ModelConfiguration, ParameterState
from ...domain.execution import ArtifactRef, CostIncurred, FailureKind
from ..base import NativeExecution, QualificationCheck, QualificationRequirements

CHARS_PER_TOKEN_ESTIMATE = 3  # conservative; real tokenizers usually yield fewer tokens


def explicit(config: ModelConfiguration, name: str) -> Any:
    setting = config.parameters.get(name)
    return setting.value if setting is not None and setting.state is ParameterState.EXPLICIT else None


class CollectingSink:
    """Keeps qualification probe artifacts in memory; they are persisted inside the qualification record."""

    def __init__(self) -> None:
        self.items: dict[str, Any] = {}

    def write(self, name: str, data: bytes | str) -> ArtifactRef:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        self.items[name] = text
        return ArtifactRef(name=name, relpath=f"qualification-probe/{name}", sha256=sha256_text(text),
                           size_bytes=len(text.encode("utf-8")), schema_version=ARTIFACT_SCHEMA_VERSION)

    def write_json(self, name: str, value: Any) -> ArtifactRef:
        ref = self.write(name, json.dumps(value, sort_keys=True))
        self.items[name] = value
        return ref


def context_check(config: ModelConfiguration, req: QualificationRequirements, max_input: int | None
                   ) -> QualificationCheck:
    """Baseline prompt + permitted baseline answer + longest follow-up + reserved follow-up output."""
    limits = [v for v in (config.context_limit, max_input) if v is not None]
    if not limits or req.output_budget_tokens is None:
        return QualificationCheck(name="context_envelope", passed=False, detail="context limit or output budget unknown")
    needed = (math.ceil(req.baseline_prompt_chars / CHARS_PER_TOKEN_ESTIMATE) + req.output_budget_tokens
              + math.ceil(req.longest_follow_up_chars / CHARS_PER_TOKEN_ESTIMATE) + req.output_budget_tokens)
    return QualificationCheck(
        name="context_envelope", passed=needed <= min(limits),
        detail=f"required ~{needed} tokens (estimated at {CHARS_PER_TOKEN_ESTIMATE} chars/token) vs limit {min(limits)}",
    )


def failed(kind: FailureKind, error_type: str, message: str, cost: CostIncurred, **extra: Any) -> NativeExecution:
    return NativeExecution(status="failed", failure_kind=kind, cost_incurred=cost,
                           error={"type": error_type, "message": message, **extra},
                           request_id=extra.get("request_id"))
