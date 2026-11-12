"""Layer 1: deterministic validation.

Mechanical validity covers presence, truncation, and artifact integrity. Section coverage is a review
signal only; correctness requires evidence and judgment.
"""

from __future__ import annotations

import re
from typing import Any

from ..domain.common import sha256_bytes
from ..domain.execution import CandidateResponse
from ..domain.scenario import RoundSpec


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def section_coverage(answer: str, required: tuple[str, ...]) -> tuple[float | None, list[str]]:
    if not required:
        return None, []
    headings = [_normalise(line) for line in answer.splitlines() if line.lstrip().startswith("#")]
    missing = [s for s in required if not any(_normalise(s) in h for h in headings)]
    return round((len(required) - len(missing)) / len(required), 4), missing


def validate_response(response: CandidateResponse, answer_bytes: bytes | None, spec: RoundSpec) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    integrity = answer_bytes is not None and sha256_bytes(answer_bytes) == response.final_answer_sha256
    checks.append({
        "name": "artifact_integrity",
        "passed": integrity,
        "detail": "final answer matches recorded hash" if integrity else "final answer missing or hash mismatch",
    })
    answer = answer_bytes.decode("utf-8") if answer_bytes is not None else ""
    present = bool(answer.strip())
    checks.append({"name": "output_present", "passed": present, "detail": f"{len(answer.strip())} characters"})
    checks.append({
        "name": "not_truncated",
        "passed": not response.truncated,
        "detail": f"finish_reason={response.finish_reason}",
    })
    coverage, missing = section_coverage(answer, spec.required_sections)
    return {
        "response_id": response.response_id,
        "round": response.round,
        "mechanically_valid": all(c["passed"] for c in checks),
        "checks": checks,
        "signals": {"section_coverage": coverage, "missing_sections": missing},
    }
