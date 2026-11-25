"""Checkpoint boundaries shared by campaign generation and evaluation."""
from __future__ import annotations

from dataclasses import dataclass


class CheckpointStop(RuntimeError):
    """Work is durable; a later command may resume it."""


def resource_exhausted(error: str | None, failure: str | None = None) -> bool:
    text = (error or "").lower()
    return failure == "rate_limited" or any(s in text for s in (
        "insufficient_quota", "insufficient quota", "insufficient balance", "insufficient credits",
        "credit balance", "credits exhausted", "quota exhausted", "quota exceeded", "resource_exhausted",
        "billing_hard_limit", "payment required", "402", "usage limit reached",
    ))


@dataclass
class CallBudget:
    limit: int | None = None
    used: int = 0

    def before_call(self) -> None:
        if self.limit is not None and self.used >= self.limit:
            raise CheckpointStop(f"new-call limit reached ({self.used}/{self.limit}); resume the same campaign")
        self.used += 1
