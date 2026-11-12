"""Public scenario material sent to candidate models. Private rubric material lives in `evaluation`."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .common import Frozen, sha256_text


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(Frozen):
    role: Role
    content: str


class RoundSpec(Frozen):
    round: int = Field(ge=1)
    name: str
    prompt: str
    prompt_sha256: str
    parent_round: int | None = None
    required_sections: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _verify_hash(self) -> RoundSpec:
        if sha256_text(self.prompt) != self.prompt_sha256:
            raise ValueError(f"round {self.round} prompt does not match its recorded hash")
        return self


class Scenario(Frozen):
    scenario_id: str
    revision: str
    system_prompt: str
    system_prompt_sha256: str
    rounds: tuple[RoundSpec, ...]

    @model_validator(mode="after")
    def _check_structure(self) -> Scenario:
        if sha256_text(self.system_prompt) != self.system_prompt_sha256:
            raise ValueError("system prompt does not match its recorded hash")
        numbers = [r.round for r in self.rounds]
        if len(set(numbers)) != len(numbers):
            raise ValueError("duplicate round numbers")
        roots = [r for r in self.rounds if r.parent_round is None]
        if len(roots) != 1:
            raise ValueError("a scenario needs exactly one baseline round")
        root = roots[0].round
        for spec in self.rounds:
            if spec.parent_round is not None and spec.parent_round != root:
                raise ValueError(
                    f"round {spec.round} must branch from baseline round {root}; "
                    "cumulative follow-up chains are not allowed"
                )
        return self

    @property
    def baseline(self) -> RoundSpec:
        return next(r for r in self.rounds if r.parent_round is None)

    @property
    def follow_ups(self) -> tuple[RoundSpec, ...]:
        return tuple(sorted((r for r in self.rounds if r.parent_round is not None), key=lambda r: r.round))

    def round_spec(self, number: int) -> RoundSpec:
        for spec in self.rounds:
            if spec.round == number:
                return spec
        raise KeyError(f"scenario {self.scenario_id} has no round {number}")
