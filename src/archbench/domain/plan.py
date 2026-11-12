"""The frozen experiment plan compiled before scored execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import ARTIFACT_SCHEMA_VERSION, PLAN_SCHEMA_VERSION, Frozen, content_hash
from .configuration import ModelConfiguration
from .execution import FailureKind
from .scenario import Scenario


class FrozenFile(Frozen):
    """Exact bytes of an input file. A Git commit alone does not identify uncommitted changes."""

    path: str
    sha256: str
    content: str


class SourceRevision(Frozen):
    available: bool
    commit: str | None = None
    dirty: bool | None = None
    dirty_state_sha256: str | None = None

    @property
    def label(self) -> str:
        if not self.available or self.commit is None:
            return "unknown"
        if self.dirty:
            return f"{self.commit}+dirty.{(self.dirty_state_sha256 or '')[:12]}"
        return self.commit


class RetryPolicy(Frozen):
    """Persisted before running. Every retry is a new attempt; weak answers are never regenerated."""

    max_attempts: int = Field(default=3, ge=1)
    retry_on: tuple[FailureKind, ...] = (
        FailureKind.TRANSIENT_TRANSPORT,
        FailureKind.RATE_LIMITED,
        FailureKind.INTERRUPTED,
    )
    selection: Literal["first_success"] = "first_success"


class SchedulingPolicy(Frozen):
    order: Literal["as_listed", "seeded_shuffle"] = "as_listed"
    seed: int | None = None
    resolved_order: tuple[str, ...] = ()
    concurrency: Literal["sequential"] = "sequential"


class ConfigBinding(Frozen):
    configuration: ModelConfiguration
    source: FrozenFile
    qualification_id: str


class ExperimentPlan(Frozen):
    plan_schema_version: str = PLAN_SCHEMA_VERSION
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION
    experiment_id: str
    benchmark_revision: str
    compiled_at: str
    experiment_file: FrozenFile
    scenario: Scenario
    scenario_files: tuple[FrozenFile, ...]
    rubric_revision: str
    rubric_sha256: str
    configurations: tuple[ConfigBinding, ...]
    repetitions: int = Field(ge=1)
    retry: RetryPolicy
    scheduling: SchedulingPolicy
    pricing_snapshot: FrozenFile | None = None
    source: SourceRevision
    dependency_versions: dict[str, str]
    host: dict[str, Any]

    @property
    def plan_sha256(self) -> str:
        return content_hash(self)

    def binding(self, configuration_key: str) -> ConfigBinding:
        for binding in self.configurations:
            if binding.configuration.key == configuration_key:
                return binding
        raise KeyError(f"configuration {configuration_key} is not part of experiment {self.experiment_id}")
