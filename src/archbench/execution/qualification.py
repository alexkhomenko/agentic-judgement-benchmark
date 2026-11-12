"""Qualification is its own persisted operation, keyed to exact configuration bytes and scenario."""

from __future__ import annotations

from ..adapters.base import QualificationRequirements
from ..adapters.registry import AdapterRegistry
from ..domain.common import content_hash
from ..domain.configuration import ModelConfiguration
from ..domain.plan import FrozenFile
from ..domain.records import Record, RecordKind
from ..domain.scenario import Scenario
from ..storage.database import Database


def qualification_requirements(scenario: Scenario, config: ModelConfiguration) -> QualificationRequirements:
    return QualificationRequirements(
        baseline_prompt_chars=len(scenario.system_prompt) + len(scenario.baseline.prompt),
        longest_follow_up_chars=max((len(r.prompt) for r in scenario.follow_ups), default=0),
        output_budget_tokens=config.explicit_int("max_output_tokens"),
    )


def qualify_configuration(
    db: Database,
    registry: AdapterRegistry,
    config: ModelConfiguration,
    source: FrozenFile,
    scenario: Scenario,
    code_revision: str | None,
) -> Record:
    adapter = registry.get(config.adapter)
    requirements = qualification_requirements(scenario, config)
    report = adapter.qualify(config, requirements)
    scenario_sha = content_hash(scenario)
    record = Record.create(
        kind=RecordKind.QUALIFICATION,
        prefix="qual",
        subject_id=config.key,
        inputs={f"config:{config.key}": source.sha256, f"scenario:{scenario.scenario_id}": scenario_sha},
        code_revision=code_revision,
        payload={
            "config_sha256": source.sha256,
            "scenario_sha256": scenario_sha,
            "report": report.model_dump(mode="json"),
        },
    )
    db.insert_record(record)
    return record
