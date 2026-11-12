"""Compile an experiment YAML into an immutable `ExperimentPlan` and persist it."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..domain.common import Frozen, content_hash, utc_now
from ..domain.configuration import ModelConfiguration
from ..domain.plan import ConfigBinding, ExperimentPlan, FrozenFile, RetryPolicy, SchedulingPolicy
from ..domain.records import Record, RecordKind
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from .loader import freeze_file, load_configuration, load_rubric, load_scenario, parse_yaml
from .provenance import dependency_versions, host_metadata, source_revision


class SchedulingSpec(Frozen):
    order: Literal["as_listed", "seeded_shuffle"] = "as_listed"
    seed: int | None = None


class ExperimentSpec(Frozen):
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    benchmark_revision: str
    scenario: str
    rubric: str
    configurations: tuple[str, ...] = Field(min_length=1)
    repetitions: int = Field(default=3, ge=1)
    retry: RetryPolicy = RetryPolicy()
    scheduling: SchedulingSpec = SchedulingSpec()
    pricing_snapshot: str | None = None


class PlanError(RuntimeError):
    pass


class LoadedExperiment(Frozen):
    spec: ExperimentSpec
    experiment_file: FrozenFile
    configurations: tuple[tuple[ModelConfiguration, FrozenFile], ...]


def load_experiment(path: Path, base: Path) -> LoadedExperiment:
    frozen = freeze_file(path, base)
    spec = ExperimentSpec.model_validate(parse_yaml(frozen))
    configs = tuple(load_configuration(base / p, base) for p in spec.configurations)
    keys = [c.key for c, _ in configs]
    if len(set(keys)) != len(keys):
        raise PlanError(f"duplicate configuration keys in experiment: {keys}")
    return LoadedExperiment(spec=spec, experiment_file=frozen, configurations=configs)


def scenario_sha256_for(path: Path, base: Path) -> str:
    scenario, _ = load_scenario(path, base)
    return content_hash(scenario)


def find_qualification(db: Database, config_key: str, config_sha256: str, scenario_sha256: str) -> Record | None:
    matches = [
        r
        for r in db.list_records(kind=RecordKind.QUALIFICATION, subject_id=config_key)
        if r.payload.get("config_sha256") == config_sha256
        and r.payload.get("scenario_sha256") == scenario_sha256
        and r.payload.get("report", {}).get("passed") is True
    ]
    return matches[-1] if matches else None


def compile_plan(experiment_path: Path, *, base: Path, db: Database) -> ExperimentPlan:
    loaded = load_experiment(experiment_path, base)
    spec = loaded.spec
    scenario, scenario_files = load_scenario(base / spec.scenario, base)
    rubric, rubric_file = load_rubric(base / spec.rubric, base)
    scenario_sha = content_hash(scenario)

    bindings: list[ConfigBinding] = []
    unqualified: list[str] = []
    for config, frozen in loaded.configurations:
        qualification = find_qualification(db, config.key, frozen.sha256, scenario_sha)
        if qualification is None:
            unqualified.append(config.key)
            continue
        bindings.append(ConfigBinding(configuration=config, source=frozen, qualification_id=qualification.record_id))
    if unqualified:
        raise PlanError(
            "configurations lack a passing qualification for these exact bytes and scenario: "
            + ", ".join(unqualified)
            + " (run `bench qualify`)"
        )

    order = [b.configuration.key for b in bindings]
    if spec.scheduling.order == "seeded_shuffle":
        if spec.scheduling.seed is None:
            raise PlanError("seeded_shuffle requires an explicit seed")
        random.Random(spec.scheduling.seed).shuffle(order)

    return ExperimentPlan(
        experiment_id=spec.experiment_id,
        benchmark_revision=spec.benchmark_revision,
        compiled_at=utc_now(),
        experiment_file=loaded.experiment_file,
        scenario=scenario,
        scenario_files=scenario_files,
        rubric_revision=rubric.revision,
        rubric_sha256=rubric_file.sha256,
        configurations=tuple(bindings),
        repetitions=spec.repetitions,
        retry=spec.retry,
        scheduling=SchedulingPolicy(
            order=spec.scheduling.order, seed=spec.scheduling.seed, resolved_order=tuple(order)
        ),
        pricing_snapshot=freeze_file(base / spec.pricing_snapshot, base) if spec.pricing_snapshot else None,
        source=source_revision(base),
        dependency_versions=dependency_versions(),
        host=host_metadata(),
    )


def plan_record_id(experiment_id: str) -> str:
    return f"plan-{experiment_id}"


def persist_plan(db: Database, store: ArtifactStore, plan: ExperimentPlan) -> Record:
    record_id = plan_record_id(plan.experiment_id)
    if db.get_record(record_id) is not None:
        raise PlanError(f"experiment {plan.experiment_id} is already frozen; use a new experiment_id")
    payload = plan.model_dump(mode="json")
    sink = store.sink(plan.experiment_id)
    sink.write_json("plan.json", payload)
    sink.write_json("manifest.json", run_manifest(plan))
    record = Record.create(
        kind=RecordKind.EXPERIMENT_PLAN,
        record_id=record_id,
        experiment_id=plan.experiment_id,
        payload=payload,
        inputs={b.qualification_id: b.source.sha256 for b in plan.configurations},
        code_revision=plan.source.label,
    )
    with db.transaction():
        db.insert_record(record)
        db.register_artifacts(record_id, sink.refs)
    return record


def load_plan(db: Database, experiment_id: str) -> ExperimentPlan:
    record = db.get_record(plan_record_id(experiment_id))
    if record is None:
        raise PlanError(f"no frozen plan for experiment {experiment_id} (run `bench plan`)")
    return ExperimentPlan.model_validate(record.payload)


def run_manifest(plan: ExperimentPlan) -> dict[str, object]:
    return {
        "experiment_id": plan.experiment_id,
        "benchmark_revision": plan.benchmark_revision,
        "plan_sha256": plan.plan_sha256,
        "compiled_at": plan.compiled_at,
        "source": plan.source.model_dump(mode="json"),
        "experiment_config_sha256": plan.experiment_file.sha256,
        "scenario": {"id": plan.scenario.scenario_id, "revision": plan.scenario.revision},
        "prompt_hashes": {
            "system": plan.scenario.system_prompt_sha256,
            **{f"round-{r.round}": r.prompt_sha256 for r in plan.scenario.rounds},
        },
        "rubric": {"revision": plan.rubric_revision, "sha256": plan.rubric_sha256},
        "configurations": [
            {
                "key": b.configuration.key,
                "track": b.configuration.track.value,
                "adapter": b.configuration.adapter,
                "provider": b.configuration.provider,
                "model": b.configuration.model,
                "config_sha256": b.source.sha256,
                "qualification_id": b.qualification_id,
            }
            for b in plan.configurations
        ],
        "repetitions": plan.repetitions,
        "retry": plan.retry.model_dump(mode="json"),
        "scheduling": plan.scheduling.model_dump(mode="json"),
        "pricing_snapshot_sha256": plan.pricing_snapshot.sha256 if plan.pricing_snapshot else None,
        "dependency_versions": plan.dependency_versions,
        "host": plan.host,
    }
