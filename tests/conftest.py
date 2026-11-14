from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from archbench.adapters.registry import AdapterRegistry, default_registry
from archbench.domain.common import content_hash
from archbench.domain.plan import ExperimentPlan
from archbench.domain.records import Record, RecordKind
from archbench.execution.qualification import qualify_configuration
from archbench.planning.compiler import compile_plan, load_experiment, persist_plan
from archbench.planning.loader import load_scenario
from archbench.storage.artifacts import ArtifactStore
from archbench.storage.database import Database

# The tests exercise the harness, not the experiments: every configuration, rubric, catalog, and scenario
# they read comes from this test-owned tree. The project's own configs/, rubrics/, and scenarios/ belong
# to the experiments and are not part of the repository's tested code.
PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent / "fixtures" / "project"
EXPERIMENT_DATA = tuple(str(PROJECT / d) + os.sep for d in ("configs", "rubrics", "scenarios"))
BASE_CONFIG = yaml.safe_load((ROOT / "configs/models/fake/fake-alpha.yaml").read_text())


def _refuse_experiment_data(event: str, args: tuple[Any, ...]) -> None:
    """Fail any test that opens the project's experiment data instead of the fixtures."""
    if event == "open" and args and isinstance(args[0], str | bytes | os.PathLike):
        path = os.path.abspath(os.fsdecode(args[0]))
        if path.startswith(EXPERIMENT_DATA):
            raise PermissionError(f"tests must not read experiment data ({path}); use tests/fixtures/project")


sys.addaudithook(_refuse_experiment_data)


@dataclass
class Env:
    base: Path
    tmp: Path
    db: Database
    store: ArtifactStore
    registry: AdapterRegistry

    def config(self, name: str, **overrides: Any) -> Path:
        data = {**BASE_CONFIG, "configuration_id": name, **overrides}
        path = self.tmp / "configs" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    def experiment(self, experiment_id: str, configs: list[Path], **overrides: Any) -> Path:
        data = {
            "experiment_id": experiment_id,
            "benchmark_revision": "1.2",
            "scenario": "scenarios/personal-workflows",
            "rubric": "rubrics/personal-workflows/rubric.yaml",
            "configurations": [str(p) for p in configs],
            "repetitions": 3,
            **overrides,
        }
        path = self.tmp / "experiments" / f"{experiment_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    def qualify(self, experiment: Path) -> list[Record]:
        loaded = load_experiment(experiment, self.base)
        scenario, _ = load_scenario(self.base / loaded.spec.scenario, self.base)
        return [
            qualify_configuration(self.db, self.registry, c, f, scenario, "test")
            for c, f in loaded.configurations
        ]

    def force_qualify(self, experiment: Path) -> None:
        """Insert passing qualification records without running checks (to test runtime guards)."""
        loaded = load_experiment(experiment, self.base)
        scenario, _ = load_scenario(self.base / loaded.spec.scenario, self.base)
        for config, frozen in loaded.configurations:
            self.db.insert_record(Record.create(
                kind=RecordKind.QUALIFICATION, prefix="qual", subject_id=config.key,
                payload={"config_sha256": frozen.sha256, "scenario_sha256": content_hash(scenario),
                         "report": {"passed": True, "checks": []}},
            ))

    def plan(self, experiment: Path, *, force: bool = False) -> ExperimentPlan:
        if force:
            self.force_qualify(experiment)
        else:
            self.qualify(experiment)
        plan = compile_plan(experiment, base=self.base, db=self.db)
        persist_plan(self.db, self.store, plan)
        return plan


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    db = Database(tmp_path / "results" / "index.sqlite")
    e = Env(base=ROOT, tmp=tmp_path, db=db, store=ArtifactStore(tmp_path / "results"), registry=default_registry())
    yield e
    db.close()


def published_snapshot(db: Database) -> dict[str, list[tuple[Any, ...]]]:
    """All published rows, for checking that downstream stages leave upstream evidence untouched."""
    conn = db.connection
    return {
        table: [tuple(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()]
        for table in ("records", "attempts", "responses", "artifacts")
    }


@pytest.fixture(autouse=True)
def _isolated_secret_guard() -> Iterator[None]:
    from archbench.secrets import GUARD

    GUARD.clear()
    yield
    GUARD.clear()
