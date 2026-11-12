"""Load versioned inputs while preserving their exact bytes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..domain.common import sha256_bytes, sha256_text
from ..domain.configuration import ModelConfiguration
from ..domain.evaluation import EvaluationPolicy, Rubric
from ..domain.plan import FrozenFile
from ..domain.scenario import RoundSpec, Scenario
from ..secrets import GUARD


def freeze_file(path: Path, base: Path) -> FrozenFile:
    data = path.read_bytes()
    GUARD.check(data, f"frozen copy of {path} (keep credentials in secrets.local.yaml, never in configs)")
    try:
        rel = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        rel = path.resolve().as_posix()
    return FrozenFile(path=rel, sha256=sha256_bytes(data), content=data.decode("utf-8"))


def parse_yaml(frozen: FrozenFile) -> dict[str, Any]:
    value = yaml.safe_load(frozen.content)
    if not isinstance(value, dict):
        raise ValueError(f"{frozen.path} must contain a YAML mapping")
    return value


def load_scenario(directory: Path, base: Path) -> tuple[Scenario, tuple[FrozenFile, ...]]:
    meta_file = freeze_file(directory / "metadata.yaml", base)
    meta = parse_yaml(meta_file)
    files = [meta_file]
    system_file = freeze_file(directory / meta["system_prompt"], base)
    files.append(system_file)
    rounds = []
    for entry in meta["rounds"]:
        prompt_file = freeze_file(directory / entry["prompt"], base)
        files.append(prompt_file)
        rounds.append(
            RoundSpec(
                round=entry["round"],
                name=entry["name"],
                prompt=prompt_file.content,
                prompt_sha256=sha256_text(prompt_file.content),
                parent_round=entry.get("parent_round"),
                required_sections=tuple(entry.get("required_sections", ())),
            )
        )
    scenario = Scenario(
        scenario_id=meta["scenario_id"],
        revision=str(meta["revision"]),
        system_prompt=system_file.content,
        system_prompt_sha256=sha256_text(system_file.content),
        rounds=tuple(rounds),
    )
    return scenario, tuple(files)


def load_configuration(path: Path, base: Path) -> tuple[ModelConfiguration, FrozenFile]:
    frozen = freeze_file(path, base)
    return ModelConfiguration.model_validate(parse_yaml(frozen)), frozen


def load_rubric(path: Path, base: Path) -> tuple[Rubric, FrozenFile]:
    frozen = freeze_file(path, base)
    return Rubric.model_validate(parse_yaml(frozen)), frozen


def load_policy(path: Path, base: Path) -> tuple[EvaluationPolicy, FrozenFile]:
    frozen = freeze_file(path, base)
    return EvaluationPolicy.model_validate(parse_yaml(frozen)), frozen
