"""Plan compilation, qualification, and immutable configuration identity."""

from __future__ import annotations

import pytest

from archbench.planning.compiler import PlanError, compile_plan, load_plan, persist_plan

from .conftest import Env


def test_unqualified_configuration_cannot_be_planned(env: Env) -> None:
    exp = env.experiment("exp-unq", [env.config("alpha")])
    with pytest.raises(PlanError, match="qualification"):
        compile_plan(exp, base=env.base, db=env.db)


def test_context_envelope_covers_baseline_answer_and_follow_up(env: Env) -> None:
    # Fits a short prompt but not baseline + permitted answer + follow-up + reserved follow-up output.
    exp = env.experiment("exp-small", [env.config("small", context_limit=8000)])
    (record,) = env.qualify(exp)
    report = record.payload["report"]
    assert report["passed"] is False
    check = next(c for c in report["checks"] if c["name"] == "context_envelope")
    assert check["passed"] is False
    with pytest.raises(PlanError):
        compile_plan(exp, base=env.base, db=env.db)


def test_unsupported_explicit_parameter_fails_qualification(env: Env) -> None:
    params = {"max_output_tokens": {"state": "explicit", "value": 4000},
              "temperature": {"state": "explicit", "value": 0.2}}
    exp = env.experiment("exp-param", [env.config("temp", parameters=params)])
    (record,) = env.qualify(exp)
    checks = {c["name"]: c for c in record.payload["report"]["checks"]}
    assert checks["parameter_support"]["passed"] is False
    assert checks["continuation"]["passed"] is True


def test_qualification_is_bound_to_exact_config_bytes(env: Env) -> None:
    cfg = env.config("alpha")
    exp = env.experiment("exp-bytes", [cfg])
    env.qualify(exp)
    cfg.write_text(cfg.read_text() + "# a comment changes the bytes\n")
    with pytest.raises(PlanError, match="exact bytes"):
        compile_plan(exp, base=env.base, db=env.db)


def test_plan_preserves_bytes_and_cannot_be_refrozen(env: Env) -> None:
    cfg = env.config("alpha")
    exp = env.experiment("exp-freeze", [cfg])
    plan = env.plan(exp)
    assert plan.configurations[0].source.content == cfg.read_text()
    assert plan.experiment_file.content == exp.read_text()
    assert {f.path.rsplit("/", 1)[-1] for f in plan.scenario_files} >= {"metadata.yaml", "baseline.md"}
    assert load_plan(env.db, plan.experiment_id).plan_sha256 == plan.plan_sha256
    with pytest.raises(PlanError, match="already frozen"):
        persist_plan(env.db, env.store, compile_plan(exp, base=env.base, db=env.db))


def test_seeded_shuffle_is_reproducible_and_requires_seed(env: Env) -> None:
    cfgs = [env.config(n) for n in ("a1", "a2", "a3", "a4")]
    exp = env.experiment("exp-order", cfgs, scheduling={"order": "seeded_shuffle", "seed": 7})
    env.qualify(exp)
    first = compile_plan(exp, base=env.base, db=env.db).scheduling.resolved_order
    second = compile_plan(exp, base=env.base, db=env.db).scheduling.resolved_order
    assert first == second and sorted(first) == ["a1@1", "a2@1", "a3@1", "a4@1"]
    bad = env.experiment("exp-noseed", cfgs, scheduling={"order": "seeded_shuffle"})
    with pytest.raises(PlanError, match="seed"):
        compile_plan(bad, base=env.base, db=env.db)


def test_scenario_rejects_cumulative_follow_up_chains() -> None:
    from archbench.domain.common import sha256_text
    from archbench.domain.scenario import RoundSpec, Scenario

    def spec(n: int, parent: int | None) -> RoundSpec:
        return RoundSpec(round=n, name=f"r{n}", prompt=f"p{n}", prompt_sha256=sha256_text(f"p{n}"), parent_round=parent)

    with pytest.raises(ValueError, match="cumulative"):
        Scenario(scenario_id="s", revision="1", system_prompt="s", system_prompt_sha256=sha256_text("s"),
                 rounds=(spec(1, None), spec(2, 1), spec(3, 2)))


def test_parameter_state_is_separate_from_value() -> None:
    from archbench.domain.configuration import ParameterSetting, ParameterState

    with pytest.raises(ValueError):
        ParameterSetting(state=ParameterState.EXPLICIT)
    with pytest.raises(ValueError):
        ParameterSetting(state=ParameterState.UNKNOWN, value=0.7)
    unknown = ParameterSetting(state=ParameterState.UNKNOWN)
    assert unknown.value is None and unknown.observed_effective is None
