"""Campaign progress matrix: cells count answers made with the current bytes, targets and overrides apply."""

from __future__ import annotations

from pathlib import Path

import yaml

from archbench.execution.controller import Controller
from archbench.reporting.matrix import build_matrix, render_html

from .conftest import Env


def targets(env: Env, target: int, overrides: list[dict[str, object]] | None = None) -> Path:
    path = env.tmp / "targets.yaml"
    path.write_text(yaml.safe_dump({"conditions": {"run-x": {"experiments": str(env.tmp / "experiments"),
                                                              "target": target}},
                                    "overrides": overrides or []}))
    return path


def test_cells_count_answers_with_the_current_bytes_and_apply_targets(env: Env) -> None:
    alpha, beta = env.config("alpha"), env.config("beta")
    plan = env.plan(env.experiment("exp-m", [alpha, beta], repetitions=2))
    Controller(env.db, env.store, env.registry).run(plan)
    path = targets(env, 3, [{"condition": "run-x", "model": "beta@1", "target": 1}])
    data = build_matrix(env.db, env.store, env.base, path)
    cells = {c["model"]: c for c in data["cells"]}
    assert (cells["alpha@1"]["done"], cells["alpha@1"]["target"]) == (2, 3)  # round-1 answers only
    assert (cells["beta@1"]["done"], cells["beta@1"]["target"]) == (2, 1)
    assert cells["alpha@1"]["experiments"] == ["exp-m"]
    summary = data["summary"]["run-x"]
    assert (summary["answers"], summary["target"], summary["failed"]) == (4, 4, 0)
    assert summary["spent_usd"] == 0 and summary["spent_unknown"] == 0  # the fake adapter bills nothing
    page = render_html(data)
    assert "2/3" in page and "2/1" in page and "run-x" in page

    # Changing a configuration's bytes starts its cell from zero: old answers no longer count.
    alpha.write_text(alpha.read_text() + "# revised\n")
    data = build_matrix(env.db, env.store, env.base, path)
    cells = {c["model"]: c for c in data["cells"]}
    assert cells["alpha@1"]["done"] == 0 and cells["beta@1"]["done"] == 2


def test_failed_tasks_are_counted_separately(env: Env) -> None:
    flaky = env.config("flaky", adapter_options={"permanent_failure_rounds": [1]})
    plan = env.plan(env.experiment("exp-f", [flaky], repetitions=1))
    Controller(env.db, env.store, env.registry).run(plan)
    data = build_matrix(env.db, env.store, env.base, targets(env, 1))
    (cell,) = data["cells"]
    assert (cell["done"], cell["failed"], cell["attempts"]) == (0, 1, 1)
