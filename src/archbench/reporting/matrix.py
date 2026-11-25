"""Progress matrix for the v2 campaign: question x model, answers done / target, spend and projected cost.

Targets live in a small YAML file (see `load_targets`). Each condition names a directory of experiment configs;
every experiment there contributes one row (its scenario) and one column per configuration. A cell counts the
answers whose plan used exactly these bytes: the same scenario content hash and the same configuration file
SHA-256. Answers from any experiment qualify (pilot slices, top-up runs), answers to an older wording or from an
older config revision do not.

Spend is real money recorded per attempt (`api_cost_usd` in metrics.json), failed billed attempts included;
attempts whose cost is unknown are counted separately and never treated as free. Evaluation spend is the sum of
the evaluator calls stored under each matching experiment's evaluations. Projections use the mean observed cost
per answer of the same model, condition and decision target, else of the same model and condition.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..domain.common import content_hash, utc_now
from ..domain.execution import CostIncurred, TaskStatus, observed_spend
from ..domain.plan import ExperimentPlan
from ..domain.records import RecordKind
from ..evaluation.selection import scenario_selection
from ..planning.compiler import load_experiment
from ..planning.loader import load_scenario
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database


@dataclass
class Spend:
    usd: float = 0.0
    unknown: int = 0  # billed (or possibly billed) attempts whose cost was not recorded

    def add(self, other: Spend) -> None:
        self.usd += other.usd
        self.unknown += other.unknown


@dataclass
class Cell:
    condition: str
    question: str  # scenario id without the v2 / search prefix, e.g. s1-personal-family-model-top-three
    target_kind: str  # provider | model | framework | infrastructure (other outside v2)
    model: str  # configuration key
    target: int
    done: int = 0
    in_progress: int = 0
    failed: int = 0
    attempts: int = 0
    spend: Spend = field(default_factory=Spend)
    experiments: set[str] = field(default_factory=set)

    @property
    def remaining(self) -> int:
        return max(self.target - self.done, 0)


def load_targets(path: Path) -> dict[str, Any]:
    """{conditions: {name: {experiments: <dir>, target: N}}, overrides: [{condition, question?, model?, target}]}"""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("conditions"), dict):
        raise ValueError(f"{path}: expected a mapping with 'conditions'")
    for name, spec in data["conditions"].items():
        if not isinstance(spec, dict) or not isinstance(spec.get("target"), int) or spec["target"] < 0:
            raise ValueError(f"{path}: condition {name} needs experiments and a non-negative integer target")
    for override in data.get("overrides") or []:
        if not isinstance(override.get("target"), int) or override.get("condition") not in data["conditions"]:
            raise ValueError(f"{path}: bad override {override}")
    return data


def _target(targets: dict[str, Any], condition: str, question: str, model: str) -> int:
    value = int(targets["conditions"][condition]["target"])
    for o in targets.get("overrides") or []:  # later overrides win
        if o["condition"] == condition and o.get("question", question) == question and o.get("model", model) == model:
            value = int(o["target"])
    return value


def _question(scenario_id: str) -> str:
    return scenario_id.removeprefix("agentic-design-bias-v2-").removeprefix("search-")


def _attempt_cost(db: Database, store: ArtifactStore, attempt_id: str, incurred: CostIncurred) -> Spend:
    if incurred is CostIncurred.NO:
        return Spend()
    ref = db.artifacts_for(attempt_id).get("metrics.json")
    usd, complete = observed_spend(store.read_json(ref) if ref else None)
    return Spend(usd=usd, unknown=0 if complete else 1)


def _evaluation_spend(db: Database, store: ArtifactStore, experiment_id: str) -> Spend:
    spend = Spend()
    for ref in db.artifacts_under(f"{experiment_id}/evaluations/"):
        if ref.name != "metrics.json":
            continue
        usd, complete = observed_spend(store.read_json(ref))
        spend.add(Spend(usd=usd, unknown=0 if complete else 1))
    return spend


def build_matrix(db: Database, store: ArtifactStore, project: Path, targets_path: Path) -> dict[str, Any]:
    targets = load_targets(targets_path)
    # Plans by (scenario hash, configuration bytes) -> experiment ids, across every frozen plan.
    by_bytes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in db.list_records(kind=RecordKind.EXPERIMENT_PLAN):
        plan = ExperimentPlan.model_validate(record.payload)
        scenario_sha = content_hash(plan.scenario)
        for binding in plan.configurations:
            by_bytes[(scenario_sha, binding.source.sha256)].add(plan.experiment_id)

    conditions: list[dict[str, Any]] = []
    cells: list[Cell] = []
    for condition, spec in targets["conditions"].items():
        directory = project / spec["experiments"]
        models: list[dict[str, str]] = []
        questions: list[str] = []
        for path in sorted(directory.glob("*.yaml")):
            loaded = load_experiment(path, project)
            scenario, _ = load_scenario(project / loaded.spec.scenario, project)
            scenario_sha = content_hash(scenario)
            question = _question(scenario.scenario_id)
            kind: str
            try:
                kind = scenario_selection(scenario.scenario_id)[0]
            except ValueError:
                kind = "other"  # not a v2 selection question (e.g. the v1 test fixtures)
            questions.append(question)
            for config, frozen in loaded.configurations:
                if config.key not in {m["key"] for m in models}:
                    models.append({"key": config.key, "label": config.model.rsplit("/", 1)[-1],
                                   "provider": config.provider})
                cell = Cell(condition=condition, question=question, target_kind=kind, model=config.key,
                            target=_target(targets, condition, question, config.key))
                cell.experiments = by_bytes.get((scenario_sha, frozen.sha256), set())
                cells.append(cell)
        conditions.append({"name": condition, "models": models, "questions": questions,
                           "experiments": str(spec["experiments"])})

    # Count answers, open work and spend per cell from the experiments that used the same bytes.
    tasks_cache: dict[str, Any] = {}
    evaluation: dict[str, Spend] = defaultdict(Spend)
    seen_eval: set[str] = set()
    for cell in cells:
        for experiment_id in sorted(cell.experiments):
            if experiment_id not in tasks_cache:
                tasks = [t for t in db.list_tasks(experiment_id) if t.round == 1]
                attempts = db.list_attempts(experiment_id=experiment_id)
                tasks_cache[experiment_id] = (tasks, attempts)
            tasks, attempts = tasks_cache[experiment_id]
            mine = {t.task_id for t in tasks if t.configuration_key == cell.model}
            for task in tasks:
                if task.task_id not in mine:
                    continue
                if db.response_for_task(task.task_id) is not None:
                    cell.done += 1
                elif task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    cell.in_progress += 1
                else:
                    cell.failed += 1
            for attempt in attempts:
                if attempt.task_id in mine:
                    cell.attempts += 1
                    cell.spend.add(_attempt_cost(db, store, attempt.attempt_id, attempt.cost_incurred))
            if experiment_id not in seen_eval:
                seen_eval.add(experiment_id)
                evaluation[cell.condition].add(_evaluation_spend(db, store, experiment_id))

    # Mean cost per answer: same model + condition + decision target, else same model + condition.
    def mean(keyfn: Any) -> dict[Any, float]:
        usd: dict[Any, float] = defaultdict(float)
        answers: dict[Any, int] = defaultdict(int)
        for c in cells:
            if c.done and not c.spend.unknown:
                usd[keyfn(c)] += c.spend.usd
                answers[keyfn(c)] += c.done
        return {k: usd[k] / answers[k] for k in answers}

    fine = mean(lambda c: (c.condition, c.model, c.target_kind))
    coarse = mean(lambda c: (c.condition, c.model))
    # Rough fallbacks for models with no answers yet: the condition's mean for the decision target, else overall.
    rough_kind = mean(lambda c: (c.condition, c.target_kind))
    rough = mean(lambda c: c.condition)

    def unit_cost(c: Cell) -> float | None:
        return fine.get((c.condition, c.model, c.target_kind), coarse.get((c.condition, c.model)))

    def estimate(c: Cell) -> tuple[float | None, bool]:
        """Cost per answer and whether it is a rough condition-wide fallback."""
        own = unit_cost(c)
        if own is not None:
            return own, False
        fallback = rough_kind.get((c.condition, c.target_kind), rough.get(c.condition))
        return fallback, fallback is not None

    summary: dict[str, Any] = {}
    for cond in conditions:
        cond_cells = [c for c in cells if c.condition == cond["name"]]
        spent = Spend()
        for c in cond_cells:
            spent.add(c.spend)
        open_cells = [(*estimate(c), c.remaining) for c in cond_cells if c.remaining]
        per_rep = [estimate(c) for c in cond_cells]
        summary[cond["name"]] = {
            "cells": len(cond_cells), "answers": sum(c.done for c in cond_cells), "target": sum(c.target for c in cond_cells),
            "in_progress": sum(c.in_progress for c in cond_cells), "failed": sum(c.failed for c in cond_cells),
            "spent_usd": round(spent.usd, 4), "spent_unknown": spent.unknown,
            "evaluation_usd": round(evaluation[cond["name"]].usd, 4),
            "evaluation_unknown": evaluation[cond["name"]].unknown,
            "projected_to_target_usd": round(sum(x * n for x, _, n in open_cells if x is not None), 4),
            "projection_unpriced_cells": sum(1 for x, _, _ in open_cells if x is None),
            "projection_rough_cells": sum(1 for x, r, _ in open_cells if x is not None and r),
            "per_extra_repetition_usd": round(sum(x for x, _ in per_rep if x is not None), 4),
            "per_extra_repetition_unpriced_cells": sum(1 for x, _ in per_rep if x is None),
            "per_extra_repetition_rough_cells": sum(1 for x, r in per_rep if x is not None and r),
        }
    return {
        "generated_at": utc_now(), "targets_file": str(targets_path), "conditions": conditions, "summary": summary,
        "cells": [{"condition": c.condition, "question": c.question, "target_kind": c.target_kind, "model": c.model,
                   "target": c.target, "done": c.done, "in_progress": c.in_progress, "failed": c.failed,
                   "attempts": c.attempts, "spent_usd": round(c.spend.usd, 6), "spent_unknown": c.spend.unknown,
                   "unit_cost_usd": unit_cost(c), "experiments": sorted(c.experiments)} for c in cells],
    }


def _usd(value: float | None) -> str:
    return "?" if value is None else f"${value:,.2f}" if value >= 1 else f"${value:.3f}"


def render_html(data: dict[str, Any], *, refresh_s: int | None = None) -> str:
    e = html.escape
    refresh = f'<meta http-equiv="refresh" content="{refresh_s}">' if refresh_s else ""
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>v2 campaign matrix</title>
<meta name="viewport" content="width=device-width, initial-scale=1">{refresh}
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e2de;--none:#f1f0ec;--part:#fbe8b8;--done:#cfe8d2;
--fail:#b3261e;--run:#2f5fb3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#191917;--fg:#ecebe7;--muted:#a3a29c;--line:#34332f;--none:#262522;
--part:#5a4a1e;--done:#244a2c;--fail:#f2b8b5;--run:#9ab8f0}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif;margin:24px 16px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} .muted{{color:var(--muted)}}
.stats{{display:flex;flex-wrap:wrap;gap:8px 24px;margin:8px 0 12px}} .stats b{{font-variant-numeric:tabular-nums}}
.wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:6px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:center;white-space:nowrap}}
th{{font-weight:600;font-size:12px}} th.q,td.q{{text-align:left;position:sticky;left:0;background:var(--bg)}}
thead th{{vertical-align:bottom}} thead th span{{writing-mode:vertical-rl;transform:rotate(180deg)}}
td.c{{min-width:34px}} td.none{{background:var(--none)}} td.part{{background:var(--part)}} td.done{{background:var(--done)}}
.f{{color:var(--fail);font-size:11px}} .r{{color:var(--run);font-size:11px}} tr.sit td.q{{font-weight:600;padding-top:10px}}
tfoot td{{font-size:12px;color:var(--muted)}}
</style></head><body>
<h1>v2 campaign matrix</h1>
<div class="muted">Answers done / target per question and model. Targets: {e(data['targets_file'])}.
Updated {e(data['generated_at'])}. Red: failed tasks; blue: pending or running.</div>"""]
    cells = {(c["condition"], c["question"], c["model"]): c for c in data["cells"]}
    for cond in data["conditions"]:
        s = data["summary"][cond["name"]]
        unknown = f" + {s['spent_unknown']} unpriced" if s["spent_unknown"] else ""
        eval_unknown = f" + {s['evaluation_unknown']} unpriced" if s["evaluation_unknown"] else ""
        def gaps(rough: int, unpriced: int) -> str:
            notes = ([f"{rough} cells estimated from other models"] if rough else []) + (
                [f"{unpriced} cells without any cost yet"] if unpriced else [])
            return f' <span class="muted">({"; ".join(notes)})</span>' if notes else ""
        proj_gap = gaps(s["projection_rough_cells"], s["projection_unpriced_cells"])
        rep_gap = gaps(s["per_extra_repetition_rough_cells"], s["per_extra_repetition_unpriced_cells"])
        parts.append(f"""<h2>{e(cond['name'])} <span class="muted">({e(cond['experiments'])})</span></h2>
<div class="stats"><span>Answers <b>{s['answers']}</b> / {s['target']}</span>
<span>In progress <b>{s['in_progress']}</b></span><span>Failed <b>{s['failed']}</b></span>
<span>Answer spend <b>{_usd(s['spent_usd'])}</b>{unknown}</span>
<span>Evaluation spend <b>{_usd(s['evaluation_usd'])}</b>{eval_unknown}</span>
<span>To reach target <b>{_usd(s['projected_to_target_usd'])}</b>{proj_gap}</span>
<span>Each +1 repetition <b>{_usd(s['per_extra_repetition_usd'])}</b>{rep_gap}</span></div>
<div class="wrap"><table><thead><tr><th class="q">Question</th>""")
        parts += [f'<th title="{e(m["key"])}"><span>{e(m["label"])}</span></th>' for m in cond["models"]]
        parts.append("</tr></thead><tbody>")
        situation = None
        for q in cond["questions"]:
            sit = q.split("-", 1)[0]
            if sit != situation:
                situation = sit
                parts.append(f'<tr class="sit"><td class="q">{e(sit.upper())}</td>'
                             + "<td></td>" * len(cond["models"]) + "</tr>")
            parts.append(f'<tr><td class="q">{e(q.split("-", 1)[1])}</td>')
            for m in cond["models"]:
                c = cells.get((cond["name"], q, m["key"]))
                if c is None:
                    parts.append("<td></td>")
                    continue
                state = "done" if c["done"] >= c["target"] else "part" if c["done"] else "none"
                tip = (f"{m['key']} | {c['done']}/{c['target']} answers, {c['attempts']} attempts, "
                       f"spent {_usd(c['spent_usd'])}" + (f" + {c['spent_unknown']} unpriced" if c["spent_unknown"]
                                                          else "")
                       + f", ~{_usd(c['unit_cost_usd'])}/answer | " + ", ".join(c["experiments"]))
                extra = (f'<span class="f"> ✕{c["failed"]}</span>' if c["failed"] else "") + (
                    f'<span class="r"> ●{c["in_progress"]}</span>' if c["in_progress"] else "")
                parts.append(f'<td class="c {state}" title="{e(tip)}">{c["done"]}/{c["target"]}{extra}</td>')
            parts.append("</tr>")
        parts.append('</tbody><tfoot><tr><td class="q">spent / answer</td>')
        for m in cond["models"]:
            mine = [c for c in data["cells"] if c["condition"] == cond["name"] and c["model"] == m["key"]]
            spent, done = sum(c["spent_usd"] for c in mine), sum(c["done"] for c in mine)
            parts.append(f'<td title="{e(m["key"])}">{_usd(spent)}<br>{_usd(spent / done) if done else "–"}</td>')
        parts.append("</tr></tfoot></table></div>")
    embedded = json.dumps(data).replace("</", "<\\/")
    parts.append(f'<script type="application/json" id="data">{embedded}</script></body></html>')
    return "".join(parts)
