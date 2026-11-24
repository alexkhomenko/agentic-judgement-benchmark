"""Cross-model report: many configurations against the same probes, as one static, self-contained HTML file.

Inputs are named explicitly: the evaluation runs to include. Nothing looks up a "latest" run. Two runs that
cover the same probe and configuration are rejected rather than silently merged. The page embeds the exact
evaluation runs, policies, judges, extractors, catalogs, and a hash of the aggregated data, so a published
copy can be traced back to its records.

Quality is the adjudicated score where one exists. Otherwise it is the mean of the judges' raw scores, shown
as provisional, because no human has checked that judgement. Self-preference is measured across every run
included for a probe: how often a model puts its own vendor first, against how often the other vendors'
models put that same vendor first.

Reading persisted records only; never invokes a model.
"""

from __future__ import annotations

import html
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..domain.common import content_hash, utc_now
from ..domain.recommendations import UNKNOWN_VENDOR, UNLISTED, FrameworkCatalog
from ..domain.records import RecordKind
from ..evaluation.checks import RunIndex
from ..evaluation.records import effective, run_records
from ..evaluation.review import judge_raw_scores
from ..evaluation.runner import RunContext, evaluation_status, load_run
from ..planning.compiler import load_plan
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from .bias import bias_tables, wilson

CROSS_MODEL_REVISION = "cross-model-v1"

VENDOR_NAMES = {
    "anthropic": "Anthropic", "openai": "OpenAI", "google": "Google", "microsoft": "Microsoft", "aws": "AWS",
    "meta": "Meta", "mistral": "Mistral", "alibaba": "Alibaba", "deepseek": "DeepSeek", "xai": "xAI",
    "cohere": "Cohere", "moonshot": "Moonshot", "zhipu": "Zhipu", "minimax": "MiniMax",
    "independent": "Independent / open source", UNKNOWN_VENDOR: "Not in catalog", "other": "Other vendors",
    "none": "Nothing recommended", "bedrock": "AWS", "ollama": "Ollama", "zai": "Zhipu",
}
SERIES_SLOTS = 7  # categorical slots for vendors; the rest fold into "Other vendors"


class CrossModelError(ValueError):
    pass


# ----- collection --------------------------------------------------------------------------------------


def _probe(scenario_id: str) -> tuple[str, int]:
    match = re.search(r"-p(\d+)-", scenario_id + "-")
    number = int(match.group(1)) if match else 0
    return (f"P{number}" if number else scenario_id), number


def _vendor_name(vendor: str | None) -> str:
    return VENDOR_NAMES.get(vendor or "none", (vendor or "none").replace("-", " ").title())


def _counts_as_product(catalog: FrameworkCatalog, item_id: str) -> bool:
    return bool(UNLISTED.match(item_id)) or catalog.counts_for_preference(item_id)


def _item_vendor(catalog: FrameworkCatalog, item_id: str) -> str:
    return UNKNOWN_VENDOR if UNLISTED.match(item_id) else catalog.vendor_of(item_id)


def _item_name(catalog: FrameworkCatalog, item_id: str) -> str:
    entry = catalog.by_id.get(item_id)
    return entry.name if entry else item_id.removeprefix("unlisted:")


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": wilson(k, n)}


def _run_cells(db: Database, store: ArtifactStore, ctx: RunContext) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exp, run_id, rubric = ctx.experiment_id, ctx.run.record_id, ctx.rubric
    status = evaluation_status(db, run_id)
    index = RunIndex.build(db, ctx)
    adjudications = {str(r.subject_id): r.payload
                     for r in effective(run_records(db, RecordKind.ADJUDICATION, exp, run_id))}
    by_response = {str(p.subject_id): p for p in index.packets}
    probe, number = _probe(ctx.plan.scenario.scenario_id)
    catalog = ctx.catalog
    bias = None
    if catalog is not None:
        bias = bias_tables(ctx.plan, ctx.manifest, run_records(db, RecordKind.RECOMMENDATION_EXTRACTION, exp, run_id),
                           catalog, ctx.extractor.configuration.provider if ctx.extractor else None)
    bias_rows = {r["response_id"]: r for r in bias["responses"]} if bias else {}

    run_info: dict[str, Any] = {
        "evaluation_run_id": run_id, "experiment_id": exp, "probe": probe,
        "scenario": f"{ctx.plan.scenario.scenario_id}@{ctx.plan.scenario.revision}",
        "policy": str(ctx.run.policy_revision), "rubric": f"{rubric.rubric_id}@{rubric.revision}",
        "judges": [{"judge_id": j, "model": m.configuration.model, "provider": m.configuration.provider}
                   for j, m in ctx.judges.items()],
        "extractor": {"model": ctx.extractor.configuration.model, "provider": ctx.extractor.configuration.provider}
        if ctx.extractor else None,
        "catalog": {"id": catalog.catalog_id, "revision": catalog.revision, "status": catalog.status}
        if catalog else None,
        "human_checks": f"{status.human_checks.done}/{status.human_checks.expected}",
        "adjudicated": f"{status.adjudications.done}/{status.adjudications.expected}",
        "judge_evaluations": f"{status.judge_evaluations.done}/{status.judge_evaluations.expected}",
        "complete": status.complete,
        "plan_sha256": ctx.plan.plan_sha256,
        "run_sha256": ctx.run.payload_sha256,
    }

    cells = []
    candidate_providers = {b.configuration.provider for b in ctx.plan.configurations}
    for binding in ctx.plan.configurations:
        cfg = binding.configuration
        entries = [e for e in ctx.manifest["responses"] if e["configuration_key"] == cfg.key]
        own = catalog.vendor_for_provider(cfg.provider) if catalog else None
        scores: list[float] = []
        basis: Counter[str] = Counter()
        dims: dict[str, list[float]] = defaultdict(list)
        judge_gaps: list[float] = []
        cf_candidates = 0
        answers: list[dict[str, Any]] = []
        for e in sorted(entries, key=lambda e: e["repetition"]):
            rid = e["response_id"]
            packet = by_response.get(rid)
            if packet is None:
                continue
            judges = {j: r.payload["output"] for j, r in index.chosen_judges(packet.record_id).items()}
            adj = adjudications.get(rid)
            if adj is not None:
                scores.append(adj["score"]["adjusted_score"])
                basis["adjudicated"] += 1
                for d, v in adj["final_ratings"].items():
                    dims[d].append(v)
            elif judges:
                raw = judge_raw_scores(rubric, e["round"], judges)
                scores.append(statistics.fmean(raw.values()))
                basis["judge_mean"] += 1
                if len(raw) > 1:
                    judge_gaps.append(max(raw.values()) - min(raw.values()))
                for d in next(iter(judges.values()))["ratings"]:
                    dims[d].append(statistics.fmean(o["ratings"][d]["rating"] for o in judges.values()))
            cf_candidates += any(o["critical_failure_candidates"] for o in judges.values())
            row = bias_rows.get(rid)
            if row and row["extracted"] and catalog is not None:
                ids = row["recommended"].split(" > ") if row["recommended"] else []
                products = [i for i in ids if _counts_as_product(catalog, i)]
                answers.append({
                    "first": ids[0] if ids else None,
                    "first_name": _item_name(catalog, ids[0]) if ids else None,
                    "first_vendor": _item_vendor(catalog, ids[0]) if ids else "none",
                    "product_vendors": sorted({_item_vendor(catalog, i) for i in products}),
                    "as_written": row["recommended_as_written"],
                    "recommends_model": row["recommends_model"],
                    "open_weights": row["recommends_open_weights"],
                    "criteria": bool(row.get("selection_criteria_stated")),
                    "caveat": bool(row.get("landscape_caveat")),
                })
        cells.append({
            "probe": probe, "probe_number": number, "configuration": cfg.key,
            "model": cfg.model, "provider": cfg.provider, "vendor": own, "track": cfg.track.value,
            "adapter": cfg.adapter, "evaluation_run_id": run_id,
            "planned": ctx.plan.repetitions * len(ctx.plan.scenario.rounds), "responses": len(entries),
            "quality": {
                "n": len(scores), "mean": statistics.fmean(scores) if scores else None,
                "min": min(scores) if scores else None, "max": max(scores) if scores else None,
                "basis": dict(basis),
            },
            "judge_gap": statistics.fmean(judge_gaps) if judge_gaps else None,
            "dimensions": {d: statistics.fmean(v) for d, v in dims.items()},
            "critical_failure_candidates": cf_candidates,
            "answers": answers,
            "judges_share_vendor": [j["judge_id"] for j in run_info["judges"] if j["provider"] == cfg.provider],
            "extractor_shares_vendor": bool(ctx.extractor and ctx.extractor.configuration.provider == cfg.provider),
            "extractor_is_candidate": bool(ctx.extractor and ctx.extractor.configuration.provider in candidate_providers),
        })
    run_info["dimensions"] = [{"id": d.id, "title": d.title, "weight": d.weight} for d in rubric.dimensions]
    return run_info, cells


def _pending_experiments(db: Database, covered: set[tuple[str, str]], family: str) -> list[dict[str, Any]]:
    """Answers in the database for a probe of this family that no included evaluation run covers."""
    found = []
    experiments = sorted({str(r.experiment_id) for r in db.list_records(kind=RecordKind.EXPERIMENT_PLAN)})
    for exp in experiments:
        try:
            plan = load_plan(db, exp)
        except Exception:  # noqa: BLE001 - an unreadable plan is just not listed
            continue
        if not plan.scenario.scenario_id.startswith(family):
            continue
        probe, number = _probe(plan.scenario.scenario_id)
        responses = Counter(r.configuration_key for r in db.list_responses(exp))
        for binding in plan.configurations:
            cfg = binding.configuration
            if (probe, cfg.key) in covered or not responses.get(cfg.key):
                continue
            evaluated = any(r.experiment_id == exp for r in db.list_records(kind=RecordKind.EVALUATION_RUN))
            found.append({"probe": probe, "probe_number": number, "configuration": cfg.key, "model": cfg.model,
                          "provider": cfg.provider, "track": cfg.track.value, "adapter": cfg.adapter,
                          "experiment_id": exp, "responses": responses[cfg.key],
                          "state": "evaluated in a run not included" if evaluated else "answered, not evaluated"})
    return found


def _scenario_probes(directory: Path | None) -> dict[str, dict[str, Any]]:
    probes: dict[str, dict[str, Any]] = {}
    if directory is None:
        return probes
    system = directory / "system.md"
    for meta in sorted(directory.glob("*/metadata.yaml")):
        data = yaml.safe_load(meta.read_text())
        probe, number = _probe(str(data["scenario_id"]))
        first = data["rounds"][0]
        probes[probe] = {
            "probe": probe, "number": number, "scenario_id": data["scenario_id"], "name": first["name"],
            "prompt": (meta.parent / first["prompt"]).read_text().strip(),
            "system_prompt": system.read_text().strip() if system.exists() else "",
        }
    return probes


def collect(db: Database, store: ArtifactStore, run_ids: list[str], *, scenarios: Path | None = None,
            title: str = "Agentic design bias") -> dict[str, Any]:
    if not run_ids:
        raise CrossModelError("name at least one evaluation run")
    runs: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    probes = _scenario_probes(scenarios)
    family = ""
    for run_id in run_ids:
        try:
            ctx = load_run(db, run_id)
        except KeyError as exc:
            raise CrossModelError(str(exc)) from exc
        info, run_cells = _run_cells(db, store, ctx)
        clash = {(c["probe"], c["configuration"]) for c in run_cells} & {(c["probe"], c["configuration"]) for c in cells}
        if clash:
            raise CrossModelError(f"{run_id} covers {sorted(clash)} again; include one evaluation run per probe and model")
        runs.append(info)
        cells.extend(run_cells)
        scenario = ctx.plan.scenario
        probe, number = _probe(scenario.scenario_id)
        family = family or scenario.scenario_id.rsplit("-p", 1)[0]
        first = scenario.rounds[0]
        probes.setdefault(probe, {})
        probes[probe].update({
            "probe": probe, "number": number, "scenario_id": scenario.scenario_id, "name": first.name,
            "prompt": first.prompt.strip(), "system_prompt": scenario.system_prompt.strip(),
        })
    pending = _pending_experiments(db, {(c["probe"], c["configuration"]) for c in cells}, family) if family else []

    # Self-preference against a baseline pooled over every included run for the probe.
    for cell in cells:
        own = cell["vendor"]
        n = len(cell["answers"])
        if own is None:
            cell["own_first"] = cell["own_included"] = cell["baseline"] = None
            continue
        cell["own_first"] = _rate(sum(a["first_vendor"] == own for a in cell["answers"]), n)
        cell["own_included"] = _rate(sum(own in a["product_vendors"] for a in cell["answers"]), n)
        others = [c for c in cells if c["probe"] == cell["probe"] and c["vendor"] != own and c["answers"]]
        cell["baseline"] = {
            "models": len(others),
            "first": statistics.fmean(sum(a["first_vendor"] == own for a in o["answers"]) / len(o["answers"])
                                      for o in others) if others else None,
            "included": statistics.fmean(sum(own in a["product_vendors"] for a in o["answers"]) / len(o["answers"])
                                         for o in others) if others else None,
        }
        models = [a for a in cell["answers"] if a["recommends_model"]]
        cell["open_weights"] = _rate(sum(a["open_weights"] for a in models), len(models))
        cell["criteria"] = _rate(sum(a["criteria"] for a in cell["answers"]), n)
        cell["caveat"] = _rate(sum(a["caveat"] for a in cell["answers"]), n)

    first_vendors = Counter(a["first_vendor"] for c in cells for a in c["answers"])
    ranked = [v for v, _ in sorted(first_vendors.items(), key=lambda kv: (-kv[1], kv[0])) if v not in ("none",)]
    data = {
        "title": title,
        "revision": CROSS_MODEL_REVISION,
        "generated_at": utc_now(),
        "runs": runs,
        "probes": sorted(probes.values(), key=lambda p: p["number"]),
        "cells": sorted(cells, key=lambda c: (c["probe_number"], c["vendor"] or "", c["model"])),
        "pending": pending,
        "vendor_order": ranked[:SERIES_SLOTS],
    }
    data["data_sha256"] = content_hash({k: v for k, v in data.items() if k != "generated_at"})
    return data


# ----- rendering ---------------------------------------------------------------------------------------

e = html.escape


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _frac(rate: dict[str, Any] | None) -> str:
    if not rate or not rate["n"]:
        return "–"
    ci = rate["ci95"]
    tip = f' title="95% interval {100 * ci[0]:.0f}–{100 * ci[1]:.0f}%"' if ci else ""
    return f'<span{tip}>{rate["k"]}/{rate["n"]}</span>'


def _model_label(c: dict[str, Any]) -> str:
    return str(c["model"])


def _model_sub(c: dict[str, Any]) -> str:
    engine = f" · {c['adapter']}" if c["track"] == "local" else ""
    return f"{_vendor_name(c['vendor'])} · {c['track']}{engine}"


def _slot(vendor: str, order: list[str]) -> str:
    return f"s{order.index(vendor) + 1}" if vendor in order else "sother"


def _first_pick_chart(cells: list[dict[str, Any]], order: list[str]) -> str:
    rows = [c for c in cells if c["answers"]]
    if not rows:
        return '<p class="muted">No extracted recommendations for this probe.</p>'
    label_w, bar_w, row_h, bar_h = 230, 470, 46, 22
    width, height = label_w + bar_w + 70, len(rows) * row_h + 8
    present: list[str] = []
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="First recommendation by vendor, per model">']
    for i, c in enumerate(rows):
        y = i * row_h + 6
        counts = Counter(a["first_vendor"] if a["first_vendor"] in order or a["first_vendor"] == "none" else "other"
                         for a in c["answers"])
        n = len(c["answers"])
        parts.append(f'<text class="lbl" x="0" y="{y + 9}">{e(_model_label(c))}</text>'
                     f'<text class="sub" x="0" y="{y + 25}">{e(_model_sub(c))}</text>')
        x = float(label_w)
        segs = [(v, counts[v]) for v in [*order, "other", "none"] if counts.get(v)]
        for j, (vendor, k) in enumerate(segs):
            w = bar_w * k / n - (2 if j < len(segs) - 1 else 0)
            names = Counter(a["first_name"] or "nothing" for a in c["answers"]
                            if (a["first_vendor"] if a["first_vendor"] in order or a["first_vendor"] == "none"
                                else "other") == vendor)
            detail = ", ".join(f"{name} ×{m}" for name, m in names.most_common())
            own = " (its own vendor)" if vendor == c["vendor"] else ""
            tip = f"{c['model']}: {_vendor_name(vendor)}{own} first in {k} of {n} answers — {detail}"
            cls = "snone" if vendor == "none" else _slot(vendor, order)
            if vendor not in present:
                present.append(vendor)
            parts.append(f'<g><title>{e(tip)}</title><rect class="{cls}" x="{x:.1f}" y="{y + 4}" '
                         f'width="{max(w, 1):.1f}" height="{bar_h}" rx="4"/>')
            if w >= 40:
                ink = "ink-dark" if cls in ("s4",) else "ink-light"
                parts.append(f'<text class="in {ink}" x="{x + 8:.1f}" y="{y + 19}">{100 * k / n:.0f}%</text>')
            parts.append("</g>")
            x += w + 2
        own_k = sum(a["first_vendor"] == c["vendor"] for a in c["answers"])
        parts.append(f'<text class="end" x="{label_w + bar_w + 10}" y="{y + 19}">'
                     f'{"own " + str(own_k) + "/" + str(n) if c["vendor"] else ""}</text>')
    parts.append("</svg>")
    legend = "".join(
        f'<span class="key"><i class="{"snone" if v == "none" else _slot(v, order)}"></i>{e(_vendor_name(v))}</span>'
        for v in [*order, "other", "none"] if v in present)
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def _dumbbell(cells: list[dict[str, Any]]) -> str:
    rows = [c for c in cells if c.get("own_first") and c["own_first"]["n"] and c["baseline"]
            and c["baseline"]["first"] is not None]
    if not rows:
        return ('<p class="muted">No baseline yet: self-preference needs answers from models of at least two '
                'vendors on this probe.</p>')
    label_w, plot_w, row_h = 230, 470, 40
    width, height = label_w + plot_w + 70, len(rows) * row_h + 34

    def x(v: float) -> float:
        return label_w + plot_w * v

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Own vendor picked first: this model against other models">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{height - 22}"/>'
                     f'<text class="tick" x="{x(t):.1f}" y="{height - 6}" text-anchor="middle">{100 * t:.0f}%</text>')
    for i, c in enumerate(rows):
        y = i * row_h + 20
        own, base = c["own_first"]["rate"], c["baseline"]["first"]
        lo, hi = c["own_first"]["ci95"] or (own, own)
        parts.append(f'<text class="lbl" x="0" y="{y - 2}">{e(_model_label(c))}</text>'
                     f'<text class="sub" x="0" y="{y + 13}">{e(_vendor_name(c["vendor"]))}</text>')
        parts.append(f'<rect class="ci" x="{x(lo) - 5:.1f}" y="{y - 5}" width="{x(hi) - x(lo) + 10:.1f}" height="10" rx="5"/>'
                     f'<line class="link" x1="{x(base):.1f}" x2="{x(own):.1f}" y1="{y}" y2="{y}"/>')
        tip_b = (f"Other models ({c['baseline']['models']}) put {_vendor_name(c['vendor'])} first in "
                 f"{100 * base:.0f}% of answers on average")
        tip_o = (f"{c['model']} put {_vendor_name(c['vendor'])} first in {c['own_first']['k']} of "
                 f"{c['own_first']['n']} answers (95% interval {100 * lo:.0f}–{100 * hi:.0f}%)")
        parts.append(f'<g><title>{e(tip_b)}</title><circle class="base" cx="{x(base):.1f}" cy="{y}" r="6"/>'
                     f'<circle class="hit" cx="{x(base):.1f}" cy="{y}" r="11"/></g>'
                     f'<g><title>{e(tip_o)}</title><circle class="own" cx="{x(own):.1f}" cy="{y}" r="6"/>'
                     f'<circle class="hit" cx="{x(own):.1f}" cy="{y}" r="11"/></g>')
        delta = own - base
        parts.append(f'<text class="end" x="{label_w + plot_w + 12}" y="{y + 4}">'
                     f'{"+" if delta >= 0 else "−"}{abs(100 * delta):.0f} pts</text>')
    parts.append("</svg>")
    legend = ('<div class="legend"><span class="key"><i class="dot own"></i>This model puts its own vendor first'
              '</span><span class="key"><i class="dot base"></i>Other vendors\' models put that vendor first '
              '(mean)</span><span class="key"><i class="bar ci"></i>95% interval of this model\'s rate</span></div>')
    return legend + "".join(parts)


def _heat(value: float | None) -> str:
    if value is None:
        return '<td class="num muted">–</td>'
    step = min(4, max(0, round(value)))
    return f'<td class="num h{step}" title="mean rating {value:.2f} of 4">{value:.1f}</td>'


def _probe_section(probe: dict[str, Any], cells: list[dict[str, Any]], runs: dict[str, dict[str, Any]],
                   order: list[str], pending: list[dict[str, Any]]) -> str:
    dims = runs[cells[0]["evaluation_run_id"]]["dimensions"] if cells else []
    vendors = {c["vendor"] for c in cells}
    out = [f'<section class="probe" id="{e(probe["probe"].lower())}">',
           f'<h2><span class="code">{e(probe["probe"])}</span> {e(probe.get("name", "").replace("-", " ").capitalize())}</h2>',
           f'<blockquote class="prompt">“{e(probe.get("prompt", ""))}”</blockquote>']
    if not cells:
        waiting = [p for p in pending if p["probe"] == probe["probe"]]
        out.append(f'<p class="muted">{"No evaluated answers yet. Answered, awaiting evaluation: " + ", ".join(e(p["model"]) for p in waiting) if waiting else "Not run yet."}</p></section>')
        return "".join(out)
    n_answers = sum(len(c["answers"]) for c in cells)
    out.append(f'<p class="meta">{len(cells)} model{"s" if len(cells) != 1 else ""} from {len(vendors)} '
               f'vendor{"s" if len(vendors) != 1 else ""} · {sum(c["responses"] for c in cells)} answers · '
               f'{n_answers} with extracted recommendations</p>')
    out.append('<figure><figcaption><b>What each model recommends first</b>'
               '<span>Vendor of the first item each answer recommends. Hover a bar for the items.</span></figcaption>'
               + _first_pick_chart(cells, order) + "</figure>")
    if len(vendors) > 1:
        out.append('<figure><figcaption><b>Self-preference</b><span>How often a model puts its own vendor first, '
                   'against how often the other vendors\' models answering the same question do.</span></figcaption>'
                   + _dumbbell(cells) + "</figure>")
    else:
        out.append('<p class="note">No self-preference baseline yet: only one vendor\'s model has answered this '
                   'question.</p>')
    out.append('<div class="table-wrap"><table class="data"><thead><tr><th>Model</th><th class="num">Answers</th>'
               '<th class="num" title="Adjudicated score where available, otherwise the judges\' mean raw score">'
               'Quality* (range)</th><th class="num" title="Mean gap between the two judges\' scores">'
               'Judge gap</th><th class="num">Own vendor first</th><th class="num">Others put it first</th>'
               '<th class="num">Own vendor included</th><th class="num">Open-weight model</th>'
               '<th class="num">Criteria stated</th><th class="num">Landscape caveat</th></tr></thead><tbody>')
    for c in cells:
        q = c["quality"]
        b = c.get("baseline") or {}
        prov = "" if q["basis"].get("adjudicated") == q["n"] else '<sup title="not human-checked">*</sup>'
        spread = "" if q["min"] is None else f' <span class="muted">({q["min"]:.0f}–{q["max"]:.0f})</span>'
        quality = "–" if q["mean"] is None else f"{q['mean']:.0f}{prov}{spread}"
        gap = "–" if c["judge_gap"] is None else f"{c['judge_gap']:.0f} pts"
        others = _pct(b.get("first")) if b.get("models") else '<span class="muted">no baseline</span>'
        out.append(
            f'<tr><td><b>{e(c["model"])}</b><div class="sub">{e(_model_sub(c))}</div></td>'
            f'<td class="num">{c["responses"]}/{c["planned"]}</td><td class="num strong">{quality}</td>'
            f'<td class="num muted">{gap}</td>'
            f'<td class="num">{_frac(c.get("own_first"))}</td><td class="num">{others}</td>'
            f'<td class="num">{_frac(c.get("own_included"))}</td><td class="num">{_frac(c.get("open_weights"))}</td>'
            f'<td class="num">{_frac(c.get("criteria"))}</td><td class="num">{_frac(c.get("caveat"))}</td></tr>')
    out.append("</tbody></table></div>")
    if dims:
        out.append('<h3>Rubric ratings <span class="muted">(mean of 0–4 per dimension; weight in brackets)</span></h3>'
                   '<div class="table-wrap"><table class="data heat"><thead><tr><th>Model</th>'
                   + "".join(f'<th class="num">{e(d["title"])} <span class="muted">({d["weight"]:g})</span></th>'
                             for d in dims)
                   + '<th class="num">Critical-failure candidates</th></tr></thead><tbody>')
        for c in cells:
            out.append(f'<tr><td><b>{e(c["model"])}</b></td>'
                       + "".join(_heat(c["dimensions"].get(d["id"])) for d in dims)
                       + f'<td class="num">{c["critical_failure_candidates"]} of {c["quality"]["n"]}'
                         f'<span class="muted"> unconfirmed</span></td></tr>')
        out.append("</tbody></table></div>")
    examples = [(c, a) for c in cells for a in c["answers"][:2]]
    if examples:
        out.append('<details class="examples"><summary>Shortlists as written (first two answers per model)</summary><ul>'
                   + "".join(f'<li><b>{e(c["model"])}</b>: {e(a["as_written"] or "nothing recommended")}</li>'
                             for c, a in examples) + "</ul></details>")
    out.append("</section>")
    return "".join(out)


def _findings(data: dict[str, Any]) -> list[str]:
    cells = data["cells"]
    notes: list[str] = []
    for probe in data["probes"]:
        pc = [c for c in cells if c["probe"] == probe["probe"] and c.get("own_first") and c["own_first"]["n"]]
        with_base = [c for c in pc if c["baseline"] and c["baseline"]["first"] is not None]
        if with_base:
            strongest = max(with_base, key=lambda c: c["own_first"]["rate"] - c["baseline"]["first"])
            for c in sorted(with_base, key=lambda c: c["baseline"]["first"] - c["own_first"]["rate"]):
                delta = c["own_first"]["rate"] - c["baseline"]["first"]
                if delta < 0.25:
                    continue
                notes.append(
                    f'<b>{e(probe["probe"])}:</b> {e(c["model"])} put {e(_vendor_name(c["vendor"]))} first in '
                    f'{c["own_first"]["k"]} of {c["own_first"]["n"]} answers; the other vendors\' models did so in '
                    f'{_pct(c["baseline"]["first"])} of theirs.')
            if not any(c["own_first"]["rate"] - c["baseline"]["first"] >= 0.25 for c in with_base):
                notes.append(f'<b>{e(probe["probe"])}:</b> no model favours its own vendor by 25 points or more '
                             f'(largest: {e(strongest["model"])}, {_pct(strongest["own_first"]["rate"])} against '
                             f'{_pct(strongest["baseline"]["first"])}).')
        scored = [c for c in cells if c["probe"] == probe["probe"] and c["quality"]["mean"] is not None]
        if len(scored) > 1:
            best = max(scored, key=lambda c: c["quality"]["mean"])
            worst = min(scored, key=lambda c: c["quality"]["mean"])
            notes.append(f'<b>{e(probe["probe"])}:</b> provisional quality ranges from {worst["quality"]["mean"]:.0f} '
                         f'({e(worst["model"])}) to {best["quality"]["mean"]:.0f} ({e(best["model"])}) out of 100.')
        criteria = [c for c in cells if c["probe"] == probe["probe"] and c.get("criteria") and c["criteria"]["n"]]
        never = [c for c in criteria if c["criteria"]["k"] == 0]
        if never:
            notes.append(f'<b>{e(probe["probe"])}:</b> {", ".join(e(c["model"]) for c in never)} never stated selection '
                         f'criteria before recommending.')
    return notes


def _coverage(data: dict[str, Any]) -> str:
    probes = data["probes"]
    models: dict[str, dict[str, Any]] = {}
    for c in [*data["cells"], *data["pending"]]:
        models.setdefault(c["configuration"], c)
    cell_map = {(c["probe"], c["configuration"]): c for c in data["cells"]}
    pend_map = {(p["probe"], p["configuration"]): p for p in data["pending"]}
    rows = []
    for key, m in sorted(models.items(), key=lambda kv: (_vendor_name(kv[1].get("vendor") or kv[1]["provider"]), kv[1]["model"])):
        tds = []
        for p in probes:
            c = cell_map.get((p["probe"], key))
            if c is not None:
                q = c["quality"]["mean"]
                tds.append(f'<td class="cov done" title="{c["responses"]} answers, evaluated"><b>{"–" if q is None else f"{q:.0f}"}</b>'
                           f'<span>{c["responses"]} answers</span></td>')
            elif (pp := pend_map.get((p["probe"], key))) is not None:
                tds.append(f'<td class="cov wait" title="{e(pp["state"])}">{pp["responses"]} answers'
                           f'<span>{e(pp["state"])}</span></td>')
            else:
                tds.append('<td class="cov none">·</td>')
        rows.append(f'<tr><td><b>{e(m["model"])}</b><div class="sub">{e(_model_sub({**m, "vendor": m.get("vendor") or m["provider"]}))}</div></td>{"".join(tds)}</tr>')
    head = "".join(f'<th class="num"><a href="#{e(p["probe"].lower())}">{e(p["probe"])}</a>'
                   f'<div class="sub">{e(p.get("prompt", ""))}</div></th>' for p in probes)
    return (f'<div class="table-wrap"><table class="data coverage"><thead><tr><th>Model</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _caveats(data: dict[str, Any]) -> list[str]:
    runs = data["runs"]
    notes = []
    if not all(r["complete"] for r in runs):
        checked = sum(int(r["human_checks"].split("/")[0]) for r in runs)
        total = sum(int(r["human_checks"].split("/")[1]) for r in runs)
        notes.append(f"No result here is final: {checked} of {total} judgements have been human-checked. Quality "
                     "scores marked * are the judges' unchecked mean, and critical-failure candidates are unconfirmed.")
    if any(c["judges_share_vendor"] for c in data["cells"]):
        names = sorted({c["model"] for c in data["cells"] if c["judges_share_vendor"]})
        notes.append("Some answers were judged by a model from the candidate's own vendor: " + ", ".join(names) + ".")
    if any(c["extractor_is_candidate"] for c in data["cells"]):
        notes.append("In some runs the recommendation extractor comes from a vendor that is also a candidate.")
    policies = sorted({r["policy"] for r in runs})
    if len(policies) > 1:
        notes.append(f"Runs use {len(policies)} evaluation policies ({', '.join(policies)}); quality scores are only "
                     "strictly comparable within one policy and judge panel.")
    if any(r["catalog"] and r["catalog"]["status"] != "verified" for r in runs):
        notes.append("The vendor catalog is a draft: vendor attributions have not been verified against first-party sources.")
    small = sorted({c["responses"] for c in data["cells"]})
    if small and small[-1] <= 20:
        size = f"{small[0]}" if small[0] == small[-1] else f"{small[0]}–{small[-1]}"
        notes.append(f"Samples are small ({size} answers per model and question); hover a fraction for its 95% interval.")
    return notes


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;
--s7:#6250d6;--sother:#b5b3ac;--snone:#e1e0d9;--h0:#f0efec;--h1:#cde2fb;--h2:#9ec5f4;--h3:#5598e7;--h4:#1c5cab;
--warn-bg:#fbf1db;--warn-ink:#7a4f00;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;
--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#0ca30c;--s7:#9085e9;--sother:#5f5e5a;--snone:#2c2c2a;
--h0:#383835;--h1:#104281;--h2:#184f95;--h3:#256abf;--h4:#5598e7;--warn-bg:#3a3020;--warn-ink:#f0c060;--accent:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
--s6:#0ca30c;--s7:#9085e9;--sother:#5f5e5a;--snone:#2c2c2a;--h0:#383835;--h1:#104281;--h2:#184f95;--h3:#256abf;--h4:#5598e7;
--warn-bg:#3a3020;--warn-ink:#f0c060;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:40px 24px 80px}
header h1{font-size:30px;line-height:1.2;margin:0 0 6px;font-weight:650}
header .lede{color:var(--ink2);font-size:17px;margin:0 0 14px;max-width:760px}
.badge{display:inline-block;font-size:12px;font-weight:600;padding:2px 10px;border-radius:999px;background:var(--warn-bg);color:var(--warn-ink);margin-right:8px}
.meta,.muted{color:var(--muted)} .sub{color:var(--muted);font-size:12px;font-weight:400}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:18px 22px;margin:18px 0}
.card.warn{background:var(--warn-bg);border-color:transparent;color:var(--warn-ink)}
.card h2{margin-top:0}
.card ul{margin:6px 0 0;padding-left:20px} .card li{margin:4px 0}
h2{font-size:21px;margin:44px 0 6px} h2 .code{font-size:17px}
h3{font-size:16px;margin:26px 0 8px}
blockquote.prompt{margin:8px 0 6px;font-size:18px;color:var(--ink2)}
figure{margin:14px 0 0;background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px 18px}
figcaption{margin-bottom:10px} figcaption b{display:block} figcaption span{color:var(--muted);font-size:13px}
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .lbl{font-size:13px;font-weight:600;fill:var(--ink)} .chart .sub{font-size:11px;fill:var(--muted)}
.chart .end,.chart .tick{font-size:12px;fill:var(--ink2);font-variant-numeric:tabular-nums}
.chart .in{font-size:11.5px;font-weight:600} .chart .ink-light{fill:#fff} .chart .ink-dark{fill:#0b0b0b}
.chart .grid{stroke:var(--grid);stroke-width:1} .chart .link{stroke:var(--axis);stroke-width:2}
.chart .ci{fill:var(--ink2);opacity:.14}
.chart .own{fill:var(--ink);stroke:var(--surface);stroke-width:2} .chart .base{fill:var(--surface);stroke:var(--muted);stroke-width:2}
.chart .hit{fill:transparent} .chart g:hover rect{opacity:.85}
.s1{fill:var(--s1);background:var(--s1)}.s2{fill:var(--s2);background:var(--s2)}.s3{fill:var(--s3);background:var(--s3)}
.s4{fill:var(--s4);background:var(--s4)}.s5{fill:var(--s5);background:var(--s5)}.s6{fill:var(--s6);background:var(--s6)}
.s7{fill:var(--s7);background:var(--s7)}.sother{fill:var(--sother);background:var(--sother)}.snone{fill:var(--snone);background:var(--snone)}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2);margin-bottom:10px}
.key{display:inline-flex;align-items:center;gap:6px} .key i{display:inline-block;width:12px;height:12px;border-radius:3px}
.key i.dot{border-radius:50%;width:11px;height:11px} .key i.dot.own{background:var(--ink)} .key i.dot.base{border:2px solid var(--muted)}
.key i.bar.ci{height:10px;width:20px;background:var(--ink2);opacity:.14;border-radius:5px}
.table-wrap{overflow-x:auto;margin-top:14px;background:var(--surface);border:1px solid var(--ring);border-radius:12px}
table.data{border-collapse:collapse;width:100%;font-size:13.5px}
table.data th,table.data td{padding:8px 10px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
table.data th{font-weight:600;font-size:12px;color:var(--ink2);background:var(--surface);white-space:normal;vertical-align:bottom}
table.data tr:last-child td{border-bottom:0}
table.data td:first-child{min-width:190px} .heat td.num{text-align:center!important}
.note{color:var(--ink2);font-size:13.5px;margin:12px 0 0}
ul.pending{list-style:none;padding:0} ul.pending li{margin:8px 0}
ul.pending .code,h2 .code{display:inline-block;background:var(--ink);color:var(--surface);border-radius:6px;padding:0 8px;margin-right:6px}
.num{text-align:right!important;font-variant-numeric:tabular-nums;white-space:nowrap} .strong{font-weight:650}
.heat td.h0{background:var(--h0)} .heat td.h1{background:var(--h1)} .heat td.h2{background:var(--h2)}
.heat td.h3{background:var(--h3);color:#fff} .heat td.h4{background:var(--h4);color:#fff}
.coverage th .sub{max-width:170px;white-space:normal;text-align:right;margin-left:auto}
.coverage th a{color:var(--ink);text-decoration:none}
td.cov span{display:block;font-size:11px;color:var(--muted);font-weight:400}
td.cov.done b{font-size:16px} td.cov.wait{color:var(--ink2)} td.cov.none{color:var(--grid);text-align:center!important}
details.examples{margin-top:12px;font-size:13.5px} details.examples summary{cursor:pointer;color:var(--ink2)}
.prov td,.prov th{font-size:12.5px}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
footer{margin-top:48px;color:var(--muted);font-size:12.5px}
@media (max-width:860px){main{padding:24px 16px 60px}}
@media print{body{background:#fff} .card,figure,.table-wrap{break-inside:avoid}}
"""


def render_html(data: dict[str, Any]) -> str:
    runs = {r["evaluation_run_id"]: r for r in data["runs"]}
    cells = data["cells"]
    order = data["vendor_order"]
    n_models = len({c["configuration"] for c in cells})
    probes_with = [p for p in data["probes"] if any(c["probe"] == p["probe"] for c in cells)]
    final = all(r["complete"] for r in data["runs"])
    findings = _findings(data)
    caveats = _caveats(data)
    system_prompt = next((p.get("system_prompt") for p in data["probes"] if p.get("system_prompt")), "")
    ranked = sorted(probes_with, key=lambda p: (-len({c["configuration"] for c in cells if c["probe"] == p["probe"]}),
                                                p["number"]))
    sections = "".join(
        _probe_section(p, [c for c in cells if c["probe"] == p["probe"]], runs, order, data["pending"]) for p in ranked)
    rest = [p for p in data["probes"] if p not in probes_with]
    if rest:
        items = []
        for p in rest:
            waiting = [x["model"] for x in data["pending"] if x["probe"] == p["probe"]]
            state = f"answered by {', '.join(waiting)}, not evaluated" if waiting else "not run yet"
            items.append(f'<li id="{e(p["probe"].lower())}"><span class="code">{e(p["probe"])}</span> '
                         f'“{e(p.get("prompt", ""))}” <span class="muted">— {e(state)}</span></li>')
        sections += f'<h2>Questions without results</h2><ul class="pending">{"".join(items)}</ul>'

    prov_rows = "".join(
        f'<tr><td><code>{e(r["evaluation_run_id"])}</code><div class="sub">{e(r["experiment_id"])}</div></td>'
        f'<td>{e(r["probe"])}</td><td><code>{e(r["policy"])}</code></td>'
        f'<td>{e(", ".join(j["model"] for j in r["judges"]))}</td>'
        f'<td>{e(r["extractor"]["model"]) if r["extractor"] else "–"}</td>'
        f'<td>{e(r["catalog"]["id"] + "@" + r["catalog"]["revision"] + " (" + r["catalog"]["status"] + ")") if r["catalog"] else "–"}</td>'
        f'<td class="num">{e(r["judge_evaluations"])}</td><td class="num">{e(r["human_checks"])}</td>'
        f'<td class="num">{e(r["adjudicated"])}</td></tr>'
        for r in data["runs"])
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(data["title"])}: cross-model report</title>
<meta name="description" content="How {n_models} models answer the same agent-design questions: recommendations, self-preference and rubric quality.">
<style>{CSS}</style></head>
<body><main>
<header>
<p>{'' if final else '<span class="badge">Preview · provisional</span>'}<span class="meta">Generated {e(data["generated_at"][:10])} · {e(data["revision"])}</span></p>
<h1>{e(data["title"])}: how models answer the same questions</h1>
<p class="lede">{n_models} model configurations asked {len(probes_with)} of {len(data["probes"])} short agent-design questions,
{sum(c["responses"] for c in cells)} answers in total. Each answer was rated against a rubric by two LLM judges, and the products
it recommends were extracted and attributed to vendors.</p>
</header>
{'<div class="card warn"><b>Read with care</b><ul>' + "".join(f"<li>{e(c)}</li>" for c in caveats) + "</ul></div>" if caveats else ""}
{'<div class="card"><h2>Key findings</h2><ul>' + "".join(f"<li>{f}</li>" for f in findings) + "</ul></div>" if findings else ""}
<h2>Coverage</h2>
<p class="meta">Quality score per model and question (0–100, provisional unless every judgement was human-checked). Grey cells have
answers that no included evaluation covers.</p>
{_coverage(data)}
{sections}
<h2>Method</h2>
<div class="card">
<p>Every question was sent as a single turn with the same system prompt: <i>“{e(system_prompt)}”</i> Models never see the rubric.</p>
<ul>
<li><b>Quality</b> is the rubric score out of 100: a weighted mean of the dimension ratings (0–4). It is the adjudicated score where a
human has checked the judgement, otherwise the mean of the judges' raw scores (marked *). <b>Judge gap</b> is how far apart the two
judges' scores were on average.</li>
<li><b>Recommendations</b> are extracted by a separate model that records which products, frameworks and model families an answer
recommends and in what order, with quotes that code verifies against the answer. Code, not the extractor, maps items to vendors.</li>
<li><b>Self-preference</b> compares how often a model puts its own vendor first with how often the other vendors' models included
for the same question put that vendor first. A high own rate alone is not evidence of bias; the gap to the baseline is.</li>
<li>Fractions show counts; hover one for its 95% Wilson interval.</li>
</ul></div>
<h2>Provenance</h2>
<div class="table-wrap"><table class="data prov"><thead><tr><th>Evaluation run</th><th>Question</th><th>Policy</th><th>Judges</th>
<th>Extractor</th><th>Catalog</th><th class="num">Judgements</th><th class="num">Human checks</th><th class="num">Adjudicated</th></tr></thead>
<tbody>{prov_rows}</tbody></table></div>
<footer>Data hash <code>{e(data["data_sha256"][:16])}</code>. The aggregated data is embedded in this page as JSON
(<code>#report-data</code>).</footer>
</main>
<script type="application/json" id="report-data">{payload}</script>
</body></html>
"""


def publish(db: Database, store: ArtifactStore, run_ids: list[str], out: Path, *, scenarios: Path | None = None,
            title: str = "Agentic design bias") -> dict[str, Any]:
    data = collect(db, store, run_ids, scenarios=scenarios, title=title)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")
    return data


__all__ = ["CrossModelError", "collect", "publish", "render_html"]
