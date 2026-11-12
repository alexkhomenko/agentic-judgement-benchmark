"""Report revisions with frozen input manifests.

A report revision names the exact evaluation run and every record it used; it never looks up a mutable
"latest" result. Identical inputs and code reproduce the same revision ID. Records from another policy
are rejected, and an incomplete evaluation run is reported as incomplete, never filled from an earlier
run. Reporting reads persisted records only and never invokes a model.
"""

from __future__ import annotations

import csv
import io
import statistics
from collections import Counter, defaultdict
from typing import Any

from ..domain.common import content_hash
from ..domain.records import Record, RecordKind
from ..evaluation.records import effective, run_records, selected_attempt
from ..evaluation.review import selection_record_id
from .bias import bias_tables
from ..evaluation.runner import EvaluationStatus, RunContext, evaluation_status, load_run
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database

REPORTING_POLICY_REVISION = "reporting-v3"  # v3: recommendation-bias section


class ReportInputError(RuntimeError):
    pass


def publish_report(
    db: Database,
    store: ArtifactStore,
    experiment_id: str,
    evaluation_run_id: str,
    *,
    code_revision: str | None = None,
) -> Record:
    ctx = load_run(db, evaluation_run_id)
    if ctx.experiment_id != experiment_id:
        raise KeyError(f"evaluation run {evaluation_run_id} does not belong to experiment {experiment_id}")
    run = ctx.run
    status = evaluation_status(db, evaluation_run_id)
    inputs = _inputs(db, ctx)
    input_manifest = {
        "experiment_id": experiment_id,
        "plan_sha256": ctx.plan.plan_sha256,
        "evaluation_run_id": evaluation_run_id,
        "evaluation_run_sha256": run.payload_sha256,
        "evaluation_policy": run.policy_revision,
        "records": {kind: {r.record_id: r.payload_sha256 for r in recs} for kind, recs in sorted(inputs.items())},
        "reporting_policy": REPORTING_POLICY_REVISION,
        "code_revision": code_revision,
    }
    revision = f"rpt-{content_hash(input_manifest)[:16]}"
    existing = db.get_record(revision)
    if existing is not None:
        return existing

    tables = build_tables(db, store, ctx, inputs, status)
    sink = store.sink(f"{experiment_id}/reports/{revision}")
    sink.write_json("manifest.json", {"report_revision": revision, **input_manifest})
    sink.write("completion.csv", _csv(tables["completion"]))
    sink.write("leaderboard.csv", _csv(tables["leaderboard"]))
    sink.write("responses.csv", _csv(tables["responses"]))
    if tables["bias"] is not None:
        sink.write("bias-responses.csv", _csv(tables["bias"]["responses"]))
        sink.write("bias-summary.csv", _csv([_flatten(r) for r in tables["bias"]["summary"]]))
        sink.write("bias-items.csv", _csv(tables["bias"]["items"]))
        sink.write("bias-vendor-share.csv", _csv(tables["bias"]["vendor_share"]))
    sink.write("report.md", render_markdown(revision, ctx, status, tables))
    all_inputs = {run.record_id: run.payload_sha256}
    for recs in inputs.values():
        all_inputs.update({r.record_id: r.payload_sha256 for r in recs})
    record = Record.create(
        kind=RecordKind.REPORT_REVISION,
        record_id=revision,
        experiment_id=experiment_id,
        inputs=all_inputs,
        policy_revision=REPORTING_POLICY_REVISION,
        code_revision=code_revision,
        payload={
            "input_manifest": input_manifest,
            "evaluation_complete": status.complete,
            "pending_stages": status.pending_stages,
            "leaderboard": tables["leaderboard"],
        },
    )
    with db.transaction():
        db.insert_record(record)
        db.register_artifacts(revision, sink.refs)
    return record


def _inputs(db: Database, ctx: RunContext) -> dict[str, list[Record]]:
    exp, run_id, policy = ctx.experiment_id, ctx.run.record_id, str(ctx.run.policy_revision)
    inputs = {
        "validation": run_records(db, RecordKind.VALIDATION, exp, run_id),
        "blinded_packet": run_records(db, RecordKind.BLINDED_PACKET, exp, run_id),
        "judge_evaluation": run_records(db, RecordKind.JUDGE_EVALUATION, exp, run_id),
        "architecture_classification": run_records(db, RecordKind.ARCHITECTURE_CLASSIFICATION, exp, run_id),
        "recommendation_extraction": run_records(db, RecordKind.RECOMMENDATION_EXTRACTION, exp, run_id),
        "human_review": effective(run_records(db, RecordKind.HUMAN_REVIEW, exp, run_id)),
        "adjudication": effective(run_records(db, RecordKind.ADJUDICATION, exp, run_id)),
    }
    selection = db.get_record(selection_record_id(run_id))
    inputs["review_selection"] = [selection] if selection else []
    for kind, recs in inputs.items():
        for r in recs:
            if not str(r.policy_revision).startswith(policy):
                raise ReportInputError(f"{kind} record {r.record_id} was produced under {r.policy_revision}, "
                                       f"not the run's policy {policy}")
    return inputs


# ----- tables ----------------------------------------------------------------------------------------


def _metric(db: Database, store: ArtifactStore, owner_id: str, name: str) -> float | None:
    ref = db.artifacts_for(owner_id).get("metrics.json")
    if ref is None:
        return None
    value = store.read_json(ref).get(name, {}).get("value")
    return float(value) if value is not None else None


class CostTally:
    """Known spend plus a count of calls whose cost is unknown. Unknown is never treated as zero."""

    def __init__(self) -> None:
        self.known = 0.0
        self.unknown = 0
        self.calls = 0

    def add(self, cost_incurred: str, observed_usd: float | None) -> None:
        self.calls += 1
        if observed_usd is not None:
            self.known += observed_usd
        elif cost_incurred != "no":
            self.unknown += 1

    def row(self, stage: str) -> dict[str, Any]:
        return {"stage": stage, "calls": self.calls, "known_usd": round(self.known, 6), "unknown_cost_calls": self.unknown}


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 2) if values else None


def build_tables(
    db: Database, store: ArtifactStore, ctx: RunContext, inputs: dict[str, list[Record]], status: EvaluationStatus
) -> dict[str, Any]:
    plan, manifest, policy = ctx.plan, ctx.manifest, ctx.policy
    rounds = sorted(r.round for r in plan.scenario.rounds)
    base_round = plan.scenario.baseline.round
    responses = {(e["configuration_key"], e["repetition"], e["round"]): e for e in manifest["responses"]}
    attempts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in manifest["attempts"]:
        attempts_by_task[a["task_id"]].append(a)
    tasks = db.list_tasks(plan.experiment_id)
    validity = {r.subject_id: r.payload for r in inputs["validation"]}
    adjudication = {r.subject_id: r.payload for r in inputs["adjudication"]}
    category: dict[str, str] = {}
    by_packet: dict[str, list[Record]] = defaultdict(list)
    for r in inputs["architecture_classification"]:
        by_packet[r.payload["packet_id"]].append(r)
    for recs in by_packet.values():
        chosen = selected_attempt(recs)
        if chosen is not None:
            category[str(chosen.subject_id)] = chosen.payload["output"]["category"]

    response_rows: list[dict[str, Any]] = []
    for binding in plan.configurations:
        cfg = binding.configuration
        for rep in range(1, plan.repetitions + 1):
            for rnd in rounds:
                entry = responses.get((cfg.key, rep, rnd))
                adj = adjudication.get(entry["response_id"]) if entry else None
                score = adj["score"] if adj else None
                response_rows.append({
                    "configuration": cfg.key, "track": cfg.track.value, "repetition": rep, "round": rnd,
                    "response_id": entry["response_id"] if entry else None,
                    "mechanically_valid": validity.get(entry["response_id"], {}).get("mechanically_valid")
                    if entry else None,
                    "truncated": entry["truncated"] if entry else None,
                    "basis": adj["basis"] if adj else None,
                    "raw_score": score["raw_score"] if score else None,
                    "adjusted_score": score["adjusted_score"] if score else None,
                    "critical_failure": score["critical_failure_confirmed"] if score else None,
                    "acceptable": (
                        score["adjusted_score"] >= policy.acceptable.min_adjusted_score
                        and not score["critical_failure_confirmed"]
                    ) if score else None,
                    "category": category.get(entry["response_id"]) if entry else None,
                })

    completion: list[dict[str, Any]] = []
    leaderboard: list[dict[str, Any]] = []
    per_round: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    candidate_cost, retry_cost = CostTally(), CostTally()
    for binding in plan.configurations:
        cfg = binding.configuration
        planned = plan.repetitions * len(rounds)
        rows = [r for r in response_rows if r["configuration"] == cfg.key]
        got = [responses[k] for k in responses if k[0] == cfg.key]
        cfg_tasks = [t for t in tasks if t.configuration_key == cfg.key]
        attempts = [a for t in cfg_tasks for a in attempts_by_task.get(t.task_id, [])]
        selected_attempts = {e["attempt_id"] for e in got}
        cfg_cost = CostTally()
        for a in attempts:
            usd = _metric(db, store, a["attempt_id"], "api_cost_usd")
            (candidate_cost if a["attempt_id"] in selected_attempts else retry_cost).add(a["cost_incurred"], usd)
            cfg_cost.add(a["cost_incurred"], usd)
        latencies = [v for e in got if (v := _metric(db, store, e["attempt_id"], "client_wall_ms")) is not None]

        failed = blocked = 0
        for rep in range(1, plan.repetitions + 1):
            for rnd in rounds:
                if (cfg.key, rep, rnd) in responses:
                    continue
                task_attempts = [a for t in cfg_tasks if t.repetition == rep and t.round == rnd
                                 for a in attempts_by_task.get(t.task_id, [])]
                if task_attempts:
                    failed += 1
                    reason = f"failed: {task_attempts[-1]['failure_kind']} after {len(task_attempts)} attempt(s)"
                elif rnd != base_round and (cfg.key, rep, base_round) not in responses:
                    blocked += 1
                    reason = "blocked: baseline produced no response"
                else:
                    reason = "not run"
                missing.append({"configuration": cfg.key, "repetition": rep, "round": rnd, "reason": reason})
        completion.append({
            "configuration": cfg.key, "track": cfg.track.value, "model": cfg.model, "planned": planned,
            "responses": len(got), "completion_rate": round(len(got) / planned, 4),
            "mechanically_valid": sum(1 for r in rows if r["mechanically_valid"]),
            "truncated": sum(1 for e in got if e["truncated"]), "failed": failed, "blocked": blocked,
            "attempts": len(attempts), "retries": len(attempts) - len({a["task_id"] for a in attempts}),
            "cost_unknown_attempts": sum(1 for a in attempts if a["cost_incurred"] == "unknown"),
            "mean_client_wall_ms": round(statistics.fmean(latencies), 3) if latencies else None,
        })

        scored = [r for r in rows if r["adjusted_score"] is not None]
        acceptable = sum(1 for r in scored if r["acceptable"])
        r1_categories = [r["category"] for r in rows if r["round"] == base_round and r["category"]]
        modal = Counter(r1_categories).most_common(1)
        cost_per_acceptable = (
            round(cfg_cost.known / acceptable, 6) if acceptable and not cfg_cost.unknown else None
        )
        leaderboard.append({
            "configuration": cfg.key, "track": cfg.track.value, "model": cfg.model,
            "scored": len(scored), "planned": planned,
            "quality_adjusted_mean": _mean([r["adjusted_score"] for r in scored]),
            "quality_adjusted_median": round(statistics.median(r["adjusted_score"] for r in scored), 2)
            if scored else None,
            "quality_uncapped_mean": _mean([r["raw_score"] for r in scored]),
            "critical_failure_rate": round(sum(1 for r in scored if r["critical_failure"]) / len(scored), 4)
            if scored else None,
            "acceptable": acceptable,
            "acceptable_rate_of_planned": round(acceptable / planned, 4),
            "stability_r1_modal_share": f"{modal[0][1]}/{len(r1_categories)}" if modal else None,
            "mean_client_wall_ms": completion[-1]["mean_client_wall_ms"],
            "candidate_cost_known_usd": round(cfg_cost.known, 6),
            "candidate_cost_unknown_calls": cfg_cost.unknown,
            "cost_per_acceptable_usd": cost_per_acceptable,
        })
        for rnd in rounds:
            values = [r["adjusted_score"] for r in scored if r["round"] == rnd]
            per_round.append({
                "configuration": cfg.key, "round": rnd, "scored": len(values), "planned": plan.repetitions,
                "adjusted_mean": _mean(values),
                "adjusted_min": min(values) if values else None, "adjusted_max": max(values) if values else None,
                "adjusted_sd": round(statistics.stdev(values), 2) if len(values) > 1 else None,
                "critical_failures": sum(1 for r in scored if r["round"] == rnd and r["critical_failure"]),
            })
        for rnd in ctx.classifier_rounds:
            counts = Counter(r["category"] for r in rows if r["round"] == rnd and r["category"])
            classes.append({"configuration": cfg.key, "round": rnd, **dict(sorted(counts.items()))})

    judge_cost, classifier_cost, extraction_cost = CostTally(), CostTally(), CostTally()
    providers = {e["response_id"]: e["provider"] for e in manifest["responses"]}
    self_family = 0
    for tally, recs in ((judge_cost, inputs["judge_evaluation"]), (classifier_cost, inputs["architecture_classification"]),
                        (extraction_cost, inputs["recommendation_extraction"])):
        for r in recs:
            tally.add(r.payload["cost_incurred"], _metric(db, store, r.record_id, "api_cost_usd"))
    for r in inputs["judge_evaluation"]:
        if r.payload.get("status") == "completed" and r.payload.get("provider") == providers.get(str(r.subject_id)):
            self_family += 1
    qualification_cost = CostTally()
    for binding in plan.configurations:
        qualification = db.get_record(binding.qualification_id)
        for probe in (qualification.payload.get("report", {}).get("probes", []) if qualification else []):
            qualification_cost.add(probe.get("cost_incurred", "unknown"), probe.get("cost_usd"))
    total = CostTally()
    for t in (qualification_cost, candidate_cost, retry_cost, judge_cost, classifier_cost, extraction_cost):
        total.known += t.known
        total.unknown += t.unknown
        total.calls += t.calls
    return {
        "responses": response_rows,
        "completion": completion,
        "leaderboard": leaderboard,
        "per_round": per_round,
        "missing": missing,
        "classification": classes,
        "costs": [
            qualification_cost.row("qualification probes"),
            candidate_cost.row("candidate generation (selected attempts)"),
            retry_cost.row("candidate retries and failed attempts"),
            judge_cost.row("judge inference"),
            classifier_cost.row("architecture classification"),
            extraction_cost.row("recommendation extraction"),
            total.row("total measured"),
        ],
        "self_family_judge_evaluations": self_family,
        "bias": bias_tables(
            plan, manifest, inputs["recommendation_extraction"], ctx.catalog,
            ctx.extractor.configuration.provider if ctx.extractor else None,
        ) if ctx.catalog is not None else None,
    }


# ----- rendering -------------------------------------------------------------------------------------


def _csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue()


def _flatten(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested metric dicts become prefixed CSV columns; lists and tuples become text."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        elif isinstance(value, list | tuple):
            out[name] = " ".join(str(v) for v in value)
        else:
            out[name] = value
    return out


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{value:.0%}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _leaderboard(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Configuration | Track | Quality (adjusted mean) | Uncapped mean | Critical failure rate | "
        "Acceptable / planned | Stability (R1 modal) | Scored / planned | Latency (ms) | Candidate cost (USD) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cost = _fmt(r["candidate_cost_known_usd"])
        if r["candidate_cost_unknown_calls"]:
            cost += f" + {r['candidate_cost_unknown_calls']} unknown"
        lines.append(
            f"| {r['configuration']} | {r['track']} | {_fmt(r['quality_adjusted_mean'])} | "
            f"{_fmt(r['quality_uncapped_mean'])} | {_fmt(r['critical_failure_rate'], pct=True)} | "
            f"{r['acceptable']}/{r['planned']} | {_fmt(r['stability_r1_modal_share'])} | "
            f"{r['scored']}/{r['planned']} | {_fmt(r['mean_client_wall_ms'])} | {cost} |"
        )
    return lines


def render_markdown(revision: str, ctx: RunContext, status: EvaluationStatus, tables: dict[str, Any]) -> str:
    plan, policy = ctx.plan, ctx.policy
    names = {r.round: r.name for r in plan.scenario.rounds}
    rounds = sorted(names)
    lines = [
        f"# ArchDecisionBench report {revision}",
        "",
        f"- Experiment: `{plan.experiment_id}` (benchmark {plan.benchmark_revision})",
        f"- Plan SHA-256: `{plan.plan_sha256}`",
        f"- Source: `{plan.source.label}`",
        f"- Scenario: `{plan.scenario.scenario_id}` revision {plan.scenario.revision}",
        f"- Rubric: `{ctx.rubric.rubric_id}` revision {ctx.rubric.revision}",
        f"- Evaluation run: `{ctx.run.record_id}` under policy `{ctx.run.policy_revision}`",
        f"- Evaluation status: {'complete' if status.complete else 'INCOMPLETE — pending: ' + ', '.join(status.pending_stages)}",
        f"- Acceptable response: adjusted score >= {policy.acceptable.min_adjusted_score:g} and no confirmed "
        f"critical failure ({policy.acceptable.status} threshold)",
        "",
    ]
    if not status.complete:
        lines += [
            "> This evaluation run is incomplete. Quality figures below cover only adjudicated responses and",
            "> are not filled from any other run. Read them with the scored/planned columns.",
            "",
        ]
    lines += [
        "## Evaluation coverage",
        "",
        "| Stage | Done | Expected | Exhausted |",
        "|---|---:|---:|---:|",
    ]
    for label, count in (
        ("Validation", status.validations), ("Blinded packets", status.packets),
        ("Judge evaluations", status.judge_evaluations), ("Architecture classification", status.classifications),
        ("Recommendation extraction", status.extractions),
        ("Human reviews (selected)", status.human_reviews), ("Adjudications", status.adjudications),
    ):
        lines.append(f"| {label} | {count.done} | {count.expected} | {count.exhausted} |")
    lines += [
        "",
        f"Judge evaluations where the judge shares the candidate's provider: "
        f"{tables['self_family_judge_evaluations']}.",
        "",
        "## Completion and execution by configuration",
        "",
        "| Configuration | Track | Planned | Responses | Completion | Mechanically valid | Truncated | Failed | "
        "Blocked | Attempts | Retries | Cost-unknown attempts | Mean client latency (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["completion"]:
        lines.append(
            f"| {row['configuration']} | {row['track']} | {row['planned']} | {row['responses']} | "
            f"{row['completion_rate']:.0%} | {row['mechanically_valid']} | {row['truncated']} | {row['failed']} | "
            f"{row['blocked']} | {row['attempts']} | {row['retries']} | {row['cost_unknown_attempts']} | "
            f"{_fmt(row['mean_client_wall_ms'])} |"
        )
    lines += [
        "",
        "## Quality by round",
        "",
        "Adjusted rubric score (0–100) per round: mean, with scored/planned and the range across repetitions.",
        "",
        "| Configuration | " + " | ".join(f"R{n} {names[n]}" for n in rounds) + " |",
        "|---|" + "---:|" * len(rounds),
    ]
    cells: dict[str, dict[int, str]] = defaultdict(dict)
    for row in tables["per_round"]:
        spread = (f", {row['adjusted_min']:.1f}–{row['adjusted_max']:.1f}"
                  if row["adjusted_min"] is not None and row["scored"] > 1 else "")
        cf = f", {row['critical_failures']} CF" if row["critical_failures"] else ""
        cells[row["configuration"]][row["round"]] = (
            f"{_fmt(row['adjusted_mean'])} ({row['scored']}/{row['planned']}{spread}{cf})"
        )
    for key, by_round in cells.items():
        lines.append(f"| {key} | " + " | ".join(by_round[n] for n in rounds) + " |")
    lines += ["", "## Combined leaderboard", "", *_leaderboard(tables["leaderboard"])]
    for track in sorted({r["track"] for r in tables["leaderboard"]}):
        lines += ["", f"### {track.capitalize()} track", "",
                  *_leaderboard([r for r in tables["leaderboard"] if r["track"] == track])]
    if tables["classification"]:
        categories = sorted({k for row in tables["classification"] for k in row} - {"configuration", "round"})
        lines += [
            "", "## Architecture classification", "",
            "Reported separately from quality. Choosing a category neither raises nor lowers a score; agreement "
            "between models is not evidence of correctness.", "",
            "| Configuration | Round | " + " | ".join(categories) + " |",
            "|---|---:|" + "---:|" * len(categories),
        ]
        for row in tables["classification"]:
            lines.append(f"| {row['configuration']} | {row['round']} | "
                         + " | ".join(str(row.get(c, 0)) for c in categories) + " |")
    if tables["bias"] is not None:
        lines += _bias_section(tables["bias"])
    lines += [
        "", "## Cost", "",
        "Known spend from recorded observations. Calls with unknown cost are counted, never treated as zero. "
        "Qualification covers the passing qualification of each configuration in the plan.", "",
        "| Stage | Calls | Known USD | Unknown-cost calls |", "|---|---:|---:|---:|",
        *[f"| {c['stage']} | {c['calls']} | {c['known_usd']:.4f} | {c['unknown_cost_calls']} |" for c in tables["costs"]],
    ]
    if tables["missing"]:
        lines += ["", "## Missing responses", "", "| Configuration | Repetition | Round | Reason |",
                  "|---|---:|---:|---|"]
        for row in tables["missing"]:
            lines.append(f"| {row['configuration']} | {row['repetition']} | {row['round']} | {row['reason']} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Per-round results come before any combined score. The four branches in a repetition share a "
        "baseline and are not independent scenarios.",
        f"- With one scenario and {plan.repetitions} repetitions, spreads are descriptive and cannot establish "
        "broad architecture competence.",
        "- Configurations are compared as configurations; differences between distinct checkpoints do not "
        "isolate the effect of model size.",
        "- Latency is client wall-clock around the adapter call and is not comparable across hosting types "
        "without qualification.",
        "",
        "## Provenance",
        "",
        "| Dependency | Version |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(plan.dependency_versions.items())],
        "",
    ]
    return "\n".join(lines)


def _share(rate: dict[str, Any] | None) -> str:
    if rate is None:
        return "n/a"
    if not rate["n"]:
        return "0/0"
    low, high = rate["ci95"]
    return f"{rate['k']}/{rate['n']} ({rate['rate']:.0%}, CI {low:.0%}–{high:.0%})"


def _delta(row: dict[str, Any], metric: str) -> str:
    if row["own_vendor"] is None:
        return "n/a: provider not mapped to a catalog vendor"
    baseline = row[f"{metric}_baseline"]
    if baseline is None:
        return "n/a: no other-vendor baseline"
    return (f"{row[f'{metric}_delta']:+.0%} vs {baseline['mean_rate']:.0%} "
            f"({baseline['configurations']} other-vendor config(s))")


def _bias_section(bias: dict[str, Any]) -> list[str]:
    catalog = bias["catalog"]
    lines = [
        "", "## Recommendation bias", "",
        "From the recommendation-extraction stage. Extraction records what each answer recommends and in what "
        "order; vendors are attributed in code from the frozen catalog. Rates show k/n with a 95% Wilson "
        "interval. Self-preference compares a configuration's rate of recommending its own vendor's products "
        "(" + ", ".join(bias["preference_kinds"]) + ") with the mean rate of configurations from other vendors.",
        "",
    ]
    warnings = []
    if catalog["status"] != "verified":
        warnings.append(f"Catalog `{catalog['catalog_id']}` revision {catalog['revision']} is a **{catalog['status']}**: "
                        "vendor relationships and statuses are unverified, so stale-item counts are incomplete.")
    if len(bias["vendors_with_candidates"]) < 2:
        warnings.append("Fewer than two catalog vendors have candidates, so no self-preference baseline exists. "
                        "Own-vendor rates below are descriptive only.")
    if bias["extractor_shares_candidate_provider"]:
        warnings.append("The extractor shares a provider with at least one candidate. Check the extraction signals "
                        "column and spot-check extractions by hand.")
    lines += [f"> {w}" for w in warnings] + ([""] if warnings else [])
    lines += [
        "### Self-preference", "",
        "| Configuration | Own vendor | Extracted | Recommends own product | Delta vs baseline | "
        "Own product first | Delta vs baseline |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in bias["summary"]:
        lines.append(
            f"| {row['configuration']} | {row['own_vendor'] or 'unmapped'} | {row['extracted']}/{row['responses']} | "
            f"{_share(row['own_inclusion'])} | {_delta(row, 'own_inclusion')} | {_share(row['own_first'])} | "
            f"{_delta(row, 'own_first')} |"
        )
    lines += [
        "", "### Shortlist behaviour", "",
        "| Configuration | Items per answer | Distinct items | Distinct shortlists | First-pick entropy (bits) | "
        "Criteria stated | Landscape caveat | Declined | Affiliation stated when own recommended | "
        "Open-weight model, when a model is recommended | Stale / unlisted slots | Responses with extraction signals |",
        "|---|---|---:|---:|---:|---|---|---|---|---|---:|---:|",
    ]
    for row in bias["summary"]:
        counts = ", ".join(f"{k}: {v}" for k, v in row["item_counts"].items()) or "n/a"
        lines.append(
            f"| {row['configuration']} | {counts} | {row['distinct_items']} | {row['distinct_shortlists']} | "
            f"{_fmt(row['first_pick_entropy_bits'])} | {_share(row['selection_criteria_stated'])} | "
            f"{_share(row['landscape_caveat'])} | {_share(row['declined_to_recommend'])} | "
            f"{_share(row['self_affiliation_when_own_recommended'])} | "
            f"{_share(row['open_weights_when_model_recommended'])}"
            f"{f' (+{u} mixed-weights family)' if (u := row['open_weights_unknown']) else ''} | {row['stale_slots']} / {row['unlisted_slots']} of {row['slots']} | "
            f"{row['responses_with_extraction_signals']} |"
        )
    configs = [r["configuration"] for r in bias["summary"]]
    cells: dict[str, dict[str, str]] = defaultdict(dict)
    meta: dict[str, str] = {}
    for row in bias["items"]:
        first = f", {row['first']} first" if row["first"] else ""
        cells[row["item"]][row["configuration"]] = f"{row['included']}/{row['extracted']}{first}"
        meta[row["item"]] = row["vendor"]
    lines += [
        "", "### Recommended items", "",
        "Responses recommending each item (and how often it came first), most recommended overall first. "
        "Models are counted by family; `bias-responses.csv` has the versions as written.", "",
        "| Item | Vendor | " + " | ".join(configs) + " |",
        "|---|---|" + "---:|" * len(configs),
    ]
    for item, by_config in cells.items():
        lines.append(f"| {item} | {meta[item]} | " + " | ".join(by_config.get(c, "–") for c in configs) + " |")
    vendors = sorted({k for r in bias["vendor_share"] for k in r} - {"configuration", "own_vendor", "product_slots"})
    lines += [
        "", "### Vendor share of recommended products", "",
        "| Configuration | Product slots | " + " | ".join(vendors) + " |",
        "|---|---:|" + "---:|" * len(vendors),
    ]
    for row in bias["vendor_share"]:
        lines.append(f"| {row['configuration']} | {row['product_slots']} | "
                     + " | ".join(_fmt(row.get(v), pct=True) if row.get(v) is not None else "–" for v in vendors) + " |")
    lines += [
        "",
        "Extraction signals (aliases found in an answer but not extracted, name/id conflicts, unverified quotes) "
        "are listed per response in `bias-responses.csv`. They call for a human check; they are not errors.",
    ]
    return lines
