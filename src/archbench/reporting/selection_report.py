"""Self-contained v2 reports from persisted evidence; reporting never invokes a model."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ..domain.common import content_hash, sha256_bytes
from ..domain.records import Record, RecordKind
from ..evaluation.selection_review import current_reviews
from ..evaluation.selection_runner import STAGES, _load, _records
from ..evaluation.selection_completion import publish_selection_report, selection_status
from ..planning.compiler import load_plan, plan_record_id
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database

REVISION = "selection-full-report-v2"
FLAG_EXPLANATIONS = {
    "unknown_item": "An extracted name has no exact catalog match. This can include a conditional alternative; inspect its role and rank below.",
    "factual_review_needed": "The auditor requested factual verification. Product claims and recommendation freshness have not been independently established by this evaluation.",
    "quality_audit_unavailable": "A human explicitly resolved the failed audit as unavailable. Quality scores remain missing; the automatic failures are preserved.",
}


def _evidence(db: Database, store: ArtifactStore, owner: str) -> dict[str, Any]:
    refs = db.artifacts_for(owner)
    for ref in refs.values():
        store.read_verified(ref)
    result: dict[str, Any] = {"artifacts": {n: r.model_dump(mode="json") for n, r in refs.items()}}
    for name, key in (("metrics.json", "metrics"), ("model-output.txt", "raw_output"),
                      ("reasoning.txt", "reasoning"), ("canonical-messages.json", "messages")):
        if name in refs:
            result[key] = (store.read_json(refs[name]) if name.endswith(".json")
                           else store.read_verified(refs[name]).decode("utf-8"))
    return result


def collect(db: Database, store: ArtifactStore, experiment_id: str, run_id: str) -> dict[str, Any]:
    run, policy, catalog = _load(db, run_id)
    if run.experiment_id != experiment_id:
        raise ValueError("selection evaluation does not belong to the requested experiment")
    plan = load_plan(db, experiment_id)
    selection = publish_selection_report(db, store, run_id)
    inputs = {run.record_id: run.payload_sha256, selection.record_id: selection.payload_sha256,
              plan_record_id(experiment_id): plan.plan_sha256}
    rows = []
    for row in selection.payload["responses"]:
        entry = next(e for e in run.payload["responses"] if e["response_id"] == row["response_id"])
        response = db.get_response(row["response_id"])
        current, stale = current_reviews(db, run, response.response_id)
        for r in current + stale:
            inputs[r.record_id] = r.payload_sha256
        rows.append({**row, "answer": entry["packet"]["candidate_answer"],
                     "task": entry["packet"]["task_conversation"],
                     "candidate_response": response.model_dump(mode="json"),
                     "candidate_evidence": _evidence(db, store, response.attempt_id),
                     "human_reviews": [{**r.model_dump(mode="json"), "stale": r in stale} for r in current + stale]})
    calls = []
    for attempt in db.list_attempts(experiment_id=experiment_id):
        task = db.get_task(attempt.task_id)
        calls.append({"id": attempt.attempt_id, "stage": "candidate", "configuration_key": task.configuration_key,
                      "status": attempt.status.value, "attempt_no": attempt.attempt_no,
                      "cost_incurred": attempt.cost_incurred.value, "error": attempt.error_summary,
                      "started_at": attempt.started_at, "finished_at": attempt.finished_at,
                      **_evidence(db, store, attempt.attempt_id)})
        inputs[attempt.attempt_id] = content_hash(attempt)
    for role in STAGES:
        for entry in run.payload["responses"]:
            for record in _records(db, run, role, entry["response_id"]):
                calls.append({"id": record.record_id, "stage": role, **record.payload,
                              "created_at": record.created_at, **_evidence(db, store, record.record_id)})
                inputs[record.record_id] = record.payload_sha256
    configurations, qualifications = [], []
    for binding in plan.configurations:
        configurations.append({"key": binding.configuration.key, "configuration": binding.configuration.model_dump(mode="json"),
                               "file": binding.source.model_dump(mode="json"), "qualification_id": binding.qualification_id})
        qual = db.get_record(binding.qualification_id)
        if qual:
            inputs[qual.record_id] = qual.payload_sha256
            qualifications.append(qual.model_dump(mode="json"))
    artifact_hashes = {r["relpath"]: r["sha256"] for c in calls for r in c["artifacts"].values()}
    known_cost, unknown_cost = 0.0, 0
    for call in calls:
        cost = call.get("metrics", {}).get("api_cost_usd", {}).get("value")
        if cost is not None:
            known_cost += float(cost)
        elif call.get("cost_incurred") != "no":
            unknown_cost += 1
    return {"reporting_policy": REVISION, "experiment_id": experiment_id, "evaluation_run_id": run_id,
            "selection_report_id": selection.record_id, "automatic_report_id": selection.payload["automatic_report_id"],
            "review_policy_revision": selection.payload["review_policy_revision"], "scenario_id": plan.scenario.scenario_id,
            "scenario_revision": plan.scenario.revision, "plan_sha256": plan.plan_sha256,
            "compiled_at": plan.compiled_at, "status": selection_status(db, run_id),
            "policy": policy.model_dump(mode="json"), "policy_file": run.payload["policy_file"],
            "catalog": {"id": catalog.catalog_id, "revision": catalog.revision,
                        "sha256": run.payload["catalog_file"]["sha256"]},
            "candidate_configurations": configurations, "evaluator_models": run.payload["models"],
            "qualifications": qualifications, "responses": rows, "calls": calls,
            "aggregates": selection.payload["aggregates"],
            "cost": {"known_usd": round(known_cost, 8), "unknown_cost_calls": unknown_cost,
                     "candidate_calls": sum(c["stage"] == "candidate" for c in calls),
                     "evaluator_calls": sum(c["stage"] != "candidate" for c in calls)},
            "source": plan.source.model_dump(mode="json"), "host": plan.host,
            "dependencies": plan.dependency_versions, "scenario_files": [f.model_dump(mode="json") for f in plan.scenario_files],
            "inputs": inputs, "artifact_hashes": artifact_hashes,
            "limitations": [
                "Completion applies to this frozen plan, not the full v2 roster.",
                "Small pilot samples cannot establish provider bias, statistical significance, or comparative quality.",
                "Preference labels describe recommendations; they do not establish that a choice is unjustified.",
                "Human Proceed acknowledges the displayed evidence; automated attention flags remain in the report.",
                "Proceed with audit unavailable is a human resolution with a required note; it supplies no quality score. "
                "Automatic validity and reviewed completion are reported separately.",
                "Quality ratings and rationales are auditor outputs; factual accuracy and scenario relevance require review.",
                "Recommendation freshness is not a separate measured label in this policy.",
                "Quality audits cover sampled or flagged answers; do not treat their mean as full-population quality.",
                "Cloud and local tracks must be analyzed separately. Classifier probabilities are not calibrated confidence.",
                "API costs use recorded usage and pinned prices; electricity, hardware and qualification probes are excluded.",
                "Release date does not establish a model's knowledge cutoff. Omitted settings retain native defaults.",
            ]}


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else "Unavailable"), quote=True)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _details(title: str, value: Any, *, text: bool = False) -> str:
    return f"<details><summary>{_e(title)}</summary><pre>{_e(value if text else _json(value))}</pre></details>"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{_e(h)}</th>" for h in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_e(v)}</td>" for v in row) + "</tr>" for row in rows) + \
        "</tbody></table></div>"


def _metric(call: dict[str, Any], name: str) -> Any:
    return call.get("metrics", {}).get(name, {}).get("value")


def render_html(data: dict[str, Any], revision: str) -> str:
    status, cost, rows = data["status"], data["cost"], data["responses"]
    state = "COMPLETE" if status["complete"] else "INCOMPLETE"
    if status["audit_resolution"]["human_unavailable"]:
        state += " · AUDIT UNAVAILABLE"
    tracks = ", ".join(sorted({c["configuration"]["track"] for c in data["candidate_configurations"]}))
    h = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; base-uri 'none'; form-action 'none'\">",
         "<title>V2 experiment report</title><style>" + CSS + "</style></head><body>",
         f"<header><div class='eyebrow'>AGENTIC DESIGN BIAS · V2</div><h1>Experiment report</h1><p>{_e(data['experiment_id'])}</p>",
         f"<span class='badge'>{state}</span> <span class='badge'>{_e(tracks.upper())}</span> <span class='badge'>CALIBRATION {_e(data['policy']['calibration_status'].upper())}</span></header>",
         "<nav><a href='#summary'>Overview</a><a href='#answers'>Answers & judgments</a><a href='#cost'>Runtime & cost</a><a href='#method'>Method & qualification</a><a href='#evidence'>Evidence</a></nav><main>",
         "<section id='summary'><h2>Overview</h2><div class='cards'>",
         f"<div class='card'><b>{len(data['candidate_configurations'])}</b><span>Candidate model(s)</span></div>",
         f"<div class='card'><b>1</b><span>Question · target {status['balance']['target_per_model']} per model</span></div>",
         f"<div class='card'><b>{status['human_review']['approved']} / {len(rows)}</b><span>Human approved</span></div>",
         f"<div class='card'><b>${cost['known_usd']:.8f}</b><span>Known API cost · {cost['unknown_cost_calls']} unknown-cost calls</span></div></div>",
         f"<p class='muted'>Frozen question: {_e(data['scenario_id'])} · {_e(data['scenario_revision'])}. Compiled {_e(data['compiled_at'])}.</p>",
         "<p class='notice'>This report describes the frozen plan above. A one-answer pilot demonstrates the pipeline; it cannot establish a provider-bias rate.</p>",
         _table(["Configuration", "Track", "Responses", "Target", "Balanced"],
                [[c["key"], c["configuration"]["track"], status["balance"]["counts"][c["key"]],
                  status["balance"]["target_per_model"], status["balance"]["complete"]] for c in data["candidate_configurations"]]),
         "</section><section id='answers'><h2>Answers & judgments</h2>"]
    for index, row in enumerate(rows, 1):
        labels = row["labels"] or {}
        h += [f"<article><h3>Response {index} · repetition {row['repetition']}</h3><p class='muted'>{_e(row['configuration_key'])} · {_e(row['human_review_state'])}</p>"]
        if row["flags"]:
            h += ["<div class='attention'><strong>Attention flags preserved after human review</strong><ul>" +
                  "".join(f"<li><b>{_e(f.replace('_', ' '))}</b>" +
                          (f" — {_e(FLAG_EXPLANATIONS[f])}" if f in FLAG_EXPLANATIONS else "") +
                          "</li>" for f in row["flags"]) + "</ul></div>"]
        if row["audit_resolution"]["status"] == "unavailable":
            h.append("<div class='notice'><strong>Quality audit unavailable — human resolved.</strong> "
                     "No quality score is supplied. Failed attempts remain in the history.</div>")
        h += ["<div class='columns'><div><h4>Question</h4>"]
        for message in row["task"]:
            h.append((f"<div class='prose'>{_e(message['content'])}</div>" if message["role"] == "user" else
                      _details(f"{message['role']} instructions", message["content"], text=True)))
        h += [f"<h4>Full candidate answer</h4><div class='prose answer'>{_e(row['answer'])}</div></div><div>",
              "<h4>Preference & basic compliance · Jev</h4>"]
        label_rows = []
        for key, value in labels.items():
            p = value.get("selected_probability", value.get("probability"))
            label_rows.append([key.replace("_", " "), value.get("choice", "Probability of Yes"),
                               f"{100*p:.1f}%" if p is not None else None])
        h += [_table(["Label", "Result", "Probability"], label_rows),
              "<p class='muted'>Classifier probabilities are uncalibrated model outputs.</p>",
              f"<p>First-choice owner: <strong>{_e(row['rank_one_owner'])}</strong>. Candidate's own provider first: <strong>{_e(row['own_provider_rank_one'])}</strong>.</p>",
              "<h4>Recommendation extraction · Luna</h4>"]
        extraction = row["extraction"] or {}
        h.append(_table(["Name as written", "Role", "Rank", "Catalog ID", "Evidence"],
                        [[i["name_as_written"], i["role"], i["rank"], i["canonical_id"], i["evidence"]]
                         for i in extraction.get("items", [])]))
        h.append("<h4>Quality audit · Haiku</h4><p class='muted'>Audit reasons: " + _e(", ".join(row["audit_reasons"])) + ". These are the auditor's original judgments.</p>")
        if row["quality"] is None:
            h.append("<p>Quality score: <strong>Unavailable</strong>.</p>")
        for dimension, rating in (row["quality"] or {}).items():
            h.append(f"<div class='rating'><b>{_e(dimension.replace('_',' ').title())}</b><span>{rating['rating']} / 4</span></div>" +
                     f"<p>{_e(rating['rationale'])}</p>" + "".join(f"<blockquote>{_e(q)}</blockquote>" for q in rating["evidence"]))
        h.append("<h4>Human validation</h4>")
        for review in row["human_reviews"]:
            p = review["payload"]
            h.append(f"<p><strong>{_e(p['reviewer_id'])} · {_e(p['decision'])}</strong> · {_e(review['created_at'])}" +
                     (" · STALE" if review["stale"] else "") + f"</p><p>{_e(p['note'] or 'No note supplied.')}</p>" +
                     f"<p class='muted'>Flags shown: {_e(', '.join(p['automated_flags_seen']))}. Review {_e(review['record_id'])}.</p>")
        if not row["human_reviews"]:
            h.append("<p class='notice'>Human validation pending.</p>")
        h += ["</div></div>", _details("Candidate runtime metrics", row["candidate_evidence"].get("metrics", {}))]
        if "reasoning" in row["candidate_evidence"]:
            h.append(_details("Raw candidate reasoning trace (not a verified explanation)", row["candidate_evidence"]["reasoning"], text=True))
        h += [_details("Full parsed evaluation and review records", {k: row[k] for k in
                ("extraction", "labels", "audit", "human_reviews", "selected_records", "candidate_response")}), "</article>"]
    h += ["</section><section id='cost'><h2>Runtime & cost</h2>",
          _table(["Stage", "Attempt", "Status", "Input tokens", "Output tokens", "Reasoning tokens", "Wall time (s)", "API USD"],
                 [[c["stage"], c["attempt_no"], c["status"], _metric(c, "input_tokens"), _metric(c, "output_tokens"),
                   _metric(c, "reasoning_tokens"), round(_metric(c, "client_wall_ms")/1000, 3) if _metric(c, "client_wall_ms") is not None else None,
                   f"{_metric(c, 'api_cost_usd'):.8f}" if _metric(c, "api_cost_usd") is not None else None] for c in data["calls"]]),
          f"<p><strong>Total known API cost: ${cost['known_usd']:.8f}</strong> · {cost['candidate_calls']} candidate call(s) · {cost['evaluator_calls']} evaluator call(s) · {cost['unknown_cost_calls']} unknown-cost call(s).</p>",
          "<p class='muted'>Reasoning tokens are included in output where the provider reports them that way. Missing measurements are shown as Unavailable, never zero. All persisted attempts, including invalid outputs and retries, contribute to this ledger. Qualification probes and local hardware/electricity are excluded.</p>",
          "<h3>Attempt history</h3>"]
    for c in data["calls"]:
        problems = c.get("problems") or ([c["error"]] if c.get("error") else [])
        if problems:
            h.append(f"<div class='attention'><b>{_e(c['stage'])} · attempt {c['attempt_no']} · {_e(c['status'])}</b><pre>{_e(chr(10).join(problems))}</pre></div>")
        h.append(_details(f"{c['stage']} · {c['id']} · full call evidence", c))
    h += ["</section><section id='method'><h2>Method & qualification</h2>",
          "<p>Independent one-turn candidate answers. Luna extracts recommendations; Jev classifies preference and basic compliance; Haiku audits the configured sample and flagged answers. Every answer then requires explicit human validation.</p>",
          f"<p>Policy {_e(data['policy']['policy_id'])}@{data['policy']['revision']} · audit fraction {100*data['policy']['audit_fraction']:.0f}% · calibration {_e(data['policy']['calibration_status'])}. No overall quality score is defined by this policy.</p>",
          f"<p>Human review policy: {_e(data['review_policy_revision'])}. Automatic outputs valid: {_e(status['all_required_outputs_valid'])}. Required work resolved: {_e(status['all_required_work_resolved'])}.</p>"]
    for c in data["candidate_configurations"]:
        cfg = c["configuration"]
        h += [f"<h3>{_e(cfg['model'])}</h3><p>Adapter {_e(cfg['adapter'])} · track {_e(cfg['track'])} · checkpoint revision {_e(cfg['adapter_options'].get('revision'))}.</p>",
              _table(["Parameter", "State", "Submitted value", "Documented default"],
                     [[k, v["state"], _json(v["value"]) if v["value"] is not None else "Not submitted",
                       _json(v["documented_default"])] for k, v in cfg["parameters"].items()]),
              _details("Exact candidate configuration, hashes and file manifest", c)]
    for q in data["qualifications"]:
        h += [f"<h3>Qualification {_e(q['record_id'])}</h3><p>Recorded {_e(q['created_at'])} · passed: {_e(q['payload']['report']['passed'])}</p>",
              _table(["Check", "Passed", "Evidence"], [[c["name"], c["passed"], c["detail"]] for c in q["payload"]["report"]["checks"]]),
              _details("Full qualification record", q)]
    h += [_details("Frozen evaluator models and prompts", data["evaluator_models"]),
          _details("Frozen policy", data["policy"]), "<h3>Interpretation limits</h3><ul>" +
          "".join(f"<li>{_e(v)}</li>" for v in data["limitations"]) + "</ul></section>",
          "<section id='evidence'><h2>Evidence & provenance</h2>",
          _table(["Field", "Value"], [["Report revision", revision], ["Evaluation run", data["evaluation_run_id"]],
                 ["Selection report", data["selection_report_id"]], ["Plan SHA-256", data["plan_sha256"]],
                 ["Catalog", _json(data["catalog"])], ["Source", _json(data["source"])]]),
          _details("Host and dependency metadata", {"host": data["host"], "dependencies": data["dependencies"]}),
          _details("Frozen scenario files", data["scenario_files"]),
          _details("Input record hashes", data["inputs"]), _details("Artifact hashes", data["artifact_hashes"]),
          "</section></main><footer>Generated from immutable benchmark evidence. Reporting makes no model calls.</footer></body></html>"]
    return "\n".join(h)


CSS = """
:root{font-family:system-ui,-apple-system,sans-serif;color:#182d3c;background:#f5f7f9;line-height:1.6}*{box-sizing:border-box}
body{margin:0}header{background:#152f40;color:white;padding:44px max(24px,calc((100vw - 1400px)/2));overflow-wrap:anywhere}
.eyebrow{color:#a5cdc2;letter-spacing:2px;font-size:12px;font-weight:650}h1{font-size:40px;line-height:1.2;margin:12px 0}header p{color:#cfdee6}
.badge{display:inline-block;border:1px solid #638577;border-radius:20px;padding:3px 12px;font-size:11px;letter-spacing:1px;margin:5px 5px 0 0}
nav{display:flex;flex-wrap:wrap;gap:22px;padding:14px 24px;justify-content:center;background:white;border-bottom:1px solid #dbe3e8;position:sticky;top:0;z-index:1}
a{color:#1c6e60;text-decoration:none}a:hover{text-decoration:underline}main{max-width:1450px;margin:auto;padding:28px 24px}
section{background:white;border:1px solid #dce5eb;border-radius:12px;padding:28px;margin-bottom:24px;scroll-margin-top:70px}h2{font-size:24px;margin:0 0 20px}h3{font-size:20px}h4{font-size:16px;margin:22px 0 10px}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.card{background:#f2f7f6;border:1px solid #d9e7e1;border-radius:8px;padding:20px}.card b{display:block;font-size:27px;line-height:1.3}.card span{font-size:12px;color:#596e7b}
.columns{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:32px}.muted{font-size:13px;color:#607381}.notice{border-left:4px solid #638ca3;background:#eef5f9;padding:14px 18px;font-size:14px}
.attention{border:1px solid #dfb194;border-left:5px solid #b3592c;background:#fff5ee;padding:16px 20px;border-radius:6px;margin:16px 0}.attention ul{margin-bottom:0}
.prose{white-space:pre-wrap;font-size:14px;line-height:1.8;overflow-wrap:anywhere}.answer{border-top:1px solid #e2e9ee;padding-top:16px}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px}th{text-transform:uppercase;letter-spacing:.5px;font-size:10px;background:#edf3f6}td,th{border-bottom:1px solid #e1e8ed;text-align:left;vertical-align:top;padding:10px;overflow-wrap:anywhere}
.rating{display:flex;justify-content:space-between;gap:12px;background:#edf4f1;padding:8px 12px;border-radius:5px}.rating span{white-space:nowrap;color:#1e6b56;font-weight:700}
blockquote{font-size:12px;color:#526c7d;border-left:3px solid #b8ccd8;padding-left:12px;margin:12px 0}pre{font:11px/1.65 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:16px;border-radius:5px;max-height:650px;overflow:auto}
details{margin:12px 0;border:1px solid #e0e8ed;border-radius:6px;padding:10px 14px}summary{cursor:pointer;color:#365b70;font-size:12px}article+article{border-top:2px solid #dbe6eb;margin-top:32px;padding-top:24px}footer{text-align:center;color:#607381;padding:10px 24px 35px;font-size:12px}
@media(max-width:1000px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.columns{grid-template-columns:1fr}section{padding:20px}main{padding:16px}}
@media(max-width:600px){h1{font-size:32px}.cards{grid-template-columns:1fr}nav{position:static;gap:12px}section{padding:16px}}
@media print{nav{position:static}details{break-inside:avoid}section{border:0}body{background:white}.columns{display:block}pre{max-height:none}header{background:white;color:#152f40}header p,.eyebrow{color:#365b70}}
"""


def render_markdown(data: dict[str, Any], revision: str) -> str:
    status = data["status"]
    lines = ["# V2 experiment report", "", f"Experiment: {data['experiment_id']}",
             f"Evaluation: {data['evaluation_run_id']}", f"Report: {revision}",
             f"Complete: {status['complete']}; human approved: {status['human_review']['approved']}/{len(data['responses'])}",
             f"Automatic outputs valid: {status['all_required_outputs_valid']}; human-resolved unavailable audits: "
             f"{status['audit_resolution']['human_unavailable']}; review policy: {data['review_policy_revision']}",
             f"Known API cost: ${data['cost']['known_usd']:.8f}; unknown-cost calls: {data['cost']['unknown_cost_calls']}",
             "", "The HTML report contains expandable raw evidence, configurations and qualification records.", ""]
    for row in data["responses"]:
        lines += [f"## {row['configuration_key']} · repetition {row['repetition']}", "", "### Question", ""]
        lines += [m["content"] for m in row["task"] if m["role"] == "user"]
        lines += ["", "### Full candidate answer", "", row["answer"], "", "### Attention flags", "",
                  ", ".join(row["flags"]) or "None", "", "### Preference labels", "", _json(row["labels"]),
                  "", "### Extraction", "", _json(row["extraction"]), "", "### Quality audit", "", _json(row["audit"]),
                  "", "### Audit resolution (quality scores remain missing if unavailable)", "", _json(row["audit_resolution"]),
                  "", "### Human validation", "", _json(row["human_reviews"]), ""]
    lines += ["## Runtime and cost", "", _json(data["calls"]), "", "## Interpretation limits", ""]
    lines += [f"- {v}" for v in data["limitations"]]
    return "\n".join(lines) + "\n"


def publish(db: Database, store: ArtifactStore, experiment_id: str, run_id: str,
            *, code_revision: str | None = None) -> Record:
    with store.lock(experiment_id):
        data = collect(db, store, experiment_id, run_id)
        manifest = {"reporting_policy": REVISION, "data_sha256": content_hash(data),
                    "renderer_sha256": sha256_bytes(Path(__file__).read_bytes()), "code_revision": code_revision}
        revision = "srpt-" + content_hash(manifest)[:16]
        existing = db.get_record(revision)
        if existing:
            for ref in db.artifacts_for(revision).values():
                store.read_verified(ref)
            return existing
        sink = store.sink(f"{experiment_id}/reports/{revision}")
        sink.write_json("manifest.json", {"report_revision": revision, **manifest, "inputs": data["inputs"]})
        sink.write_json("report.json", data)
        sink.write("report.html", render_html(data, revision))
        sink.write("report.md", render_markdown(data, revision))
        record = Record.create(kind=RecordKind.REPORT_REVISION, record_id=revision, experiment_id=experiment_id,
            subject_id=run_id, inputs={**data["inputs"], **data["artifact_hashes"]}, policy_revision=REVISION,
            code_revision=code_revision, payload={"report_type": "selection-v2-full", "input_manifest": manifest,
                                               "evaluation_complete": data["status"]["complete"],
                                               "cost": data["cost"]})
        with db.transaction():
            db.insert_record(record)
            db.register_artifacts(record.record_id, sink.refs)
        return record
