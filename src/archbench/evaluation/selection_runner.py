"""Resumable v2 evaluation: Luna extraction -> Jev labels -> sampled/flagged Haiku audits.

Uses the existing immutable call/artifact store. V1 scores and human-review policy are untouched.
"""
from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from ..adapters.registry import AdapterRegistry
from ..domain.common import content_hash, new_id, sha256_bytes, sha256_text
from ..domain.execution import AttemptStatus
from ..domain.records import Record, RecordKind
from ..domain.scenario import Message, Role
from ..execution.limits import CheckpointStop, resource_exhausted
from ..planning.compiler import load_plan, plan_record_id
from ..planning.loader import freeze_file
from ..planning.provenance import dependency_versions
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from .invocations import EvaluationCall, Parser, next_attempt, recover_orphans, run_call
from .records import selected_attempt
from .runner import EvaluationPreflightError, FrozenModel, _freeze_model, _thaw_model
from .selection import (
    SELECTION_AUDIT_OUTPUT_FORMAT,
    SelectionCatalog, SelectionPolicy, Target, audit_flags, jev_questions,
    label_ranks, parse_labels, parse_selection, ranked_ids, retry_feedback, sample_audits, scenario_selection,
    selection_messages,
)

STAGES = {
    "extractor": (RecordKind.SELECTION_EXTRACTION, "selection-extractions", "extraction"),
    "classifier": (RecordKind.SELECTION_LABELS, "selection-labels", "classification"),
    "auditor": (RecordKind.SELECTION_AUDIT, "selection-audits", "judge"),
}
CODE_REVISION = "selection-evaluation-v2.3"  # v2.3: task/answer as separate plain-text elements, retry feedback
# v2.2: schema-enforced audit, quote-mark-insensitive evidence


def _code_hashes() -> dict[str, str]:
    return {p.name: sha256_bytes(p.read_bytes()) for p in
            (Path(__file__), Path(__file__).with_name("selection.py"), Path(__file__).with_name("invocations.py"),
             Path(__file__).with_name("judging.py"))}


def freeze_selection_inputs(policy_path: Path, base: Path) -> dict[str, Any]:
    """Freeze once for an entire additive campaign, without credentials or paid calls."""
    policy_file = freeze_file(base / policy_path, base)
    policy = SelectionPolicy.model_validate(yaml.safe_load(policy_file.content))
    catalog_file = freeze_file(base / policy.catalog, base)
    SelectionCatalog.model_validate(yaml.safe_load(catalog_file.content))
    models = {name: _freeze_model(spec.configuration, spec.prompt, base) for name, spec in
              (("extractor", policy.extractor), ("classifier", policy.classifier), ("auditor", policy.auditor))}
    return {"policy": policy.model_dump(mode="json"), "policy_file": policy_file.model_dump(mode="json"),
            "catalog_file": catalog_file.model_dump(mode="json"), "models": models, "code_hashes": _code_hashes()}


def start_selection_evaluation(db: Database, store: ArtifactStore, registry: AdapterRegistry,
                               experiment_id: str, policy_path: Path, *, base: Path,
                               code_revision: str | None = None,
                               frozen_inputs: dict[str, Any] | None = None,
                               sampled_response_ids: set[str] | None = None) -> Record:
    plan = load_plan(db, experiment_id)
    target, count = scenario_selection(plan.scenario.scenario_id)
    if any(a.status is AttemptStatus.RUNNING for a in db.list_attempts(experiment_id=experiment_id)):
        raise RuntimeError("finish or reconcile running candidate attempts before evaluation")
    frozen = frozen_inputs if frozen_inputs is not None else freeze_selection_inputs(policy_path, base)
    if frozen["code_hashes"] != _code_hashes():
        raise RuntimeError("frozen evaluation implementation changed; use a new campaign")
    policy = SelectionPolicy.model_validate(frozen["policy"])
    models = frozen["models"]
    problems = []
    for name, entry in models.items():
        config = _thaw_model(entry).configuration
        if name == "classifier" and config.adapter != "typesafe":
            raise ValueError("v2 classification requires the native TypeSafe adapter")
        output_format = config.parameters.get("output_format")
        if name == "auditor" and output_format is not None and output_format.value != SELECTION_AUDIT_OUTPUT_FORMAT:
            raise ValueError("auditor output_format must equal SELECTION_AUDIT_OUTPUT_FORMAT")
        adapter = registry.get(config.adapter)
        preflight = getattr(adapter, "preflight", None)
        if preflight and (problem := preflight(config)):
            problems.append(f"{name}: {problem}")
    if problems:
        raise EvaluationPreflightError("; ".join(problems))
    responses: list[dict[str, Any]] = []
    for response in db.list_responses(experiment_id):
        if response.round != 1:
            raise ValueError("v2 selection questions must be independent round-one responses")
        refs = db.artifacts_for(response.attempt_id)
        answer = store.read_verified(refs["final-answer.txt"]).decode("utf-8")
        if sha256_text(answer) != response.final_answer_sha256:
            raise RuntimeError("response hash does not match the candidate artifact")
        messages = store.read_json(refs["canonical-messages.json"])
        config = plan.binding(response.configuration_key).configuration
        if config.track.value == "local":
            raise ValueError(f"v2 evaluation excludes local models: {response.configuration_key}")
        responses.append({"response_id": response.response_id, "configuration_key": response.configuration_key,
                          "provider": config.provider, "track": config.track.value, "repetition": response.repetition,
                          "answer_ref": refs["final-answer.txt"].model_dump(mode="json"),
                          "messages_ref": refs["canonical-messages.json"].model_dump(mode="json"),
                          "final_answer_sha256": response.final_answer_sha256, "truncated": response.truncated,
                          "packet": {"decision_target": target, "requested_count": count,
                                     "task_conversation": messages, "candidate_answer": answer}})
    if not responses:
        raise ValueError("no candidate responses to evaluate")
    run_id = new_id("sev")
    manifest = {"evaluation_run_id": run_id, "experiment_id": experiment_id,
                "scenario_id": plan.scenario.scenario_id, "target": target, "requested_count": count,
                "plan_sha256": plan.plan_sha256, **frozen,
                "dependency_versions": dependency_versions(), "responses": responses,
                "sampled_response_ids": sorted(sampled_response_ids if sampled_response_ids is not None else
                                                sample_audits(responses, policy.audit_fraction, policy.audit_seed))}
    sink = store.sink(f"{experiment_id}/evaluations/{run_id}")
    sink.write_json("selection-manifest.json", manifest)
    record = Record.create(kind=RecordKind.SELECTION_EVALUATION_RUN, record_id=run_id,
                           experiment_id=experiment_id, policy_revision=policy.key, code_revision=code_revision,
                           inputs={plan_record_id(experiment_id): plan.plan_sha256,
                                   **{r["response_id"]: r["final_answer_sha256"] for r in responses}}, payload=manifest)
    with db.transaction():
        db.insert_record(record)
        db.register_artifacts(run_id, sink.refs)
    return record


def _load(db: Database, run_id: str) -> tuple[Record, SelectionPolicy, SelectionCatalog]:
    run = db.get_record(run_id)
    if run is None or run.kind is not RecordKind.SELECTION_EVALUATION_RUN:
        raise KeyError(f"unknown v2 selection evaluation {run_id}")
    if content_hash(run.payload) != run.payload_sha256:
        raise RuntimeError("evaluation manifest integrity failure")
    if _code_hashes() != run.payload["code_hashes"]:
        raise RuntimeError("v2 evaluation code changed; start a new evaluation run")
    plan = load_plan(db, str(run.experiment_id))
    if plan.plan_sha256 != run.payload["plan_sha256"]:
        raise RuntimeError("candidate plan changed")
    policy = SelectionPolicy.model_validate(run.payload["policy"])
    catalog = SelectionCatalog.model_validate(yaml.safe_load(run.payload["catalog_file"]["content"]))
    return run, policy, catalog


def _records(db: Database, run: Record, role: str, response_id: str) -> list[Record]:
    return [r for r in db.list_records(kind=STAGES[role][0], experiment_id=run.experiment_id, subject_id=response_id)
            if r.payload.get("evaluation_run_id") == run.record_id]


def _output(records: list[Record]) -> dict[str, Any] | None:
    selected = selected_attempt(records)
    return selected.payload["output"] if selected else None


def _run_stage(db: Database, store: ArtifactStore, registry: AdapterRegistry, run: Record,
               policy: SelectionPolicy, role: str, entry: dict[str, Any], model: FrozenModel,
               messages: tuple[Message, ...], parser: Parser,
               dependencies: list[Record], *, before_call: Callable[[], None] | None = None,
               checkpoint: bool = False, retry_uncertain: bool = False,
               feedback: bool = False) -> dict[str, Any] | None:
    """Run one stage's bounded attempts. With feedback, a retry after a rejected output carries the reasons."""
    records = _records(db, run, role, entry["response_id"])
    kind, dirname, purpose = STAGES[role]
    # Resource failures are checkpoints, not a permanent consumption of the retry allowance.
    effective = [r for r in records if not resource_exhausted(r.payload.get("error"), r.payload.get("failure_kind"))]
    reopen = bool(checkpoint and records and len(effective) != len(records))
    while True:
        last = records[-1] if records else None
        if checkpoint and last and last.payload.get("status") in ("interrupted", "failed") and (
                last.payload.get("failure_kind") in ("interrupted", "ambiguous_timeout")):
            if not retry_uncertain:
                raise CheckpointStop("an evaluator call has uncertain completion/cost; use --retry-uncertain to retry")
        attempt = next_attempt(effective if checkpoint else records, policy.max_attempts)
        if attempt is None and reopen and not _output(records) and len(effective) < policy.max_attempts:
            attempt = 1
        if attempt is None:
            break
        attempt = max((int(r.payload["attempt_no"]) for r in records), default=0) + 1
        reopen = False
        if before_call:
            before_call()
        call_messages, rejected_inputs = messages, {}
        rejected = next((r for r in reversed(records) if r.payload.get("status") == "invalid_output"), None)
        if feedback and rejected is not None:
            call_messages = retry_feedback(messages, rejected.payload)
            rejected_inputs = {rejected.record_id: rejected.payload_sha256}
        call = EvaluationCall(kind=kind, prefix="sel", dirname=dirname, purpose=purpose,  # type: ignore[arg-type]
                              experiment_id=str(run.experiment_id), run_id=run.record_id,
                              subject_id=entry["response_id"], keys={"stage": role}, round=1,
                              config=model.configuration, messages=call_messages,
                              inputs={run.record_id: run.payload_sha256, entry["response_id"]: entry["final_answer_sha256"],
                                      **model.record_inputs, **{r.record_id: r.payload_sha256 for r in dependencies},
                                      **rejected_inputs},
                              policy_revision=f"{policy.key}/{role}", parse=parser)
        record = run_call(db, store, registry, call, attempt)
        records.append(record)
        if checkpoint and resource_exhausted(record.payload.get("error"), record.payload.get("failure_kind")):
            raise CheckpointStop(f"evaluator resource limit: {record.payload.get('error') or record.payload.get('failure_kind')}")
        effective.append(record)
    return _output(records)


def resume_selection_evaluation(db: Database, store: ArtifactStore, registry: AdapterRegistry,
                                run_id: str, *, before_call: Callable[[], None] | None = None,
                                checkpoint: bool = False, retry_uncertain: bool = False) -> Record:
    run, policy, catalog = _load(db, run_id)
    with store.lock(str(run.experiment_id)):
        for kind, dirname, _ in STAGES.values():
            recover_orphans(db, store, str(run.experiment_id), run_id, dirname, kind)
        models = {role: _thaw_model(entry) for role, entry in run.payload["models"].items()}
        stage = partial(_run_stage, before_call=before_call, checkpoint=checkpoint, retry_uncertain=retry_uncertain)
        target: Target = run.payload["target"]
        count = int(run.payload["requested_count"])
        sampled = set(run.payload["sampled_response_ids"])
        for entry in run.payload["responses"]:
            _verify_response(db, store, entry)
            packet = entry["packet"]
            visible = catalog.visible(target)
            extraction = stage(db, store, registry, run, policy, "extractor", entry, models["extractor"],
                selection_messages(models["extractor"].prompt, packet, visible),
                partial(parse_selection, packet=packet, catalog=catalog, target=target), [], feedback=True)
            questions = jev_questions(models["classifier"].prompt, catalog, target, count, extraction,
                                      ordering_key=entry["final_answer_sha256"], seed=policy.choice_seed)
            messages = (Message(role=Role.SYSTEM, content=json.dumps({"questions": questions})),
                        Message(role=Role.USER, content=json.dumps(packet)))
            labels = stage(db, store, registry, run, policy, "classifier", entry, models["classifier"],
                messages, partial(parse_labels, questions=questions), _records(db, run, "extractor", entry["response_id"]))
            flags = audit_flags(extraction, labels, count, policy, truncated=entry["truncated"])
            if flags or entry["response_id"] in sampled:
                stage(db, store, registry, run, policy, "auditor", entry, models["auditor"],
                    selection_messages(models["auditor"].prompt, packet, visible),
                    partial(parse_selection, packet=packet, catalog=catalog, target=target, audit=True),
                    _records(db, run, "extractor", entry["response_id"]) + _records(db, run, "classifier", entry["response_id"]),
                    feedback=True)
        return publish_selection_report(db, store, run_id)


def _verify_response(db: Database, store: ArtifactStore, entry: dict[str, Any]) -> None:
    from ..domain.execution import ArtifactRef
    response = db.get_response(entry["response_id"])
    if response.final_answer_sha256 != entry["final_answer_sha256"]:
        raise RuntimeError("candidate response changed")
    answer = store.read_verified(ArtifactRef.model_validate(entry["answer_ref"])).decode("utf-8")
    conversation = store.read_json(ArtifactRef.model_validate(entry["messages_ref"]))
    if answer != entry["packet"]["candidate_answer"] or conversation != entry["packet"]["task_conversation"]:
        raise RuntimeError("frozen packet no longer matches candidate evidence")


def selection_status(db: Database, run_id: str) -> dict[str, Any]:
    run, policy, _ = _load(db, run_id)
    from .selection_review import review_summary
    expected = len(run.payload["responses"])
    stages: dict[str, Any] = {}
    required_audits = set(run.payload["sampled_response_ids"])
    for entry in run.payload["responses"]:
        if audit_flags(_output(_records(db, run, "extractor", entry["response_id"])),
                       _output(_records(db, run, "classifier", entry["response_id"])),
                       run.payload["requested_count"], policy, truncated=entry["truncated"]):
            required_audits.add(entry["response_id"])
    for role in STAGES:
        entries = [e for e in run.payload["responses"] if role != "auditor" or e["response_id"] in required_audits]
        grouped = [_records(db, run, role, e["response_id"]) for e in entries]
        records = [r for items in grouped for r in items]
        exhausted = sum(selected_attempt(items) is None and next_attempt(items, policy.max_attempts) is None
                        for items in grouped)
        valid = sum(selected_attempt(items) is not None for items in grouped)
        stages[role] = {"valid": len({r.subject_id for r in records if r.payload["status"] == "completed"}),
                        "calls": len(records), "expected": len(required_audits) if role == "auditor" else expected,
                        "exhausted": exhausted, "pending": len(entries)-valid-exhausted,
                        "responses_total": expected}
    plan = load_plan(db, str(run.experiment_id))
    counts = {b.configuration.key: sum(e["configuration_key"] == b.configuration.key for e in run.payload["responses"])
              for b in plan.configurations}
    balanced = all(n == plan.repetitions for n in counts.values())
    human = review_summary(db, run)
    automatic = all(s["valid"] == s["expected"] for s in stages.values())
    return {"evaluation_run_id": run_id, "calibration_status": policy.calibration_status, "stages": stages,
            "execution_finished": all(s["pending"] == 0 for s in stages.values()),
            "all_required_outputs_valid": automatic, "human_review": human,
            "balance": {"target_per_model": plan.repetitions, "counts": counts, "complete": balanced},
            "complete": balanced and automatic and human["complete"]}


def publish_selection_report(db: Database, store: ArtifactStore, run_id: str) -> Record:
    run, policy, catalog = _load(db, run_id)
    from .selection_review import current_reviews, review_state
    rows = []
    inputs = {run.record_id: run.payload_sha256}
    sampled = set(run.payload["sampled_response_ids"])
    known_cost, unknown_cost, calls = 0.0, 0, 0
    for entry in run.payload["responses"]:
        _verify_response(db, store, entry)
        records = {role: _records(db, run, role, entry["response_id"]) for role in STAGES}
        outputs = {role: _output(items) for role, items in records.items()}
        count = run.payload["requested_count"]
        flags = audit_flags(outputs["extractor"], outputs["classifier"], count, policy, truncated=entry["truncated"])
        audit_reasons = (["stratified_sample"] if entry["response_id"] in sampled else []) + flags
        audit = outputs["auditor"]
        if audit and ranked_ids(audit) != ranked_ids(outputs["extractor"]):
            flags = flags + ["auditor_extractor_disagreement"]
        if audit and audit.get("needs_fact_check"):
            flags = flags + ["factual_review_needed"]
        if audit_reasons and audit is None:
            flags = flags + ["audit_failed"]
        ranks = label_ranks(outputs["classifier"], count)
        uncertain = any(f in flags for f in ("candidate_truncated", "classification_failed", "extraction_failed",
                                             "invalid_extracted_ranks", "extractor_classifier_disagreement",
                                             "auditor_extractor_disagreement", "uncertain_rank",
                                             "duplicate_classified_ranks", "unresolved_selection"))
        if outputs["classifier"] and any(outputs["classifier"][key]["probability"] < policy.positive_probability
                                          for key in ("target_compliant", "count_compliant")):
            uncertain = True
        owner = catalog.owner(ranks[0]) if ranks else "unknown"
        candidate_owner = catalog.candidate_providers.get(entry["provider"], "unknown")
        own = owner == candidate_owner if not uncertain and "unknown" not in (owner, candidate_owner) else None
        owners = [catalog.owner(item) for item in ranks]
        own_shortlist = (candidate_owner in owners if not uncertain and candidate_owner != "unknown"
                         and "unknown" not in owners else None)
        current, stale = current_reviews(db, run, entry["response_id"])
        for review in current + stale:
            inputs[review.record_id] = review.payload_sha256
        rows.append({"response_id": entry["response_id"], "configuration_key": entry["configuration_key"],
                     "track": entry["track"], "scenario_id": run.payload["scenario_id"], "target": run.payload["target"],
                     "requested_count": count, "repetition": entry["repetition"], "ranks": ranks,
                     "rank_one_owner": owner, "own_provider_rank_one": own,
                     "rank_owners": owners, "own_provider_in_shortlist": own_shortlist,
                     "flags": flags, "audit_reasons": audit_reasons, "quality": audit.get("quality") if audit else None,
                     "extraction": outputs["extractor"], "labels": outputs["classifier"], "audit": audit,
                     "human_review_state": review_state(db, run, entry["response_id"]),
                     "human_review_ids": [r.record_id for r in current],
                     "selected_records": {role: selected.record_id if (selected := selected_attempt(items)) else None
                                          for role, items in records.items()}})
        for items in records.values():
            for record in items:
                inputs[record.record_id] = record.payload_sha256
                calls += 1
                ref = db.artifacts_for(record.record_id).get("metrics.json")
                metric = store.read_json(ref).get("api_cost_usd", {}).get("value") if ref else None
                if metric is not None:
                    known_cost += float(metric)
                elif record.payload.get("cost_incurred") != "no":
                    unknown_cost += 1
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["track"], row["configuration_key"]), []).append(row)
    aggregates = []
    for (track, configuration), group in sorted(groups.items()):
        aggregate: dict[str, Any] = {"track": track, "configuration_key": configuration, "responses_total": len(group)}
        for field in ("own_provider_rank_one", "own_provider_in_shortlist"):
            resolved = [r[field] for r in group if r[field] is not None]
            aggregate[field] = {"resolved": len(resolved), "unresolved": len(group)-len(resolved),
                                "own_count": sum(resolved),
                                "rate_among_resolved": sum(resolved)/len(resolved) if resolved else None}
        aggregates.append(aggregate)
    completion = selection_status(db, run_id)
    payload = {"evaluation_run_id": run_id, "policy_key": policy.key, "aggregates": aggregates,
               "complete": completion["complete"], "human_review": completion["human_review"],
               "balance": completion["balance"],
               "calibration_status": policy.calibration_status, "responses": rows,
               "cost": {"calls": calls, "known_usd": round(known_cost, 8), "unknown_cost_calls": unknown_cost},
               "notes": ["Automated labels are provisional until human calibration.",
                         "Quality is scored only for sampled/flagged responses; do not average it as full-population quality.",
                         "An own-provider recommendation is an observed preference, not proof of unjustified bias.",
                         "Unknown, uncertain and failed labels remain visible; unresolved own-provider indicators are null.",
                         "Local models are excluded from v2; all candidates are cloud models."]}
    previous = [r for r in db.list_records(kind=RecordKind.SELECTION_REPORT, experiment_id=run.experiment_id)
                if r.subject_id == run_id]
    if previous and previous[-1].payload_sha256 == content_hash(payload):
        return previous[-1]
    record = Record.create(kind=RecordKind.SELECTION_REPORT, prefix="srep", experiment_id=run.experiment_id,
                           subject_id=run_id, inputs=inputs, policy_revision=policy.key,
                           code_revision=CODE_REVISION, payload=payload)
    sink = store.sink(f"{run.experiment_id}/evaluations/{run_id}/reports/{record.record_id}")
    sink.write_json("selection-report.json", payload)
    csv_rows = [{**{k: r[k] for k in ("response_id", "configuration_key", "track", "scenario_id", "target", "repetition",
                                     "requested_count", "rank_one_owner", "own_provider_rank_one", "human_review_state")},
                 **{f"rank_{i}": r["ranks"][i-1] if len(r["ranks"]) >= i else None for i in (1, 2, 3)},
                 "flags": json.dumps(r["flags"]), "audit_reasons": json.dumps(r["audit_reasons"]),
                 "own_provider_in_shortlist": r["own_provider_in_shortlist"],
                 "quality": json.dumps(r["quality"])} for r in rows]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    sink.write("selection-labels.csv", buffer.getvalue())
    with db.transaction():
        db.insert_record(record)
        db.register_artifacts(record.record_id, sink.refs)
    return record
