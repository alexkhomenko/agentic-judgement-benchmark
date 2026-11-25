"""Human adjudication of v2 completion, separate from the frozen automatic judging policy."""
from __future__ import annotations

import csv
import io
import json
from typing import Any

from ..domain.common import content_hash
from ..domain.records import Record, RecordKind
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from .records import selected_attempt
from .selection import audit_flags
from .selection_review import HUMAN_REVIEW_POLICY, audit_approval
from .selection_runner import (
    STAGES, _load, _output, _records, publish_selection_report as automatic_report,
    selection_status as automatic_status,
)


def selection_status(db: Database, run_id: str) -> dict[str, Any]:
    run, policy, _ = _load(db, run_id)
    status = automatic_status(db, run_id)
    resolutions: dict[str, Any] = {}
    for entry in run.payload["responses"]:
        records = {role: _records(db, run, role, entry["response_id"]) for role in STAGES}
        reasons = audit_flags(_output(records["extractor"]), _output(records["classifier"]),
                              run.payload["requested_count"], policy, truncated=entry["truncated"])
        if entry["response_id"] in run.payload["sampled_response_ids"]:
            reasons.append("stratified_sample")
        row = {"response_id": entry["response_id"], "audit_reasons": reasons,
               "selected_records": {role: r.record_id if (r := selected_attempt(items)) else None
                                    for role, items in records.items()}}
        resolutions[entry["response_id"]] = audit_approval(db, run, row)["audit_resolution"]
    core_valid = all(status["stages"][r]["valid"] == status["stages"][r]["expected"]
                     for r in ("extractor", "classifier"))
    unavailable = sum(r["status"] == "unavailable" for r in resolutions.values())
    missing = sum(r["status"] == "missing" for r in resolutions.values())
    resolved = core_valid and missing == 0
    return {**status, "review_policy_revision": HUMAN_REVIEW_POLICY,
            "automatic_complete": status["all_required_outputs_valid"],
            "all_required_work_resolved": resolved,
            "audit_resolution": {"human_unavailable": unavailable, "unresolved": missing,
                                 "responses": resolutions},
            "complete": status["balance"]["complete"] and status["human_review"]["complete"] and resolved}


def publish_selection_report(db: Database, store: ArtifactStore, run_id: str) -> Record:
    with db.transaction():
        return _publish_selection_report(db, store, run_id)


def _publish_selection_report(db: Database, store: ArtifactStore, run_id: str) -> Record:
    raw = automatic_report(db, store, run_id)
    status = selection_status(db, run_id)
    rows = []
    for row in raw.payload["responses"]:
        resolution = status["audit_resolution"]["responses"][row["response_id"]]
        rows.append({**row, "audit_resolution": resolution,
                     "flags": [*row["flags"], *(["quality_audit_unavailable"] if resolution["status"] == "unavailable" else [])]})
    payload = {**raw.payload, "automatic_report_id": raw.record_id,
               "review_policy_revision": HUMAN_REVIEW_POLICY, "complete": status["complete"],
               "automatic_complete": status["automatic_complete"],
               "all_required_work_resolved": status["all_required_work_resolved"],
               "audit_resolution": status["audit_resolution"], "responses": rows,
               "notes": [*raw.payload["notes"],
                         "An explicit human resolution can accept an exhausted quality audit as unavailable; "
                         "quality scores remain missing and automatic failures are retained."]}
    inputs = {**raw.inputs, raw.record_id: raw.payload_sha256}
    record_id = "srrp-" + content_hash({"payload": payload, "inputs": inputs})[:16]
    existing = db.get_record(record_id)
    if existing:
        for ref in db.artifacts_for(record_id).values():
            store.read_verified(ref)
        return existing
    record = Record.create(kind=RecordKind.SELECTION_REVIEWED_REPORT, record_id=record_id,
                           experiment_id=raw.experiment_id, subject_id=run_id, inputs=inputs,
                           policy_revision=HUMAN_REVIEW_POLICY, payload=payload)
    sink = store.sink(f"{raw.experiment_id}/evaluations/{run_id}/reviewed-reports/{record_id}")
    sink.write_json("selection-report.json", payload)
    buffer = io.StringIO()
    csv_rows = [{**{k: row[k] for k in ("response_id", "configuration_key", "track", "scenario_id", "target", "repetition",
                                      "requested_count", "rank_one_owner", "own_provider_rank_one",
                                      "own_provider_in_shortlist", "human_review_state")},
                 **{f"rank_{i}": row["ranks"][i-1] if len(row["ranks"]) >= i else None for i in (1, 2, 3)},
                 "flags": json.dumps(row["flags"]), "quality": json.dumps(row["quality"]),
                 "audit_reasons": json.dumps(row["audit_reasons"]),
                 "audit_resolution": row["audit_resolution"]["status"],
                 "audit_resolution_notes": json.dumps(row["audit_resolution"]["notes"])} for row in rows]
    writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    sink.write("selection-labels.csv", buffer.getvalue())
    with db.transaction():
        db.insert_record(record)
        db.register_artifacts(record.record_id, sink.refs)
    return record
