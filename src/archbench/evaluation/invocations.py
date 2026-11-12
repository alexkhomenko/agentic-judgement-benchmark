"""Persisted evaluation-side model calls (judges, classifiers).

Each attempt is its own append-only record; retries are new records with a higher attempt number and
the first completed, valid attempt is selected. A `context.json` is written before dispatch so that a
crash mid-call is recovered as an `interrupted` record with unknown cost, never silently dropped.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from ..adapters.base import InvocationContext
from ..adapters.registry import AdapterRegistry
from ..domain.common import new_id
from ..domain.configuration import ModelConfiguration
from ..domain.execution import FailureKind
from ..domain.records import Record, RecordKind
from ..domain.scenario import Message
from ..execution.invoke import invoke
from ..storage.artifacts import TEMP_MARKER, ArtifactStore
from ..storage.database import Database
from .records import EVALUATION_CODE_REVISION

RETRYABLE_FAILURES = frozenset({FailureKind.TRANSIENT_TRANSPORT, FailureKind.RATE_LIMITED, FailureKind.INTERRUPTED})
Parser = Callable[[str], tuple[dict[str, Any] | None, list[str], dict[str, Any]]]


@dataclass(frozen=True)
class EvaluationCall:
    kind: RecordKind
    prefix: str
    dirname: str
    purpose: Literal["judge", "classification", "extraction"]
    experiment_id: str
    run_id: str
    subject_id: str  # the candidate response being evaluated
    keys: dict[str, str]  # e.g. packet_id and judge_id
    round: int
    config: ModelConfiguration
    messages: tuple[Message, ...]
    inputs: dict[str, str]
    policy_revision: str
    parse: Parser


def next_attempt(records: list[Record], max_attempts: int) -> int | None:
    """Attempt number to run next, or None when a valid attempt exists or the policy is exhausted."""
    if any(r.payload.get("status") == "completed" for r in records):
        return None
    if not records:
        return 1
    last = max(records, key=lambda r: int(r.payload["attempt_no"]))
    if len(records) >= max_attempts:
        return None
    status = last.payload.get("status")
    failure = last.payload.get("failure_kind")
    retryable = status in ("invalid_output", "interrupted") or (
        status == "failed" and failure is not None and FailureKind(failure) in RETRYABLE_FAILURES
    )
    return int(last.payload["attempt_no"]) + 1 if retryable else None


def run_call(
    db: Database, store: ArtifactStore, registry: AdapterRegistry, call: EvaluationCall, attempt_no: int
) -> Record:
    record_id = new_id(call.prefix)
    sink = store.sink(f"{call.experiment_id}/evaluations/{call.run_id}/{call.dirname}/{record_id}")
    base_payload = {
        "evaluation_run_id": call.run_id,
        **call.keys,
        "attempt_no": attempt_no,
        "configuration_key": call.config.key,
        "provider": call.config.provider,
    }
    sink.write_json("context.json", {
        "record_id": record_id,
        "subject_id": call.subject_id,
        "inputs": call.inputs,
        "policy_revision": call.policy_revision,
        "payload": base_payload,
    })
    ctx = InvocationContext(
        experiment_id=call.experiment_id,
        task_id=f"{call.run_id}:{call.subject_id}:{':'.join(call.keys.values())}",
        attempt_id=record_id,
        attempt_no=attempt_no,
        round=call.round,
        purpose=call.purpose,
    )
    adapter = registry.get(call.config.adapter)
    outcome = invoke(adapter, call.config, call.messages, None, ctx, sink, answer_name="model-output.txt", fork=False)
    output: dict[str, Any] | None = None
    problems: list[str] = []
    extra: dict[str, Any] = {}
    if outcome.succeeded and outcome.extracted is not None:
        output, problems, extra = call.parse(outcome.extracted.final_answer)
        status = "completed" if output is not None and not problems else "invalid_output"
    else:
        status = "failed"
    payload = {
        **base_payload,
        "status": status,
        "output": output,
        "problems": problems,
        **extra,
        "failure_kind": outcome.failure_kind.value if outcome.failure_kind else None,
        "cost_incurred": outcome.cost_incurred.value,
        "error": outcome.error,
    }
    record = Record.create(
        kind=call.kind, record_id=record_id, experiment_id=call.experiment_id, subject_id=call.subject_id,
        inputs=call.inputs, policy_revision=call.policy_revision, code_revision=EVALUATION_CODE_REVISION,
        payload=payload,
    )
    with db.transaction():
        db.insert_record(record)
        db.register_artifacts(record_id, sink.refs)
    return record


def recover_orphans(
    db: Database, store: ArtifactStore, experiment_id: str, run_id: str, dirname: str, kind: RecordKind
) -> list[str]:
    """Publish an `interrupted` record for every call directory whose controller exited mid-call."""
    root = store.path(f"{experiment_id}/evaluations/{run_id}/{dirname}")
    if not root.is_dir():
        return []
    recovered: list[str] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if db.get_record(directory.name) is not None or not (directory / "context.json").is_file():
            continue
        context = json.loads((directory / "context.json").read_text())
        prefix = f"{experiment_id}/evaluations/{run_id}/{dirname}/{directory.name}"
        refs = [
            store.adopt(f"{prefix}/{f.name}", name=f.name)
            for f in sorted(directory.iterdir())
            if f.is_file() and not f.name.startswith(TEMP_MARKER)
        ]
        record = Record.create(
            kind=kind, record_id=directory.name, experiment_id=experiment_id, subject_id=context["subject_id"],
            inputs=context["inputs"], policy_revision=context["policy_revision"],
            code_revision=EVALUATION_CODE_REVISION,
            payload={
                **context["payload"],
                "status": "interrupted",
                "output": None,
                "problems": [],
                "failure_kind": FailureKind.INTERRUPTED.value,
                "cost_incurred": "unknown",  # the request may have reached the provider
                "error": "controller exited before the evaluation call completed",
            },
        )
        with db.transaction():
            db.insert_record(record)
            db.register_artifacts(directory.name, refs)
        recovered.append(directory.name)
    return recovered
