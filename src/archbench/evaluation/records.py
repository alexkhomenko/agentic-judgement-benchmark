"""Helpers for reading the derived records that belong to one evaluation run."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..domain.execution import ArtifactRef
from ..domain.records import Record, RecordKind
from ..storage.artifacts import ArtifactStore
from ..storage.database import Database

EVALUATION_CODE_REVISION = "evaluation-v5"  # v5: version digits may follow catalog names (v4: merge duplicate items)


def run_records(db: Database, kind: RecordKind, experiment_id: str, run_id: str) -> list[Record]:
    """Records of one kind produced by one evaluation run, in creation order."""
    return [
        r for r in db.list_records(kind=kind, experiment_id=experiment_id)
        if r.payload.get("evaluation_run_id") == run_id
    ]


def effective(records: Iterable[Record]) -> list[Record]:
    """Records not superseded by another record in the same collection. Superseded records are kept."""
    items = list(records)
    superseded = {r.supersedes for r in items if r.supersedes}
    return [r for r in items if r.record_id not in superseded]


def packet_ref(db: Database, packet_id: str) -> ArtifactRef:
    return db.artifacts_for(packet_id)["packet.json"]


def read_packet(db: Database, store: ArtifactStore, packet_id: str) -> dict[str, Any]:
    value = store.read_json(packet_ref(db, packet_id))
    if not isinstance(value, dict):
        raise ValueError(f"packet {packet_id} is not a JSON object")
    return value


def selected_attempt(records: list[Record]) -> Record | None:
    """Persisted selection policy for evaluation calls: the first completed, valid attempt."""
    return next((r for r in sorted(records, key=lambda r: r.payload["attempt_no"])
                 if r.payload.get("status") == "completed"), None)
