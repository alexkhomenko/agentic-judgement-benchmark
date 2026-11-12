"""Startup reconciliation. Database and filesystem writes are not one atomic transaction."""

from __future__ import annotations

from pydantic import Field

from ..domain.common import Frozen
from ..domain.execution import AttemptStatus, CostIncurred, FailureKind, TaskStatus
from .artifacts import LOCK_NAME, TEMP_MARKER, ArtifactIntegrityError, ArtifactStore
from .database import Database


class ReconcileReport(Frozen):
    interrupted_attempts: list[str] = Field(default_factory=list)
    reset_tasks: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    hash_mismatches: list[str] = Field(default_factory=list)
    unregistered_files: list[str] = Field(default_factory=list)
    stale_temp_files: list[str] = Field(default_factory=list)

    @property
    def integrity_ok(self) -> bool:
        return not (self.missing_artifacts or self.hash_mismatches or self.unregistered_files)


def reconcile(db: Database, store: ArtifactStore, experiment_id: str) -> ReconcileReport:
    interrupted: list[str] = []
    for attempt in db.list_attempts(experiment_id=experiment_id, status=AttemptStatus.RUNNING):
        prefix = f"{experiment_id}/artifacts/{attempt.attempt_id}"
        directory = store.path(prefix)
        already = set(db.artifacts_for(attempt.attempt_id))
        refs = []
        if directory.is_dir():
            for file in sorted(directory.iterdir()):
                if file.is_file() and not file.name.startswith(TEMP_MARKER) and file.name not in already:
                    refs.append(store.adopt(f"{prefix}/{file.name}", name=file.name))
        with db.transaction():
            db.register_artifacts(attempt.attempt_id, refs)
            # A request may have reached the provider before the crash: cost is unknown, not zero.
            db.finish_attempt(
                attempt.attempt_id,
                status=AttemptStatus.INTERRUPTED,
                failure_kind=FailureKind.INTERRUPTED,
                cost_incurred=CostIncurred.UNKNOWN,
                error_summary="controller exited before the attempt completed",
            )
        interrupted.append(attempt.attempt_id)

    reset: list[str] = []
    for task in db.list_tasks(experiment_id):
        if task.status is TaskStatus.RUNNING:
            db.set_task_status(task.task_id, TaskStatus.PENDING, "reset after interrupted run")
            reset.append(task.task_id)

    missing: list[str] = []
    mismatched: list[str] = []
    registered = db.artifacts_under(f"{experiment_id}/")
    for ref in registered:
        try:
            store.read_verified(ref)
        except ArtifactIntegrityError as exc:
            (missing if "missing" in str(exc) else mismatched).append(ref.relpath)

    registered_paths = {ref.relpath for ref in registered}
    unregistered: list[str] = []
    stale: list[str] = []
    root = store.path(experiment_id)
    if root.is_dir():
        for file in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = file.relative_to(store.root).as_posix()
            if file.name == LOCK_NAME and file.parent == root:
                continue
            if file.name.startswith(TEMP_MARKER):
                stale.append(rel)
            elif rel not in registered_paths:
                unregistered.append(rel)

    return ReconcileReport(
        interrupted_attempts=interrupted,
        reset_tasks=reset,
        missing_artifacts=missing,
        hash_mismatches=mismatched,
        unregistered_files=unregistered,
        stale_temp_files=stale,
    )

