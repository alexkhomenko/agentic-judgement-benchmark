"""SQLite repository.

Published results (records, terminal attempts, responses, artifact registrations) are append-only and
enforced by triggers. Operational task state is mutable and kept apart from evidence. The controller is
the only writer.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..domain.common import content_hash, utc_now
from ..domain.execution import (
    ArtifactRef,
    AttemptStatus,
    CandidateResponse,
    CostIncurred,
    FailureKind,
    InvocationAttempt,
    ResponseTask,
    TaskStatus,
)
from ..domain.records import Record, RecordKind
from ..secrets import GUARD

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id       TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    experiment_id   TEXT,
    subject_id      TEXT,
    inputs_json     TEXT NOT NULL,
    policy_revision TEXT,
    code_revision   TEXT,
    payload_json    TEXT NOT NULL,
    payload_sha256  TEXT NOT NULL,
    supersedes      TEXT REFERENCES records(record_id),
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_lookup ON records(kind, experiment_id, subject_id);

CREATE TABLE IF NOT EXISTS tasks (
    task_id           TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL,
    configuration_key TEXT NOT NULL,
    repetition        INTEGER NOT NULL,
    round             INTEGER NOT NULL,
    parent_task_id    TEXT REFERENCES tasks(task_id),
    status            TEXT NOT NULL,
    status_reason     TEXT,
    updated_at        TEXT NOT NULL,
    UNIQUE (experiment_id, configuration_key, repetition, round)
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id    TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL REFERENCES tasks(task_id),
    attempt_no    INTEGER NOT NULL,
    status        TEXT NOT NULL,
    failure_kind  TEXT,
    cost_incurred TEXT NOT NULL,
    error_summary TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    UNIQUE (task_id, attempt_no)
);

CREATE TABLE IF NOT EXISTS responses (
    response_id         TEXT PRIMARY KEY,
    task_id             TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
    attempt_id          TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
    experiment_id       TEXT NOT NULL,
    configuration_key   TEXT NOT NULL,
    repetition          INTEGER NOT NULL,
    round               INTEGER NOT NULL,
    final_answer_sha256 TEXT NOT NULL,
    truncated           INTEGER NOT NULL,
    finish_reason       TEXT,
    selection_policy    TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    owner_id       TEXT NOT NULL,
    name           TEXT NOT NULL,
    relpath        TEXT NOT NULL UNIQUE,
    sha256         TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (owner_id, name)
);

CREATE TRIGGER IF NOT EXISTS records_no_update BEFORE UPDATE ON records
BEGIN SELECT RAISE(ABORT, 'records are append-only'); END;
CREATE TRIGGER IF NOT EXISTS records_no_delete BEFORE DELETE ON records
BEGIN SELECT RAISE(ABORT, 'records are append-only'); END;

CREATE TRIGGER IF NOT EXISTS responses_no_update BEFORE UPDATE ON responses
BEGIN SELECT RAISE(ABORT, 'responses are append-only'); END;
CREATE TRIGGER IF NOT EXISTS responses_no_delete BEFORE DELETE ON responses
BEGIN SELECT RAISE(ABORT, 'responses are append-only'); END;

CREATE TRIGGER IF NOT EXISTS artifacts_no_update BEFORE UPDATE ON artifacts
BEGIN SELECT RAISE(ABORT, 'artifact registrations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_no_delete BEFORE DELETE ON artifacts
BEGIN SELECT RAISE(ABORT, 'artifact registrations are append-only'); END;

CREATE TRIGGER IF NOT EXISTS attempts_published_immutable BEFORE UPDATE ON attempts
WHEN OLD.status <> 'running'
  OR NEW.attempt_id IS NOT OLD.attempt_id
  OR NEW.task_id IS NOT OLD.task_id
  OR NEW.attempt_no IS NOT OLD.attempt_no
  OR NEW.started_at IS NOT OLD.started_at
BEGIN SELECT RAISE(ABORT, 'published attempts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS attempts_no_delete BEFORE DELETE ON attempts
BEGIN SELECT RAISE(ABORT, 'attempts are append-only'); END;

CREATE TRIGGER IF NOT EXISTS tasks_no_delete BEFORE DELETE ON tasks
BEGIN SELECT RAISE(ABORT, 'tasks cannot be deleted'); END;
"""


class RecordIntegrityError(RuntimeError):
    pass


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._depth = 0

    def close(self) -> None:
        self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")
        finally:
            self._depth = 0

    # ----- records ---------------------------------------------------------------------------------

    def insert_record(self, record: Record) -> None:
        GUARD.check(json.dumps([record.payload, record.inputs], sort_keys=True), f"record {record.record_id}")
        with self.transaction():
            self._conn.execute(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.record_id,
                    record.kind.value,
                    record.experiment_id,
                    record.subject_id,
                    json.dumps(record.inputs, sort_keys=True),
                    record.policy_revision,
                    record.code_revision,
                    json.dumps(record.payload, sort_keys=True),
                    record.payload_sha256,
                    record.supersedes,
                    record.created_at,
                ),
            )

    def get_record(self, record_id: str) -> Record | None:
        row = self._conn.execute("SELECT * FROM records WHERE record_id = ?", (record_id,)).fetchone()
        return _record(row) if row else None

    def list_records(
        self,
        *,
        kind: RecordKind | None = None,
        experiment_id: str | None = None,
        subject_id: str | None = None,
    ) -> list[Record]:
        clauses, params = _filters(kind=kind.value if kind else None, experiment_id=experiment_id, subject_id=subject_id)
        rows = self._conn.execute(f"SELECT * FROM records{clauses} ORDER BY created_at, rowid", params).fetchall()
        return [_record(r) for r in rows]

    # ----- tasks (operational) ---------------------------------------------------------------------

    def ensure_tasks(self, tasks: Iterable[ResponseTask]) -> None:
        now = utc_now()
        with self.transaction():
            for t in tasks:
                self._conn.execute(
                    "INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        t.task_id,
                        t.experiment_id,
                        t.configuration_key,
                        t.repetition,
                        t.round,
                        t.parent_task_id,
                        t.status.value,
                        t.status_reason,
                        now,
                    ),
                )

    def get_task(self, task_id: str) -> ResponseTask:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown task {task_id}")
        return _task(row)

    def list_tasks(self, experiment_id: str) -> list[ResponseTask]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE experiment_id = ? ORDER BY configuration_key, repetition, round",
            (experiment_id,),
        ).fetchall()
        return [_task(r) for r in rows]

    def set_task_status(self, task_id: str, status: TaskStatus, reason: str | None = None) -> None:
        with self.transaction():
            self._conn.execute(
                "UPDATE tasks SET status = ?, status_reason = ?, updated_at = ? WHERE task_id = ?",
                (status.value, reason, utc_now(), task_id),
            )

    # ----- attempts --------------------------------------------------------------------------------

    def start_attempt(self, attempt: InvocationAttempt) -> None:
        if attempt.status is not AttemptStatus.RUNNING:
            raise ValueError("attempts start in the running state")
        with self.transaction():
            self._conn.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    attempt.attempt_id,
                    attempt.task_id,
                    attempt.attempt_no,
                    attempt.status.value,
                    None,
                    attempt.cost_incurred.value,
                    None,
                    attempt.started_at,
                    None,
                ),
            )

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        status: AttemptStatus,
        failure_kind: FailureKind | None,
        cost_incurred: CostIncurred,
        error_summary: str | None,
    ) -> None:
        if status is AttemptStatus.RUNNING:
            raise ValueError("finish_attempt requires a terminal status")
        with self.transaction():
            cur = self._conn.execute(
                "UPDATE attempts SET status = ?, failure_kind = ?, cost_incurred = ?, error_summary = ?, "
                "finished_at = ? WHERE attempt_id = ?",
                (
                    status.value,
                    failure_kind.value if failure_kind else None,
                    cost_incurred.value,
                    error_summary,
                    utc_now(),
                    attempt_id,
                ),
            )
            if cur.rowcount != 1:
                raise KeyError(f"unknown attempt {attempt_id}")

    def get_attempt(self, attempt_id: str) -> InvocationAttempt:
        row = self._conn.execute("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown attempt {attempt_id}")
        return _attempt(row)

    def list_attempts(
        self,
        *,
        task_id: str | None = None,
        experiment_id: str | None = None,
        status: AttemptStatus | None = None,
    ) -> list[InvocationAttempt]:
        sql = "SELECT a.* FROM attempts a JOIN tasks t ON t.task_id = a.task_id"
        clauses: list[str] = []
        params: list[Any] = []
        if task_id is not None:
            clauses.append("a.task_id = ?")
            params.append(task_id)
        if experiment_id is not None:
            clauses.append("t.experiment_id = ?")
            params.append(experiment_id)
        if status is not None:
            clauses.append("a.status = ?")
            params.append(status.value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY a.task_id, a.attempt_no"
        return [_attempt(r) for r in self._conn.execute(sql, params).fetchall()]

    # ----- responses -------------------------------------------------------------------------------

    def insert_response(self, response: CandidateResponse) -> None:
        with self.transaction():
            self._conn.execute(
                "INSERT INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    response.response_id,
                    response.task_id,
                    response.attempt_id,
                    response.experiment_id,
                    response.configuration_key,
                    response.repetition,
                    response.round,
                    response.final_answer_sha256,
                    int(response.truncated),
                    response.finish_reason,
                    response.selection_policy,
                    response.created_at,
                ),
            )

    def response_for_task(self, task_id: str) -> CandidateResponse | None:
        row = self._conn.execute("SELECT * FROM responses WHERE task_id = ?", (task_id,)).fetchone()
        return _response(row) if row else None

    def get_response(self, response_id: str) -> CandidateResponse:
        row = self._conn.execute("SELECT * FROM responses WHERE response_id = ?", (response_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown response {response_id}")
        return _response(row)

    def list_responses(self, experiment_id: str) -> list[CandidateResponse]:
        rows = self._conn.execute(
            "SELECT * FROM responses WHERE experiment_id = ? ORDER BY configuration_key, repetition, round",
            (experiment_id,),
        ).fetchall()
        return [_response(r) for r in rows]

    # ----- artifacts -------------------------------------------------------------------------------

    def register_artifacts(self, owner_id: str, refs: Iterable[ArtifactRef]) -> None:
        now = utc_now()
        with self.transaction():
            for ref in refs:
                self._conn.execute(
                    "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)",
                    (owner_id, ref.name, ref.relpath, ref.sha256, ref.size_bytes, ref.schema_version, now),
                )

    def artifacts_for(self, owner_id: str) -> dict[str, ArtifactRef]:
        rows = self._conn.execute("SELECT * FROM artifacts WHERE owner_id = ? ORDER BY name", (owner_id,)).fetchall()
        return {r["name"]: _artifact(r) for r in rows}

    def artifacts_under(self, relpath_prefix: str) -> list[ArtifactRef]:
        rows = self._conn.execute(
            "SELECT * FROM artifacts WHERE relpath LIKE ? ESCAPE '\\' ORDER BY relpath",
            (_like_prefix(relpath_prefix),),
        ).fetchall()
        return [_artifact(r) for r in rows]


# ----- row mapping -------------------------------------------------------------------------------


def _filters(**values: Any) -> tuple[str, list[Any]]:
    clauses = [f"{k} = ?" for k, v in values.items() if v is not None]
    params = [v for v in values.values() if v is not None]
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


def _like_prefix(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


def _record(row: sqlite3.Row) -> Record:
    payload = json.loads(row["payload_json"])
    if content_hash(payload) != row["payload_sha256"]:
        raise RecordIntegrityError(f"record {row['record_id']} payload does not match its hash")
    return Record(
        record_id=row["record_id"],
        kind=RecordKind(row["kind"]),
        experiment_id=row["experiment_id"],
        subject_id=row["subject_id"],
        inputs=json.loads(row["inputs_json"]),
        policy_revision=row["policy_revision"],
        code_revision=row["code_revision"],
        payload=payload,
        payload_sha256=row["payload_sha256"],
        supersedes=row["supersedes"],
        created_at=row["created_at"],
    )


def _task(row: sqlite3.Row) -> ResponseTask:
    return ResponseTask(
        task_id=row["task_id"],
        experiment_id=row["experiment_id"],
        configuration_key=row["configuration_key"],
        repetition=row["repetition"],
        round=row["round"],
        parent_task_id=row["parent_task_id"],
        status=TaskStatus(row["status"]),
        status_reason=row["status_reason"],
    )


def _attempt(row: sqlite3.Row) -> InvocationAttempt:
    return InvocationAttempt(
        attempt_id=row["attempt_id"],
        task_id=row["task_id"],
        attempt_no=row["attempt_no"],
        status=AttemptStatus(row["status"]),
        failure_kind=FailureKind(row["failure_kind"]) if row["failure_kind"] else None,
        cost_incurred=CostIncurred(row["cost_incurred"]),
        error_summary=row["error_summary"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _response(row: sqlite3.Row) -> CandidateResponse:
    return CandidateResponse(
        response_id=row["response_id"],
        task_id=row["task_id"],
        attempt_id=row["attempt_id"],
        experiment_id=row["experiment_id"],
        configuration_key=row["configuration_key"],
        repetition=row["repetition"],
        round=row["round"],
        final_answer_sha256=row["final_answer_sha256"],
        truncated=bool(row["truncated"]),
        finish_reason=row["finish_reason"],
        selection_policy=row["selection_policy"],
        created_at=row["created_at"],
    )


def _artifact(row: sqlite3.Row) -> ArtifactRef:
    return ArtifactRef(
        name=row["name"],
        relpath=row["relpath"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        schema_version=row["schema_version"],
    )
