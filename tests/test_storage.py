"""Append-only storage, artifact immutability, redaction, and reconciliation."""

from __future__ import annotations

import os
import sqlite3
import stat

import pytest

from archbench.adapters.redaction import REDACTED, redact, redact_text
from archbench.domain.records import Record, RecordKind
from archbench.execution.controller import Controller
from archbench.storage.artifacts import ArtifactExistsError, ArtifactIntegrityError
from archbench.storage.reconcile import reconcile

from .conftest import Env


def test_records_are_append_only(env: Env) -> None:
    env.db.insert_record(Record.create(kind=RecordKind.VALIDATION, payload={"a": 1}, record_id="r1"))
    conn = env.db.connection
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE records SET payload_json = '{}' WHERE record_id = 'r1'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM records WHERE record_id = 'r1'")
    with pytest.raises(sqlite3.IntegrityError):
        env.db.insert_record(Record.create(kind=RecordKind.VALIDATION, payload={"a": 2}, record_id="r1"))


def test_published_rows_cannot_change(env: Env) -> None:
    plan = env.plan(env.experiment("exp-rows", [env.config("alpha")], repetitions=1))
    Controller(env.db, env.store, env.registry).run(plan)
    conn = env.db.connection
    for sql in (
        "UPDATE responses SET truncated = 1",
        "DELETE FROM responses",
        "UPDATE artifacts SET sha256 = 'x'",
        "DELETE FROM artifacts",
        "UPDATE attempts SET status = 'failed'",
        "DELETE FROM attempts",
        "DELETE FROM tasks",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(sql)


def test_tampered_record_payload_detected(env: Env) -> None:
    env.db.insert_record(Record.create(kind=RecordKind.VALIDATION, payload={"a": 1}, record_id="r1"))
    conn = env.db.connection
    conn.execute("DROP TRIGGER records_no_update")
    conn.execute("UPDATE records SET payload_json = '{\"a\": 2}' WHERE record_id = 'r1'")
    from archbench.storage.database import RecordIntegrityError

    with pytest.raises(RecordIntegrityError):
        env.db.get_record("r1")


def test_artifact_paths_never_overwritten(env: Env) -> None:
    ref = env.store.write("exp/a.txt", "first")
    with pytest.raises(ArtifactExistsError):
        env.store.write("exp/a.txt", "second")
    assert env.store.read_verified(ref) == b"first"
    mode = env.store.path("exp/a.txt").stat().st_mode
    assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    with pytest.raises(ValueError, match="escapes"):
        env.store.write("../outside.txt", "x")


def test_tampered_artifact_detected_on_read_and_reconcile(env: Env) -> None:
    plan = env.plan(env.experiment("exp-tamper", [env.config("alpha")], repetitions=1))
    Controller(env.db, env.store, env.registry).run(plan)
    (response, *_) = env.db.list_responses(plan.experiment_id)
    ref = env.db.artifacts_for(response.attempt_id)["final-answer.txt"]
    path = env.store.path(ref.relpath)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    path.write_text("rewritten")
    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        env.store.read_verified(ref)
    (env.store.path(plan.experiment_id) / "stray.txt").write_text("unregistered")
    report = reconcile(env.db, env.store, plan.experiment_id)
    assert report.hash_mismatches == [ref.relpath]
    assert report.unregistered_files == [f"{plan.experiment_id}/stray.txt"]
    assert not report.integrity_ok


def test_redaction_of_keys_and_text() -> None:
    value = {
        "headers": {"Authorization": "Bearer abc.def", "x-api-key": "k", "content-type": "json"},
        "body": {"max_tokens": 10, "note": "used sk-abcdefghijkl and api_key=hunter2"},
    }
    out = redact(value)
    assert out["headers"]["Authorization"] == REDACTED
    assert out["headers"]["x-api-key"] == REDACTED
    assert out["headers"]["content-type"] == "json"
    assert out["body"]["max_tokens"] == 10
    assert "sk-abcdefghijkl" not in out["body"]["note"] and "hunter2" not in out["body"]["note"]
    assert redact_text("authorization: Bearer xyz") .count("xyz") == 0


def test_no_secret_reaches_results_directory(env: Env) -> None:
    plan = env.plan(env.experiment("exp-secret", [env.config("alpha")], repetitions=1))
    Controller(env.db, env.store, env.registry).run(plan)
    for path in env.store.path(plan.experiment_id).rglob("*"):
        if path.is_file():
            assert b"fake-not-a-secret" not in path.read_bytes(), path
