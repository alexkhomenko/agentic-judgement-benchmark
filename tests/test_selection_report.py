"""Full v2 report preserves evidence, human decisions, failures and cost without new model calls."""
from __future__ import annotations

import pytest

from archbench.reporting.selection_report import collect, publish, render_html
from archbench.storage.artifacts import ArtifactIntegrityError

from .conftest import Env
from .test_selection_review import app_for, decision


def test_full_report_is_reproducible_and_tracks_human_completion(env: Env) -> None:
    app = app_for(env)
    before = publish(env.db, env.store, str(app.run.experiment_id), app.run.record_id)
    assert not before.payload["evaluation_complete"]
    app.submit(decision(app))
    record = publish(env.db, env.store, str(app.run.experiment_id), app.run.record_id)
    assert record.record_id != before.record_id and record.payload["evaluation_complete"]
    refs = env.db.artifacts_for(record.record_id)
    assert set(refs) == {"manifest.json", "report.json", "report.html", "report.md"}
    data = env.store.read_json(refs["report.json"])
    assert data["responses"][0]["human_review_state"] == "approved"
    assert data["cost"]["candidate_calls"] == 1 and data["cost"]["evaluator_calls"] == 3
    assert data["cost"]["unknown_cost_calls"] == 0
    assert data["qualifications"] and data["artifact_hashes"]
    human_id = data["responses"][0]["human_reviews"][0]["record_id"]
    assert human_id in record.inputs
    page = env.store.read_verified(refs["report.html"]).decode()
    assert "COMPLETE" in page and "Full candidate answer" in page and "human · proceed" in page
    assert "statistical significance" in page
    assert "CALIBRATION PENDING" in page
    assert publish(env.db, env.store, str(app.run.experiment_id), app.run.record_id).record_id == record.record_id


def test_report_accounts_for_invalid_audited_calls_and_preserves_attention(env: Env) -> None:
    app = app_for(env, malformed=1, probability=.6)
    data = collect(env.db, env.store, str(app.run.experiment_id), app.run.record_id)
    assert data["cost"]["evaluator_calls"] == 4
    assert any(c["status"] == "invalid_output" and c["problems"] for c in data["calls"])
    page = render_html(data, "test-report")
    assert "uncertain rank" in page and "invalid_output" in page
    assert "INCOMPLETE" in page and "Human validation pending" in page
    assert "Classifier probabilities are uncalibrated" in page


def test_report_escapes_untrusted_answer_and_refuses_wrong_run_or_corrupt_evidence(env: Env) -> None:
    app = app_for(env)
    exp = str(app.run.experiment_id)
    with pytest.raises(ValueError, match="belong"):
        collect(env.db, env.store, "another-experiment", app.run.record_id)
    data = collect(env.db, env.store, exp, app.run.record_id)
    data["responses"][0]["answer"] = "<script>alert('unsafe')</script>"
    page = render_html(data, "test-report")
    assert "<script>" not in page and "&lt;script&gt;" in page
    call = data["calls"][-1]
    ref = env.db.artifacts_for(call["id"])["metrics.json"]
    path = env.store.path(ref.relpath)
    path.chmod(0o600)
    path.write_text("{}")
    with pytest.raises(ArtifactIntegrityError):
        collect(env.db, env.store, exp, app.run.record_id)
