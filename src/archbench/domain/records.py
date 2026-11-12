"""Append-only derived records with explicit lineage.

Qualification, plans, evaluation runs, validations, packets, judge evaluations, human reviews,
adjudications, classifications, and report revisions all share this envelope.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from .common import Frozen, content_hash, new_id, utc_now


class RecordKind(StrEnum):
    QUALIFICATION = "qualification"
    EXPERIMENT_PLAN = "experiment_plan"
    EVALUATION_RUN = "evaluation_run"
    VALIDATION = "validation"
    BLINDED_PACKET = "blinded_packet"
    JUDGE_EVALUATION = "judge_evaluation"
    REVIEW_SELECTION = "review_selection"
    HUMAN_REVIEW = "human_review"
    ADJUDICATION = "adjudication"
    ARCHITECTURE_CLASSIFICATION = "architecture_classification"
    RECOMMENDATION_EXTRACTION = "recommendation_extraction"
    REPORT_REVISION = "report_revision"


class Record(Frozen):
    record_id: str
    kind: RecordKind
    experiment_id: str | None
    subject_id: str | None
    inputs: dict[str, str]  # exact input IDs -> content hashes
    policy_revision: str | None
    code_revision: str | None
    payload: dict[str, Any]
    payload_sha256: str
    supersedes: str | None = None
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        kind: RecordKind,
        payload: dict[str, Any],
        experiment_id: str | None = None,
        subject_id: str | None = None,
        inputs: dict[str, str] | None = None,
        policy_revision: str | None = None,
        code_revision: str | None = None,
        supersedes: str | None = None,
        record_id: str | None = None,
        prefix: str = "rec",
    ) -> Record:
        return cls(
            record_id=record_id or new_id(prefix),
            kind=kind,
            experiment_id=experiment_id,
            subject_id=subject_id,
            inputs=inputs or {},
            policy_revision=policy_revision,
            code_revision=code_revision,
            payload=payload,
            payload_sha256=content_hash(payload),
            supersedes=supersedes,
            created_at=utc_now(),
        )
