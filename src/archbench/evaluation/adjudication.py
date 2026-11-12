"""Adjudication: a separate record per response that names the exact judge and human records it used.

Original judge evaluations and human reviews are never modified. Rule `human_precedence`:
- a response with effective human reviews takes the per-dimension mean of those reviews, and a critical
  failure is confirmed when any reviewer confirms it;
- a response selected for review but not yet reviewed stays pending (no record);
- an unselected response, whose judges are complete and agree with no critical-failure candidate, takes
  the per-dimension mean of the selected judge evaluations.
When the inputs change (for example a corrected human review), a new adjudication supersedes the old.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from pydantic import Field

from ..domain.common import Frozen, new_id
from ..domain.evaluation import EvaluationPolicy, Rubric
from ..domain.records import Record, RecordKind
from ..storage.database import Database
from .records import EVALUATION_CODE_REVISION, effective, run_records, selected_attempt
from .review import selection_record_id
from .scoring import compute_score


class AdjudicationProgress(Frozen):
    evaluation_run_id: str
    added: int
    superseded: int
    pending_human_review: list[str] = Field(default_factory=list)
    adjudicated: int = 0


def _mean_ratings(ratings: list[dict[str, int]]) -> dict[str, float]:
    return {d: round(statistics.fmean(r[d] for r in ratings), 4) for d in sorted(ratings[0])}


def adjudicate(db: Database, run: Record, policy: EvaluationPolicy, rubric: Rubric) -> AdjudicationProgress:
    experiment_id = str(run.experiment_id)
    selection = db.get_record(selection_record_id(run.record_id))
    if selection is None:
        raise RuntimeError("judge evaluation is incomplete; no review selection exists yet")
    selected = {i["response_id"]: i for i in selection.payload["items"]}
    packets = {r.subject_id: r for r in run_records(db, RecordKind.BLINDED_PACKET, experiment_id, run.record_id)}
    judges: dict[str, list[Record]] = defaultdict(list)
    for r in run_records(db, RecordKind.JUDGE_EVALUATION, experiment_id, run.record_id):
        judges[str(r.subject_id)].append(r)
    reviews: dict[str, list[Record]] = defaultdict(list)
    for r in effective(run_records(db, RecordKind.HUMAN_REVIEW, experiment_id, run.record_id)):
        reviews[str(r.subject_id)].append(r)
    existing = {
        str(r.subject_id): r
        for r in effective(run_records(db, RecordKind.ADJUDICATION, experiment_id, run.record_id))
    }

    added = superseded = 0
    pending: list[str] = []
    for response_id, packet in sorted(packets.items(), key=lambda kv: str(kv[0])):
        rid = str(response_id)
        round_number = int(packet.payload["round"])
        chosen = {
            j.judge_id: a
            for j in policy.judges
            if (a := selected_attempt([r for r in judges[rid] if r.payload["judge_id"] == j.judge_id])) is not None
        }
        human = sorted(reviews.get(rid, []), key=lambda r: r.record_id)
        if rid in selected and not human:
            pending.append(rid)
            continue
        if human:
            basis = "human_review"
            ratings = _mean_ratings([
                {k: v["rating"] for k, v in h.payload["submission"]["ratings"].items()} for h in human
            ])
            confirmed = sorted({f for h in human for f in h.payload["confirmed_critical_failures"]})
        else:
            if len(chosen) < len(policy.judges):  # cannot happen for unselected responses; guard anyway
                pending.append(rid)
                continue
            basis = "judge_mean"
            ratings = _mean_ratings([
                {k: v["rating"] for k, v in a.payload["output"]["ratings"].items()} for a in chosen.values()
            ])
            confirmed = []
        score = compute_score(rubric, round_number, ratings, critical_failure_confirmed=bool(confirmed))
        judge_scores = selection.payload["judge_raw_scores"].get(rid, {})
        disagreements: list[dict[str, Any]] = [
            {"kind": reason} for reason in selected.get(rid, {}).get("reasons", [])
            if reason in ("score_disagreement", "dimension_disagreement", "critical_failure_candidate",
                          "insufficient_judge_evaluations")
        ]
        for h in human:
            for judge_id, judge_score in judge_scores.items():
                disagreements.append({
                    "kind": "judge_vs_human", "judge_id": judge_id, "reviewer_id": h.payload["reviewer_id"],
                    "distance": round(abs(judge_score - h.payload["score"]["raw_score"]), 4),
                })
        inputs = {
            selection.record_id: selection.payload_sha256,
            **{a.record_id: a.payload_sha256 for a in chosen.values()},
            **{h.record_id: h.payload_sha256 for h in human},
        }
        prior = existing.get(rid)
        if prior is not None and prior.inputs == inputs:
            continue
        db.insert_record(Record.create(
            kind=RecordKind.ADJUDICATION, record_id=new_id("adj"), experiment_id=experiment_id, subject_id=rid,
            inputs=inputs, policy_revision=f"{policy.key}/adjudication:{policy.adjudication.rule}",
            code_revision=EVALUATION_CODE_REVISION, supersedes=prior.record_id if prior else None,
            payload={
                "evaluation_run_id": run.record_id,
                "packet_id": packet.record_id,
                "round": round_number,
                "rule": policy.adjudication.rule,
                "basis": basis,
                "judge_evaluation_ids": {j: a.record_id for j, a in sorted(chosen.items())},
                "human_review_ids": [h.record_id for h in human],
                "judge_raw_scores": judge_scores,
                "final_ratings": ratings,
                "confirmed_critical_failures": confirmed,
                "disagreements": disagreements,
                "score": score.model_dump(mode="json"),
            },
        ))
        added += 1
        superseded += prior is not None
    total = len(effective(run_records(db, RecordKind.ADJUDICATION, experiment_id, run.record_id)))
    return AdjudicationProgress(
        evaluation_run_id=run.record_id, added=added, superseded=superseded, pending_human_review=pending,
        adjudicated=total,
    )
