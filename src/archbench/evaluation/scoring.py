"""Rubric score computed in code.

    raw_score      = 100 * sum(applicable_weight * rating) / (4 * sum(applicable_weight))
    adjusted_score = min(raw_score, 40) if critical_failure_confirmed else raw_score
"""

from __future__ import annotations

from collections.abc import Mapping

from ..domain.evaluation import Rubric, RubricScore

CRITICAL_FAILURE_CAP = 40.0


def compute_score(
    rubric: Rubric,
    round_number: int,
    ratings: Mapping[str, float],
    *,
    critical_failure_confirmed: bool,
) -> RubricScore:
    applicable = rubric.applicable_dimensions(round_number)
    applicable_ids = {d.id for d in applicable}
    missing = applicable_ids - ratings.keys()
    if missing:
        raise ValueError(f"missing ratings for applicable dimensions: {sorted(missing)}")
    extra = ratings.keys() - applicable_ids
    if extra:
        raise ValueError(f"ratings supplied for dimensions not applicable to round {round_number}: {sorted(extra)}")
    for dim, rating in ratings.items():
        # Individual judge and human ratings are integers (enforced by their schemas); adjudicated ratings
        # may be means of several integer ratings.
        if not isinstance(rating, int | float) or isinstance(rating, bool) or not 0 <= rating <= 4:
            raise ValueError(f"rating for {dim} must be a number 0-4, got {rating!r}")

    total_weight = sum(d.weight for d in applicable)
    raw = 100.0 * sum(d.weight * ratings[d.id] for d in applicable) / (4.0 * total_weight)
    adjusted = min(raw, CRITICAL_FAILURE_CAP) if critical_failure_confirmed else raw
    return RubricScore(
        raw_score=round(raw, 4),
        adjusted_score=round(adjusted, 4),
        critical_failure_confirmed=critical_failure_confirmed,
        applicable_dimensions=tuple(d.id for d in applicable),
    )
