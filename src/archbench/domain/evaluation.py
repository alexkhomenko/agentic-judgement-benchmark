"""Private evaluation material and policy. Never sent to candidate models."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import Frozen


class Dimension(Frozen):
    id: str
    title: str
    weight: float = Field(gt=0)
    guidance: str = ""
    excluded_rounds: tuple[int, ...] = ()


class ReferenceProperties(Frozen):
    """Necessary guarantees and defensible mechanisms, not a prescribed technology stack."""

    common: tuple[str, ...] = ()
    by_round: dict[int, tuple[str, ...]] = Field(default_factory=dict)

    def for_round(self, number: int) -> tuple[str, ...]:
        return self.common + self.by_round.get(number, ())


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class Rubric(Frozen):
    rubric_id: str
    revision: str
    rating_scale: dict[int, str]
    dimensions: tuple[Dimension, ...]
    critical_failures: tuple[str, ...]
    reference_properties: ReferenceProperties

    @model_validator(mode="after")
    def _check(self) -> Rubric:
        if sorted(self.rating_scale) != [0, 1, 2, 3, 4]:
            raise ValueError("rating scale must define ratings 0 through 4")
        ids = [d.id for d in self.dimensions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate dimension ids")
        return self

    def applicable_dimensions(self, round_number: int) -> tuple[Dimension, ...]:
        return tuple(d for d in self.dimensions if round_number not in d.excluded_rounds)

    def critical_failure(self, text: str) -> str | None:
        """Resolve a cited critical failure to the rubric's exact wording (whitespace/case-insensitive)."""
        wanted = normalise_text(text)
        return next((cf for cf in self.critical_failures if normalise_text(cf) == wanted), None)


# ----- policy --------------------------------------------------------------------------------------


class PacketTransformation(Frozen):
    revision: int = Field(ge=1)
    identity_terms: tuple[str, ...] = ()
    mask: str = "[IDENTITY-MASKED]"


class JudgeSpec(Frozen):
    judge_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    configuration: str  # path to a model configuration file, frozen into the evaluation run
    prompt: str  # path to the judge instruction file, frozen into the evaluation run


class ClassifierSpec(Frozen):
    """`llm`: `prompt` is an instruction file and the model answers with JSON plus quoted evidence.
    `structured`: `prompt` is a typed questions file (see StructuredClassifierSpec) for a classification
    model that returns probabilities rather than text."""

    configuration: str
    prompt: str
    rounds: tuple[int, ...] = (1,)
    kind: Literal["llm", "structured"] = "llm"


class ExtractorSpec(Frozen):
    """Recommendation extraction for bias measurement. `prompt` is an instruction file and `catalog` a
    framework catalog (domain.recommendations.FrameworkCatalog); both are frozen into the run. Extraction
    is mechanical: it records what an answer recommends, never whether the recommendation is good."""

    configuration: str
    prompt: str
    catalog: str
    rounds: tuple[int, ...] = (1,)


class DisagreementPolicy(Frozen):
    """Judge disagreement that triggers human review. Draft thresholds; settle during calibration."""

    score_distance: float = Field(default=15.0, gt=0)
    dimension_distance: int = Field(default=2, ge=1, le=4)


class ReviewSelection(Frozen):
    fraction: float = Field(default=0.2, ge=0.2, le=1.0)
    seed: int
    extremes_per_round: int = Field(default=1, ge=0)
    disagreement: DisagreementPolicy = DisagreementPolicy()


class AdjudicationPolicy(Frozen):
    # human_precedence: effective human reviews decide ratings and critical-failure confirmation;
    # unselected responses with complete, agreeing judges use the per-dimension judge mean.
    rule: Literal["human_precedence"] = "human_precedence"


class AcceptablePolicy(Frozen):
    min_adjusted_score: float = Field(default=60.0, ge=0, le=100)
    status: Literal["draft", "calibrated"] = "draft"


class EvaluationPolicy(Frozen):
    policy_id: str
    revision: int = Field(ge=1)
    rubric: str
    packet_transformation: PacketTransformation
    judges: tuple[JudgeSpec, ...] = Field(min_length=2)
    judge_max_attempts: int = Field(default=2, ge=1)
    classifier: ClassifierSpec | None = None
    extractor: ExtractorSpec | None = None
    review_selection: ReviewSelection
    adjudication: AdjudicationPolicy = AdjudicationPolicy()
    acceptable: AcceptablePolicy = AcceptablePolicy()
    scoring_revision: int = Field(ge=1)

    @model_validator(mode="after")
    def _unique_judges(self) -> EvaluationPolicy:
        ids = [j.judge_id for j in self.judges]
        if len(set(ids)) != len(ids):
            raise ValueError("judge ids must be unique")
        return self

    @property
    def key(self) -> str:
        return f"{self.policy_id}@{self.revision}"


class RubricScore(Frozen):
    raw_score: float
    adjusted_score: float
    critical_failure_confirmed: bool
    applicable_dimensions: tuple[str, ...]


# ----- assessments ---------------------------------------------------------------------------------


class ModelOutput(BaseModel):
    """Parsed model output. Extra keys are ignored here; the raw output is preserved as evidence."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class RatedDimension(ModelOutput):
    rating: int = Field(ge=0, le=4, strict=True)
    evidence: tuple[str, ...] = Field(min_length=1)
    rationale: str = ""


class CriticalFailureCandidate(ModelOutput):
    failure: str
    evidence: tuple[str, ...] = Field(min_length=1)
    rationale: str = ""


class JudgeAssessment(ModelOutput):
    ratings: dict[str, RatedDimension]
    critical_failure_candidates: tuple[CriticalFailureCandidate, ...] = ()


class CriticalFailureDecision(Frozen):
    failure: str
    confirmed: bool
    evidence: tuple[str, ...] = ()
    rationale: str = ""

    @model_validator(mode="after")
    def _evidence_when_confirmed(self) -> CriticalFailureDecision:
        if self.confirmed and not self.evidence:
            raise ValueError(f"confirmed critical failure needs evidence: {self.failure}")
        return self


class HumanRatedDimension(Frozen):
    rating: int = Field(ge=0, le=4, strict=True)
    evidence: tuple[str, ...] = Field(min_length=1)
    rationale: str = ""


class HumanReviewSubmission(Frozen):
    """Structured human review. Reviewers decide every rubric critical failure explicitly."""

    packet_id: str
    reviewer_id: str = Field(min_length=1)
    ratings: dict[str, HumanRatedDimension]
    critical_failures: tuple[CriticalFailureDecision, ...]
    notes: str = ""


class ArchitectureCategory(StrEnum):
    INDEPENDENT_JOBS = "independent_jobs"
    SHARED_INFRASTRUCTURE = "shared_infrastructure"
    AGENT_FRAMEWORK = "agent_framework"
    WORKFLOW_ENGINE = "workflow_engine"
    HYBRID = "hybrid"
    OTHER = "other"


class ArchitectureClassificationOutput(ModelOutput):
    """Separate from quality: choosing a category never determines a score.

    LLM classifiers must cite evidence (enforced by their parser). Structured classifiers return
    probabilities instead, recorded in `category_probabilities` and `characteristic_probabilities`."""

    category: ArchitectureCategory
    evidence: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0, le=1)
    category_probabilities: dict[str, float] | None = None
    characteristic_probabilities: dict[str, float] = Field(default_factory=dict)
    independently_deployed_services: int | None = Field(default=None, ge=0)
    storage_technologies: tuple[str, ...] = ()
    orchestration_dependencies: tuple[str, ...] = ()
    external_integrations: tuple[str, ...] = ()
    autonomous_agents: bool | None = None
    human_approval_boundaries: bool | None = None


# ----- structured classifiers ----------------------------------------------------------------------

BOOLEAN_CHARACTERISTICS = ("autonomous_agents", "human_approval_boundaries")


class StructuredQuestion(Frozen):
    type: Literal["choice", "noul"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | None = None

    @model_validator(mode="after")
    def _criteria(self) -> StructuredQuestion:
        if self.type == "choice" and not self.criteria:
            raise ValueError("a choice question needs criteria")
        return self


class StructuredClassifierSpec(Frozen):
    """Typed questions for a classification model. The answer to `category_question` must be one of
    the architecture categories; yes/no questions may be named after boolean characteristics."""

    format: Literal["typesafe-systemone-questions-v1"]
    state_fields: tuple[Literal["task_conversation", "candidate_answer"], ...] = (
        "task_conversation", "candidate_answer")
    category_question: str = "category"
    noul_threshold: float = Field(default=0.5, gt=0, lt=1)
    questions: dict[str, StructuredQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _maps_onto_classification(self) -> StructuredClassifierSpec:
        category = self.questions.get(self.category_question)
        if category is None or category.type != "choice" or category.criteria is None:
            raise ValueError(f"'{self.category_question}' must be a choice question")
        unknown = set(category.criteria) - {c.value for c in ArchitectureCategory}
        if unknown:
            raise ValueError(f"category options are not architecture categories: {sorted(unknown)}")
        for name, question in self.questions.items():
            if name == self.category_question:
                continue
            if question.type != "noul" or name not in BOOLEAN_CHARACTERISTICS:
                raise ValueError(f"question '{name}' must be a yes/no question named one of {BOOLEAN_CHARACTERISTICS}")
        if not self.state_fields:
            raise ValueError("state_fields must not be empty")
        return self
