"""Evaluation runs over frozen responses.

Starting a run freezes the exact response IDs/hashes and the full evaluation policy, including the
rubric, judge and classifier configurations, and their prompt bytes. Resuming appends missing work to
the same run without replacing completed results and never re-reads policy files from disk. A new
rubric or policy is a new run.

Automatic stages: validation -> blinded packets -> independent judges -> classification ->
recommendation extraction -> review selection. Human review import and adjudication are separate operations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from ..adapters.registry import AdapterRegistry
from ..domain.common import Frozen, content_hash, new_id
from ..domain.configuration import ModelConfiguration
from ..domain.evaluation import EvaluationPolicy, Rubric, StructuredClassifierSpec
from ..domain.plan import ExperimentPlan, FrozenFile
from ..domain.recommendations import FrameworkCatalog
from ..domain.records import Record, RecordKind
from ..domain.scenario import Message
from ..planning.compiler import load_plan, plan_record_id
from ..planning.provenance import dependency_versions
from ..planning.loader import freeze_file, load_configuration, load_policy, load_rubric
from ..storage.artifacts import ArtifactIntegrityError, ArtifactStore
from ..storage.database import Database
from .extraction import extraction_messages, extraction_packet, parse_extraction
from .invocations import EvaluationCall, Parser, next_attempt, recover_orphans, run_call
from .judging import (
    CLASSIFIER_TAG,
    JUDGE_TAG,
    classification_packet,
    evaluation_messages,
    parse_classification_output,
    parse_judge_output,
    parse_structured_classification,
    structured_messages,
)
from .packets import build_packet
from .records import EVALUATION_CODE_REVISION, effective, read_packet, run_records, selected_attempt
from .review import build_selection, publish_selection, selection_record_id
from .validation import validate_response


class EvaluationPreflightError(RuntimeError):
    pass


class StageCount(Frozen):
    done: int
    expected: int
    exhausted: int = 0


class EvaluationStatus(Frozen):
    evaluation_run_id: str
    responses: int
    validations: StageCount
    packets: StageCount
    judge_evaluations: StageCount
    classifications: StageCount
    extractions: StageCount
    review_selection: str | None
    human_reviews: StageCount
    adjudications: StageCount
    pending_stages: list[str] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.pending_stages


@dataclass(frozen=True)
class FrozenModel:
    configuration: ModelConfiguration
    prompt: str
    record_inputs: dict[str, str]


@dataclass(frozen=True)
class RunContext:
    run: Record
    plan: ExperimentPlan
    policy: EvaluationPolicy
    rubric: Rubric
    judges: dict[str, FrozenModel]
    classifier: FrozenModel | None
    classifier_rounds: tuple[int, ...]
    classifier_spec: StructuredClassifierSpec | None = None  # set for structured classifiers
    extractor: FrozenModel | None = None
    extractor_rounds: tuple[int, ...] = ()
    catalog: FrameworkCatalog | None = None

    @property
    def experiment_id(self) -> str:
        return self.plan.experiment_id

    @property
    def manifest(self) -> dict[str, Any]:
        return self.run.payload


def _freeze_model(path: str, prompt: str, base: Path) -> dict[str, Any]:
    config, config_file = load_configuration(base / path, base)
    prompt_file = freeze_file(base / prompt, base)
    return {
        "configuration": config.model_dump(mode="json"),
        "configuration_file": config_file.model_dump(mode="json"),
        "prompt_file": prompt_file.model_dump(mode="json"),
    }


def _thaw_model(entry: dict[str, Any]) -> FrozenModel:
    config_file = FrozenFile.model_validate(entry["configuration_file"])
    prompt_file = FrozenFile.model_validate(entry["prompt_file"])
    return FrozenModel(
        configuration=ModelConfiguration.model_validate(entry["configuration"]),
        prompt=prompt_file.content,
        record_inputs={config_file.path: config_file.sha256, prompt_file.path: prompt_file.sha256},
    )


def start_evaluation(
    db: Database,
    store: ArtifactStore,
    experiment_id: str,
    policy_path: Path,
    *,
    base: Path,
    registry: AdapterRegistry,
    code_revision: str | None = None,
) -> Record:
    plan = load_plan(db, experiment_id)
    policy, policy_file = load_policy(policy_path, base)
    rubric, rubric_file = load_rubric(base / policy.rubric, base)
    responses = db.list_responses(experiment_id)
    attempts = db.list_attempts(experiment_id=experiment_id)
    if any(a.status.value == "running" for a in attempts):
        raise RuntimeError("experiment has running attempts; finish or reconcile execution before evaluating")

    judges = {j.judge_id: _freeze_model(j.configuration, j.prompt, base) for j in policy.judges}
    classifier = (
        {**_freeze_model(policy.classifier.configuration, policy.classifier.prompt, base),
         "rounds": list(policy.classifier.rounds), "kind": policy.classifier.kind}
        if policy.classifier else None
    )
    extractor = _freeze_extractor(policy, base)
    problems = []
    for entry in [*judges.values(), *([classifier] if classifier else []), *([extractor] if extractor else [])]:
        adapter = registry.get(entry["configuration"]["adapter"])  # fail before freezing, not mid-run
        preflight = getattr(adapter, "preflight", None)
        if preflight is not None and (problem := preflight(ModelConfiguration.model_validate(entry["configuration"]))):
            problems.append(f"{entry['configuration']['configuration_id']}: {problem}")
    if problems:
        raise EvaluationPreflightError("; ".join(sorted(set(problems))))
    if classifier is not None:
        _structured_spec(classifier)  # a malformed questions file fails before the run is frozen
    providers = [j["configuration"]["provider"] for j in judges.values()]

    run_id = new_id("evr")
    manifest: dict[str, Any] = {
        "evaluation_run_id": run_id,
        "experiment_id": experiment_id,
        "plan_sha256": plan.plan_sha256,
        "policy": policy.model_dump(mode="json"),
        "policy_file": policy_file.model_dump(mode="json"),
        "rubric": rubric.model_dump(mode="json"),
        "rubric_file": rubric_file.model_dump(mode="json"),
        "judges": judges,
        "judge_providers_distinct": len(set(providers)) == len(providers),
        "classifier": classifier,
        "extractor": extractor,
        "scoring_revision": policy.scoring_revision,
        "evaluation_code_revision": EVALUATION_CODE_REVISION,
        "dependency_versions": dependency_versions(),
        "responses": [
            {
                "response_id": r.response_id,
                "task_id": r.task_id,
                "attempt_id": r.attempt_id,
                "configuration_key": r.configuration_key,
                "track": plan.binding(r.configuration_key).configuration.track.value,
                "provider": plan.binding(r.configuration_key).configuration.provider,
                "repetition": r.repetition,
                "round": r.round,
                "final_answer_sha256": r.final_answer_sha256,
                "truncated": r.truncated,
            }
            for r in responses
        ],
        "attempts": [
            {
                "attempt_id": a.attempt_id,
                "task_id": a.task_id,
                "attempt_no": a.attempt_no,
                "status": a.status.value,
                "failure_kind": a.failure_kind.value if a.failure_kind else None,
                "cost_incurred": a.cost_incurred.value,
            }
            for a in attempts
        ],
    }
    sink = store.sink(f"{experiment_id}/evaluations/{run_id}")
    sink.write_json("manifest.json", manifest)
    inputs = {plan_record_id(experiment_id): content_hash(plan.model_dump(mode="json"))}
    inputs.update({r.response_id: r.final_answer_sha256 for r in responses})
    record = Record.create(
        kind=RecordKind.EVALUATION_RUN,
        record_id=run_id,
        experiment_id=experiment_id,
        inputs=inputs,
        policy_revision=policy.key,
        code_revision=code_revision,
        payload=manifest,
    )
    with db.transaction():
        db.insert_record(record)
        db.register_artifacts(run_id, sink.refs)
    return record


def load_run(db: Database, evaluation_run_id: str) -> RunContext:
    run = db.get_record(evaluation_run_id)
    if run is None or run.kind is not RecordKind.EVALUATION_RUN:
        raise KeyError(f"unknown evaluation run {evaluation_run_id}")
    manifest = run.payload
    plan = load_plan(db, str(manifest["experiment_id"]))
    if plan.plan_sha256 != manifest["plan_sha256"]:
        raise RuntimeError("frozen plan no longer matches the evaluation run manifest")
    classifier = manifest.get("classifier")
    extractor = manifest.get("extractor")
    return RunContext(
        run=run,
        plan=plan,
        policy=EvaluationPolicy.model_validate(manifest["policy"]),
        rubric=Rubric.model_validate(manifest["rubric"]),
        judges={judge_id: _thaw_model(entry) for judge_id, entry in manifest["judges"].items()},
        classifier=_thaw_model(classifier) if classifier else None,
        classifier_rounds=tuple(classifier["rounds"]) if classifier else (),
        classifier_spec=_structured_spec(classifier),
        extractor=_thaw_model(extractor) if extractor else None,
        extractor_rounds=tuple(extractor["rounds"]) if extractor else (),
        catalog=_catalog(extractor),
    )


def _freeze_extractor(policy: EvaluationPolicy, base: Path) -> dict[str, Any] | None:
    if policy.extractor is None:
        return None
    catalog_file = freeze_file(base / policy.extractor.catalog, base)
    _catalog({"catalog_file": catalog_file.model_dump(mode="json")})  # a malformed catalog fails before freezing
    return {
        **_freeze_model(policy.extractor.configuration, policy.extractor.prompt, base),
        "catalog_file": catalog_file.model_dump(mode="json"),
        "rounds": list(policy.extractor.rounds),
    }


def _catalog(extractor: dict[str, Any] | None) -> FrameworkCatalog | None:
    if not extractor:
        return None
    frozen = FrozenFile.model_validate(extractor["catalog_file"])
    return FrameworkCatalog.model_validate(yaml.safe_load(frozen.content))


def _structured_spec(classifier: dict[str, Any] | None) -> StructuredClassifierSpec | None:
    if not classifier or classifier.get("kind", "llm") != "structured":
        return None
    prompt = FrozenFile.model_validate(classifier["prompt_file"])
    return StructuredClassifierSpec.model_validate(yaml.safe_load(prompt.content))


def resume_evaluation(
    db: Database, store: ArtifactStore, registry: AdapterRegistry, evaluation_run_id: str
) -> EvaluationStatus:
    ctx = load_run(db, evaluation_run_id)
    _validate_and_packetize(db, store, ctx)
    recover_orphans(db, store, ctx.experiment_id, evaluation_run_id, "judgements", RecordKind.JUDGE_EVALUATION)
    recover_orphans(db, store, ctx.experiment_id, evaluation_run_id, "classifications",
                    RecordKind.ARCHITECTURE_CLASSIFICATION)
    recover_orphans(db, store, ctx.experiment_id, evaluation_run_id, "extractions",
                    RecordKind.RECOMMENDATION_EXTRACTION)
    _judge(db, store, registry, ctx)
    _classify(db, store, registry, ctx)
    _extract(db, store, registry, ctx)
    _select(db, ctx)
    return evaluation_status(db, evaluation_run_id)


# ----- stages ----------------------------------------------------------------------------------------


def _validate_and_packetize(db: Database, store: ArtifactStore, ctx: RunContext) -> None:
    run, policy, rubric = ctx.run, ctx.policy, ctx.rubric
    done_validations = {r.subject_id for r in run_records(db, RecordKind.VALIDATION, ctx.experiment_id, run.record_id)}
    done_packets = {r.subject_id for r in run_records(db, RecordKind.BLINDED_PACKET, ctx.experiment_id, run.record_id)}
    for entry in ctx.manifest["responses"]:
        response = db.get_response(entry["response_id"])
        if response.final_answer_sha256 != entry["final_answer_sha256"]:
            raise RuntimeError(f"response {response.response_id} no longer matches the frozen manifest")
        artifacts = db.artifacts_for(response.attempt_id)
        answer = _read_optional(store, artifacts.get("final-answer.txt"))

        if response.response_id not in done_validations:
            payload = validate_response(response, answer, ctx.plan.scenario.round_spec(response.round))
            payload["evaluation_run_id"] = run.record_id
            db.insert_record(Record.create(
                kind=RecordKind.VALIDATION, prefix="val", experiment_id=ctx.experiment_id,
                subject_id=response.response_id,
                inputs={run.record_id: run.payload_sha256, response.response_id: response.final_answer_sha256},
                policy_revision=policy.key, code_revision=EVALUATION_CODE_REVISION, payload=payload,
            ))

        if response.response_id not in done_packets and answer is not None:
            messages_ref = artifacts["canonical-messages.json"]
            conversation = [Message.model_validate(m) for m in store.read_json(messages_ref)]
            packet_id = new_id("pkt")
            packet, masked = build_packet(
                packet_id=packet_id, round_number=response.round, conversation=conversation,
                final_answer=answer.decode("utf-8"), rubric=rubric, transformation=policy.packet_transformation,
            )
            sink = store.sink(f"{ctx.experiment_id}/evaluations/{run.record_id}/packets/{packet_id}")
            ref = sink.write_json("packet.json", packet)
            record = Record.create(
                kind=RecordKind.BLINDED_PACKET, record_id=packet_id, experiment_id=ctx.experiment_id,
                subject_id=response.response_id,
                inputs={
                    run.record_id: run.payload_sha256,
                    response.response_id: response.final_answer_sha256,
                    messages_ref.relpath: messages_ref.sha256,
                },
                policy_revision=f"{policy.key}/packet-transformation@{policy.packet_transformation.revision}",
                code_revision=EVALUATION_CODE_REVISION,
                payload={
                    "evaluation_run_id": run.record_id,
                    "round": response.round,
                    "packet_sha256": ref.sha256,
                    "packet_relpath": ref.relpath,
                    "identity_masking": masked,
                },
            )
            with db.transaction():
                db.insert_record(record)
                db.register_artifacts(packet_id, sink.refs)


def _packets(db: Database, ctx: RunContext) -> list[Record]:
    return run_records(db, RecordKind.BLINDED_PACKET, ctx.experiment_id, ctx.run.record_id)


def _judge(db: Database, store: ArtifactStore, registry: AdapterRegistry, ctx: RunContext) -> None:
    existing: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in run_records(db, RecordKind.JUDGE_EVALUATION, ctx.experiment_id, ctx.run.record_id):
        existing[(r.payload["packet_id"], r.payload["judge_id"])].append(r)
    for packet_record in _packets(db, ctx):
        packet = read_packet(db, store, packet_record.record_id)
        for judge_id, judge in ctx.judges.items():
            # Each judge sees only the packet: never another judge's assessment.
            records = list(existing[(packet_record.record_id, judge_id)])
            while (attempt_no := next_attempt(records, ctx.policy.judge_max_attempts)) is not None:
                call = EvaluationCall(
                    kind=RecordKind.JUDGE_EVALUATION, prefix="jdg", dirname="judgements", purpose="judge",
                    experiment_id=ctx.experiment_id, run_id=ctx.run.record_id,
                    subject_id=str(packet_record.subject_id),
                    keys={"packet_id": packet_record.record_id, "judge_id": judge_id},
                    round=int(packet["round"]), config=judge.configuration,
                    messages=evaluation_messages(judge.prompt, JUDGE_TAG, packet),
                    inputs={packet_record.record_id: packet_record.payload["packet_sha256"], **judge.record_inputs},
                    policy_revision=f"{ctx.policy.key}/judge:{judge_id}",
                    parse=partial(parse_judge_output, packet=packet, rubric=ctx.rubric),
                )
                records.append(run_call(db, store, registry, call, attempt_no))


def _classify(db: Database, store: ArtifactStore, registry: AdapterRegistry, ctx: RunContext) -> None:
    if ctx.classifier is None:
        return
    classifier = ctx.classifier
    existing: dict[str, list[Record]] = defaultdict(list)
    for r in run_records(db, RecordKind.ARCHITECTURE_CLASSIFICATION, ctx.experiment_id, ctx.run.record_id):
        existing[r.payload["packet_id"]].append(r)
    for packet_record in _packets(db, ctx):
        if int(packet_record.payload["round"]) not in ctx.classifier_rounds:
            continue
        packet = classification_packet(read_packet(db, store, packet_record.record_id))
        if ctx.classifier_spec is not None:
            messages = structured_messages(ctx.classifier_spec, packet)
            parse: Parser = partial(parse_structured_classification, spec=ctx.classifier_spec)
        else:
            messages = evaluation_messages(classifier.prompt, CLASSIFIER_TAG, packet)
            parse = partial(parse_classification_output, packet=packet)
        records = list(existing[packet_record.record_id])
        while (attempt_no := next_attempt(records, ctx.policy.judge_max_attempts)) is not None:
            call = EvaluationCall(
                kind=RecordKind.ARCHITECTURE_CLASSIFICATION, prefix="cls", dirname="classifications",
                purpose="classification", experiment_id=ctx.experiment_id, run_id=ctx.run.record_id,
                subject_id=str(packet_record.subject_id), keys={"packet_id": packet_record.record_id},
                round=int(packet["round"]), config=classifier.configuration,
                messages=messages,
                inputs={packet_record.record_id: packet_record.payload["packet_sha256"], **classifier.record_inputs},
                policy_revision=f"{ctx.policy.key}/classifier",
                parse=parse,
            )
            records.append(run_call(db, store, registry, call, attempt_no))


def _extract(db: Database, store: ArtifactStore, registry: AdapterRegistry, ctx: RunContext) -> None:
    if ctx.extractor is None or ctx.catalog is None:
        return
    extractor, catalog = ctx.extractor, ctx.catalog
    catalog_file = FrozenFile.model_validate(ctx.manifest["extractor"]["catalog_file"])
    existing: dict[str, list[Record]] = defaultdict(list)
    for r in run_records(db, RecordKind.RECOMMENDATION_EXTRACTION, ctx.experiment_id, ctx.run.record_id):
        existing[r.payload["packet_id"]].append(r)
    for packet_record in _packets(db, ctx):
        if int(packet_record.payload["round"]) not in ctx.extractor_rounds:
            continue
        packet = extraction_packet(read_packet(db, store, packet_record.record_id), catalog)
        records = list(existing[packet_record.record_id])
        while (attempt_no := next_attempt(records, ctx.policy.judge_max_attempts)) is not None:
            call = EvaluationCall(
                kind=RecordKind.RECOMMENDATION_EXTRACTION, prefix="ext", dirname="extractions",
                purpose="extraction", experiment_id=ctx.experiment_id, run_id=ctx.run.record_id,
                subject_id=str(packet_record.subject_id), keys={"packet_id": packet_record.record_id},
                round=int(packet["round"]), config=extractor.configuration,
                messages=extraction_messages(extractor.prompt, packet),
                inputs={packet_record.record_id: packet_record.payload["packet_sha256"],
                        catalog_file.path: catalog_file.sha256, **extractor.record_inputs},
                policy_revision=f"{ctx.policy.key}/extractor",
                parse=partial(parse_extraction, packet=packet, catalog=catalog),
            )
            records.append(run_call(db, store, registry, call, attempt_no))


def _judging_settled(db: Database, ctx: RunContext) -> tuple[bool, list[Record]]:
    records = run_records(db, RecordKind.JUDGE_EVALUATION, ctx.experiment_id, ctx.run.record_id)
    grouped: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        grouped[(r.payload["packet_id"], r.payload["judge_id"])].append(r)
    settled = all(
        next_attempt(grouped[(p.record_id, j)], ctx.policy.judge_max_attempts) is None
        for p in _packets(db, ctx) for j in ctx.judges
    )
    return settled, records


def _select(db: Database, ctx: RunContext) -> None:
    if db.get_record(selection_record_id(ctx.run.record_id)) is not None:
        return
    settled, records = _judging_settled(db, ctx)
    if not settled:
        return
    tracks = {e["response_id"]: e["track"] for e in ctx.manifest["responses"]}
    responses = [
        {"response_id": p.subject_id, "packet_id": p.record_id, "round": int(p.payload["round"]),
         "track": tracks[str(p.subject_id)]}
        for p in _packets(db, ctx)
    ]
    by_response: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_response[str(r.subject_id)].append(r)
    selection = build_selection(policy=ctx.policy, rubric=ctx.rubric, responses=responses, judge_records=by_response)
    publish_selection(db, ctx.run, selection, records)


# ----- status ----------------------------------------------------------------------------------------


def _model_stage(records: list[Record], groups: list[tuple[str, ...]], key: tuple[str, ...], max_attempts: int
                 ) -> StageCount:
    grouped: dict[tuple[str, ...], list[Record]] = defaultdict(list)
    for r in records:
        grouped[tuple(r.payload[k] for k in key)].append(r)
    done = sum(1 for g in groups if selected_attempt(grouped[g]) is not None)
    exhausted = sum(
        1 for g in groups if selected_attempt(grouped[g]) is None and next_attempt(grouped[g], max_attempts) is None
    )
    return StageCount(done=done, expected=len(groups), exhausted=exhausted)


def evaluation_status(db: Database, evaluation_run_id: str) -> EvaluationStatus:
    ctx = load_run(db, evaluation_run_id)
    exp, run_id = ctx.experiment_id, ctx.run.record_id
    total = len(ctx.manifest["responses"])
    packets = _packets(db, ctx)
    validations = run_records(db, RecordKind.VALIDATION, exp, run_id)
    answered = sum(
        1 for v in validations
        if next(c for c in v.payload["checks"] if c["name"] == "artifact_integrity")["passed"]
    )
    max_attempts = ctx.policy.judge_max_attempts
    judges = _model_stage(
        run_records(db, RecordKind.JUDGE_EVALUATION, exp, run_id),
        [(p.record_id, j) for p in packets for j in ctx.judges], ("packet_id", "judge_id"), max_attempts,
    )
    classifications = _model_stage(
        run_records(db, RecordKind.ARCHITECTURE_CLASSIFICATION, exp, run_id),
        [(p.record_id,) for p in packets if int(p.payload["round"]) in ctx.classifier_rounds],
        ("packet_id",), max_attempts,
    )
    extractions = _model_stage(
        run_records(db, RecordKind.RECOMMENDATION_EXTRACTION, exp, run_id),
        [(p.record_id,) for p in packets if int(p.payload["round"]) in ctx.extractor_rounds],
        ("packet_id",), max_attempts,
    )
    selection = db.get_record(selection_record_id(run_id))
    selected = {i["packet_id"] for i in selection.payload["items"]} if selection else set()
    reviewed = {r.payload["packet_id"] for r in effective(run_records(db, RecordKind.HUMAN_REVIEW, exp, run_id))}
    adjudications = effective(run_records(db, RecordKind.ADJUDICATION, exp, run_id))

    pending = []
    if len(validations) < total or len(packets) < answered:
        pending.append("validation_and_packets")
    if judges.done + judges.exhausted < judges.expected:
        pending.append("judge_evaluation")
    if classifications.done + classifications.exhausted < classifications.expected:
        pending.append("architecture_classification")
    if extractions.done + extractions.exhausted < extractions.expected:
        pending.append("recommendation_extraction")
    if selection is None:
        pending.append("review_selection")
    if selection is None or selected - reviewed:
        pending.append("human_review")
    if len(adjudications) < len(packets) or selection is None:
        pending.append("adjudication")
    return EvaluationStatus(
        evaluation_run_id=run_id,
        responses=total,
        validations=StageCount(done=len(validations), expected=total),
        packets=StageCount(done=len(packets), expected=answered),
        judge_evaluations=judges,
        classifications=classifications,
        extractions=extractions,
        review_selection=selection.record_id if selection else None,
        human_reviews=StageCount(done=len(selected & reviewed), expected=len(selected)),
        adjudications=StageCount(done=len(adjudications), expected=len(packets)),
        pending_stages=pending,
    )


def _read_optional(store: ArtifactStore, ref: Any) -> bytes | None:
    if ref is None:
        return None
    try:
        return store.read_verified(ref)
    except ArtifactIntegrityError:
        return None
