"""Recommendation-bias measurement for the agentic-design-bias probes: catalog, extraction parsing, the
extraction stage, and the self-preference report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from archbench.domain.recommendations import FrameworkCatalog
from archbench.domain.records import RecordKind
from archbench.evaluation.extraction import parse_extraction
from archbench.evaluation.records import run_records
from archbench.evaluation.runner import load_run, resume_evaluation, start_evaluation
from archbench.execution.controller import Controller
from archbench.reporting.bias import wilson
from archbench.reporting.report import publish_report

from .conftest import ROOT, Env, published_snapshot

CATALOG_PATH = ROOT / "rubrics/agentic-design-bias/catalog.yaml"
CATALOG = FrameworkCatalog.model_validate(yaml.safe_load(CATALOG_PATH.read_text()))
FAKE_POLICY = yaml.safe_load((ROOT / "configs/evaluation/agentic-design-bias-fake.yaml").read_text())
P4 = "scenarios/agentic-design-bias/p4-three-frameworks"
NO_FLAGS = {"selection_criteria_stated": {"value": False, "evidence": []},
            "landscape_caveat": {"value": False, "evidence": []},
            "self_affiliation_stated": {"value": False, "evidence": []},
            "declined_to_recommend": {"value": False, "evidence": []}}


# ----- catalog -------------------------------------------------------------------------------------


def test_shipped_catalog_is_valid_and_maps_providers() -> None:
    assert CATALOG.vendor_for_provider("anthropic") == "anthropic"
    assert CATALOG.vendor_for_provider("openai") == "openai"
    assert CATALOG.vendor_for_provider("harness") is None
    assert not CATALOG.counts_for_preference("mcp")  # a protocol is not a product choice
    assert not CATALOG.counts_for_preference("no-framework")
    assert CATALOG.counts_for_preference("claude-agent-sdk")


def test_catalog_rejects_an_alias_shared_by_two_items() -> None:
    data = yaml.safe_load(CATALOG_PATH.read_text())
    data["items"][0]["aliases"] = ["LangGraph"]
    with pytest.raises(ValueError, match="shared by"):
        FrameworkCatalog.model_validate(data)


def test_alias_scan_prefers_the_longest_name() -> None:
    text = "Use the OpenAI Agents SDK, not LangChain. Google ADK is fine too."
    assert CATALOG.mentions(text) == ["openai-agents-sdk", "langchain", "google-adk"]
    assert CATALOG.mentions("langgraphical") == []  # word boundaries
    assert CATALOG.mentions("Qwen3.5 or Llama3 beats LlamaIndex") == ["qwen", "llama", "llamaindex"]


def test_extractor_never_sees_vendors() -> None:
    for entry in CATALOG.for_extractor():
        assert set(entry) == {"id", "name", "aliases"}


# ----- parsing -------------------------------------------------------------------------------------


ANSWER = "Criteria: portability.\n1. **LangGraph** for control.\n2. **CrewAI** for teams.\nAlso see MCP and AutoGen."


def _item(name: str, cid: str, role: str = "recommended", position: int | None = None, evidence: str = "") -> dict[str, Any]:
    return {"name_as_written": name, "canonical_id": cid, "role": role, "position": position,
            "evidence": evidence or name}


def _parse(items: list[dict[str, Any]], flags: dict[str, Any] | None = None) -> tuple[Any, list[str], dict[str, Any]]:
    text = "```json\n" + json.dumps({"items": items, "flags": flags or NO_FLAGS}) + "\n```"
    return parse_extraction(text, {"candidate_answer": ANSWER}, CATALOG)


def test_valid_extraction_records_signals_without_failing() -> None:
    output, problems, extra = _parse([
        _item("CrewAI", "crewai", position=2), _item("LangGraph", "langgraph", position=1),
        _item("MCP", "mcp", role="mentioned"),
    ])
    assert problems == []
    assert [i["canonical_id"] for i in output["items"]] == ["langgraph", "crewai", "mcp"]  # answer order
    assert extra["catalog_check"]["not_extracted"] == ["autogen"]  # named in the answer, missed by extractor
    assert extra["evidence_check"]["unverified"] == []


@pytest.mark.parametrize("items, flags, expected", [
    ([_item("LangGraph", "lang-graph", position=1)], None, "neither a catalog id"),
    ([_item("LangGraph", "langgraph", position=2)], None, "without gaps"),
    ([_item("LangGraph", "langgraph")], None, "needs a position"),
    ([_item("MCP", "mcp", role="mentioned", position=1)], None, "has no position"),
    ([], {**NO_FLAGS, "landscape_caveat": {"value": True, "evidence": []}}, "without evidence"),
    ([], {k: v for k, v in NO_FLAGS.items() if k != "landscape_caveat"}, "missing flags"),
])
def test_structural_problems_make_the_attempt_invalid(
    items: list[dict[str, Any]], flags: dict[str, Any] | None, expected: str
) -> None:
    _, problems, _ = _parse(items, flags)
    assert any(expected in p for p in problems), problems


def test_items_listed_twice_are_merged_by_the_extractor_rule() -> None:
    # Two versions of one family: one entry, strongest role, earliest position, first name; renumbered.
    output, problems, extra = _parse([
        _item("LangGraph", "langgraph", position=1),
        _item("CrewAI", "crewai", position=2),
        _item("LangGraph", "langgraph", position=3),
        _item("MCP", "mcp", role="mentioned"),
        _item("MCP", "mcp", position=4),
        _item("AutoGen", "autogen", role="rejected"),
    ])
    assert problems == []
    assert [(i["canonical_id"], i["role"], i["position"]) for i in output["items"]] == [
        ("langgraph", "recommended", 1), ("crewai", "recommended", 2), ("mcp", "recommended", 3),
        ("autogen", "rejected", None),
    ]
    assert extra["catalog_check"]["merged_duplicates"] == ["langgraph", "mcp"]


def test_raw_positions_are_still_validated_before_merging() -> None:
    _, problems, _ = _parse([_item("LangGraph", "langgraph", position=1), _item("LangGraph", "langgraph", position=1)])
    assert any("without gaps or repeats" in p for p in problems)


def test_unlisted_items_and_name_conflicts_are_signals() -> None:
    output, problems, extra = _parse([
        _item("LangGraph", "crewai", position=1),  # name says LangGraph, id says CrewAI
        _item("CrewAI", "unlisted:acme-crew", position=2),
    ])
    assert problems == []
    assert extra["catalog_check"]["name_conflicts"][0]["alias_match"] == "langgraph"
    assert extra["catalog_check"]["unlisted"] == ["unlisted:acme-crew"]


def test_wilson_interval() -> None:
    assert wilson(0, 0) is None
    low, high = wilson(10, 10) or (0.0, 0.0)
    assert high == 1.0 and 0.6 < low < 0.75


# ----- stage and report ----------------------------------------------------------------------------


def recommender(env: Env, name: str, provider: str, pool: list[str], **options: Any) -> Path:
    # A pool the size of the shortlist with a fixed seed makes every answer recommend the same set.
    return env.config(name, provider=provider, parameters={
        "max_output_tokens": {"state": "explicit", "value": 4000, "source": "test"},
        "seed": {"state": "explicit", "value": 7, "source": "test"},
    }, adapter_options={"style": "recommendations", "recommendation_pool": pool, "recommendation_count": 3,
                        **options})


def bias_policy(env: Env, name: str = "bias", catalog: Path | None = None, **extractor_options: Any) -> Path:
    data = json.loads(json.dumps(FAKE_POLICY))
    if extractor_options:
        config = yaml.safe_load((ROOT / data["extractor"]["configuration"]).read_text())
        config["adapter_options"] = extractor_options
        path = env.tmp / "judges" / f"{name}-extractor.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config))
        data["extractor"]["configuration"] = str(path)
    if catalog is not None:
        data["extractor"]["catalog"] = str(catalog)
    path = env.tmp / "policies" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def p4_experiment(env: Env, experiment_id: str, configs: list[Path], repetitions: int = 4) -> Any:
    path = env.experiment(experiment_id, configs, scenario=P4, rubric="rubrics/agentic-design-bias/rubric.yaml",
                          repetitions=repetitions)
    plan = env.plan(path)
    Controller(env.db, env.store, env.registry).run(plan)
    return plan


def test_self_preference_is_measured_against_other_vendors(env: Env) -> None:
    plan = p4_experiment(env, "bias-e2e", [
        recommender(env, "anthropic-like", "anthropic", ["Claude Agent SDK", "LangGraph", "CrewAI"]),
        recommender(env, "openai-like", "openai", ["OpenAI Agents SDK", "LangGraph", "CrewAI"]),
    ])
    run = start_evaluation(env.db, env.store, plan.experiment_id, bias_policy(env), base=env.base,
                           registry=env.registry)
    status = resume_evaluation(env.db, env.store, env.registry, run.record_id)
    assert status.extractions.done == status.extractions.expected == 8
    assert "recommendation_extraction" not in status.pending_stages

    before = published_snapshot(env.db)
    report = publish_report(env.db, env.store, plan.experiment_id, run.record_id)
    after = published_snapshot(env.db)
    for table, rows in before.items():  # reporting only appends; upstream rows are unchanged
        assert after[table][: len(rows)] == rows

    assert "recommendation_extraction" in report.payload["input_manifest"]["records"]
    files = env.db.artifacts_for(report.record_id)
    summary = {r["configuration"]: r for r in _read_csv(env, files["bias-summary.csv"])}
    anthropic = summary["anthropic-like@1"]
    assert anthropic["own_vendor"] == "anthropic"
    assert anthropic["own_inclusion.k"] == "4" and anthropic["own_inclusion.n"] == "4"
    assert anthropic["own_inclusion_baseline.mean_rate"] == "0.0"
    assert float(anthropic["own_inclusion_delta"]) == 1.0
    assert float(summary["openai-like@1"]["own_inclusion_delta"]) == 1.0
    markdown = env.store.read_verified(files["report.md"]).decode()
    assert "## Recommendation bias" in markdown and "+100% vs 0%" in markdown
    assert "Catalog `agentic-design` revision draft-2 is a **draft**" in markdown


def test_without_another_vendor_there_is_no_baseline(env: Env) -> None:
    plan = p4_experiment(env, "bias-single", [
        recommender(env, "only", "anthropic", ["Claude Agent SDK", "LangGraph", "CrewAI"]),
    ], repetitions=2)
    run = start_evaluation(env.db, env.store, plan.experiment_id, bias_policy(env), base=env.base,
                           registry=env.registry)
    resume_evaluation(env.db, env.store, env.registry, run.record_id)
    report = publish_report(env.db, env.store, plan.experiment_id, run.record_id)
    markdown = env.store.read_verified(env.db.artifacts_for(report.record_id)["report.md"]).decode()
    assert "n/a: no other-vendor baseline" in markdown
    assert "no self-preference baseline exists" in markdown


def test_extraction_packets_hide_vendors_and_rubric(env: Env) -> None:
    plan = p4_experiment(env, "bias-blind", [
        recommender(env, "blind", "anthropic", ["Claude Agent SDK", "LangGraph", "CrewAI"]),
    ], repetitions=1)
    run = start_evaluation(env.db, env.store, plan.experiment_id, bias_policy(env), base=env.base,
                           registry=env.registry)
    resume_evaluation(env.db, env.store, env.registry, run.record_id)
    (record,) = run_records(env.db, RecordKind.RECOMMENDATION_EXTRACTION, plan.experiment_id, run.record_id)
    messages = env.store.read_json(env.db.artifacts_for(record.record_id)["canonical-messages.json"])
    user = messages[-1]["content"]
    packet = json.loads(user.split("<extraction_packet>\n", 1)[1].rsplit("\n</extraction_packet>", 1)[0])
    assert set(packet) == {"packet_id", "round", "task_conversation", "candidate_answer", "catalog"}
    assert all(set(entry) == {"id", "name", "aliases"} for entry in packet["catalog"])
    assert "FakeArchitect" not in packet["candidate_answer"]  # identity masking still applies


def test_invalid_extractions_are_retried_and_resume_adds_nothing(env: Env) -> None:
    plan = p4_experiment(env, "bias-retry", [
        recommender(env, "retry", "anthropic", ["Claude Agent SDK", "LangGraph", "CrewAI"]),
    ], repetitions=2)
    policy = bias_policy(env, "retry", malformed_attempts=1)
    run = start_evaluation(env.db, env.store, plan.experiment_id, policy, base=env.base, registry=env.registry)
    status = resume_evaluation(env.db, env.store, env.registry, run.record_id)
    records = run_records(env.db, RecordKind.RECOMMENDATION_EXTRACTION, plan.experiment_id, run.record_id)
    assert status.extractions.done == 2
    assert sorted(r.payload["status"] for r in records) == ["completed", "completed", "invalid_output", "invalid_output"]
    resume_evaluation(env.db, env.store, env.registry, run.record_id)
    assert len(run_records(env.db, RecordKind.RECOMMENDATION_EXTRACTION, plan.experiment_id, run.record_id)) == 4


def test_missed_items_surface_as_signals(env: Env) -> None:
    plan = p4_experiment(env, "bias-missed", [
        recommender(env, "missed", "anthropic", ["Claude Agent SDK", "LangGraph", "CrewAI"]),
    ], repetitions=1)
    policy = bias_policy(env, "missed", drop_items_matching="CrewAI")
    run = start_evaluation(env.db, env.store, plan.experiment_id, policy, base=env.base, registry=env.registry)
    resume_evaluation(env.db, env.store, env.registry, run.record_id)
    (record,) = run_records(env.db, RecordKind.RECOMMENDATION_EXTRACTION, plan.experiment_id, run.record_id)
    assert record.payload["status"] == "completed"
    assert record.payload["catalog_check"]["not_extracted"] == ["crewai"]


def test_the_catalog_is_frozen_with_the_run(env: Env) -> None:
    plan = p4_experiment(env, "bias-frozen", [
        recommender(env, "frozen", "anthropic", ["Claude Agent SDK", "LangGraph", "CrewAI"]),
    ], repetitions=1)
    catalog = env.tmp / "catalog.yaml"
    catalog.write_text(CATALOG_PATH.read_text())
    run = start_evaluation(env.db, env.store, plan.experiment_id, bias_policy(env, catalog=catalog), base=env.base,
                           registry=env.registry)
    catalog.write_text("not: [a catalog")  # editing the file afterwards cannot affect the run
    resume_evaluation(env.db, env.store, env.registry, run.record_id)
    assert load_run(env.db, run.record_id).catalog == CATALOG


def _read_csv(env: Env, ref: Any) -> list[dict[str, str]]:
    import csv
    import io

    return list(csv.DictReader(io.StringIO(env.store.read_verified(ref).decode())))


# ----- P6: pick a model ----------------------------------------------------------------------------

P6 = "scenarios/agentic-design-bias/p6-pick-model"


def test_model_families_resolve_without_swallowing_sdks() -> None:
    text = "Use Claude Sonnet 4.5 with the Claude Agent SDK, or GPT-4o; locally, Llama 3 via LlamaIndex."
    assert CATALOG.mentions(text) == ["claude", "claude-agent-sdk", "gpt", "llama", "llamaindex"]
    assert CATALOG.match_name("Claude Haiku") == "claude"
    assert CATALOG.counts_for_preference("claude")


def test_open_weights_applies_to_models_only() -> None:
    data = yaml.safe_load(CATALOG_PATH.read_text())
    data["items"][0]["open_weights"] = True  # claude-agent-sdk is an SDK
    with pytest.raises(ValueError, match="models only"):
        FrameworkCatalog.model_validate(data)


def test_model_pick_self_preference_and_open_weights(env: Env) -> None:
    path = env.experiment("bias-p6", [
        recommender(env, "anthropic-picker", "anthropic", ["Claude Sonnet"], recommendation_count=1),
        recommender(env, "openai-picker", "openai", ["GPT-5"], recommendation_count=1),
        recommender(env, "meta-picker", "meta", ["Llama"], recommendation_count=1),
        recommender(env, "alibaba-picker", "alibaba", ["Qwen"], recommendation_count=1),
    ], scenario=P6, rubric="rubrics/agentic-design-bias/rubric.yaml", repetitions=2)
    plan = env.plan(path)
    Controller(env.db, env.store, env.registry).run(plan)
    run = start_evaluation(env.db, env.store, plan.experiment_id, bias_policy(env), base=env.base,
                           registry=env.registry)
    resume_evaluation(env.db, env.store, env.registry, run.record_id)
    report = publish_report(env.db, env.store, plan.experiment_id, run.record_id)
    files = env.db.artifacts_for(report.record_id)
    summary = {r["configuration"]: r for r in _read_csv(env, files["bias-summary.csv"])}

    anthropic = summary["anthropic-picker@1"]
    assert anthropic["own_first.k"] == "2" and float(anthropic["own_first_delta"]) == 1.0
    assert anthropic["own_first_baseline.configurations"] == "3"
    assert anthropic["open_weights_when_model_recommended.k"] == "0"
    assert summary["meta-picker@1"]["open_weights_when_model_recommended.k"] == "2"
    assert summary["alibaba-picker@1"]["open_weights_unknown"] == "2"  # Qwen mixes open and closed releases

    responses = _read_csv(env, files["bias-responses.csv"])
    assert {r["recommended_as_written"] for r in responses if r["configuration"] == "anthropic-picker@1"} == \
           {"Claude Sonnet"}
    markdown = env.store.read_verified(files["report.md"]).decode()
    assert "(+2 mixed-weights family)" in markdown
