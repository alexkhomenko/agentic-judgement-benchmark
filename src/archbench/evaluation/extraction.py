"""Recommendation extraction: packets, messages, and strict parsing.

The extractor sees the task, the blinded answer, and the catalog's ids/names/aliases. It records which
items the answer names and in what role and order; it never judges quality and never sees vendors. Code
then cross-checks the output against the answer text and the catalog. A structural problem makes the
attempt invalid, so it is retried under the run's attempt policy. A disagreement with the deterministic
alias scan is recorded as a review signal, because an alias match can be a false positive.

An item listed more than once (for example two versions of one model family) is merged by the rule the
extractor is given: one entry per item, with its strongest role, earliest position, and first name as
written. Positions are validated on the raw output first and renumbered after a merge; the raw output is
kept as evidence and the merged ids are recorded.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..domain.recommendations import (
    FLAGS,
    RECOMMENDING_ROLES,
    UNLISTED,
    FrameworkCatalog,
    RecommendationExtraction,
)
from ..domain.scenario import Message
from .judging import _errors, evaluation_messages, evidence_check, extract_json

EXTRACTION_TAG = "extraction_packet"
ROLE_STRENGTH = ("primary", "recommended", "conditional", "mentioned", "rejected")


def extraction_packet(judge_packet: dict[str, Any], catalog: FrameworkCatalog) -> dict[str, Any]:
    """Task and blinded answer from the judge packet (never the rubric), plus the catalog."""
    return {
        "packet_id": judge_packet["packet_id"],
        "round": judge_packet["round"],
        "task_conversation": judge_packet["task_conversation"],
        "candidate_answer": judge_packet["candidate_answer"],
        "catalog": catalog.for_extractor(),
    }


def extraction_messages(instructions: str, packet: dict[str, Any]) -> tuple[Message, ...]:
    return evaluation_messages(instructions, EXTRACTION_TAG, packet)


def parse_extraction(
    text: str, packet: dict[str, Any], catalog: FrameworkCatalog
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    raw = extract_json(text)
    if not isinstance(raw, dict):
        return None, ["no JSON object found in extractor output"], {}
    try:
        extraction = RecommendationExtraction.model_validate(raw)
    except ValidationError as exc:
        return None, _errors(exc), {}

    problems: list[str] = []
    for item in extraction.items:
        if item.canonical_id not in catalog.by_id and not UNLISTED.match(item.canonical_id):
            problems.append(f"{item.canonical_id!r} is neither a catalog id nor unlisted:<slug>")
        recommending = item.role in RECOMMENDING_ROLES
        if recommending and item.position is None:
            problems.append(f"{item.canonical_id}: a recommended item needs a position")
        if not recommending and item.position is not None:
            problems.append(f"{item.canonical_id}: a {item.role} item has no position")
    positions = sorted(i.position for i in extraction.items if i.position is not None)
    if positions != list(range(1, len(positions) + 1)):
        problems.append(f"positions must run 1..n without gaps or repeats, got {positions}")
    if missing := sorted(set(FLAGS) - extraction.flags.keys()):
        problems.append(f"missing flags: {missing}")
    if extra := sorted(extraction.flags.keys() - set(FLAGS)):
        problems.append(f"unknown flags: {extra}")
    for name, flag in extraction.flags.items():
        if flag.value and not flag.evidence:
            problems.append(f"flag {name} is true without evidence")

    answer = str(packet["candidate_answer"])
    quotes = [(f"item:{i.canonical_id}", i.evidence) for i in extraction.items]
    quotes += [(f"name:{i.canonical_id}", i.name_as_written) for i in extraction.items]
    quotes += [(f"flag:{n}", q) for n, f in extraction.flags.items() for q in f.evidence]

    # Deterministic cross-check against the catalog's aliases. Signals for human spot-checks only.
    conflicts = [
        {"name_as_written": i.name_as_written, "canonical_id": i.canonical_id, "alias_match": matched}
        for i in extraction.items
        if (matched := catalog.match_name(i.name_as_written)) is not None and matched != i.canonical_id
    ]
    ids = list(dict.fromkeys(i.canonical_id for i in extraction.items))
    scanned = catalog.mentions(answer)
    catalog_check = {
        "alias_scan": scanned,
        "not_extracted": [i for i in scanned if i not in ids],
        "name_conflicts": conflicts,
        "unlisted": [i for i in ids if UNLISTED.match(i)],
    }
    output = extraction.model_dump(mode="json")
    merged_items, merged = merge_duplicates(output["items"])
    catalog_check["merged_duplicates"] = merged
    output["items"] = sorted(merged_items, key=lambda i: (i["position"] is None, i["position"] or 0))
    return output, problems, {"evidence_check": evidence_check(quotes, answer), "catalog_check": catalog_check}


def merge_duplicates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """One entry per canonical id: strongest role, earliest position, first name as written."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["canonical_id"], []).append(item)
    merged = [cid for cid, group in groups.items() if len(group) > 1]
    if not merged:
        return items, []
    out = []
    for group in groups.values():
        strongest = min(group, key=lambda i: ROLE_STRENGTH.index(i["role"]))
        positions = [i["position"] for i in group if i["position"] is not None]
        recommending = strongest["role"] in RECOMMENDING_ROLES
        out.append({**strongest, "name_as_written": group[0]["name_as_written"],
                    "position": min(positions) if recommending and positions else None})
    for n, item in enumerate(sorted((i for i in out if i["position"] is not None), key=lambda i: i["position"]), 1):
        item["position"] = n
    return out, merged
