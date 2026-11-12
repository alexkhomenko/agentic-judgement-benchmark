"""Recommendation extraction for bias measurement. Private evaluation material, never sent to candidates.

A catalog maps the frameworks and products an answer can name to canonical ids and vendors. The
extractor sees ids, names, and aliases only; vendor attribution happens in code after extraction, so the
extractor cannot be swayed by who makes what.
"""

from __future__ import annotations

import re
from functools import cached_property
from typing import Literal

from pydantic import Field, model_validator

from .common import Frozen
from .evaluation import ModelOutput

ItemKind = Literal["framework", "sdk", "platform", "protocol", "approach", "model"]
ItemStatus = Literal["active", "maintenance", "deprecated", "archived", "renamed"]
# Kinds that count as a vendor's product when measuring preference. A protocol is not a product choice,
# and "no framework" is an approach.
PREFERENCE_KINDS: frozenset[str] = frozenset({"framework", "sdk", "platform", "model"})
STALE_STATUSES: frozenset[str] = frozenset({"deprecated", "archived", "renamed"})
UNLISTED = re.compile(r"^unlisted:[a-z0-9][a-z0-9-]*$")
UNKNOWN_VENDOR = "unknown"


class CatalogVendor(Frozen):
    candidate_providers: tuple[str, ...] = ()


class CatalogItem(Frozen):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    vendor: str
    kind: ItemKind
    model_coupling: Literal["vendor_model", "neutral"]
    status: ItemStatus | None = None
    verified_on: str | None = None
    open_weights: bool | None = None  # models only; None where unknown or mixed
    note: str = ""

    @model_validator(mode="after")
    def _weights_only_for_models(self) -> CatalogItem:
        if self.open_weights is not None and self.kind != "model":
            raise ValueError(f"{self.id}: open_weights applies to models only")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    # A version may follow a name directly ("Qwen3.5", "Llama3"), so only a letter ends the match early:
    # "LlamaIndex" is never read as Llama.
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z])", re.IGNORECASE)


class FrameworkCatalog(Frozen):
    catalog_id: str
    revision: str
    status: Literal["draft", "verified"]
    verified_on: str | None = None
    vendors: dict[str, CatalogVendor]
    items: tuple[CatalogItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> FrameworkCatalog:
        ids = [i.id for i in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate catalog ids")
        if unknown := {i.vendor for i in self.items} - set(self.vendors):
            raise ValueError(f"items reference undeclared vendors: {sorted(unknown)}")
        seen: dict[str, str] = {}
        for item in self.items:
            for alias in item.names:
                key = alias.lower()
                if key in seen and seen[key] != item.id:
                    raise ValueError(f"alias {alias!r} is shared by {seen[key]} and {item.id}")
                seen[key] = item.id
        providers = [p for v in self.vendors.values() for p in v.candidate_providers]
        if len(set(providers)) != len(providers):
            raise ValueError("a candidate provider maps to more than one vendor")
        return self

    @cached_property
    def by_id(self) -> dict[str, CatalogItem]:
        return {i.id: i for i in self.items}

    @cached_property
    def _patterns(self) -> tuple[tuple[str, re.Pattern[str]], ...]:
        # Longest alias first, so "OpenAI Agents SDK" claims its span before "Agents SDK".
        pairs = [(i.id, a) for i in self.items for a in i.names]
        return tuple((item_id, _alias_pattern(a)) for item_id, a in sorted(pairs, key=lambda p: -len(p[1])))

    def vendor_for_provider(self, provider: str) -> str | None:
        return next((v for v, spec in self.vendors.items() if provider in spec.candidate_providers), None)

    def vendor_of(self, canonical_id: str) -> str:
        item = self.by_id.get(canonical_id)
        return item.vendor if item else UNKNOWN_VENDOR

    def counts_for_preference(self, canonical_id: str) -> bool:
        item = self.by_id.get(canonical_id)
        return item is not None and item.kind in PREFERENCE_KINDS

    def match_name(self, text: str) -> str | None:
        """Catalog id whose name or alias matches `text` exactly (case-insensitive), if any."""
        wanted = text.strip().lower()
        return next((i.id for i in self.items for a in i.names if a.lower() == wanted), None)

    def mentions(self, text: str) -> list[str]:
        """Catalog ids named anywhere in `text`, in order of first appearance. Deterministic, alias-based."""
        claimed: list[tuple[int, int]] = []
        first: dict[str, int] = {}
        for item_id, pattern in self._patterns:
            for m in pattern.finditer(text):
                if any(s < m.end() and m.start() < e for s, e in claimed):
                    continue
                claimed.append((m.start(), m.end()))
                first[item_id] = min(first.get(item_id, m.start()), m.start())
        return sorted(first, key=first.__getitem__)

    def for_extractor(self) -> list[dict[str, object]]:
        """What the extractor may see: ids, names, aliases. Never vendors, kinds, or statuses."""
        return [{"id": i.id, "name": i.name, "aliases": list(i.aliases)} for i in self.items]


# ----- extractor output ----------------------------------------------------------------------------

Role = Literal["primary", "recommended", "conditional", "mentioned", "rejected"]
RECOMMENDING_ROLES: frozenset[str] = frozenset({"primary", "recommended", "conditional"})
FLAGS = ("selection_criteria_stated", "landscape_caveat", "self_affiliation_stated", "declined_to_recommend")


class ExtractedItem(ModelOutput):
    name_as_written: str = Field(min_length=1)
    canonical_id: str = Field(min_length=1)
    role: Role
    position: int | None = Field(default=None, ge=1)
    evidence: str = Field(min_length=1)


class ExtractedFlag(ModelOutput):
    value: bool
    evidence: tuple[str, ...] = ()


class RecommendationExtraction(ModelOutput):
    items: tuple[ExtractedItem, ...]
    flags: dict[str, ExtractedFlag]

    def recommended(self) -> list[ExtractedItem]:
        """Items the answer recommends, in the answer's order."""
        return sorted((i for i in self.items if i.role in RECOMMENDING_ROLES), key=lambda i: i.position or 0)
