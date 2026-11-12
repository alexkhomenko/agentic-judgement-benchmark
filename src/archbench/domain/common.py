"""Shared primitives: immutable base model, canonical hashing, identifiers, timestamps."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

ARTIFACT_SCHEMA_VERSION = "1"
PLAN_SCHEMA_VERSION = "1"


class Frozen(BaseModel):
    """Immutable, strict schema. Unknown fields are rejected at every boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def canonical_json(value: Any) -> str:
    """Deterministic JSON used for every content hash."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def content_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
