"""Redaction applied before any request, response, or error is persisted."""

from __future__ import annotations

import re
from typing import Any

from ..secrets import GUARD

REDACTED = "[REDACTED]"

SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "api_key",
        "apikey",
        "x-goog-api-key",
        "cookie",
        "set-cookie",
        "password",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
    }
)

_TEXT_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    re.compile(r"\bxai-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|x-api-key|authorization)(\s*[=:]\s*)\S+"),
)


def redact(value: Any) -> Any:
    """Recursively replace values of sensitive keys. Matching is exact, so `max_tokens` is untouched."""
    if isinstance(value, dict):
        return {k: (REDACTED if str(k).lower() in SENSITIVE_KEYS else redact(v)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(text: str) -> str:
    text = GUARD.scrub(text)
    text = _TEXT_PATTERNS[0].sub(f"Bearer {REDACTED}", text)
    text = _TEXT_PATTERNS[1].sub(REDACTED, text)
    text = _TEXT_PATTERNS[2].sub(REDACTED, text)
    text = _TEXT_PATTERNS[3].sub(REDACTED, text)
    return _TEXT_PATTERNS[4].sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
