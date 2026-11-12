"""Transport-boundary capture shared by cloud adapters.

Every HTTP exchange is recorded after client serialization: exact request and response bodies (with
their hashes), status, and headers. `wire_record` redacts before anything is persisted.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2

from ...domain.common import sha256_bytes, utc_now
from ..redaction import redact


class CapturingTransport(httpx2.BaseTransport):
    """Records each exchange at the transport boundary, after SDK serialization."""

    def __init__(self, inner: httpx2.BaseTransport) -> None:
        self.inner = inner
        self.exchanges: list[dict[str, Any]] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        request.read()
        entry: dict[str, Any] = {
            "started_at": utc_now(),
            "method": request.method,
            "url": str(request.url),
            "request_headers": dict(request.headers),
            "request_body": request.content,
        }
        self.exchanges.append(entry)
        try:
            response = self.inner.handle_request(request)
            response.read()
        except Exception as exc:
            entry["transport_error"] = f"{type(exc).__name__}: {exc}"
            raise
        entry.update(
            finished_at=utc_now(),
            status=response.status_code,
            response_headers=dict(response.headers),
            response_body=response.content,
        )
        return response

    def close(self) -> None:
        self.inner.close()


def _body(data: bytes | None) -> dict[str, Any]:
    if data is None:
        return {"sha256": None, "json": None}
    try:
        return {"sha256": sha256_bytes(data), "json": json.loads(data)}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"sha256": sha256_bytes(data), "text": data.decode("utf-8", errors="replace")}


def wire_record(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = redact([
        {
            **{k: v for k, v in e.items() if k not in ("request_body", "response_body")},
            "request_body": _body(e.get("request_body")),
            "response_body": _body(e.get("response_body")),
        }
        for e in exchanges
    ])
    return records
