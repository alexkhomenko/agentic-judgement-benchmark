"""Local web app for human checks: cycle through every response and approve or flag its judgement.

Serves one page on 127.0.0.1 and a small JSON API. Requests run on the serving thread, so the SQLite
connection is used from one thread only. Every API call must carry the per-session token embedded in the
page (a custom header, which a cross-origin page cannot send without a preflight this server never grants)
and a localhost Host header (against DNS rebinding). Candidate answers and model outputs are untrusted: the
page renders them as escaped text.
"""

from __future__ import annotations

import json
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import resources
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from ..storage.artifacts import ArtifactStore
from ..storage.database import Database
from .checks import CheckError, RunIndex, item_view, queue_view, record_check
from .runner import load_run

MAX_BODY = 256 * 1024


class ReviewSurface(Protocol):
    token: str
    page: bytes

    def queue(self) -> dict[str, Any]: ...
    def item(self, packet_id: str) -> dict[str, Any]: ...
    def submit(self, body: dict[str, Any]) -> dict[str, Any]: ...


class ReviewApp:
    def __init__(self, db: Database, store: ArtifactStore, evaluation_run_id: str, reviewer_id: str) -> None:
        self.db, self.store = db, store
        self.ctx = load_run(db, evaluation_run_id)
        self.reviewer_id = reviewer_id
        self.token = secrets.token_urlsafe(24)
        page = resources.files(__package__).joinpath("review_app.html").read_text(encoding="utf-8")
        self.page = page.replace("__REVIEW_TOKEN__", self.token).encode("utf-8")

    def index(self) -> RunIndex:
        return RunIndex.build(self.db, self.ctx)

    def queue(self) -> dict[str, Any]:
        return queue_view(self.index(), self.reviewer_id)

    def item(self, packet_id: str) -> dict[str, Any]:
        return item_view(self.index(), self.store, self.db, packet_id, self.reviewer_id)

    def submit(self, body: dict[str, Any]) -> dict[str, Any]:
        index = self.index()
        allowed = {"packet_id", "flags", "critical_failures", "note"}
        if not isinstance(body, dict) or set(body) - allowed:
            raise CheckError(f"a check has only the fields {sorted(allowed)}")
        record, created = record_check(self.db, self.store, index, {**body, "reviewer_id": self.reviewer_id})
        return {"check_id": record.record_id, "created": created, "queue": queue_view(index, self.reviewer_id)}


def make_handler(app: ReviewSurface, port_ref: list[int]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "bench-review"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet console
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; img-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: Any) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in {f"127.0.0.1:{port_ref[0]}", f"localhost:{port_ref[0]}"}

        def _api_ok(self) -> bool:
            if not self._host_ok() or not secrets.compare_digest(self.headers.get("X-Review-Token", ""), app.token):
                self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            url = urlsplit(self.path)
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            elif url.path == "/":
                self._send(HTTPStatus.OK, app.page, "text/html; charset=utf-8")
            elif url.path == "/report":
                query = parse_qs(url.query)
                if not secrets.compare_digest(query.get("token", [""])[0], app.token):
                    self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
                    return
                report = getattr(app, "report", None)
                if not callable(report):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "report not available"})
                    return
                try:
                    self._send(HTTPStatus.OK, report(query.get("packet", [""])[0]), "text/html; charset=utf-8")
                except CheckError as exc:
                    self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            elif not url.path.startswith("/api/"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            elif not self._api_ok():
                return
            elif url.path == "/api/queue":
                self._json(HTTPStatus.OK, app.queue())
            elif url.path == "/api/item":
                packet_id = parse_qs(url.query).get("packet", [""])[0]
                try:
                    self._json(HTTPStatus.OK, app.item(packet_id))
                except CheckError as exc:
                    self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            elif url.path == "/api/recovery":
                recovery = getattr(app, "recovery", None)
                if not callable(recovery):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "recovery not available"})
                    return
                try:
                    packet_id = parse_qs(url.query).get("packet", [""])[0]
                    self._json(HTTPStatus.OK, recovery(packet_id))
                except CheckError as exc:
                    self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            if urlsplit(self.path).path != "/api/check":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self._api_ok():
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send application/json"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "missing or oversized body"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                self._json(HTTPStatus.OK, app.submit(body))
            except (CheckError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return Handler


def serve(app: ReviewSurface, port: int = 0) -> tuple[HTTPServer, str]:
    """Bind on localhost (port 0 picks a free one). The caller runs `serve_forever()`."""
    port_ref = [port]
    server = HTTPServer(("127.0.0.1", port), make_handler(app, port_ref))
    port_ref[0] = server.server_address[1]
    return server, f"http://127.0.0.1:{port_ref[0]}/"
