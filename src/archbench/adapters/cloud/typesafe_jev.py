"""TypeSafe Jev adapter (System One API, `POST /v1/systemone`).

Jev is a classification model: it takes a `state` and typed `questions` and returns typed,
probabilistic answers. It has no conversation, no generation parameters, and no continuation.

The canonical request is two messages built by `evaluation.judging.structured_messages`: a system
message holding `{"questions": ...}` and a user message holding the state, both as JSON.

Evidence and control:
- Plain HTTP (documented REST API) through a capturing transport, so the artifact holds the exact wire
  exchange. No client-side retries: every retry is a new harness attempt.
- The model must be a pinned version (for example `jev-1.14.0`); aliases such as `jev-latest` are
  rejected because the answering model could change between calls. A response from another model is a
  failure.
- Output tokens are free per TypeSafe pricing; cost is input tokens times the configured price.
adapter_options:
    base_url (default https://api.typesafe.ai), timeout_s (default 30), long_context (bool),
    pricing: {input_usd_per_mtok, output_usd_per_mtok, source, verified_on}
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2

from ...domain.configuration import ModelConfiguration, ParameterState, Track
from ...domain.execution import CostIncurred, FailureKind, Observation
from ...domain.scenario import Message, Role
from ..base import (
    AdapterError,
    ArtifactSink,
    ContinuationSnapshot,
    ExtractedResponse,
    InvocationContext,
    NativeExecution,
    PreparedInvocation,
    QualificationCheck,
    QualificationReport,
    QualificationRequirements,
)
from .wire import CapturingTransport, wire_record

DEFAULT_BASE_URL = "https://api.typesafe.ai"
ALIASES = frozenset({"jev", "jev-latest", "jev-preview"})
TransportFactory = Callable[[], httpx2.BaseTransport]


class TypeSafeAdapter:
    name = "typesafe"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if config.adapter_options.get("long_context") is True:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        return httpx2.Client(
            base_url=str(config.adapter_options.get("base_url", DEFAULT_BASE_URL)),
            headers=headers,
            timeout=float(str(config.adapter_options.get("timeout_s", 30))),
            transport=transport,
        )

    def preflight(self, config: ModelConfiguration) -> str | None:
        if not self._api_key:
            return ("no TypeSafe credentials found: add typesafe.api_key to secrets.local.yaml "
                    "(see secrets.example.yaml) or set TYPESAFE_API_KEY")
        if config.model in ALIASES:
            return f"model {config.model!r} is an alias; pin a version such as jev-1.14.0"
        return None

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        explicit = sorted(n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "typesafe",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(name="pinned_model", passed=config.model not in ALIASES,
                               detail=f"model={config.model}"),
            QualificationCheck(name="parameter_support", passed=not explicit,
                               detail=f"Jev takes no generation parameters; explicit: {explicit}" if explicit else "ok"),
            self._model_check(config),
        ]
        if requirements.requires_continuation:
            checks.append(QualificationCheck(name="continuation", passed=False,
                                             detail="Jev is a classifier and has no conversation state"))
        return QualificationReport.from_checks(config, self.name, requirements, checks)

    def _model_check(self, config: ModelConfiguration) -> QualificationCheck:
        if not self._api_key:
            return QualificationCheck(name="model_identity", passed=None, detail="not evaluated: no credentials")
        try:
            with self._client(config, CapturingTransport(self._transport_factory())) as client:
                response = client.get("/v1/models")
            response.raise_for_status()
            listing = response.json()
        except (httpx2.HTTPError, ValueError) as exc:
            return QualificationCheck(name="model_identity", passed=False, detail=f"{type(exc).__name__}: {exc}")
        models = listing.get("data", listing) if isinstance(listing, dict) else listing
        names = {m.get("name") or m.get("id") for m in models if isinstance(m, dict)} if isinstance(models, list) else set()
        return QualificationCheck(name="model_identity", passed=config.model in names,
                                  detail=f"listed models: {sorted(n for n in names if n)}")

    # ----- invocation ------------------------------------------------------------------------------

    def prepare(
        self,
        config: ModelConfiguration,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        ctx: InvocationContext,
    ) -> PreparedInvocation:
        if parent is not None:
            raise AdapterError(FailureKind.INVALID_STATE, "Jev has no conversation to continue")
        if [m.role for m in messages] != [Role.SYSTEM, Role.USER]:
            raise AdapterError(FailureKind.INVALID_STATE, "expected a questions message and a state message")
        try:
            questions = json.loads(messages[0].content)["questions"]
            state = json.loads(messages[1].content)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AdapterError(FailureKind.INVALID_STATE, f"malformed structured request: {exc}") from None
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": "POST /v1/systemone",
                "client": {"package": "httpx2", "version": httpx2.__version__, "retries": 0},
                "body": {"model": config.model, "state": state, "questions": questions},
            },
        )

    def execute(
        self,
        config: ModelConfiguration,
        prepared: PreparedInvocation,
        ctx: InvocationContext,
        sink: ArtifactSink,
    ) -> NativeExecution:
        transport = CapturingTransport(self._transport_factory())
        try:
            return self._execute(config, prepared.native_request["body"], transport)
        finally:
            sink.write_json("wire-exchanges.json", wire_record(transport.exchanges))

    def _execute(self, config: ModelConfiguration, body: dict[str, Any], transport: CapturingTransport
                 ) -> NativeExecution:
        if not self._api_key:
            return _failed(FailureKind.PROVIDER_ERROR, "credentials", "no TypeSafe API key configured", CostIncurred.NO)
        try:
            with self._client(config, transport) as client:
                response = client.post("/v1/systemone", json=body)
        except httpx2.TimeoutException as exc:
            return _failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return _failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
        if response.status_code != 200:
            text = response.text
            if response.status_code == 429:
                kind = FailureKind.RATE_LIMITED
            elif response.status_code >= 500:  # includes 529 overloaded
                kind = FailureKind.TRANSIENT_TRANSPORT
            elif response.status_code == 400 and "max_tokens_exceeded" in text:
                kind = FailureKind.CONTEXT_EXCEEDED
            else:
                kind = FailureKind.PROVIDER_ERROR
            return _failed(kind, f"http_{response.status_code}", text[:2000], CostIncurred.NO,
                           status=response.status_code, request_id=request_id)

        try:
            payload = response.json()
        except ValueError:
            return _failed(FailureKind.PROVIDER_ERROR, "invalid_json", response.text[:2000], CostIncurred.YES,
                           request_id=request_id)
        observations = _observations(config, payload.get("usage") or {})
        if payload.get("model") != config.model:
            return NativeExecution(
                status="failed", native_response=payload, request_id=request_id, cost_incurred=CostIncurred.YES,
                failure_kind=FailureKind.PROVIDER_ERROR, observations=observations,
                error={"type": "model_mismatch", "requested": config.model, "served": payload.get("model")},
            )
        return NativeExecution(status="completed", native_response=payload, finish_reason="completed",
                               request_id=request_id, cost_incurred=CostIncurred.YES, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        body = execution.native_response
        if execution.status != "completed" or body is None or not isinstance(body.get("answers"), dict):
            raise AdapterError(FailureKind.HARNESS_ERROR, "no answers in Jev response")
        return ExtractedResponse(final_answer=json.dumps(body["answers"], sort_keys=True, ensure_ascii=False),
                                 truncated=False, finish_reason="completed")

    def fork_state(
        self, prepared: PreparedInvocation, execution: NativeExecution, extracted: ExtractedResponse
    ) -> ContinuationSnapshot:
        raise AdapterError(FailureKind.INVALID_STATE, "Jev has no conversation state")


def _failed(kind: FailureKind, error_type: str, message: str, cost: CostIncurred, **extra: Any) -> NativeExecution:
    return NativeExecution(status="failed", failure_kind=kind, cost_incurred=cost,
                           error={"type": error_type, "message": message, **extra},
                           request_id=extra.get("request_id"))


def _observations(config: ModelConfiguration, usage: dict[str, Any]) -> dict[str, Observation]:
    source = "typesafe usage"
    obs = {
        name: (Observation(value=int(usage[name]), unit="tokens", source=source) if usage.get(name) is not None
               else Observation.missing("tokens", source, "not reported in usage"))
        for name in ("input_tokens", "output_tokens")
    }
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        obs["api_cost_usd"] = Observation.missing("usd", "configuration pricing", "no pricing in configuration")
        return obs
    rate_in, rate_out = pricing.get("input_usd_per_mtok"), pricing.get("output_usd_per_mtok")
    tokens_in, tokens_out = obs["input_tokens"].value, obs["output_tokens"].value
    if not isinstance(rate_in, int | float) or not isinstance(rate_out, int | float) or tokens_in is None:
        obs["api_cost_usd"] = Observation.missing("usd", "configuration pricing", "price or input usage missing")
    else:
        total = tokens_in * rate_in / 1_000_000 + (tokens_out or 0) * rate_out / 1_000_000
        obs["api_cost_usd"] = Observation(value=round(total, 10), unit="usd",
                                          source=f"configuration pricing ({pricing.get('source', 'unspecified')})")
    return obs
