"""Native Amazon Bedrock adapter for Amazon's own models (Converse API, `POST /model/{modelId}/converse`).

AMAZON MODELS ONLY. Bedrock also serves Anthropic, Meta, Mistral and other vendors' models, but this harness
calls each vendor through its own first-party API, and the bias report maps `provider: bedrock` to the AWS
vendor. Qualification therefore rejects any model whose Bedrock provider is not Amazon (in practice: Nova).

Plain HTTP with a Bedrock API key (`Authorization: Bearer`), not boto3 and SigV4: the recorded request is
exactly the JSON body on the wire, and there are no hidden SDK retries (boto3 retries ModelNotReadyException
up to five times by default). Every retry is a new harness attempt.
- REGION: `adapter_options.region` is required; both endpoints are derived from it
  (`bedrock-runtime.{region}.amazonaws.com` for inference, `bedrock.{region}.amazonaws.com` for model lookups).
  API keys only work in the Regions AWS lists for them (no us-east-2 or us-west-1).
- CROSS-REGION ROUTING: inference profile ids (`us.`, `eu.`, `jp.`, `apac.`, `ca.`, `global.` prefixes) may be
  served from any Region in the profile. Nova 2 Lite requires one. Qualification records the profile's
  destination Regions; responses do not say which Region served a request (only CloudTrail does).
- MODEL IDENTITY: Converse responses do not name the model. Qualification checks the id against the control
  plane (`GET /inference-profiles/{id}`, then `GET /foundation-models/{id}`): Amazon provider, ACTIVE lifecycle,
  text output, and ON_DEMAND support for a bare model id. A profile id is proven by the profile lookup and the live
  probe; its `inferenceTypesSupported` (live: "INFERENCE_PROFILE", absent from AWS's documented enum) is only
  recorded. Individual responses cannot be checked.
- Only explicitly configured parameters are sent. Nova defaults: temperature 0.7, topP 0.9, maxTokens "dynamic",
  reasoning disabled.
- CONTENT MODERATION: Nova's built-in filters cannot be configured. A blocked prompt is HTTP 400
  ValidationException ("blocked by our content filters"); a blocked output is stopReason `content_filtered`
  or a 200 body with an `error` field. All are recorded as refusals. No guardrail is ever sent.
- Continuation replays the assistant message exactly as returned, including `reasoningContent` blocks (Nova 2
  returns their text as "[REDACTED]").

Cost uses configured prices. Bedrock reports cache reads and writes separately from `inputTokens`; the adapter
confirms this from `totalTokens` on each response and reports the cost as missing if the counts do not add up.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens   explicit, required*  -> inferenceConfig.maxTokens
    temperature         explicit or omitted  -> inferenceConfig.temperature
    top_p               explicit or omitted  -> inferenceConfig.topP
    top_k               explicit or omitted  -> additionalModelRequestFields.inferenceConfig.topK
    reasoning           explicit or omitted  -> additionalModelRequestFields.reasoningConfig.type ("enabled"|"disabled")
    reasoning_effort    explicit or omitted  -> additionalModelRequestFields.reasoningConfig.maxReasoningEffort
                                                ("low"|"medium"|"high"; requires reasoning "enabled")
    web_search          explicit or omitted  -> toolConfig.tools: [value] (Nova Web Grounding system tool, sent
                                                verbatim: {systemTool: {name: nova_grounding}})
    * With reasoning_effort "high", AWS requires maxTokens, temperature, topP and topK to be unset, so
      max_output_tokens must then be omitted.
adapter_options:
    region: AWS Region (required)                timeout_s: 3600 (default; AWS recommends at least 3600)
    max_output_tokens_limit: the model's documented maximum output (required)
    pricing: {input_usd_per_mtok, output_usd_per_mtok, cache_read_usd_per_mtok, cache_write_usd_per_mtok,
              web_search_usd_per_request?, source, verified_on}

NOVA WEB GROUNDING (verified 2026-11-24; https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html,
https://docs.aws.amazon.com/nova/latest/nova2-userguide/code-library.html#nova-grounding, AWS Price List API
AmazonBedrock us-east-1 publicationDate 2026-11-23 usage type USE1-Nova2.0Lite-nova-grounding):
- Only US Regions and US cross-region inference profiles (`us.` ids); qualification rejects other ids. The caller's
  IAM identity needs bedrock:InvokeTool on arn:aws:bedrock::{account}:system-tool/amazon.nova_grounding, which
  AmazonBedrockLimitedAccess (the policy behind Bedrock API keys) does not grant.
- The model decides whether to search and may search several times inside one Converse call.
- Billing unit is one "Request" ($0.03 for Nova 2 Lite in us-east-1). Converse usage reports no search count, so
  web_search_requests is 1 when the returned message shows grounding (citationsContent blocks, or toolUse /
  toolResult blocks for nova_grounding) and 0 otherwise; web_search_usd_per_request is the per-Request price.
- Grounded responses interleave `text` and `citationsContent` blocks; toolUse / toolResult blocks are accepted only
  for nova_grounding. The final answer is the text blocks plus any generated text inside citationsContent.content;
  citations stay in the native response, and the whole message is replayed unchanged on continuation (AWS does
  not document how system-tool blocks must be replayed).
- stopReason `tool_use`, or `end_turn` without final answer text after the last tool block, is an incomplete
  failure. `max_tokens` keeps the truncation handling.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote

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
from .common import CollectingSink, context_check, explicit, failed
from .wire import CapturingTransport, wire_record

STATE_FORMAT = "bedrock-converse"
DEFAULT_TIMEOUT_S = 3600
PROBE_MAX_TOKENS = 2048
# Regions where Bedrock API keys work (docs.aws.amazon.com/bedrock/latest/userguide/api-keys-supported.html).
API_KEY_REGIONS = frozenset({
    "ap-northeast-1", "ap-northeast-2", "ap-northeast-3", "ap-south-1", "ap-south-2", "ap-southeast-1",
    "ap-southeast-2", "ca-central-1", "eu-central-1", "eu-central-2", "eu-north-1", "eu-south-1", "eu-south-2",
    "eu-west-1", "eu-west-2", "eu-west-3", "sa-east-1", "us-east-1", "us-gov-east-1", "us-gov-west-1", "us-west-2",
})
PROFILE_PREFIX = re.compile(r"^(us|us-gov|eu|apac|jp|au|ca|global)\.")
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "temperature", "top_p", "top_k", "reasoning", "reasoning_effort",
                                  "web_search"})
CONTENT_TYPES = frozenset({"text", "reasoningContent"})
GROUNDING_TOOL = "nova_grounding"
GROUNDING_CONTENT_TYPES = CONTENT_TYPES | {"citationsContent", "toolUse", "toolResult"}
REDACTED_REASONING = "[REDACTED]"
CONTENT_FILTER_TEXT = "blocked by our content filters"
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no Amazon Bedrock API key found: add bedrock.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set AWS_BEARER_TOKEN_BEDROCK")


class BedrockAdapter:
    name = "bedrock"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport, *, plane: str) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        timeout = float(str(config.adapter_options.get("timeout_s", DEFAULT_TIMEOUT_S)))
        return httpx2.Client(base_url=_endpoint(config, plane), transport=transport, timeout=timeout,
                             headers={"authorization": f"Bearer {self._api_key}", "content-type": "application/json",
                                      "accept": "application/json"})

    def preflight(self, config: ModelConfiguration) -> str | None:
        return None if self._api_key else _NO_KEY

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        parameter_detail = _parameter_problem(config) or "ok"
        region = config.adapter_options.get("region")
        region_ok = isinstance(region, str) and region in API_KEY_REGIONS
        static = [
            QualificationCheck(
                name="adapter_identity",
                passed=config.adapter == self.name and config.track is Track.CLOUD and config.provider == "bedrock",
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider}",
            ),
            QualificationCheck(name="endpoint", passed=region_ok,
                               detail=_endpoint(config, "bedrock-runtime") if region_ok
                               else f"adapter_options.region {region!r} is not a Region that accepts Bedrock API keys"),
            QualificationCheck(name="first_party_model", passed=_base_model(config.model).startswith("amazon."),
                               detail=f"{config.model}: only Amazon's own models are benchmarked through Bedrock"),
            QualificationCheck(name="parameter_support", passed=parameter_detail == "ok", detail=parameter_detail),
        ]
        probes: list[dict[str, Any]] = []
        static_failures = [c.name for c in static if c.passed is not True]
        if static_failures:
            model_check = QualificationCheck(name="model_identity", passed=None,
                                             detail=f"not evaluated: {', '.join(static_failures)} failed")
        else:
            model_check = self._model_check(config, probes)
        checks = [*static, model_check, self._context_check(config, requirements)]
        live_name = "continuation" if requirements.requires_continuation else "chat_probe"
        if model_check.passed is True:
            checks.append(self._live_check(config, probes, branches=requirements.requires_continuation))
        else:
            checks.append(QualificationCheck(name=live_name, passed=None,
                                             detail="not evaluated: model identity was not verified"))
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    @staticmethod
    def _context_check(config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationCheck:
        """With max_output_tokens omitted (required at high reasoning effort), maxTokens is unset and the model may
        generate up to its documented maximum, so that maximum is reserved for each answer."""
        limit = _output_limit(config)
        if requirements.output_budget_tokens is not None or limit is None:
            return context_check(config, requirements, None)
        check = context_check(config, requirements.model_copy(update={"output_budget_tokens": limit}), None)
        return check.model_copy(update={"detail": f"{check.detail}; max_output_tokens omitted, so each answer "
                                                  f"reserves max_output_tokens_limit ({limit})"})

    def _model_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Resolve an inference profile to its foundation model, then verify that model on the control plane."""
        transport = CapturingTransport(self._transport_factory())
        problems: list[str] = []
        details: list[str] = []
        base = _base_model(config.model)
        try:
            with self._client(config, transport, plane="bedrock") as client:
                if base != config.model:
                    response = client.get(f"inference-profiles/{_path(config.model)}")
                    if response.status_code != 200:
                        return QualificationCheck(name="model_identity", passed=False,
                                                  detail=f"inference profile: HTTP {response.status_code}: "
                                                         f"{_error_message(response)}")
                    profile = _error_body(response)  # the JSON object body, or {} if it is not one
                    probes.append({"step": "inference_profile", "profile": profile})
                    arns = [str(m.get("modelArn", "")) for m in profile.get("models") or [] if isinstance(m, dict)]
                    if profile.get("status") != "ACTIVE":
                        problems.append(f"inference profile status {profile.get('status')!r}")
                    if not arns or any(not a.endswith(f"foundation-model/{base}") for a in arns):
                        problems.append(f"inference profile routes to {arns}, not only {base}")
                    regions = sorted({a.split(":")[3] for a in arns if a.count(":") >= 5})
                    details.append(f"profile {config.model} ({profile.get('type')}) may be served from {regions}")
                response = client.get(f"foundation-models/{_path(base)}")
        except (AdapterError, httpx2.HTTPError) as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}")
        finally:
            probes.append({"step": "model_lookup", "wire": wire_record(transport.exchanges)})
        if response.status_code != 200:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"foundation model: HTTP {response.status_code}: {_error_message(response)}")
        model = response.json().get("modelDetails") or {}
        probes.append({"step": "foundation_model", "model": model})
        lifecycle = (model.get("modelLifecycle") or {}).get("status")
        invocation = model.get("inferenceTypesSupported") or []
        if model.get("modelId") != base:
            problems.append(f"control plane names {model.get('modelId')!r}, not {base!r}")
        if model.get("providerName") != "Amazon":
            problems.append(f"model provider is {model.get('providerName')!r}, not Amazon")
        if lifecycle != "ACTIVE":
            problems.append(f"model lifecycle is {lifecycle!r}")
        if "TEXT" not in (model.get("outputModalities") or []):
            problems.append("model does not output text")
        # A bare model id needs ON_DEMAND (documented). For a profile id, the profile lookup above and the live probe
        # are the evidence: the service reports "INFERENCE_PROFILE" here, but AWS does not document that value, so it
        # is recorded, not required.
        if base == config.model and "ON_DEMAND" not in invocation:
            problems.append(f"model does not support ON_DEMAND invocation (supports {invocation})")
        details.insert(0, f"{model.get('modelName')} ({base}): provider={model.get('providerName')}, "
                          f"lifecycle={model.get('modelLifecycle')}, inference_types={invocation}")
        return QualificationCheck(name="model_identity", passed=not problems,
                                  detail="; ".join(problems) or "; ".join(details))

    def _live_check(self, config: ModelConfiguration, probes: list[dict[str, Any]], *, branches: bool
                    ) -> QualificationCheck:
        """Live probe with the configured parameters: one baseline, then (when the scenario branches) two
        sibling follow-ups forked from the same snapshot."""
        name = "continuation" if branches else "chat_probe"
        budget = explicit(config, "max_output_tokens")
        probe = config
        if isinstance(budget, int):
            params = dict(config.parameters)
            params["max_output_tokens"] = params["max_output_tokens"].model_copy(
                update={"value": min(budget, PROBE_MAX_TOKENS)})
            probe = config.model_copy(update={"parameters": params})

        def run(step: str, messages: tuple[Message, ...], parent: ContinuationSnapshot | None
                ) -> tuple[PreparedInvocation, NativeExecution]:
            ctx = InvocationContext(experiment_id="qualification", task_id=f"qualification:{config.key}",
                                    attempt_id=step, attempt_no=1, round=1 if parent is None else 2,
                                    purpose="qualification")
            sink = CollectingSink()
            prepared = self.prepare(probe, messages, parent, ctx)
            execution = self.execute(probe, prepared, ctx, sink)
            cost = execution.observations.get("api_cost_usd")
            probes.append({
                "step": step, "status": execution.status, "request_id": execution.request_id,
                "failure_kind": execution.failure_kind.value if execution.failure_kind else None,
                "cost_incurred": execution.cost_incurred.value,
                "cost_usd": cost.value if cost else None,
                "max_output_tokens": explicit(probe, "max_output_tokens"),
                "wire": sink.items.get("wire-exchanges.json"),
            })
            return prepared, execution

        base_messages = (Message(role=Role.SYSTEM, content="You are answering a harness qualification probe."),
                         Message(role=Role.USER, content="Reply with exactly one word: baseline"))
        try:
            prepared, execution = run("baseline", base_messages, None)
            if execution.status != "completed":
                return QualificationCheck(name=name, passed=False,
                                          detail=f"baseline probe failed: {execution.failure_kind} {execution.error}")
            returned = _output_message(execution.native_response or {})
            reasoning = any("reasoningContent" in b for b in returned.get("content") or [])
            if not branches:
                return QualificationCheck(name=name, passed=True,
                                          detail="baseline completed with the configured parameters"
                                                 + (" (with reasoning blocks)" if reasoning else ""))
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = json.dumps(snapshot.native_state, sort_keys=True)
            base_sent = prepared.native_request["body"]["messages"]
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = run(f"branch-{word}", follow, snapshot)
                sent = child.native_request["body"]["messages"]
                if sent[: len(base_sent) + 1] != base_sent + [returned]:
                    return QualificationCheck(name="continuation", passed=False,
                                              detail=f"branch {word} did not replay the baseline message exactly")
                if word == "beta" and "alpha" in json.dumps(sent):
                    return QualificationCheck(name="continuation", passed=False, detail="sibling branch leaked")
                if result.status != "completed":
                    return QualificationCheck(name="continuation", passed=False,
                                              detail=f"branch {word} failed: {result.failure_kind} {result.error}")
            if json.dumps(snapshot.native_state, sort_keys=True) != before:
                return QualificationCheck(name="continuation", passed=False, detail="forking mutated baseline state")
        except AdapterError as exc:
            return QualificationCheck(name=name, passed=False, detail=str(exc))
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact message replay"
                   + (" (including reasoning blocks)" if reasoning else ""),
        )

    # ----- invocation ------------------------------------------------------------------------------

    def prepare(
        self,
        config: ModelConfiguration,
        messages: tuple[Message, ...],
        parent: ContinuationSnapshot | None,
        ctx: InvocationContext,
    ) -> PreparedInvocation:
        max_tokens = explicit(config, "max_output_tokens")
        if max_tokens is None and explicit(config, "reasoning_effort") != "high":
            raise AdapterError(FailureKind.INVALID_STATE, "max_output_tokens must be an explicit integer")
        if parent is not None:
            parent.verify_shape()
            state = parent.native_state
            if state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if state.get("model") != config.model:
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state belongs to a different model")
            if messages[: len(parent.messages)] != parent.messages:
                raise AdapterError(FailureKind.INVALID_STATE, "conversation does not extend the baseline snapshot")
            system = copy.deepcopy(state["system"])
            native = copy.deepcopy(state["messages"])
            new, leading = messages[len(parent.messages):], 0
        else:
            leading = next((i for i, m in enumerate(messages) if m.role is not Role.SYSTEM), len(messages))
            system = [{"text": m.content} for m in messages[:leading]]
            native, new = [], messages[leading:]
            leading = 0
        if any(m.role is Role.SYSTEM for m in new[leading:]):
            raise AdapterError(FailureKind.INVALID_STATE, "system messages must lead the conversation")
        for m in new:
            if m.role is Role.ASSISTANT:
                raise AdapterError(FailureKind.INVALID_STATE,
                                   "assistant turns must come from a continuation snapshot, not plain text")
            native.append({"role": m.role.value, "content": [{"text": m.content}]})
        inference: dict[str, Any] = {}
        if max_tokens is not None:
            inference["maxTokens"] = max_tokens
        if (value := explicit(config, "temperature")) is not None:
            inference["temperature"] = value
        if (value := explicit(config, "top_p")) is not None:
            inference["topP"] = value
        body: dict[str, Any] = {"messages": native}
        if system:
            body["system"] = system
        if inference:
            body["inferenceConfig"] = inference
        extra: dict[str, Any] = {}
        reasoning: dict[str, Any] = {}
        if (kind := explicit(config, "reasoning")) is not None:
            reasoning["type"] = kind
        if (effort := explicit(config, "reasoning_effort")) is not None:
            reasoning["maxReasoningEffort"] = effort
        if reasoning:
            extra["reasoningConfig"] = reasoning
        if (top_k := explicit(config, "top_k")) is not None:
            extra["inferenceConfig"] = {"topK": top_k}
        if extra:
            body["additionalModelRequestFields"] = extra
        if (web_search := explicit(config, "web_search")) is not None:
            if problem := _web_search_problem(web_search):
                raise AdapterError(FailureKind.INVALID_STATE, problem)
            body["toolConfig"] = {"tools": [copy.deepcopy(web_search)]}
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": f"POST /model/{_path(config.model)}/converse",
                "region": config.adapter_options.get("region"),
                "model_id": config.model,
                "client": {"package": "httpx2", "version": httpx2.__version__, "retries": 0},
                "body": body,
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
            return self._execute(config, prepared, transport)
        finally:
            sink.write_json("wire-exchanges.json", wire_record(transport.exchanges))

    def _execute(self, config: ModelConfiguration, prepared: PreparedInvocation, transport: CapturingTransport
                 ) -> NativeExecution:
        body = prepared.native_request["body"]
        try:
            with self._client(config, transport, plane="bedrock-runtime") as client:
                response = client.post(f"model/{_path(config.model)}/converse", content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        request_id = response.headers.get("x-amzn-requestid")
        if response.status_code != 200:
            error_type, message = _error_type(response), _error_message(response)
            kind, cost = _http_failure(response.status_code, error_type, message)
            return failed(kind, error_type or f"http_{response.status_code}", message, cost,
                          status=response.status_code, request_id=request_id)
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        data = response.json()
        searching = "toolConfig" in body
        observations = _observations(config, data, searching=searching)
        reason = data.get("stopReason")

        def failure(kind: FailureKind, error: dict[str, Any], cost: CostIncurred = CostIncurred.YES
                    ) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=request_id,
                                   finish_reason=reason, cost_incurred=cost, failure_kind=kind,
                                   observations=observations, error=error)

        if data.get("error"):  # Nova's output filter can answer 200 with an error instead of a message
            error_text = str(data["error"])[:500]
            kind = FailureKind.REFUSAL if CONTENT_FILTER_TEXT in error_text.lower() else FailureKind.PROVIDER_ERROR
            return failure(kind, {"type": "response_error", "message": error_text})
        if reason in ("content_filtered", "guardrail_intervened"):
            return failure(FailureKind.REFUSAL, {"type": "stop_reason", "reason": reason})
        if reason == "model_context_window_exceeded":
            return failure(FailureKind.CONTEXT_EXCEEDED, {"type": "stop_reason", "reason": reason})
        if searching and reason == "tool_use":
            return failure(FailureKind.PROVIDER_ERROR, {"type": "incomplete", "reason": reason,
                                                        "detail": "response ended on a tool use without a final answer"})
        if reason not in ("end_turn", "max_tokens"):  # tool_use, stop_sequence (none sent), malformed_*, undocumented
            return failure(FailureKind.PROVIDER_ERROR, {"type": "stop_reason", "reason": reason})
        returned = _output_message(data)
        content = returned.get("content")
        if returned.get("role") != "assistant" or not isinstance(content, list) or not content:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_message"})
        valid = _grounded_block if searching else _plain_block
        if not all(valid(b) for b in content):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "unexpected_content", "content_types": sorted(
                {",".join(sorted(b)) if isinstance(b, dict) else type(b).__name__ for b in content})})
        if searching and reason == "end_turn":
            last_tool = max((i for i, b in enumerate(content) if "toolUse" in b or "toolResult" in b), default=-1)
            if not _answer_text(content[last_tool + 1:]):
                return failure(FailureKind.PROVIDER_ERROR, {"type": "incomplete", "reason": reason,
                                                            "detail": "response ended without final answer text"})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason,
                               request_id=request_id, cost_incurred=CostIncurred.YES, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        content = _output_message(data).get("content") or []
        text = _answer_text(content)
        thoughts = [str((b["reasoningContent"].get("reasoningText") or {}).get("text") or "")
                    for b in content if "reasoningContent" in b]
        visible = "".join(t for t in thoughts if t != REDACTED_REASONING)
        return ExtractedResponse(
            final_answer=text,
            truncated=data.get("stopReason") == "max_tokens",
            finish_reason=data.get("stopReason"),
            reasoning_text=visible or None,  # Nova 2 redacts its reasoning; the blocks stay in the native response
        )

    def fork_state(
        self, prepared: PreparedInvocation, execution: NativeExecution, extracted: ExtractedResponse
    ) -> ContinuationSnapshot:
        data = execution.native_response or {}
        body = prepared.native_request["body"]
        model = str(prepared.native_request["model_id"])
        return ContinuationSnapshot(
            adapter=self.name,
            configuration_key=model,
            messages=prepared.messages + (Message(role=Role.ASSISTANT, content=extracted.final_answer),),
            native_state={
                "format": STATE_FORMAT,
                "model": model,
                "region": prepared.native_request.get("region"),
                "request_id": execution.request_id,
                "system": copy.deepcopy(body.get("system") or []),
                # The assistant message (reasoning and text blocks) is replayed exactly as returned.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(_output_message(data))],
            },
        )


def _endpoint(config: ModelConfiguration, plane: str) -> str:
    return f"https://{plane}.{config.adapter_options.get('region')}.amazonaws.com/"


def _base_model(model: str) -> str:
    """The foundation model id behind an inference profile id (`us.amazon.nova-2-lite-v1:0` -> `amazon.nova-2-lite-v1:0`)."""
    return PROFILE_PREFIX.sub("", model, count=1)


def _path(model: str) -> str:
    return quote(model, safe=":")


def _output_message(data: Mapping[str, Any]) -> dict[str, Any]:
    output = data.get("output")
    message = output.get("message") if isinstance(output, dict) else None
    return message if isinstance(message, dict) else {}


def _plain_block(block: Any) -> bool:
    return isinstance(block, dict) and len(block) == 1 and next(iter(block)) in CONTENT_TYPES


def _grounded_block(block: Any) -> bool:
    """A content block allowed in a grounded response. The Nova 2 guide's example puts `text` and
    `citationsContent` in one block (the API reference says ContentBlock is a union), so both forms pass."""
    if not isinstance(block, dict) or not block or not set(block) <= GROUNDING_CONTENT_TYPES:
        return False
    if len(block) > 1:
        return set(block) == {"text", "citationsContent"}
    tool_use = block.get("toolUse")
    return tool_use is None or (isinstance(tool_use, dict) and tool_use.get("name") == GROUNDING_TOOL)


def _answer_text(content: list[Any]) -> str:
    """Final answer: text blocks, plus generated text a citationsContent block carries in its `content`."""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            parts.append(str(block["text"]))
        elif isinstance(cited := block.get("citationsContent"), dict):
            parts += [str(c["text"]) for c in cited.get("content") or [] if isinstance(c, dict) and "text" in c]
    return "".join(parts)


def _grounded(content: Any) -> bool:
    """Whether the returned message shows that Web Grounding ran (citations or a nova_grounding tool use)."""
    return isinstance(content, list) and any(
        isinstance(b, dict) and ("citationsContent" in b or (
            isinstance(b.get("toolUse"), dict) and b["toolUse"].get("name") == GROUNDING_TOOL)) for b in content)


def _web_search_problem(value: Any) -> str | None:
    tool = value.get("systemTool") if isinstance(value, dict) and len(value) == 1 else None
    if not isinstance(tool, dict) or tool.get("name") != GROUNDING_TOOL:
        return f"web_search must be the Nova grounding system tool: {{systemTool: {{name: {GROUNDING_TOOL}}}}}"
    return None


def _parameter_problem(config: ModelConfiguration) -> str | None:
    if config.model.startswith("arn:"):
        return "use a model or inference profile id, not an ARN"
    names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
    if unsupported := sorted(names - SUPPORTED_PARAMETERS):
        return f"unsupported: {unsupported}"
    if "web_search" in names:
        if problem := _web_search_problem(explicit(config, "web_search")):
            return problem
        if not config.model.startswith("us."):
            return "web_search (Nova Web Grounding) is supported only through US cross-region inference profiles"
    reasoning, effort = explicit(config, "reasoning"), explicit(config, "reasoning_effort")
    if reasoning is not None and reasoning not in ("enabled", "disabled"):
        return "reasoning must be 'enabled' or 'disabled'"
    if effort is not None:
        if reasoning != "enabled":
            return "reasoning_effort requires reasoning: enabled"
        if effort not in ("low", "medium", "high"):
            return "reasoning_effort must be 'low', 'medium' or 'high'"
    if reasoning == "enabled" and effort is None:
        return "reasoning: enabled requires reasoning_effort (Bedrock rejects reasoningConfig without it)"
    limit = _output_limit(config)
    if limit is None:
        return "adapter_options.max_output_tokens_limit must state the model's maximum output"
    if effort == "high":
        if clash := sorted(names & {"max_output_tokens", "temperature", "top_p", "top_k"}):
            return f"reasoning_effort high requires these to be omitted: {clash}"
        return None
    budget = explicit(config, "max_output_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= limit:
        return f"max_output_tokens must be an explicit integer in 1..{limit}"
    return None


def _output_limit(config: ModelConfiguration) -> int | None:
    limit = config.adapter_options.get("max_output_tokens_limit")
    return None if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 else limit


def _error_body(response: httpx2.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def _error_type(response: httpx2.Response) -> str | None:
    """AWS REST-JSON errors name the exception in `x-amzn-ErrorType` (sometimes suffixed with `:<namespace>`)."""
    raw = response.headers.get("x-amzn-errortype") or _error_body(response).get("__type")
    return str(raw).split(":")[0].split("#")[-1] if raw else None


def _error_message(response: httpx2.Response) -> str:
    body = _error_body(response)
    message = body.get("message") or body.get("Message")
    return str(message)[:500] if message else response.text[:500]


def _http_failure(status: int, error_type: str | None, message: str = "") -> tuple[FailureKind, CostIncurred]:
    text = message.lower()
    if error_type == "ThrottlingException" or (status == 429 and error_type != "ModelNotReadyException"):
        return FailureKind.RATE_LIMITED, CostIncurred.NO
    if status == 400 and CONTENT_FILTER_TEXT in text:  # Nova's input filter; whether it is billed is not documented
        return FailureKind.REFUSAL, CostIncurred.UNKNOWN
    if status == 400 and ("too long" in text or "context window" in text or "context length" in text):
        return FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO
    if error_type == "ModelTimeoutException" or status in (408, 504):
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if error_type == "ModelErrorException" or status == 424:  # the model ran and failed
        return FailureKind.PROVIDER_ERROR, CostIncurred.UNKNOWN
    if error_type == "ModelNotReadyException" or status in (500, 502, 503, 529):
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    # 400 validation, 403 access denied or expired key, 404 unknown model, 413 too large: retrying cannot help
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, data: Mapping[str, Any], *, searching: bool = False
                  ) -> dict[str, Observation]:
    source = "bedrock usage"
    raw = data.get("usage")
    usage: dict[str, Any] = raw if isinstance(raw, dict) else {}
    values = {
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
        "total_tokens": usage.get("totalTokens"),
        # Optional fields, omitted when no cache was read or written.
        "cached_input_tokens": usage.get("cacheReadInputTokens", 0 if usage else None),
        "cache_write_input_tokens": usage.get("cacheWriteInputTokens", 0 if usage else None),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if isinstance(v, int | float)
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    metrics = data.get("metrics")
    latency = metrics.get("latencyMs") if isinstance(metrics, dict) else None
    obs["provider_latency_ms"] = (Observation(value=int(latency), unit="ms", source="bedrock metrics.latencyMs")
                                  if isinstance(latency, int | float)
                                  else Observation.missing("ms", "bedrock metrics", "not reported"))
    if searching:
        grounded = _grounded(_output_message(data).get("content"))
        obs["web_search_requests"] = Observation(
            value=int(grounded), unit="requests",
            source="bedrock output message: 1 grounded request if citationsContent or nova_grounding toolUse blocks "
                   "are present (billed per Request; usage reports no search count)")
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")
    parts = [(obs["input_tokens"].value, "input_usd_per_mtok"),
             (obs["cached_input_tokens"].value, "cache_read_usd_per_mtok"),
             (obs["cache_write_input_tokens"].value, "cache_write_usd_per_mtok"),
             (obs["output_tokens"].value, "output_usd_per_mtok")]
    if not all(isinstance(tokens, int) for tokens, _ in parts):
        return Observation.missing("usd", "configuration pricing", "token counts not reported")
    total_tokens = obs["total_tokens"].value
    if isinstance(total_tokens, int) and total_tokens != sum(int(t or 0) for t, _ in parts):
        return Observation.missing("usd", "configuration pricing",
                                   "totalTokens does not equal input + cache read + cache write + output")
    total = 0.0
    for tokens, key in parts:
        if not tokens:
            continue
        rate = pricing.get(key)
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", f"no price for {key}")
        total += int(tokens) * rate / 1_000_000
    if (searches := obs.get("web_search_requests")) and searches.value:
        rate = pricing.get("web_search_usd_per_request")
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", "no price for web_search_requests")
        total += searches.value * rate
    return Observation(value=round(total, 10), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')})")
