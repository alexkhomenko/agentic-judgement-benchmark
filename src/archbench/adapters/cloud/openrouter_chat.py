"""OpenRouter chat-completions adapter, pinned to the model vendor's own serving infrastructure
(`POST https://openrouter.ai/api/v1/chat/completions`, `Authorization: Bearer`).

This adapter exists for vendors whose first-party API is unavailable to us (Qwen: the Alibaba Model Studio
account is closed; Gemini: the direct Google key's quota is too small). It reaches the SAME vendor's servers through
OpenRouter and nothing else: `provider` in the configuration names the vendor (so the bias report attributes results
to it), and only the routes in `ROUTES` are allowed, each the vendor's own OpenRouter provider:
    alibaba -> `alibaba` ("Alibaba")
    google  -> `google-ai-studio` ("Google AI Studio"): the Gemini Developer API the direct gemini adapter calls.
               Never Vertex (`google-vertex`, reported as "Google"). AI Studio also lists flex and priority tier
               endpoints; a base slug never routes to a closed-weight model's tier endpoints (provider-selection and
               service-tiers docs, read 2026-11-25), so qualification ignores them, a tier-suffixed provider_endpoint
               is rejected, and a response reporting any `service_tier` other than "default" is a failure.
`VENDORS` holds what differs per vendor: the efforts that map onto the vendor's native control, whether web search
can run on the vendor's own engine, and the shape of the body OpenRouter forwards (see UPSTREAM CHECK). Every
request carries
`provider: {only: [<vendor slug>], allow_fallbacks: false, require_parameters: true}`: no other provider, no
fallback, and no routing at all if a sent parameter is not supported. Model-level fallbacks (`models`, `route`)
and model variants (`:nitro`, `:floor`, ...) are never used.

Plain HTTP against the documented REST API: the recorded request is exactly the JSON body on the wire.
- No client-side retries: every retry is a new harness attempt. OpenRouter may retry within the pinned provider;
  `openrouter_metadata.attempt` (kept in the native response) shows when it did.
- Every request opts into routing metadata (`X-OpenRouter-Metadata: enabled`). The served provider is read from
  `openrouter_metadata` (the selected endpoint, else the last attempt) and must be the pinned vendor; successful
  responses carry no documented top-level `provider`. A response that does not say who served it is a failure.
- Only explicitly configured parameters are sent; omitted ones are left to the provider. OpenRouter's own default
  when `reasoning` is omitted is undocumented, so a native default can only be reproduced by sending it.
- Reasoning arrives as `message.reasoning` and `message.reasoning_details`. Continuation replays the assistant
  message exactly as returned. Whether that reaches the vendor is not documented: qualification checks it.
- UPSTREAM CHECK (Alibaba rules below; Google next): OpenRouter translates the request into the vendor's format.
  Qualification sends one streamed follow-up with `debug.echo_upstream_body` (streaming only, best-effort
  redacted) and checks the body OpenRouter
  forwarded: the vendor model id must be `adapter_options.upstream_model`; parameters the configuration omits must
  not be injected; an output cap (`max_tokens` or `max_completion_tokens`) must be present and unchanged; the
  replayed answer and reasoning must be forwarded with the baseline's exact text; `enable_search` must not be
  on; and every key in `adapter_options.upstream_expect` must match. Reasoning controls (enable_thinking,
  reasoning_effort, thinking_budget, preserve_thinking) must follow from the configuration: the configured effort,
  else an omitted parameter's documented default. A configured effort must reach the vendor as a control that
  carries its level (reasoning_effort, or a pinned thinking_budget); enable_thinking alone carries only effort
  "none", so an effort reduced to the thinking switch fails. Any forwarded key the configuration does
  not determine (e.g. a thinking_budget OpenRouter derived from the effort) leaves the check NOT EVALUATED, so
  qualification cannot pass, until it is checked against the vendor's docs and pinned in `upstream_expect`.
  A value the echo hides (e.g. "<included to upstream but not in debug info>") stays unverified whatever is
  pinned, and pinning the placeholder itself fails; only a hidden enable_thinking is checked, by behaviour.
  The echo is recorded in the probes.
- UPSTREAM CHECK, Google: OpenRouter forwards a Gemini generateContent body (a body without `contents` leaves the
  check NOT EVALUATED). Checked: the Gemini model id (`model`, "models/" prefix ignored) is upstream_model, and an
  echo without one stays unverified; generationConfig.maxOutputTokens is present and unchanged; no sampling value
  (temperature, topP, topK, presencePenalty, frequencyPenalty, seed) the configuration omits is forwarded;
  generationConfig.thinkingConfig.thinkingLevel is the configured effort (OpenRouter maps reasoning.effort onto
  thinkingLevel one to one: reasoning-tokens docs), and with the effort omitted it is absent or its documented
  default; the replayed model turn carries the baseline answer and every thought signature the baseline returned
  (`reasoning.encrypted` details, except OpenRouter's `gemini-continuation` token, which it drops from earlier
  turns), with any forwarded thought text equal to the baseline's; and `tools` is exactly one Google Search
  grounding tool (googleSearch / google_search) when web_search is configured and has none otherwise. Any other
  forwarded value (e.g. thinkingConfig.includeThoughts, thinkingBudget, safetySettings) leaves the check NOT
  EVALUATED until pinned in upstream_expect, whose keys are dotted camelCase paths here (e.g.
  `generationConfig.thinkingConfig.includeThoughts`; Gemini accepts snake_case too, so the echo's keys are
  normalised to camelCase before checking). If OpenRouter sends no echo for Google, the check stays NOT
  EVALUATED, so qualification cannot pass on behaviour alone; the per-call checks below still run.
- Cancellation is not supported for Alibaba: an abandoned request runs, and is billed, to completion. Google
  requests are treated the same way (a timeout is an ambiguous, possibly billed failure).
- Refusals: OpenRouter moderation (HTTP 403 with flagged metadata), a vendor content block (Alibaba's
  `data_inspection_failed`), or finish_reason `content_filter`.
- Google reasoning: Gemini may return reasoning only as thought signatures (`reasoning.encrypted` details, format
  google-gemini-v1) without readable text. A signature is replayable reasoning, so a response with reasoning
  tokens and signatures but no text is complete; with neither it is a failure as for Alibaba.

WEB SEARCH (Google only): `web_search` is OpenRouter's `openrouter:web_search` server tool, sent verbatim in
`tools` (the `plugins: [{id: web}]` plugin and `:online` are deprecated and never used). It must pin
`parameters: {engine: native}` and nothing else: `auto` (the default) falls back to Exa, other engines are not
Google, and Google's native tool takes no domain filters (OpenRouter answers 400 with engine native). Because even
`native` falls back to another engine when the endpoint lacks native search, Google-ness is proven three ways:
qualification requires the pinned endpoint's `native_tools["openrouter:web_search"].type` to be google_search and
the forwarded body to carry the googleSearch tool; and every response with `usage.server_tool_use_details.
tool_calls_requested` > 0 (searches OpenRouter orchestrated itself; native searches are not counted there) or a
routing-metadata `server_tools` pipeline stage whose mode is not "native" is a failure (an absent mode is not
evidence either way). `web_search_requests` is `usage.server_tool_use_details.web_search_requests` ("search
queries the model made", server-tools docs); with no count it is 0 unless the message carries citations, in which
case the count and the cost are missing. Google bills Gemini 3+ grounding per search query ($14 / 1,000, the
OpenRouter endpoint's `pricing.web_search` 0.014); whether OpenRouter's count is Google's billed query count
(unique, non-empty queries) is not documented, so compare `openrouter_charge` (usage.cost, which includes server
tool charges) with api_cost_usd in the pilot. A search response that ends on a tool call, with finish_reason
`tool_calls`, a native finish of TOO_MANY_TOOL_CALLS / MALFORMED_FUNCTION_CALL / UNEXPECTED_TOOL_CALL, or `stop`
without answer text is an incomplete failure, never a partial answer. The final answer is the message text only;
`annotations` (url_citation) stay in the native response. Continuation replays the assistant message exactly as
returned, annotations included: OpenRouter's request schema has no assistant `annotations` field, and whether it
ignores or forwards them is shown by the upstream check (the forwarded model turn must hold the baseline answer
text only); for that reason the search configuration's upstream baseline asks the model to search.

Cost uses configured prices, which qualification compares with the endpoint's live OpenRouter pricing (and, when
listed, requires `internal_reasoning` to equal the output price, since reasoning is costed as output).
`usage.prompt_tokens` is the total input: cache reads (`prompt_tokens_details.cached_tokens`), cache writes
(`prompt_tokens_details.cache_write_tokens`, reported only for explicit caching, which this adapter never
requests) and uncached input sum to it. OpenRouter's own charge (`usage.cost`, in credits) is recorded beside it.
Searches are priced with `web_search_usd_per_request`; searches without that rate leave the cost missing.

Configuration parameters (all others are rejected at qualification):
    max_output_tokens  explicit, required   -> max_tokens (includes reasoning tokens)
    reasoning_effort   explicit or omitted  -> reasoning.effort (Google: minimal, low, medium, high, xhigh, the
                                               Gemini thinking levels; none and max are rejected)
    temperature, top_p, top_k, presence_penalty   explicit or omitted -> same names
    web_search         explicit or omitted  -> tools: [value] (Google only; {type: openrouter:web_search,
                                               parameters: {engine: native}})
adapter_options:
    provider_endpoint: optional vendor endpoint slug with suffix, e.g. alibaba/us-east-1; must belong to vendor
    endpoint_model: OpenRouter's endpoint label, e.g. qwen4-max-20261115 (required; listing name, routing metadata)
    upstream_model: the vendor's model id OpenRouter forwards, from the debug echo (required; may differ from the
                    label: Qwen4-Max is forwarded as the moving `qwen4-max` alias)
    upstream_expect: {key: value} that must appear in the forwarded body (optional)
    timeout_s: 900 (default)
    pricing: {input_usd_per_mtok, cached_input_usd_per_mtok, cache_write_usd_per_mtok, output_usd_per_mtok,
              web_search_usd_per_request?, source, verified_on}
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
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
from .common import CollectingSink, context_check, explicit, failed
from .wire import CapturingTransport, wire_record


@dataclass(frozen=True)
class Route:
    slug: str  # OpenRouter provider slug, sent in provider.only
    name: str  # provider name OpenRouter reports (endpoints JSON provider_name, routing metadata)


# configuration `provider` (the vendor) -> the vendor's own OpenRouter provider. Never a third-party host.
# Google AI Studio is the Gemini Developer API (generativelanguage.googleapis.com), the service the direct gemini
# adapter calls; Vertex AI ("Google", google-vertex/*) is excluded. Slugs and names from
# /api/v1/models/google/<model>/endpoints, read 2026-11-25.
ROUTES = {"alibaba": Route(slug="alibaba", name="Alibaba"),
          "google": Route(slug="google-ai-studio", name="Google AI Studio")}

OPENAI_CHAT_BODY = "openai-chat"  # the vendor receives an OpenAI-style chat body (Alibaba)
GEMINI_BODY = "gemini-generate-content"  # the vendor receives a Gemini generateContent body (Google)


@dataclass(frozen=True)
class Vendor:
    """What differs per vendor behind its OpenRouter route."""
    upstream_body: str  # shape of the body OpenRouter forwards, which decides how the upstream check reads it
    efforts: frozenset[str]  # reasoning_effort values that map onto the vendor's native control
    native_search: frozenset[str]  # native_tools types that prove web search runs on the vendor's engine
    tier_endpoints_opt_in: bool  # tier endpoints (flex, priority) never serve a base-slug request
    signed_reasoning: bool  # reasoning may come back only as signatures, which are replayable reasoning
    reported: tuple[str, ...]  # forwarded keys shown in the upstream check's detail


SAMPLING = ("temperature", "top_p", "top_k", "presence_penalty")
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
UPSTREAM_REPORTED = ("model", "max_tokens", "max_completion_tokens", "enable_thinking", "reasoning_effort",
                     "thinking_budget", "preserve_thinking", "enable_search", *SAMPLING, "stream")
VENDORS = {
    "alibaba": Vendor(upstream_body=OPENAI_CHAT_BODY, efforts=REASONING_EFFORTS, native_search=frozenset(),
                      tier_endpoints_opt_in=False, signed_reasoning=False, reported=UPSTREAM_REPORTED),
    # Gemini thinking levels (aistudio.google.com/docs/thinking, read 2026-11-25); reasoning is mandatory, so
    # "none" is not offered. native_tools type as listed on every Gemini endpoint (read 2026-11-25).
    "google": Vendor(upstream_body=GEMINI_BODY, efforts=frozenset({"minimal", "low", "medium", "high", "xhigh"}),
                     native_search=frozenset({"google_search", "googleSearch"}), tier_endpoints_opt_in=True,
                     signed_reasoning=True,
                     reported=("model", "generationConfig", "tools", "toolConfig", "safetySettings")),
}

STATE_FORMAT = "openrouter-chat-v1"
BASE_URL = "https://openrouter.ai/api/v1/"
CHAT_PATH = "chat/completions"
DEFAULT_TIMEOUT_S = 900
PROBE_MAX_TOKENS = 2048
METADATA_HEADER = {"x-openrouter-metadata": "enabled"}
SUPPORTED_PARAMETERS = frozenset({"max_output_tokens", "reasoning_effort", *SAMPLING, "web_search"})
WEB_SEARCH_TOOL = "openrouter:web_search"
TIER_SEGMENTS = frozenset({"flex", "priority", "fast", "turbo", "highspeed", "prime", "ultrafast"})
STANDARD_TIERS = (None, "default", "standard")
# Native finish reasons meaning the server-side tool loop stopped before an answer (Gemini FinishReason).
INCOMPLETE_NATIVE_FINISH = frozenset({"TOO_MANY_TOOL_CALLS", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"})
GEMINI_SAMPLING = {"temperature": "temperature", "top_p": "topP", "top_k": "topK",
                   "presence_penalty": "presencePenalty"}
GEMINI_UNCONFIGURABLE_SAMPLING = ("frequencyPenalty", "seed", "candidateCount")  # never configured: never forwarded
# ...except at Google's own default: OpenRouter always forwards candidateCount 1 (seen 2026-11-25), which is the
# generateContent default (https://ai.google.dev/api/generate-content#generationconfig), so it changes nothing.
GEMINI_DEFAULT_SAMPLING = {"candidateCount": 1}
GEMINI_EXPANDED = ("generationConfig", "generationConfig.thinkingConfig")  # dicts checked key by key
GEMINI_SEARCH_KEYS = frozenset({"googleSearch", "google_search"})
GEMINI_CONTINUATION_ID = "gemini-continuation"  # long-decode token; OpenRouter drops it from earlier turns
BASELINE_PROMPT = "Reply with exactly one word: baseline"
SEARCH_BASELINE_PROMPT = ("Search the web for today's date, then reply with exactly one word: baseline")
OUTPUT_LIMIT_KEYS = ("max_tokens", "max_completion_tokens")  # vendor-side names for the output cap
# Vendor-side reasoning controls (Alibaba: enable_thinking, reasoning_effort, thinking_budget, preserve_thinking).
THINKING_KEYS = ("enable_thinking", "reasoning_effort", "thinking_budget", "preserve_thinking")
# Forwarded keys that need no verification: the request itself and streaming plumbing (the probe streams).
NEUTRAL_KEYS = frozenset({"model", "messages", "stream", "stream_options"})
REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")  # where a vendor body may carry earlier reasoning
# OpenRouter replaces some forwarded values in the debug echo, e.g. "<included to upstream but not in debug info>"
# for enable_thinking (observed 2026-11-20).
HIDDEN_VALUE = re.compile(r"(?i)^<.*(not in debug info|redacted).*>$")
CONTEXT_OVERFLOW = re.compile(r"(?i)context.?(length|window)|maximum context|too (long|many tokens)|range of input")
VENDOR_BLOCK = re.compile(r"(?i)data_inspection_failed|inappropriate content")
TransportFactory = Callable[[], httpx2.BaseTransport]
_NO_KEY = ("no OpenRouter key found: add openrouter.api_key to secrets.local.yaml (see secrets.example.yaml) "
           "or set OPENROUTER_API_KEY")


class OpenRouterAdapter:
    name = "openrouter"

    def __init__(self, *, api_key: str | None = None, transport_factory: TransportFactory | None = None) -> None:
        self._api_key = api_key
        self._transport_factory = transport_factory or httpx2.HTTPTransport

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> httpx2.Client:
        if not self._api_key:
            raise AdapterError(FailureKind.PROVIDER_ERROR, _NO_KEY)
        timeout = float(str(config.adapter_options.get("timeout_s", DEFAULT_TIMEOUT_S)))
        # No HTTP-Referer or X-OpenRouter-Title: nothing identifying is sent beyond the key.
        return httpx2.Client(base_url=BASE_URL, transport=transport, timeout=timeout,
                             headers={"authorization": f"Bearer {self._api_key}",
                                      "content-type": "application/json", **METADATA_HEADER})

    def preflight(self, config: ModelConfiguration) -> str | None:
        return None if self._api_key else _NO_KEY

    # ----- qualification ---------------------------------------------------------------------------

    def qualify(self, config: ModelConfiguration, requirements: QualificationRequirements) -> QualificationReport:
        parameter_check = _parameter_check(config)
        probes: list[dict[str, Any]] = []
        model_check, context = self._model_check(config, probes)
        checks = [
            QualificationCheck(
                name="adapter_identity",
                passed=(config.adapter == self.name and config.track is Track.CLOUD and config.provider in ROUTES
                        and ":" not in config.model),
                detail=f"adapter={config.adapter} track={config.track} provider={config.provider} "
                       f"model={config.model} (allowed vendors: {sorted(ROUTES)}; no model variants)",
            ),
            parameter_check,
            model_check,
            context_check(config, requirements, context),
        ]
        # Probes are paid requests built from the configured parameters: send none for an invalid configuration.
        live: list[tuple[str, Callable[[], QualificationCheck]]] = [
            ("upstream_request", lambda: self._upstream_check(config, probes))]
        if requirements.requires_continuation:
            live.append(("continuation", lambda: self._continuation_check(config, probes)))
        for name, run in live:
            if parameter_check.passed is not True or checks[0].passed is not True:
                checks.append(QualificationCheck(name=name, passed=None,
                                                 detail="not evaluated: configuration parameters are invalid"))
            elif model_check.passed is not True:
                checks.append(QualificationCheck(name=name, passed=None,
                                                 detail="not evaluated: model identity was not verified"))
            else:
                checks.append(run())
        return QualificationReport.from_checks(config, self.name, requirements, checks, probes)

    def _model_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]
                     ) -> tuple[QualificationCheck, int | None]:
        """The model's OpenRouter endpoints: exactly one from the vendor, serving `upstream_model`, with enough
        context and output, every sent parameter supported, and the configured prices."""
        route = ROUTES.get(config.provider)
        endpoint_model = config.adapter_options.get("endpoint_model")
        if route is None or ":" in config.model:
            return QualificationCheck(name="model_identity", passed=False, detail="no allowed route"), None
        try:
            selected_slug = _route_slug(config, route)
        except AdapterError as exc:
            return QualificationCheck(name="model_identity", passed=False, detail=str(exc)), None
        for option in ("endpoint_model", "upstream_model"):
            if not isinstance(config.adapter_options.get(option), str) or not config.adapter_options.get(option):
                return QualificationCheck(name="model_identity", passed=False,
                                          detail=f"adapter_options.{option} is required"), None
        if config.context_limit is None:
            return QualificationCheck(name="model_identity", passed=False, detail="context_limit is required"), None
        transport = CapturingTransport(self._transport_factory())
        try:
            with self._client(config, transport) as client:
                response = client.get(f"models/{config.model}/endpoints")
        except (AdapterError, httpx2.HTTPError) as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}"), None
        finally:
            # A free metadata read: recorded as a call that incurs no cost, so reports do not count it as unknown.
            listing: dict[str, Any] = {"step": "endpoints", "status": None, "cost_incurred": CostIncurred.NO.value,
                                       "cost_usd": None, "wire": wire_record(transport.exchanges)}
            probes.append(listing)
        if response.status_code != 200:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"HTTP {response.status_code}: {_error_message(response)}"), None
        data = response.json().get("data") or {}
        endpoints = [e for e in data.get("endpoints") or [] if isinstance(e, dict)]
        ours = [e for e in endpoints if str(e.get("tag") or "").split("/")[0] == route.slug]
        if selected_slug != route.slug:
            ours = [e for e in ours if e.get("tag") == selected_slug]
        elif _vendor(config).tier_endpoints_opt_in:
            # A base slug never routes to a closed-weight model's tier endpoints (flex, priority): they are not
            # candidates, and the standard endpoint is the one that must be unique.
            ours = [e for e in ours if e.get("service_tier") in STANDARD_TIERS]
        problems: list[str] = []
        if data.get("id") != config.model:
            problems.append(f"endpoints list is for {data.get('id')!r}, not {config.model!r}")
        if len(ours) != 1:
            names = [e.get("name") for e in endpoints]
            detail = f"expected exactly one {route.name} endpoint for {config.model}, found {len(ours)} in {names}"
            return QualificationCheck(name="model_identity", passed=False, detail=detail), None
        endpoint = ours[0]
        listing.update(status="completed", endpoint=endpoint)
        served = str(endpoint.get("name") or "").split("|")[-1].strip().rsplit("/", 1)[-1]
        if served != endpoint_model:
            problems.append(f"endpoint {endpoint.get('name')!r} serves {served!r}, not endpoint_model "
                            f"{endpoint_model!r}")
        if endpoint.get("provider_name") != route.name:
            problems.append(f"endpoint provider is {endpoint.get('provider_name')!r}, not {route.name!r}")
        context, output_cap = endpoint.get("context_length"), endpoint.get("max_completion_tokens")
        if isinstance(context, int) and config.context_limit > context:
            problems.append(f"context_limit {config.context_limit} > endpoint context_length {context}")
        requested = explicit(config, "max_output_tokens")
        if isinstance(output_cap, int) and isinstance(requested, int) and requested > output_cap:
            problems.append(f"max_output_tokens {requested} > endpoint max_completion_tokens {output_cap}")
        supported = set(endpoint.get("supported_parameters") or [])
        if missing := sorted(set(_wire_parameters(config)) - supported):
            problems.append(f"endpoint does not support {missing}")
        problems.extend(_pricing_problems(config, endpoint.get("pricing") or {}))
        detail = (f"{endpoint.get('name')} ({endpoint.get('provider_name')}, quantization "
                  f"{endpoint.get('quantization')}, status {endpoint.get('status')}): context_length={context}, "
                  f"max_completion_tokens={output_cap}; prices match configuration")
        if _searching(config):
            # With engine "native" OpenRouter still falls back to another engine on an endpoint without native
            # search: the endpoint itself must run openrouter:web_search as the vendor's own tool.
            native = (endpoint.get("native_tools") or {}).get(WEB_SEARCH_TOOL)
            native_type = native.get("type") if isinstance(native, dict) else None
            if native_type not in _vendor(config).native_search:
                problems.append(f"endpoint runs {WEB_SEARCH_TOOL} as {native!r}, not the vendor's own search "
                                f"({sorted(_vendor(config).native_search)}): searches would use another engine")
            detail += f"; {WEB_SEARCH_TOOL} runs natively as {native_type}"
        return (QualificationCheck(name="model_identity", passed=not problems, detail="; ".join(problems) or detail),
                context if isinstance(context, int) else None)

    def _probe_config(self, config: ModelConfiguration) -> ModelConfiguration:
        budget = explicit(config, "max_output_tokens")
        params = dict(config.parameters)
        params["max_output_tokens"] = params["max_output_tokens"].model_copy(
            update={"value": min(budget, PROBE_MAX_TOKENS) if isinstance(budget, int) else PROBE_MAX_TOKENS})
        return config.model_copy(update={"parameters": params})

    def _run_probe(self, config: ModelConfiguration, probes: list[dict[str, Any]], step: str,
                   messages: tuple[Message, ...], parent: ContinuationSnapshot | None
                   ) -> tuple[PreparedInvocation, NativeExecution]:
        ctx = InvocationContext(experiment_id="qualification", task_id=f"qualification:{config.key}",
                                attempt_id=step, attempt_no=1, round=1 if parent is None else 2,
                                purpose="qualification")
        sink = CollectingSink()
        prepared = self.prepare(config, messages, parent, ctx)
        execution = self.execute(config, prepared, ctx, sink)
        cost = execution.observations.get("api_cost_usd")
        probes.append({
            "step": step, "status": execution.status, "request_id": execution.request_id,
            "failure_kind": execution.failure_kind.value if execution.failure_kind else None,
            "cost_incurred": execution.cost_incurred.value, "cost_usd": cost.value if cost else None,
            "max_output_tokens": explicit(config, "max_output_tokens"),
            "wire": sink.items.get("wire-exchanges.json"),
        })
        return prepared, execution

    def _upstream_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """A baseline, then one streamed follow-up (configured max_output_tokens, replayed assistant message) with
        `debug.echo_upstream_body`: the body OpenRouter forwarded to the vendor must match the configuration.
        With web search configured the baseline asks for a search, so the replay of a message carrying search
        artefacts is what the echo shows (the model may still not search; the detail says so)."""
        searching = _searching(config)
        base_messages = (Message(role=Role.SYSTEM, content="You are answering a harness qualification probe."),
                         Message(role=Role.USER, content=SEARCH_BASELINE_PROMPT if searching else BASELINE_PROMPT))
        try:
            prepared, execution = self._run_probe(self._probe_config(config), probes, "upstream-baseline",
                                                  base_messages, None)
            if execution.status != "completed":
                return QualificationCheck(name="upstream_request", passed=False,
                                          detail=f"baseline failed: {execution.failure_kind} {execution.error}")
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            follow = snapshot.messages + (Message(role=Role.USER, content="Reply with exactly one word: gamma"),)
            ctx = InvocationContext(experiment_id="qualification", task_id=f"qualification:{config.key}",
                                    attempt_id="upstream-echo", attempt_no=1, round=2, purpose="qualification")
            body = {**self.prepare(config, follow, snapshot, ctx).native_request["body"],
                    "stream": True, "debug": {"echo_upstream_body": True}}
        except AdapterError as exc:
            return QualificationCheck(name="upstream_request", passed=False, detail=str(exc))
        transport = CapturingTransport(self._transport_factory())
        record: dict[str, Any] = {"step": "upstream-echo", "max_output_tokens": explicit(config, "max_output_tokens")}
        try:
            with self._client(config, transport) as client:
                response = client.post(CHAT_PATH, content=json.dumps(body))
        except AdapterError as exc:  # no key: nothing was sent
            record.update(status="failed", failure_kind=FailureKind.PROVIDER_ERROR.value,
                          cost_incurred=CostIncurred.NO.value, cost_usd=None)
            return QualificationCheck(name="upstream_request", passed=None, detail=f"not evaluated: {exc}")
        except httpx2.HTTPError as exc:  # Alibaba requests cannot be cancelled: a sent request may still bill
            sent = isinstance(exc, httpx2.TimeoutException) or bool(transport.exchanges)
            record.update(status="failed", failure_kind=FailureKind.TRANSIENT_TRANSPORT.value,
                          cost_incurred=(CostIncurred.UNKNOWN if sent else CostIncurred.NO).value, cost_usd=None)
            return QualificationCheck(name="upstream_request", passed=None, detail=f"not evaluated: {exc}")
        finally:
            record["wire"] = wire_record(transport.exchanges)
            probes.append(record)
        if response.status_code != 200:
            kind, cost = _status_failure(response.status_code, _error(response))
            record.update(status="failed", failure_kind=kind.value, cost_incurred=cost.value, cost_usd=None,
                          request_id=response.headers.get("x-generation-id"))
            return QualificationCheck(name="upstream_request", passed=False,
                                      detail=f"echo probe: HTTP {response.status_code}: {_error_message(response)}")
        chunks = _sse_chunks(response.text)
        record.update(_stream_accounting(config, chunks, response.headers.get("x-generation-id")))
        debug = next((c for c in chunks if isinstance(c.get("debug"), dict)), None)
        echo = (debug or {}).get("debug", {}).get("echo_upstream_body")
        if not isinstance(echo, dict):
            return QualificationCheck(name="upstream_request", passed=None,
                                      detail="not evaluated: the stream carried no debug.echo_upstream_body chunk")
        record.update(upstream_provider=(debug or {}).get("provider"), upstream_body=echo)
        returned = _choice(execution.native_response or {}).get("message") or {}
        vendor = _vendor(config)
        unverified: dict[str, Any]
        if vendor.upstream_body == GEMINI_BODY:
            # Gemini's REST API accepts snake_case and camelCase field names alike: check one spelling. The probe
            # records the echo as received.
            echo = _camel_keys(echo)
            if not isinstance(echo.get("contents"), list):
                return QualificationCheck(name="upstream_request", passed=None,
                                          detail=f"not evaluated: the forwarded body is not a Gemini generateContent "
                                                 f"body (top-level keys {sorted(echo)})")
            system = "\n\n".join(m.content for m in base_messages if m.role is Role.SYSTEM)
            problems, unverified, notes = _gemini_upstream_problems(config, echo, (debug or {}).get("provider"),
                                                                    returned, system)
            if searching:
                notes.append(_search_replay_note(execution, returned))
        else:
            problems, unverified_keys, notes = _upstream_problems(config, echo, (debug or {}).get("provider"),
                                                                  returned)
            unverified = {k: echo[k] for k in unverified_keys}
        reported = {k: echo[k] for k in vendor.reported if k in echo}
        detail = (f"forwarded to {(debug or {}).get('provider')}: {json.dumps(reported, sort_keys=True)}; "
                  f"top-level keys {sorted(echo)}")
        if problems:
            return QualificationCheck(name="upstream_request", passed=False,
                                      detail="; ".join(problems) + f" ({detail})")
        if unverified:
            # Not a contradiction, but not shown to match the configuration either: qualification stays incomplete
            # until the values are checked against the vendor's docs and pinned in adapter_options.upstream_expect.
            pins = json.dumps(unverified, sort_keys=True)
            return QualificationCheck(name="upstream_request", passed=None,
                                      detail=f"not verified: OpenRouter forwarded {pins}, which the configuration "
                                             f"does not determine or the echo hides; pin them in upstream_expect "
                                             f"({detail})")
        return QualificationCheck(name="upstream_request", passed=True,
                                  detail="; ".join([detail, *notes]))

    def _continuation_check(self, config: ModelConfiguration, probes: list[dict[str, Any]]) -> QualificationCheck:
        """Live probe: one baseline, then two sibling follow-ups forked from the same snapshot."""
        probe = self._probe_config(config)
        base_messages = (Message(role=Role.SYSTEM, content="You are answering a harness qualification probe."),
                         Message(role=Role.USER, content="Reply with exactly one word: baseline"))
        try:
            prepared, execution = self._run_probe(probe, probes, "baseline", base_messages, None)
            if execution.status != "completed":
                return QualificationCheck(name="continuation", passed=False,
                                          detail=f"baseline failed: {execution.failure_kind} {execution.error}")
            returned = _choice(execution.native_response or {}).get("message") or {}
            snapshot = self.fork_state(prepared, execution, self.extract(execution))
            before = json.dumps(snapshot.native_state, sort_keys=True)
            base_sent = prepared.native_request["body"]["messages"]
            for word in ("alpha", "beta"):
                follow = snapshot.messages + (Message(role=Role.USER, content=f"Reply with exactly one word: {word}"),)
                child, result = self._run_probe(probe, probes, f"branch-{word}", follow, snapshot)
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
            return QualificationCheck(name="continuation", passed=False, detail=str(exc))
        return QualificationCheck(
            name="continuation", passed=True,
            detail="baseline forked into two independent branches with exact message replay"
                   + (" (including reasoning and reasoning_details)" if _reasoning_text(returned) else ""),
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
        if not isinstance(max_tokens, int):
            raise AdapterError(FailureKind.INVALID_STATE, "max_output_tokens must be an explicit integer")
        route = ROUTES.get(config.provider)
        if route is None or ":" in config.model:
            raise AdapterError(FailureKind.INVALID_STATE,
                               f"no OpenRouter route for provider {config.provider!r} and model {config.model!r}")
        if parent is not None:
            parent.verify_shape()
            state = parent.native_state
            if state.get("format") != STATE_FORMAT:
                raise AdapterError(FailureKind.INVALID_STATE, "unrecognised continuation state format")
            if state.get("model") != config.model:
                raise AdapterError(FailureKind.INVALID_STATE, "continuation state belongs to a different model")
            if messages[: len(parent.messages)] != parent.messages:
                raise AdapterError(FailureKind.INVALID_STATE, "conversation does not extend the baseline snapshot")
            native = copy.deepcopy(state["messages"])
            new, leading = messages[len(parent.messages):], 0
        else:
            native, new = [], messages
            leading = next((i for i, m in enumerate(messages) if m.role is not Role.SYSTEM), len(messages))
        if any(m.role is Role.SYSTEM for m in new[leading:]):
            raise AdapterError(FailureKind.INVALID_STATE, "system messages must lead the conversation")
        for m in new:
            if m.role is Role.ASSISTANT:
                raise AdapterError(FailureKind.INVALID_STATE,
                                   "assistant turns must come from a continuation snapshot, not plain text")
            native.append({"role": m.role.value, "content": m.content})
        body: dict[str, Any] = {"model": config.model, "messages": native, "max_tokens": max_tokens}
        for name in SAMPLING:
            if (value := explicit(config, name)) is not None:
                body[name] = value
        if (effort := explicit(config, "reasoning_effort")) is not None:
            body["reasoning"] = {"effort": effort}
        if (web_search := explicit(config, "web_search")) is not None:
            if (problem := _web_search_problem(config, web_search)) is not None:
                raise AdapterError(FailureKind.INVALID_STATE, problem)
            body["tools"] = [copy.deepcopy(web_search)]
        body["provider"] = {"only": [_route_slug(config, route)], "allow_fallbacks": False,
                            "require_parameters": True}
        return PreparedInvocation(
            adapter=self.name,
            messages=messages,
            native_request={
                "endpoint": f"POST /api/v1/{CHAT_PATH}",
                "client": {"package": "httpx2", "version": httpx2.__version__, "retries": 0},
                "headers": dict(METADATA_HEADER),
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
            with self._client(config, transport) as client:
                response = client.post(CHAT_PATH, content=json.dumps(body))
        except AdapterError as exc:
            return failed(FailureKind.PROVIDER_ERROR, "credentials", str(exc), CostIncurred.NO)
        except httpx2.TimeoutException as exc:  # Alibaba requests cannot be cancelled: the model keeps running
            return failed(FailureKind.AMBIGUOUS_TIMEOUT, "timeout", str(exc), CostIncurred.UNKNOWN)
        except httpx2.TransportError as exc:
            cost = CostIncurred.UNKNOWN if transport.exchanges else CostIncurred.NO
            return failed(FailureKind.TRANSIENT_TRANSPORT, "connection", str(exc), cost)

        request_id = response.headers.get("x-generation-id")
        if response.status_code != 200:
            error = _error(response)
            kind, cost = _status_failure(response.status_code, error)
            return failed(kind, f"http_{response.status_code}", _error_message(response), cost,
                          status=response.status_code, request_id=request_id, metadata=error.get("metadata"))
        if json.loads(transport.exchanges[-1]["request_body"]) != body:
            raise AdapterError(FailureKind.HARNESS_ERROR, "request body on the wire differs from the prepared body",
                               CostIncurred.YES)
        try:
            data = response.json()  # leading keep-alive whitespace is valid JSON whitespace
        except json.JSONDecodeError:
            return failed(FailureKind.PROVIDER_ERROR, "invalid_body", response.text[:500], CostIncurred.UNKNOWN,
                          request_id=request_id)
        request_id = data.get("id") or request_id
        choice = _choice(data)
        reason = choice.get("finish_reason")
        message = choice.get("message") or {}
        observations = _observations(config, data.get("usage") or {}, message)
        vendor = _vendor(config)
        searching = _searching(config)

        def failure(kind: FailureKind, error: dict[str, Any], cost: CostIncurred = CostIncurred.YES
                    ) -> NativeExecution:
            return NativeExecution(status="failed", native_response=data, request_id=request_id,
                                   finish_reason=reason, cost_incurred=cost, failure_kind=kind,
                                   observations=observations, error=error)

        # An error after the provider connected arrives with HTTP 200: as a bare error body, or as
        # finish_reason "error" beside it. Tokens may have been generated.
        if isinstance(data.get("error"), dict) or reason == "error":
            body_error: dict[str, Any] = data["error"] if isinstance(data.get("error"), dict) else {}
            code = body_error["code"] if isinstance(body_error.get("code"), int) else 502
            kind, _ = _status_failure(code, body_error)
            billed = CostIncurred.YES if (data.get("usage") or {}).get("completion_tokens") else CostIncurred.UNKNOWN
            return failure(kind, {"type": "error_in_body", "code": body_error.get("code"),
                                  "message": str(body_error.get("message") or "")[:500],
                                  "metadata": body_error.get("metadata")}, billed)
        if data.get("model") != config.model:
            return failure(FailureKind.PROVIDER_ERROR,
                           {"type": "model_mismatch", "requested": config.model, "served": data.get("model")})
        route = ROUTES[config.provider]
        served = _served_provider(data)
        if served != route.name:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "provider_mismatch", "expected": route.name,
                                                        "served": served})
        endpoint = _served_endpoint_model(data)
        expected_endpoint = _expected_endpoint(config)
        if endpoint != expected_endpoint:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "endpoint_mismatch", "expected": expected_endpoint,
                                                        "served": endpoint})
        # Same label, same provider name: only the reported tier tells a flex or priority endpoint apart.
        if vendor.tier_endpoints_opt_in and data.get("service_tier") not in STANDARD_TIERS:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "tier_mismatch", "expected": "default",
                                                        "served": data.get("service_tier")})
        if searching and (engine := _search_engine_problem(data)) is not None:
            return failure(FailureKind.PROVIDER_ERROR, engine)
        if not choice:
            return failure(FailureKind.PROVIDER_ERROR, {"type": "no_choices"})
        if reason == "content_filter" or message.get("refusal"):
            return failure(FailureKind.REFUSAL, {"type": "content_filter", "refusal": message.get("refusal"),
                                                 "native_finish_reason": choice.get("native_finish_reason")})
        native_reason = choice.get("native_finish_reason")
        if searching and (message.get("tool_calls") or reason == "tool_calls"
                          or native_reason in INCOMPLETE_NATIVE_FINISH
                          or (reason == "stop" and not _content_text(message).strip())):
            # The server-side search loop stopped before a final answer: never a partial answer.
            return failure(FailureKind.PROVIDER_ERROR, {"type": "incomplete", "reason": reason,
                                                        "native_finish_reason": native_reason,
                                                        "detail": "server tool loop stopped before a final answer"})
        if reason not in ("stop", "length"):  # tool_calls (no client tools are sent), ...
            return failure(FailureKind.PROVIDER_ERROR, {"type": "finish_reason", "reason": reason,
                                                        "native_finish_reason": choice.get("native_finish_reason")})
        # Without reasoning in the message a follow-up would silently drop it from the model's context. A missing
        # reasoning text is acceptable only when the response reports zero reasoning tokens; unreported is not zero.
        # Gemini thought signatures are replayable reasoning even without readable text.
        if observations["reasoning_tokens"].value != 0 and not _reasoning_text(message) and not (
                vendor.signed_reasoning and _signatures(message)):
            return failure(FailureKind.PROVIDER_ERROR, {"type": "missing_reasoning",
                                                        "reasoning_tokens": observations["reasoning_tokens"].value})
        return NativeExecution(status="completed", native_response=data, finish_reason=reason,
                               request_id=request_id, cost_incurred=CostIncurred.YES, observations=observations)

    def extract(self, execution: NativeExecution) -> ExtractedResponse:
        data = execution.native_response
        if execution.status != "completed" or data is None:
            raise AdapterError(FailureKind.HARNESS_ERROR, "cannot extract from a failed execution")
        choice = _choice(data)
        message = choice.get("message") or {}
        return ExtractedResponse(
            final_answer=_content_text(message),  # text only: annotations stay in the native response
            truncated=choice.get("finish_reason") == "length",
            finish_reason=choice.get("finish_reason"),
            reasoning_text=_reasoning_text(message),
        )

    def fork_state(
        self, prepared: PreparedInvocation, execution: NativeExecution, extracted: ExtractedResponse
    ) -> ContinuationSnapshot:
        data = execution.native_response or {}
        body = prepared.native_request["body"]
        return ContinuationSnapshot(
            adapter=self.name,
            configuration_key=str(data.get("model")),
            messages=prepared.messages + (Message(role=Role.ASSISTANT, content=extracted.final_answer),),
            native_state={
                "format": STATE_FORMAT,
                "model": data.get("model"),
                "response_id": data.get("id"),
                "served_by": _served_provider(data),
                # The assistant message (content, reasoning, reasoning_details, and search annotations) is replayed
                # exactly as returned; the upstream check shows what of it reaches the vendor.
                "messages": copy.deepcopy(body["messages"]) + [copy.deepcopy(_choice(data).get("message") or {})],
            },
        )


def _vendor(config: ModelConfiguration) -> Vendor:
    """The vendor's profile; an unknown provider gets Alibaba's rules (it fails adapter_identity anyway)."""
    return VENDORS.get(config.provider, VENDORS["alibaba"])


def _searching(config: ModelConfiguration) -> bool:
    return explicit(config, "web_search") is not None


def _web_search_problem(config: ModelConfiguration, value: Any) -> str | None:
    """web_search must be OpenRouter's server tool pinned to the vendor's own engine, with nothing else set:
    `auto` (the default) and every named engine but `native` can run a search that is not the vendor's, and the
    other parameters either do not apply to native search or make OpenRouter leave it (domain filters)."""
    vendor = _vendor(config)
    if not vendor.native_search:
        return f"web_search: no native web search is allowed on the {config.provider!r} route"
    if (not isinstance(value, dict) or value.get("type") != WEB_SEARCH_TOOL or set(value) - {"type", "parameters"}
            or not isinstance(value.get("parameters"), dict)):
        return (f"web_search must be the {WEB_SEARCH_TOOL} server tool definition, e.g. "
                f"{{type: {WEB_SEARCH_TOOL}, parameters: {{engine: native}}}}")
    parameters = value["parameters"]
    if parameters.get("engine") != "native":
        return (f"web_search must pin parameters.engine: native, not {parameters.get('engine')!r}: auto and the "
                f"other engines (exa, firecrawl, parallel, perplexity) can search with a non-vendor engine")
    if extra := sorted(set(parameters) - {"engine"}):
        return f"web_search parameters {extra} are not part of the vendor's native search tool"
    return None


def _wire_parameters(config: ModelConfiguration) -> list[str]:
    names = ["max_tokens"] + [n for n in SAMPLING if explicit(config, n) is not None]
    names += ["reasoning"] if explicit(config, "reasoning_effort") is not None else []
    return names + (["tools"] if _searching(config) else [])


def _route_slug(config: ModelConfiguration, route: Route) -> str:
    """A suffixed slug targets one vendor endpoint; a bare slug matches every vendor endpoint (for a vendor whose
    tier endpoints are opt-in, every standard-tier endpoint, and a tier-suffixed slug is refused)."""
    endpoint = config.adapter_options.get("provider_endpoint")
    if endpoint is None:
        return route.slug
    if (not isinstance(endpoint, str) or not endpoint.startswith(route.slug + "/")
            or re.fullmatch(r"[a-z0-9-]+(?:/[a-z0-9-]+)+", endpoint) is None):
        raise AdapterError(FailureKind.INVALID_STATE,
                           f"provider_endpoint must be a suffixed endpoint of {route.slug!r}")
    if _vendor(config).tier_endpoints_opt_in and endpoint.rsplit("/", 1)[-1] in TIER_SEGMENTS:
        raise AdapterError(FailureKind.INVALID_STATE,
                           f"provider_endpoint must be a standard-tier endpoint of {route.slug!r}, not {endpoint!r}")
    return endpoint


def _parameter_check(config: ModelConfiguration) -> QualificationCheck:
    explicit_names = {n for n, s in config.parameters.items() if s.state is ParameterState.EXPLICIT}
    problems = []
    if unsupported := sorted(explicit_names - SUPPORTED_PARAMETERS):
        problems.append(f"unsupported: {unsupported}")
    if "max_output_tokens" not in explicit_names:
        problems.append("max_output_tokens must be explicit")
    effort = explicit(config, "reasoning_effort")
    efforts = _vendor(config).efforts
    if effort is not None and effort not in efforts:
        problems.append(f"reasoning_effort {effort!r} is not one of {sorted(efforts)}")
    if _searching(config) and (problem := _web_search_problem(config, explicit(config, "web_search"))):
        problems.append(problem)
    return QualificationCheck(name="parameter_support", passed=not problems, detail="; ".join(problems) or "ok")


def _pricing_problems(config: ModelConfiguration, live: dict[str, Any]) -> list[str]:
    """OpenRouter publishes USD per token as strings; the configuration holds USD per million tokens."""
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return ["no pricing in configuration"]
    problems = []
    for live_key, key in (("prompt", "input_usd_per_mtok"), ("completion", "output_usd_per_mtok"),
                          ("input_cache_read", "cached_input_usd_per_mtok"),
                          ("input_cache_write", "cache_write_usd_per_mtok")):
        try:
            published = float(live[live_key]) * 1_000_000 if live.get(live_key) is not None else None
        except (TypeError, ValueError):
            published = None
        configured = pricing.get(key)
        if published is None and configured is None:
            continue
        if published is None or not isinstance(configured, int | float) or abs(published - configured) > 1e-6:
            problems.append(f"{key}: configured {configured}, OpenRouter {published}")
    # Reasoning tokens are costed at the output price: an endpoint pricing them differently would be miscosted.
    reasoning, completion = _float(live.get("internal_reasoning")), _float(live.get("completion"))
    if live.get("internal_reasoning") is not None and (reasoning is None or reasoning != completion):
        problems.append(f"OpenRouter prices internal_reasoning at {live.get('internal_reasoning')} per token, "
                        f"completion at {live.get('completion')}; reasoning is costed as output")
    # A search price is per request (USD), not per token. Checked whenever searches can happen or one is set.
    if _searching(config) or "web_search_usd_per_request" in pricing:
        published_search, configured_search = _float(live.get("web_search")), pricing.get("web_search_usd_per_request")
        if (published_search is None or not isinstance(configured_search, int | float)
                or abs(published_search - configured_search) > 1e-9):
            problems.append(f"web_search_usd_per_request: configured {configured_search}, "
                            f"OpenRouter {published_search}")
    return problems


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _upstream_problems(config: ModelConfiguration, echo: dict[str, Any], provider: Any, returned: dict[str, Any]
                       ) -> tuple[list[str], list[str], list[str]]:
    """Contradictions between the forwarded body and the configuration, forwarded keys nothing accounts for, and
    notes on how hidden values were verified."""
    route = ROUTES[config.provider]
    problems: list[str] = []
    if provider is not None and str(provider) != route.name:
        problems.append(f"forwarded to {provider!r}, not {route.name!r}")
    if echo.get("model") != config.adapter_options.get("upstream_model"):
        problems.append(f"vendor model {echo.get('model')!r}, not upstream_model "
                        f"{config.adapter_options.get('upstream_model')!r}")
    for name in SAMPLING:  # an omitted parameter must be left to the vendor, not filled in by OpenRouter
        configured = explicit(config, name)
        if configured is None and name in echo:
            problems.append(f"{name} omitted in configuration but forwarded as {echo[name]!r}")
        elif configured is not None and echo.get(name) != configured:
            problems.append(f"{name} {configured!r} forwarded as {echo.get(name)!r}")
    # The output cap must reach the vendor unchanged: without it the vendor's default (up to 262144 thinking-mode
    # tokens on Qwen4-Max) would set output length and spend.
    budget = explicit(config, "max_output_tokens")
    limits = {key: echo[key] for key in OUTPUT_LIMIT_KEYS if key in echo}
    if not limits:
        problems.append(f"no output cap forwarded (none of {list(OUTPUT_LIMIT_KEYS)}), configured {budget}")
    for key, value in limits.items():
        if value != budget:
            problems.append(f"{key} forwarded as {value!r}, configured {budget}")
    if echo.get("enable_search"):
        problems.append("enable_search is on")
    if (replay := _replay_problem(echo, returned)) is not None:
        problems.append(replay)
    expect = config.adapter_options.get("upstream_expect") or {}
    expect = expect if isinstance(expect, dict) else {}
    # A hidden echo value proves nothing about what was forwarded, so no pin can verify it: such keys stay
    # unverified (enable_thinking alone is checked by behaviour in _thinking_check).
    hidden = {key for key, value in echo.items() if _hidden(value)}
    for key, value in expect.items():
        if _hidden(value):
            problems.append(f"upstream_expect {key} pins an echo placeholder ({value!r}), not a forwarded value")
        elif key not in hidden and echo.get(key) != value:
            problems.append(f"upstream_expect {key}={value!r}, forwarded {echo.get(key)!r}")
    explained, reasoning_problems, notes = _thinking_check(config, echo, expect, bool(_reasoning_text(returned)))
    problems.extend(reasoning_problems)
    explained |= NEUTRAL_KEYS | set(OUTPUT_LIMIT_KEYS) | set(SAMPLING) | (set(expect) - hidden)
    if echo.get("enable_search") is False:
        explained.add("enable_search")
    return problems, sorted(set(echo) - explained), notes


def _thinking_check(config: ModelConfiguration, echo: dict[str, Any], expect: dict[str, Any], reasoned: bool
                    ) -> tuple[set[str], list[str], list[str]]:
    """The forwarded reasoning controls must follow from the configuration: the configured effort, else the
    documented default of an omitted parameter of the same name, else a value pinned in upstream_expect.
    A hidden `enable_thinking` is checked by behaviour instead: the baseline returned reasoning exactly when the
    configuration implies thinking on. Any other hidden value stays unverified."""
    effort = explicit(config, "reasoning_effort")
    explained: set[str] = set()
    problems: list[str] = []
    notes: list[str] = []
    for key in THINKING_KEYS:
        if key not in echo or (key in expect and not _hidden(echo[key])):
            continue
        value = echo[key]
        setting = config.parameters.get(key)
        if key == "reasoning_effort" and effort is not None:
            wanted: Any = effort
        elif key == "enable_thinking" and effort is not None:
            wanted = effort != "none"
        elif (setting is not None and setting.state is ParameterState.OMITTED
              and setting.documented_default is not None):
            wanted = setting.documented_default
        else:
            continue  # nothing in the configuration determines it: left unverified
        if _hidden(value):
            if key != "enable_thinking" or not isinstance(wanted, bool):
                continue  # hidden and not checkable by behaviour: left unverified
            explained.add(key)
            if reasoned != wanted:
                problems.append(f"enable_thinking is hidden in the echo and the baseline "
                                f"{'returned' if reasoned else 'returned no'} reasoning; configuration implies "
                                f"{wanted!r}")
            else:
                notes.append(f"enable_thinking is hidden in the echo; the baseline "
                             f"{'returned' if reasoned else 'returned no'} reasoning, consistent with {wanted!r}")
            continue
        explained.add(key)
        if value != wanted:
            problems.append(f"{key} forwarded as {value!r}, configuration implies {wanted!r}")
    # The effort level itself must survive translation. enable_thinking only says whether the model thinks, not how
    # much, so it carries a configured effort only when that effort is "none". A thinking_budget carries a level
    # but stays unverified until pinned in upstream_expect.
    carriers = ("reasoning_effort", "thinking_budget") + (("enable_thinking",) if effort == "none" else ())
    if effort is not None and not any(k in echo for k in carriers):
        forwarded = [k for k in THINKING_KEYS if k in echo]
        problems.append(f"reasoning_effort {effort!r} configured but no control carrying the effort level was "
                        f"forwarded (only {forwarded}); the vendor default would set the effort")
    return explained, problems, notes


def _gemini_upstream_problems(config: ModelConfiguration, echo: dict[str, Any], provider: Any,
                              returned: dict[str, Any], system: str
                              ) -> tuple[list[str], dict[str, Any], list[str]]:
    """The Google counterpart of _upstream_problems, for a forwarded generateContent body: contradictions, the
    forwarded values nothing accounts for (dotted paths -> values), and notes. Pins in upstream_expect are dotted
    paths into the body, e.g. generationConfig.thinkingConfig.includeThoughts."""
    route = ROUTES[config.provider]
    problems: list[str] = []
    notes: list[str] = []
    unverified: dict[str, Any] = {}
    # `labels` carries only OpenRouter's per-account safety_identifier (abuse-monitoring metadata, seen 2026-11-25);
    # it does not affect generation.
    explained: set[str] = {"contents", "systemInstruction", "labels"}
    if provider is not None and str(provider) != route.name:
        problems.append(f"forwarded to {provider!r}, not {route.name!r}")
    expected_model = config.adapter_options.get("upstream_model")
    model = echo.get("model")
    if model is None:
        # generateContent names the model in the URL, which the echo does not show (seen 2026-11-25). The model is
        # proven instead by the routing metadata every call must carry: Google AI Studio and the configured dated
        # endpoint label (checked in model_identity and on every response).
        notes.append("model id not in the forwarded body (Gemini puts it in the URL); identified by the pinned "
                     f"endpoint {config.adapter_options.get('endpoint_model')!r}")
    else:
        explained.add("model")
        if str(model).removeprefix("models/") != expected_model:
            problems.append(f"vendor model {model!r}, not upstream_model {expected_model!r}")
    forwarded_system = _gemini_text(_dict(echo.get("systemInstruction")).get("parts"))
    if (forwarded_system or "").strip() != system.strip():
        problems.append(f"system instruction forwarded as {_preview(forwarded_system)}, sent {_preview(system)}")
    generation = _dict(echo.get("generationConfig"))
    budget = explicit(config, "max_output_tokens")
    explained.add("generationConfig.maxOutputTokens")
    if "maxOutputTokens" not in generation:
        problems.append(f"no output cap forwarded (generationConfig.maxOutputTokens), configured {budget}")
    elif generation["maxOutputTokens"] != budget:
        problems.append(f"generationConfig.maxOutputTokens forwarded as {generation['maxOutputTokens']!r}, "
                        f"configured {budget}")
    for name, key in GEMINI_SAMPLING.items():  # an omitted parameter must be left to Google, not filled in
        configured = explicit(config, name)
        explained.add(f"generationConfig.{key}")
        if configured is None and key in generation:
            problems.append(f"{name} omitted in configuration but forwarded as generationConfig.{key}="
                            f"{generation[key]!r}")
        elif configured is not None and generation.get(key) != configured:
            problems.append(f"{name} {configured!r} forwarded as generationConfig.{key}={generation.get(key)!r}")
    for key in GEMINI_UNCONFIGURABLE_SAMPLING:
        explained.add(f"generationConfig.{key}")
        if key in generation and not (key in GEMINI_DEFAULT_SAMPLING and generation[key] == GEMINI_DEFAULT_SAMPLING[key]):
            problems.append(f"generationConfig.{key}={generation[key]!r} forwarded; the configuration sets none")
    problems.extend(_gemini_thinking_problems(config, generation, explained, unverified, notes))
    problems.extend(_gemini_tool_problems(config, echo, explained, unverified, notes))
    replay, hidden_signature = _gemini_replay_problem(echo, returned)
    if replay is not None:
        problems.append(replay)
    if hidden_signature is not None:
        notes.append("thought signatures forwarded in the replayed turn, one per baseline signature; their values "
                     f"are redacted in OpenRouter's debug echo ({hidden_signature!r}) and could not be compared")
    expect = config.adapter_options.get("upstream_expect") or {}
    expect = expect if isinstance(expect, dict) else {}
    leaves = _flatten(echo)
    hidden = {path for path, value in leaves.items() if _hidden(value)}
    for key, value in expect.items():
        if _hidden(value):
            problems.append(f"upstream_expect {key} pins an echo placeholder ({value!r}), not a forwarded value")
        elif key not in hidden and _at_path(echo, key) != value:
            problems.append(f"upstream_expect {key}={value!r}, forwarded {_at_path(echo, key)!r}")
    pinned = set(expect) - hidden  # a hidden value proves nothing, whatever is pinned

    def covered(path: str, by: set[str]) -> bool:
        return any(path == done or path.startswith(done + ".") for done in by)

    for path, value in leaves.items():
        if not covered(path, explained | pinned):
            unverified.setdefault(path, value)
    for path in [p for p in unverified if covered(p, pinned)]:
        del unverified[path]  # checked against its pin above
    return problems, unverified, notes


def _gemini_thinking_problems(config: ModelConfiguration, generation: dict[str, Any], explained: set[str],
                              unverified: dict[str, Any], notes: list[str]) -> list[str]:
    """thinkingLevel must be the configured effort (OpenRouter maps effort onto it one to one), else, with the
    effort omitted, absent or the documented default. Other thinkingConfig keys are not determined here."""
    thinking = _dict(generation.get("thinkingConfig"))
    effort = explicit(config, "reasoning_effort")
    setting = config.parameters.get("reasoning_effort")
    default = (setting.documented_default if setting is not None and setting.state is ParameterState.OMITTED
               else None)
    path = "generationConfig.thinkingConfig.thinkingLevel"
    level = thinking.get("thinkingLevel")
    if level is None:
        if effort is not None:
            return [f"reasoning_effort {effort!r} configured but no {path} was forwarded (thinkingConfig keys "
                    f"{sorted(thinking)}); Google's default level would apply"]
        notes.append("no thinkingLevel forwarded: Google's native default applies")
        return []
    wanted = effort if effort is not None else default
    if _hidden(level) or wanted is None:
        unverified[path] = level
        return []
    explained.add(path)
    return [] if level == wanted else [f"{path} forwarded as {level!r}, configuration implies {wanted!r}"]


def _gemini_tool_problems(config: ModelConfiguration, echo: dict[str, Any], explained: set[str],
                          unverified: dict[str, Any], notes: list[str]) -> list[str]:
    """With web_search configured the forwarded tools are exactly one Google Search grounding tool (the native
    engine); without it no search tool is forwarded. No other tool may appear."""
    tools = echo.get("tools")
    if tools is None:
        tools = []
    elif not isinstance(tools, list):
        return [f"tools forwarded as {tools!r}"]
    explained.add("tools")
    search = [t for t in tools if isinstance(t, dict) and len(t) == 1 and set(t) <= GEMINI_SEARCH_KEYS]
    others = [t for t in tools if t not in search]
    problems = []
    if others:
        problems.append(f"tools forwarded that the configuration does not request: {others!r}")
    if _searching(config):
        if len(search) != 1:
            problems.append(f"web_search configured but {len(search)} Google Search tools (googleSearch / "
                            f"google_search) were forwarded: the search would not be Google's own grounding")
        elif next(iter(search[0].values())) != {}:
            unverified["tools"] = tools  # options the configuration does not set: pin after review
        else:
            notes.append(f"Google Search grounding forwarded as {search[0]!r}")
    elif search:
        problems.append(f"Google Search tool forwarded although web_search is not configured: {search!r}")
    return problems


def _gemini_replay_problem(echo: dict[str, Any], returned: dict[str, Any]) -> tuple[str | None, Any]:
    """The baseline's model turn must reach Google with the same answer text and every thought signature the
    baseline returned; forwarded thought text, if any, must be the baseline's. Returns the problem and, when the
    echo hides the signatures, the placeholder (unverifiable, not a contradiction)."""
    contents = [c for c in echo.get("contents") or [] if isinstance(c, dict)]
    turns = [c for c in contents if c.get("role") == "model"]
    if len(turns) != 1:
        return f"expected the one replayed model turn to be forwarded, found {len(turns)}", None
    parts = [p for p in turns[0].get("parts") or [] if isinstance(p, dict)]
    answer = "".join(p["text"] for p in parts if not p.get("thought") and isinstance(p.get("text"), str))
    baseline = _content_text(returned)
    if answer.strip() != baseline.strip():
        return f"replayed answer forwarded as {_preview(answer)}, baseline {_preview(baseline)}", None
    signatures = _signatures(returned)
    sent = [p.get("thoughtSignature", p.get("thought_signature")) for p in parts
            if p.get("thoughtSignature", p.get("thought_signature")) is not None]
    if signatures:
        if not sent:
            return "thought signatures the baseline returned are not forwarded to Google", None
        if all(_hidden(s) for s in sent):
            # OpenRouter's debug echo redacts base64 values (seen 2026-11-25: "<base64_redacted>"); the field is
            # forwarded, only its value is hidden from us. Presence and count are checked; the value cannot be.
            if len(sent) != len(signatures):
                return (f"{len(sent)} thought signatures forwarded (values redacted), the baseline returned "
                        f"{len(signatures)}"), None
            return None, sent[0]
        if missing := [s for s in signatures if s not in sent]:
            return f"{len(missing)} of {len(signatures)} baseline thought signatures not forwarded unchanged", None
    elif sent:
        return "thought signatures forwarded although the baseline returned none", None
    thought = "".join(p["text"] for p in parts if p.get("thought") and isinstance(p.get("text"), str))
    reasoning = _reasoning_text(returned)
    # Thought text may be split into several parts: only whitespace differences are tolerated.
    if thought and (not reasoning or "".join(thought.split()) != "".join(reasoning.split())):
        return (f"forwarded thought text {_preview(thought)} differs from the baseline's "
                f"{_preview(reasoning or '')}"), None
    if reasoning and not thought and not signatures:
        return "earlier reasoning is not forwarded to Google (it was returned and replayed)", None
    return None, None


def _search_replay_note(execution: NativeExecution, returned: dict[str, Any]) -> str:
    searches = execution.observations.get("web_search_requests")
    annotations = _list(returned.get("annotations"))
    if annotations or (searches is not None and searches.value):
        return (f"the baseline searched ({searches.value if searches else None} web_search_requests) and its "
                f"replayed message carried {len(annotations)} annotations")
    return "the baseline did not search: replay of a message with search annotations was not exercised"


def _camel_keys(value: Any) -> Any:
    """Every dict key in camelCase (thinking_config -> thinkingConfig); values are left as they are."""
    if isinstance(value, dict):
        return {re.sub(r"_([a-z0-9])", lambda m: m.group(1).upper(), str(k)): _camel_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_camel_keys(item) for item in value]
    return value


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _gemini_text(parts: Any) -> str | None:
    if not isinstance(parts, list):
        return None
    return "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))


def _flatten(echo: dict[str, Any]) -> dict[str, Any]:
    """Forwarded values by dotted path; generationConfig and its thinkingConfig are opened key by key."""
    leaves: dict[str, Any] = {}

    def walk(prefix: str, value: dict[str, Any]) -> None:
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if path in GEMINI_EXPANDED and isinstance(item, dict):
                walk(path, item)
            else:
                leaves[path] = item

    walk("", echo)
    return leaves


def _at_path(echo: dict[str, Any], path: str) -> Any:
    value: Any = echo
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _hidden(value: Any) -> bool:
    """An echo placeholder OpenRouter shows instead of a forwarded value."""
    return isinstance(value, str) and HIDDEN_VALUE.match(value) is not None


def _stream_accounting(config: ModelConfiguration, chunks: list[dict[str, Any]], request_id: str | None
                       ) -> dict[str, Any]:
    """Probe accounting for a streamed response, from the usage chunk OpenRouter always sends just before [DONE].
    Without one the cost is unknown: the request reached the provider and Alibaba requests run to completion."""
    usage = next((c["usage"] for c in reversed(chunks) if isinstance(c.get("usage"), dict)), None)
    errors = [c["error"] for c in chunks if isinstance(c.get("error"), dict)]
    finishes = [ch.get("finish_reason") for c in chunks for ch in c.get("choices") or [] if isinstance(ch, dict)]
    failed_stream = bool(errors) or "error" in finishes
    fields: dict[str, Any] = {
        "request_id": request_id or next((c.get("id") for c in chunks if c.get("id")), None),
        "status": "failed" if failed_stream else "completed",
        "failure_kind": None,
    }
    if failed_stream:
        code = errors[0].get("code") if errors and isinstance(errors[0].get("code"), int) else 502
        fields["failure_kind"] = _status_failure(code, errors[0] if errors else {})[0].value
        fields["error"] = errors[0] if errors else {"type": "finish_reason", "reason": "error"}
    if usage is None:
        return {**fields, "cost_incurred": CostIncurred.UNKNOWN.value, "cost_usd": None,
                "cost_missing_reason": "the stream carried no usage chunk"}
    observations = _observations(config, usage)
    if "web_search_requests" in observations:
        fields["web_search_requests"] = observations["web_search_requests"].value
    return {
        **fields,
        "cost_incurred": CostIncurred.YES.value,
        "cost_usd": observations["api_cost_usd"].value,
        "openrouter_charge": observations["openrouter_charge"].value,
        "usage": {name: obs.value for name, obs in observations.items() if obs.unit == "tokens"},
    }


def _sse_chunks(text: str) -> list[dict[str, Any]]:
    """Data lines of an SSE stream as JSON objects; comment lines (`: OPENROUTER PROCESSING`) are skipped."""
    chunks = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            chunks.append(value)
    return chunks


def _expected_endpoint(config: ModelConfiguration) -> str:
    """OpenRouter's label for the endpoint, as routing metadata reports it (e.g. qwen/qwen4-max-20261115)."""
    return f"{config.model.split('/', 1)[0]}/{config.adapter_options.get('endpoint_model')}"


def _served_endpoint_model(data: dict[str, Any]) -> str | None:
    meta = data.get("openrouter_metadata")
    available = ((meta.get("endpoints") or {}).get("available") or []) if isinstance(meta, dict) else []
    selected = [e.get("model") for e in available if isinstance(e, dict) and e.get("selected")]
    return selected[0] if len(selected) == 1 and isinstance(selected[0], str) else None


def _served_provider(data: dict[str, Any]) -> str | None:
    """The provider that produced this response, from routing metadata: the selected endpoint, else the last
    attempt. A top-level `provider` is accepted only when the metadata names none."""
    meta = data.get("openrouter_metadata")
    if isinstance(meta, dict):
        available = (meta.get("endpoints") or {}).get("available") or []
        selected = [e.get("provider") for e in available if isinstance(e, dict) and e.get("selected")]
        if len(selected) == 1 and isinstance(selected[0], str):
            return selected[0]
        attempts = [a for a in meta.get("attempts") or [] if isinstance(a, dict)]
        last = attempts[-1].get("provider") if attempts else None
        if isinstance(last, str):
            return last
    provider = data.get("provider")
    return provider if isinstance(provider, str) else None


def _replay_problem(echo: dict[str, Any], returned: dict[str, Any]) -> str | None:
    """The baseline's assistant turn must reach the vendor faithfully: the same answer and, when the baseline
    reasoned, the same reasoning text (as `message.reasoning` or the joined `reasoning_details`), not merely some
    reasoning. Only leading and trailing whitespace is ignored."""
    assistants = [m for m in echo.get("messages") or [] if isinstance(m, dict) and m.get("role") == "assistant"]
    if len(assistants) != 1:
        return f"expected the one replayed assistant message to be forwarded, found {len(assistants)}"
    forwarded = assistants[0]
    answer = str(returned.get("content") or "").strip()
    if (_flat_text(forwarded.get("content")) or "").strip() != answer:
        return f"replayed answer forwarded as {_preview(forwarded.get('content'))}, baseline {_preview(answer)}"
    baseline = {t.strip() for t in (returned.get("reasoning"), _reasoning_text(returned)) if isinstance(t, str) and t}
    sent = {k: _flat_text(forwarded.get(k)) for k in REASONING_KEYS if forwarded.get(k)}
    if not baseline:
        return f"reasoning forwarded under {sorted(sent)} although the baseline returned none" if sent else None
    if not sent:
        return "earlier reasoning is not forwarded to the vendor (it was returned and replayed)"
    if not any(isinstance(t, str) and t.strip() in baseline for t in sent.values()):
        key, text = next(iter(sent.items()))
        return (f"forwarded reasoning ({key}: {_preview(text)}) differs from the baseline's "
                f"({_preview(next(iter(baseline)))})")
    return None


def _flat_text(value: Any) -> str | None:
    """A message field as text: a string, or the text of a list of content or thinking blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [b.get("text") or b.get("thinking") for b in value if isinstance(b, dict)]
        return "".join(p for p in parts if isinstance(p, str))
    return None


def _preview(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return repr(text if len(text) <= limit else text[:limit] + "...")


def _content_text(message: dict[str, Any]) -> str:
    """The assistant's answer text: a string, or the text parts of a content list."""
    content = message.get("content")
    if isinstance(content, list):
        return "".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return str(content or "")


def _signatures(message: dict[str, Any]) -> list[str]:
    """Opaque reasoning (Gemini thought signatures) the message carries, in order. OpenRouter's long-decode
    continuation token is excluded: it is dropped from every turn but the last, so it never reaches the vendor."""
    return [d["data"] for d in message.get("reasoning_details") or []
            if isinstance(d, dict) and d.get("type") == "reasoning.encrypted" and isinstance(d.get("data"), str)
            and d["data"] and d.get("id") != GEMINI_CONTINUATION_ID]


def _search_engine_problem(data: dict[str, Any]) -> dict[str, Any] | None:
    """Evidence that OpenRouter ran a search itself instead of the provider's native search: server-orchestrated
    tool calls (native searches are not counted in tool_calls_requested), or a server_tools pipeline stage whose
    mode is explicitly not "native". An absent mode is not evidence either way."""
    details = (data.get("usage") or {}).get("server_tool_use_details")
    requested = details.get("tool_calls_requested") if isinstance(details, dict) else None
    if isinstance(requested, int) and requested > 0:
        return {"type": "search_engine_mismatch", "tool_calls_requested": requested,
                "detail": "OpenRouter orchestrated server tool calls: the search was not provider-native"}
    meta = data.get("openrouter_metadata")
    for stage in (meta.get("pipeline") or []) if isinstance(meta, dict) else []:
        if not isinstance(stage, dict) or stage.get("type") != "server_tools":
            continue
        stage_data = _dict(stage.get("data"))
        mode = stage_data.get("mode", stage.get("mode"))
        if mode is not None and mode != "native":
            return {"type": "search_engine_mismatch", "pipeline_stage": stage,
                    "detail": f"server tools ran in mode {mode!r}, not native"}
    return None


def _searches(usage: dict[str, Any], message: dict[str, Any] | None) -> Observation:
    """Searches as OpenRouter reports them. Google bills Gemini 3+ grounding per search query; OpenRouter documents
    web_search_requests as the search queries the model made, priced at the endpoint's per-request rate."""
    source = "openrouter usage.server_tool_use_details.web_search_requests (provider-native search)"
    details = usage.get("server_tool_use_details")
    count = details.get("web_search_requests") if isinstance(details, dict) else None
    if isinstance(count, int):
        return Observation(value=count, unit="requests", source=source)
    annotations = (message or {}).get("annotations")
    if isinstance(annotations, list) and annotations:
        return Observation.missing("requests", source, "search citations returned but no web_search_requests "
                                                       "reported")
    return Observation(value=0, unit="requests", source=f"{source} (none reported)")


def _reasoning_text(message: dict[str, Any]) -> str | None:
    text = message.get("reasoning")
    if isinstance(text, str) and text:
        return text
    parts = [d.get("text") or d.get("summary") for d in message.get("reasoning_details") or [] if isinstance(d, dict)]
    joined = "".join(p for p in parts if isinstance(p, str))
    return joined or None


def _choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    return first if isinstance(first, dict) else {}


def _error(response: httpx2.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except json.JSONDecodeError:
        return {}
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def _error_message(response: httpx2.Response) -> str:
    message = _error(response).get("message")
    return str(message)[:500] if message else response.text[:500]


def _status_failure(status: int, error: dict[str, Any]) -> tuple[FailureKind, CostIncurred]:
    """Map OpenRouter's error codes (errors-and-debugging page, read 2026-11-20) to a failure kind."""
    metadata: dict[str, Any] = error["metadata"] if isinstance(error.get("metadata"), dict) else {}
    text = f"{error.get('message') or ''} {metadata.get('raw') or ''}"
    if status == 403 and (metadata.get("reasons") or metadata.get("flagged_input")):
        return FailureKind.REFUSAL, CostIncurred.NO  # OpenRouter moderation: blocked before the provider
    if VENDOR_BLOCK.search(text):
        return FailureKind.REFUSAL, CostIncurred.UNKNOWN  # the vendor's content inspection; an output hit may bill
    if status in (400, 413) and CONTEXT_OVERFLOW.search(text):
        return FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO
    if status == 429:
        return FailureKind.RATE_LIMITED, CostIncurred.NO
    if status in (408, 504, 524):  # timed out; the model may have run (Alibaba requests are not cancellable)
        return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
    if status in (502, 503, 529):  # 502 provider down ("failed generations are not billed"), 503 no eligible provider
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
    if status >= 500:
        return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.UNKNOWN
    # 400 invalid request, 401 key, 402 credits, 403 permission or guardrail, 404 unknown model: retrying cannot help
    return FailureKind.PROVIDER_ERROR, CostIncurred.NO


def _observations(config: ModelConfiguration, usage: dict[str, Any], message: dict[str, Any] | None = None
                  ) -> dict[str, Observation]:
    source = "openrouter usage"
    details = usage.get("prompt_tokens_details")
    prompt = usage.get("prompt_tokens")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    # cache_write_tokens is reported only for explicit caching (cache_control), which is never sent.
    writes = details.get("cache_write_tokens", 0) if isinstance(details, dict) else None
    values = {
        "input_tokens": prompt,
        "cached_input_tokens": cached,
        "cache_write_tokens": writes,
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }
    obs = {
        name: Observation(value=int(v), unit="tokens", source=source) if v is not None
        else Observation.missing("tokens", source, "not reported in usage")
        for name, v in values.items()
    }
    if isinstance(prompt, int) and isinstance(cached, int) and isinstance(writes, int):
        obs["uncached_input_tokens"] = Observation(value=prompt - cached - writes, unit="tokens",
                                                   source=f"{source}: prompt_tokens - cached_tokens - cache_write_tokens")
    else:
        obs["uncached_input_tokens"] = Observation.missing("tokens", source, "input breakdown not reported")
    charged = usage.get("cost")
    obs["openrouter_charge"] = (
        Observation(value=float(charged), unit="credits", source="openrouter usage.cost")
        if isinstance(charged, int | float) else Observation.missing("credits", "openrouter usage.cost", "not reported"))
    if _searching(config):
        obs["web_search_requests"] = _searches(usage, message)
    obs["api_cost_usd"] = _cost(config, obs)
    return obs


def _cost(config: ModelConfiguration, obs: dict[str, Observation]) -> Observation:
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return Observation.missing("usd", "configuration pricing", "no pricing in configuration")
    parts = [
        (obs["uncached_input_tokens"].value, "input_usd_per_mtok"),
        (obs["cached_input_tokens"].value, "cached_input_usd_per_mtok"),
        (obs["cache_write_tokens"].value, "cache_write_usd_per_mtok"),
        (obs["output_tokens"].value, "output_usd_per_mtok"),
    ]
    if not all(isinstance(tokens, int) for tokens, _ in parts):
        return Observation.missing("usd", "configuration pricing", "input breakdown or output tokens not reported")
    total = 0.0
    for tokens, key in parts:
        if not tokens:
            continue
        rate = pricing.get(key)
        if not isinstance(rate, int | float):
            return Observation.missing("usd", "configuration pricing", f"no price for {key}")
        total += int(tokens) * rate / 1_000_000
    if searches := obs.get("web_search_requests"):
        if searches.value is None:
            return Observation.missing("usd", "configuration pricing", "web_search_requests not reported")
        if searches.value:
            search_rate = pricing.get("web_search_usd_per_request")
            if not isinstance(search_rate, int | float):
                return Observation.missing("usd", "configuration pricing", "no price for web_search_requests")
            total += int(searches.value) * search_rate
    return Observation(value=round(total, 10), unit="usd",
                       source=f"configuration pricing ({pricing.get('source', 'unspecified source')})")
