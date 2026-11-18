"""Native Meta adapter (Muse models on the Meta Model API): the Responses API at https://api.meta.ai/v1, which
Meta documents as the recommended protocol, through the `openai` SDK as Meta's own samples do. Meta has no
native protocol or SDK of its own.

Request handling, stateless continuation (`store: false`, encrypted reasoning replayed unchanged), refusal
handling, model-echo checks and extraction are inherited from `OpenAIAdapter`. Differences:
- Credential isolation, as for xAI: no client is built without an explicit Meta key, and OpenAI organization,
  project and admin identifiers are cleared so nothing from OPENAI_* reaches Meta.
- STANDARD TIER ONLY. The Contributor tier is selected per request by calling a `-contributor` model id with the
  same key; under it Meta may use inputs and outputs to train and evaluate its models, and rate limits are lower.
  Qualification rejects contributor ids, and every configured id must be a Muse model listed by `GET /v1/models`.
- SERVER-SIDE PROMPT CHANGE: Meta adds "Meta-injected steering context: a hidden system prompt, a default
  developer message" to every request; it is not billed, not counted in usage, and cannot be disabled.
  Configurations record this; it cannot be avoided.
- Reasoning is always on (effort `none` is a 400); raw reasoning is never returned, only optional summaries and
  encrypted content, which is replayed exactly.
- Error mapping: a too-long prompt is HTTP 400 with `code: null` and a message about the context length; a
  safety block is HTTP 400 `content_policy_violation` (a refusal with unknown cost); 402 is a billing problem;
  504 means a non-streaming request ran too long (ambiguous, cost unknown). Meta documents no request-id header,
  but responses carry `x-request-id` (observed 2026-11-16), which is recorded as the request id.
- Wonder and Spark have no long-context price, so no guard applies to them. Ember prices a whole request at
  long-context rates above 200K input tokens: configure `pricing.long_context` with `threshold_input_tokens: 200000`,
  or such requests are reported with unknown cost instead of base rates.

Configuration parameters: those of `OpenAIAdapter` except `verbosity` (not implemented by Meta):
    max_output_tokens (required, includes reasoning), reasoning_effort, reasoning_summary, temperature.
adapter_options: as for `OpenAIAdapter` (timeout_s, replay_reasoning, pricing).
"""

from __future__ import annotations

import re

import openai

from ...domain.configuration import ModelConfiguration
from ...domain.execution import CostIncurred, FailureKind
from ..base import QualificationCheck
from .openai_responses import OpenAIAdapter, _error_code
from .wire import CapturingTransport

META_BASE_URL = "https://api.meta.ai/v1"
CONTRIBUTOR_SUFFIX = "-contributor"
# Models whose whole request is priced at long-context rates above this many input tokens (docs/pricing-rate-limits).
LONG_CONTEXT_THRESHOLDS = {"muse-ember-": 200_000}
CONTEXT_OVERFLOW = re.compile(r"(?i)context (length|window)|maximum context|too long")


class MetaAdapter(OpenAIAdapter):
    name = "meta"
    provider = "meta"
    base_url = META_BASE_URL
    credentials_hint = ("no Meta Model API key found: add meta.api_key to secrets.local.yaml "
                        "(see secrets.example.yaml) or set MODEL_API_KEY")
    supported_parameters = frozenset({"max_output_tokens", "reasoning_effort", "reasoning_summary", "temperature"})

    def _long_context_guard(self, config: ModelConfiguration) -> int | None:
        """Ember's long-context threshold; None for Wonder and Spark, which Meta prices at one rate."""
        return next((t for prefix, t in LONG_CONTEXT_THRESHOLDS.items() if config.model.startswith(prefix)), None)

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> openai.OpenAI:
        if not self._api_key:
            # Never let the SDK fall back to OPENAI_API_KEY: that would send an OpenAI key to Meta.
            raise openai.OpenAIError(self.credentials_hint)
        client = super()._client(config, transport)
        client.organization = None  # read from OPENAI_ORG_ID when unset; never sent to Meta
        client.project = None  # read from OPENAI_PROJECT_ID when unset
        client.admin_api_key = None  # read from OPENAI_ADMIN_KEY when unset
        return client

    def _model_check(self, config: ModelConfiguration) -> QualificationCheck:
        if config.model.endswith(CONTRIBUTOR_SUFFIX):
            return QualificationCheck(
                name="model_identity", passed=False,
                detail=f"{config.model} is the Contributor tier: Meta may train on its inputs and outputs. "
                       "Benchmark the Standard-tier id instead")
        if not config.model.startswith("muse-"):
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"{config.model} is not a Muse model")
        if not self._api_key:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {self.credentials_hint}")
        check = super()._model_check(config)
        if check.passed is True:
            return check.model_copy(update={"detail": check.detail + "; Meta publishes no dated snapshots"})
        return check

    def _status_failure(self, exc: openai.APIStatusError) -> tuple[FailureKind, CostIncurred]:
        code = _error_code(exc)
        if code == "content_policy_violation":
            return FailureKind.REFUSAL, CostIncurred.UNKNOWN
        if exc.status_code == 400 and code is None and CONTEXT_OVERFLOW.search(str(exc)):
            return FailureKind.CONTEXT_EXCEEDED, CostIncurred.NO
        if exc.status_code == 504:  # gateway_timeout: the request ran too long; the model may have run
            return FailureKind.AMBIGUOUS_TIMEOUT, CostIncurred.UNKNOWN
        if exc.status_code == 429:
            return FailureKind.RATE_LIMITED, CostIncurred.NO
        if exc.status_code >= 500:
            return FailureKind.TRANSIENT_TRANSPORT, CostIncurred.NO
        # 400, 401, 402 billing, 403 no access, 404 unknown model, 413 too large: retrying cannot help
        return FailureKind.PROVIDER_ERROR, CostIncurred.NO
