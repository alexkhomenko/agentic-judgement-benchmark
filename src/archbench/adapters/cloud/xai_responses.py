"""Native xAI (Grok) adapter: the Responses API at https://api.x.ai/v1, through the official `openai` SDK
as xAI's own examples do.

Request handling, stateless continuation (`store: false`, encrypted reasoning replayed unchanged), failure
mapping, and extraction are inherited from `OpenAIAdapter`. Differences:
- Credential isolation. The SDK falls back to OPENAI_API_KEY and sends OPENAI_ORG_ID / OPENAI_PROJECT_ID
  as headers when they are unset. This adapter refuses to build a client without an explicit xAI key and
  clears the organization, project, and admin key, so no OpenAI credential or identifier reaches xAI.
- Cost. xAI reports `usage.cost_in_usd_ticks` (1 USD = 10^10 ticks), which becomes api_cost_usd; the
  configured price table is kept beside it as api_cost_usd_configured.
- Qualification reads xAI's Models API: context length, aliases, default reasoning effort, the model's
  build fingerprint and version (`/v1/language-models`), and its published prices, which must match the
  configuration's pricing so a price change cannot silently skew cost reports.
- `max_output_tokens` counts visible output only on xAI (reasoning tokens are not included).

Configuration parameters and adapter_options are those of `OpenAIAdapter` (pricing may set long_context
with threshold_input_tokens; xAI prices prompts of 200k tokens or more at long-context rates).
"""

from __future__ import annotations

from typing import Any

import httpx2
import openai

from ...domain.configuration import ModelConfiguration
from ..base import QualificationCheck
from .openai_responses import OpenAIAdapter
from .wire import CapturingTransport

XAI_BASE_URL = "https://api.x.ai/v1"
# Models API prices are USD cents per 100M tokens; configuration prices are USD per 1M tokens.
CENTS_PER_100M_TO_USD_PER_1M = 1 / 10_000
PRICE_FIELDS = {
    "input_usd_per_mtok": "prompt_text_token_price",
    "cached_input_usd_per_mtok": "cached_prompt_text_token_price",
    "output_usd_per_mtok": "completion_text_token_price",
}


class XAIAdapter(OpenAIAdapter):
    name = "xai"
    provider = "xai"
    base_url = XAI_BASE_URL
    credentials_hint = ("no xAI credentials found: add xai.api_key to secrets.local.yaml "
                        "(see secrets.example.yaml) or set XAI_API_KEY")

    def _client(self, config: ModelConfiguration, transport: CapturingTransport) -> openai.OpenAI:
        if not self._api_key:
            # Never let the SDK fall back to OPENAI_API_KEY: that would send an OpenAI key to xAI.
            raise openai.OpenAIError(self.credentials_hint)
        client = super()._client(config, transport)
        client.organization = None  # read from OPENAI_ORG_ID when unset; never sent to xAI
        client.project = None  # read from OPENAI_PROJECT_ID when unset
        client.admin_api_key = None  # read from OPENAI_ADMIN_KEY when unset
        return client

    def _model_check(self, config: ModelConfiguration) -> QualificationCheck:
        if not self._api_key:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {self.credentials_hint}")
        transport = CapturingTransport(self._transport_factory())
        try:
            with httpx2.Client(base_url=XAI_BASE_URL, transport=transport,
                               timeout=float(str(config.adapter_options.get("timeout_s", 600))),
                               headers={"authorization": f"Bearer {self._api_key}"}) as client:
                model = client.get(f"/models/{config.model}")
                language = client.get(f"/language-models/{config.model}")
        except httpx2.HTTPError as exc:
            return QualificationCheck(name="model_identity", passed=None, detail=f"not evaluated: {exc}")
        if model.status_code != 200:
            return QualificationCheck(name="model_identity", passed=False,
                                      detail=f"HTTP {model.status_code}: {model.text[:300]}")
        card: dict[str, Any] = model.json()
        build: dict[str, Any] = language.json() if language.status_code == 200 else {}
        problems = []
        if card.get("id") != config.model:
            problems.append(f"API reports id {card.get('id')} (aliases {card.get('aliases')})")
        context = card.get("context_length")
        if isinstance(context, int) and config.context_limit is not None and config.context_limit > context:
            problems.append(f"context_limit {config.context_limit} > context_length {context}")
        problems += _price_drift(config, {**build, **card})
        capabilities = card.get("capabilities") or {}
        detail = (f"{card.get('id')}: context_length={context}, aliases={card.get('aliases')}, "
                  f"default_reasoning_effort={capabilities.get('default_reasoning_effort')}, "
                  f"version={build.get('version')}, fingerprint={build.get('fingerprint')}")
        return QualificationCheck(name="model_identity", passed=not problems, detail="; ".join(problems) or detail)


def _price_drift(config: ModelConfiguration, card: dict[str, Any]) -> list[str]:
    """Configured prices must match the prices xAI publishes for the model."""
    pricing = config.adapter_options.get("pricing")
    if not isinstance(pricing, dict):
        return []
    problems = []
    for key, field in PRICE_FIELDS.items():
        published, configured = card.get(field), pricing.get(key)
        if isinstance(published, int | float) and isinstance(configured, int | float):
            usd = round(published * CENTS_PER_100M_TO_USD_PER_1M, 6)
            if abs(usd - configured) > 1e-9:
                problems.append(f"pricing {key} is {configured} but xAI publishes {usd}")
    return problems
