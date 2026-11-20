"""Adapter lookup by the configuration's `adapter` field. No fallback to another adapter or model."""

from __future__ import annotations

import os
from collections.abc import Mapping

from ..secrets import GUARD, LocalSecrets
from .base import InferenceAdapter
from .cloud.anthropic_messages import AnthropicAdapter
from .cloud.bedrock_converse import BedrockAdapter
from .cloud.cohere_chat import CohereAdapter
from .cloud.deepseek_chat import DeepSeekAdapter
from .cloud.gemini_generate import GeminiAdapter
from .cloud.glm_chat import GlmAdapter
from .cloud.kimi_chat import KimiAdapter
from .cloud.mai_chat import MaiAdapter
from .cloud.meta_responses import MetaAdapter
from .cloud.minimax_messages import MiniMaxAdapter
from .cloud.mistral_chat import MistralAdapter
from .cloud.openai_responses import OpenAIAdapter
from .cloud.openrouter_chat import OpenRouterAdapter
from .cloud.typesafe_jev import TypeSafeAdapter
from .cloud.xai_responses import XAIAdapter
from .fake import FakeAdapter
from .fake_judge import FakeJudgeAdapter
from .local.mlx_lm import MlxLmAdapter
from .local.ollama_chat import OllamaAdapter


class AdapterRegistry:
    def __init__(self, adapters: Mapping[str, InferenceAdapter] | None = None) -> None:
        self._adapters: dict[str, InferenceAdapter] = dict(adapters or {})

    def register(self, adapter: InferenceAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"adapter {adapter.name} already registered")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> InferenceAdapter:
        try:
            return self._adapters[name]
        except KeyError:
            raise KeyError(f"no adapter registered as '{name}' (available: {sorted(self._adapters)})") from None


def default_registry(secrets: LocalSecrets | None = None) -> AdapterRegistry:
    """Adapters for this process. Keys from the local secrets file are passed directly to adapters;
    without one, each SDK resolves credentials from its own environment defaults."""
    secrets = secrets or LocalSecrets()
    # TypeSafe has no SDK credential chain here; fall back to its documented environment variable.
    typesafe_key = secrets.api_key("typesafe") or os.environ.get("TYPESAFE_API_KEY") or None
    if typesafe_key:
        GUARD.register(typesafe_key)
    # The Gemini adapter uses plain HTTP, so it applies the SDK's environment convention itself.
    google_key = (secrets.api_key("google") or os.environ.get("GOOGLE_API_KEY")
                  or os.environ.get("GEMINI_API_KEY") or None)
    if google_key:
        GUARD.register(google_key)
    mistral_key = secrets.api_key("mistral") or os.environ.get("MISTRAL_API_KEY") or None
    if mistral_key:
        GUARD.register(mistral_key)
    xai_key = secrets.api_key("xai") or os.environ.get("XAI_API_KEY") or None
    if xai_key:
        GUARD.register(xai_key)
    deepseek_key = secrets.api_key("deepseek") or os.environ.get("DEEPSEEK_API_KEY") or None
    if deepseek_key:
        GUARD.register(deepseek_key)
    zhipu_key = secrets.api_key("zhipu") or os.environ.get("ZAI_API_KEY") or None
    if zhipu_key:
        GUARD.register(zhipu_key)
    # Never ANTHROPIC_API_KEY: MiniMax's Anthropic-compatible API must not receive an Anthropic credential.
    minimax_key = secrets.api_key("minimax") or os.environ.get("MINIMAX_API_KEY") or None
    if minimax_key:
        GUARD.register(minimax_key)
    cohere_key = secrets.api_key("cohere") or os.environ.get("CO_API_KEY") or None
    if cohere_key:
        GUARD.register(cohere_key)
    moonshot_key = secrets.api_key("moonshot") or os.environ.get("MOONSHOT_API_KEY") or None
    if moonshot_key:
        GUARD.register(moonshot_key)
    # Bedrock API key (a bearer token); AWS SDKs read the same environment variable. SigV4 credentials are not used.
    bedrock_key = secrets.api_key("bedrock") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or None
    if bedrock_key:
        GUARD.register(bedrock_key)
    # Meta Model API key; MODEL_API_KEY is the variable Meta's samples use (no SDK reads it automatically).
    meta_key = secrets.api_key("meta") or os.environ.get("MODEL_API_KEY") or None
    if meta_key:
        GUARD.register(meta_key)
    # Microsoft Foundry resource key for MAI deployments; AZURE_INFERENCE_CREDENTIAL is the Azure AI Inference samples'
    # variable. AZURE_OPENAI_API_KEY is never read: it belongs to a different service and may name another resource.
    microsoft_key = secrets.api_key("microsoft") or os.environ.get("AZURE_INFERENCE_CREDENTIAL") or None
    if microsoft_key:
        GUARD.register(microsoft_key)
    # OpenRouter key, used only for vendor-pinned routes (see openrouter_chat.ROUTES); no SDK reads it automatically.
    openrouter_key = secrets.api_key("openrouter") or os.environ.get("OPENROUTER_API_KEY") or None
    if openrouter_key:
        GUARD.register(openrouter_key)
    return AdapterRegistry({
        "fake": FakeAdapter(),
        "fake-judge": FakeJudgeAdapter(),
        "anthropic": AnthropicAdapter(api_key=secrets.api_key("anthropic")),
        "openai": OpenAIAdapter(api_key=secrets.api_key("openai")),
        "gemini": GeminiAdapter(api_key=google_key),
        "mistral": MistralAdapter(api_key=mistral_key),
        "xai": XAIAdapter(api_key=xai_key),
        "meta": MetaAdapter(api_key=meta_key),
        "mai": MaiAdapter(api_key=microsoft_key),
        "deepseek": DeepSeekAdapter(api_key=deepseek_key),
        "glm": GlmAdapter(api_key=zhipu_key),
        "kimi": KimiAdapter(api_key=moonshot_key),
        "minimax": MiniMaxAdapter(api_key=minimax_key),
        "cohere": CohereAdapter(api_key=cohere_key),
        "bedrock": BedrockAdapter(api_key=bedrock_key),
        "openrouter": OpenRouterAdapter(api_key=openrouter_key),
        "ollama": OllamaAdapter(),
        "mlx-lm": MlxLmAdapter(),
        "typesafe": TypeSafeAdapter(api_key=typesafe_key),
    })
