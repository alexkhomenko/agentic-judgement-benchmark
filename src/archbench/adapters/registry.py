"""Adapter lookup by the configuration's `adapter` field. No fallback to another adapter or model."""

from __future__ import annotations

import os
from collections.abc import Mapping

from ..secrets import GUARD, LocalSecrets
from .base import InferenceAdapter
from .cloud.anthropic_messages import AnthropicAdapter
from .cloud.deepseek_chat import DeepSeekAdapter
from .cloud.gemini_generate import GeminiAdapter
from .cloud.glm_chat import GlmAdapter
from .cloud.mistral_chat import MistralAdapter
from .cloud.openai_responses import OpenAIAdapter
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
    return AdapterRegistry({
        "fake": FakeAdapter(),
        "fake-judge": FakeJudgeAdapter(),
        "anthropic": AnthropicAdapter(api_key=secrets.api_key("anthropic")),
        "openai": OpenAIAdapter(api_key=secrets.api_key("openai")),
        "gemini": GeminiAdapter(api_key=google_key),
        "mistral": MistralAdapter(api_key=mistral_key),
        "xai": XAIAdapter(api_key=xai_key),
        "deepseek": DeepSeekAdapter(api_key=deepseek_key),
        "glm": GlmAdapter(api_key=zhipu_key),
        "ollama": OllamaAdapter(),
        "mlx-lm": MlxLmAdapter(),
        "typesafe": TypeSafeAdapter(api_key=typesafe_key),
    })
