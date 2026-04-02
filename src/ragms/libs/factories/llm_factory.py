"""Factory for text LLM provider instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragms.libs.abstractions import BaseLLM
from ragms.libs.providers.llm.deepseek_llm import DeepSeekLLM
from ragms.libs.providers.llm.openai_llm import OpenAILLM
from ragms.libs.providers.llm.qwen_llm import QwenLLM
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings, LLMSettings


def _coerce_llm_config(config: AppSettings | LLMSettings | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, AppSettings):
        return config.llm.model_dump(mode="python")
    if isinstance(config, LLMSettings):
        return config.model_dump(mode="python")
    return dict(config)


class LLMFactory:
    """Create text-generation model implementations from provider configuration."""

    _PROVIDERS = {
        "openai": OpenAILLM,
        "qwen": QwenLLM,
        "deepseek": DeepSeekLLM,
    }

    @staticmethod
    def create(config: AppSettings | LLMSettings | Mapping[str, Any] | None = None) -> BaseLLM:
        """Return an LLM implementation for the configured provider."""

        options = _coerce_llm_config(config)
        if isinstance(config, Mapping) and "provider" not in options:
            raise RagMSError("Missing llm provider in configuration")

        provider = str(options.pop("provider", "openai")).strip().lower() or "openai"
        provider_class = LLMFactory._PROVIDERS.get(provider)
        if provider_class is None:
            raise RagMSError(f"Unknown llm provider: {provider}")
        return provider_class(**options)
