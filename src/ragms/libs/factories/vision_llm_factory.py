"""Factory for vision LLM provider instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragms.libs.abstractions import BaseVisionLLM
from ragms.libs.providers.vision_llms.gpt4o_vision_llm import GPT4oVisionLLM
from ragms.libs.providers.vision_llms.qwen_vl_llm import QwenVLLLM
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings, VisionLLMSettings


def _coerce_vision_config(
    config: AppSettings | VisionLLMSettings | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, AppSettings):
        return config.vision_llm.model_dump(mode="python")
    if isinstance(config, VisionLLMSettings):
        return config.model_dump(mode="python")
    return dict(config)


class VisionLLMFactory:
    """Create vision-language model implementations from provider configuration."""

    _PROVIDERS = {
        "gpt4o": GPT4oVisionLLM,
        "qwen_vl": QwenVLLLM,
    }

    @staticmethod
    def create(
        config: AppSettings | VisionLLMSettings | Mapping[str, Any] | None = None,
        *,
        document_language: str | None = None,
        deployment_environment: str | None = None,
    ) -> BaseVisionLLM:
        """Return a vision LLM implementation using explicit or routed provider selection."""

        options = _coerce_vision_config(config)
        if isinstance(config, Mapping) and "provider" not in options:
            raise RagMSError("Missing vision llm provider in configuration")

        provider = str(options.pop("provider", "auto")).strip().lower() or "auto"
        language_providers = {
            str(key).lower(): str(value).lower()
            for key, value in dict(options.pop("language_providers", {})).items()
        }
        environment_providers = {
            str(key).lower(): str(value).lower()
            for key, value in dict(options.pop("environment_providers", {})).items()
        }
        if provider == "auto":
            provider = VisionLLMFactory._resolve_routed_provider(
                document_language=document_language,
                deployment_environment=deployment_environment,
                language_providers=language_providers,
                environment_providers=environment_providers,
            )

        provider_class = VisionLLMFactory._PROVIDERS.get(provider)
        if provider_class is None:
            raise RagMSError(f"Unknown vision llm provider: {provider}")
        return provider_class(**options)

    @staticmethod
    def _resolve_routed_provider(
        *,
        document_language: str | None,
        deployment_environment: str | None,
        language_providers: dict[str, str],
        environment_providers: dict[str, str],
    ) -> str:
        normalized_language = (document_language or "").strip().lower()
        if normalized_language:
            if normalized_language in language_providers:
                return language_providers[normalized_language]
            for prefix, provider in language_providers.items():
                if normalized_language.startswith(prefix):
                    return provider

        normalized_environment = (deployment_environment or "").strip().lower()
        if normalized_environment and normalized_environment in environment_providers:
            return environment_providers[normalized_environment]

        return "gpt4o"
