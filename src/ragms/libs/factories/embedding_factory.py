"""Factory for embedding provider instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragms.libs.abstractions import BaseEmbedding
from ragms.libs.providers.embeddings.openai_embedding import OpenAIEmbedding
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings, EmbeddingSettings


def _coerce_embedding_config(
    config: AppSettings | EmbeddingSettings | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, AppSettings):
        return config.embedding.model_dump(mode="python")
    if isinstance(config, EmbeddingSettings):
        return config.model_dump(mode="python")
    return dict(config)


class EmbeddingFactory:
    """Create embedding implementations from provider configuration."""

    _PROVIDERS = {
        "openai": OpenAIEmbedding,
    }

    @staticmethod
    def create(
        config: AppSettings | EmbeddingSettings | Mapping[str, Any] | None = None,
    ) -> BaseEmbedding:
        """Return an embedding implementation for the configured provider."""

        options = _coerce_embedding_config(config)
        if isinstance(config, Mapping) and "provider" not in options:
            raise RagMSError("Missing embedding provider in configuration")

        provider = str(options.pop("provider", "openai")).strip().lower() or "openai"
        provider_class = EmbeddingFactory._PROVIDERS.get(provider)
        if provider_class is None:
            raise RagMSError(f"Unknown embedding provider: {provider}")
        return provider_class(**options)
