"""Factory for reranker provider instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragms.libs.abstractions import BaseReranker
from ragms.libs.providers.rerankers.cross_encoder_reranker import CrossEncoderReranker
from ragms.libs.providers.rerankers.disabled_reranker import DisabledReranker
from ragms.libs.providers.rerankers.llm_reranker import LLMReranker
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings


def _coerce_reranker_config(config: AppSettings | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, AppSettings):
        return {"provider": config.retrieval.rerank_backend}
    return dict(config)


class RerankerFactory:
    """Create reranker implementations from provider configuration."""

    _PROVIDERS = {
        "disabled": DisabledReranker,
        "cross_encoder": CrossEncoderReranker,
        "llm_reranker": LLMReranker,
    }

    @staticmethod
    def create(config: AppSettings | Mapping[str, Any] | None = None) -> BaseReranker:
        """Return a reranker implementation for the configured provider."""

        options = _coerce_reranker_config(config)
        if isinstance(config, Mapping) and "provider" not in options and "backend" not in options:
            raise RagMSError("Missing reranker provider in configuration")

        provider = str(options.pop("provider", options.pop("backend", "disabled"))).strip().lower()
        provider = provider or "disabled"
        provider_class = RerankerFactory._PROVIDERS.get(provider)
        if provider_class is None:
            raise RagMSError(f"Unknown reranker provider: {provider}")
        return provider_class(**options)
