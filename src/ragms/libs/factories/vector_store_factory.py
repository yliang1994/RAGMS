"""Factory for vector store provider instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragms.libs.abstractions import BaseVectorStore
from ragms.libs.providers.vector_stores.chroma_store import ChromaStore
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings, VectorStoreSettings


def _coerce_vector_store_config(
    config: AppSettings | VectorStoreSettings | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, AppSettings):
        return config.vector_store.model_dump(mode="python")
    if isinstance(config, VectorStoreSettings):
        return config.model_dump(mode="python")
    return dict(config)


class VectorStoreFactory:
    """Create vector store implementations from provider configuration."""

    _PROVIDERS = {
        "chroma": ChromaStore,
    }

    @staticmethod
    def create(
        config: AppSettings | VectorStoreSettings | Mapping[str, Any] | None = None,
    ) -> BaseVectorStore:
        """Return a vector store implementation for the configured backend."""

        options = _coerce_vector_store_config(config)
        backend = str(options.pop("backend", options.pop("provider", "chroma"))).strip().lower()
        backend = backend or "chroma"
        provider_class = VectorStoreFactory._PROVIDERS.get(backend)
        if provider_class is None:
            raise RagMSError(f"Unknown vector store provider: {backend}")
        return provider_class(**options)
