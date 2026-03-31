from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseVectorStore
from ragms.libs.providers.vector_stores.chroma_store import ChromaStore


class VectorStoreFactory:
    _REGISTRY = {
        "chroma": ChromaStore,
    }

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseVectorStore:
        provider = config.get("provider", config.get("backend"))
        try:
            vector_store_class = cls._REGISTRY[provider]
        except KeyError as exc:
            raise ValueError(f"Unknown vector store provider: {provider}") from exc
        return vector_store_class(
            path=str(config["path"]),
            collection_prefix=config.get("collection_prefix", "ragms_"),
        )

