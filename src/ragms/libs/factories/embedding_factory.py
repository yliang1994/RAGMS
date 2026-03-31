from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseEmbedding
from ragms.libs.providers.embeddings.bge_embedding import BGEEmbedding
from ragms.libs.providers.embeddings.jina_embedding import JinaEmbedding
from ragms.libs.providers.embeddings.openai_embedding import OpenAIEmbedding


class EmbeddingFactory:
    _REGISTRY = {
        "openai": OpenAIEmbedding,
        "bge": BGEEmbedding,
        "jina": JinaEmbedding,
    }

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseEmbedding:
        provider = config.get("provider")
        model = config.get("model")
        if not provider or not model:
            raise ValueError("Embedding config requires provider and model")
        try:
            embedding_class = cls._REGISTRY[provider]
        except KeyError as exc:
            raise ValueError(f"Unknown embedding provider: {provider}") from exc
        return embedding_class(
            model=model,
            api_key=config.get("api_key"),
            batch_size=config.get("batch_size", 64),
        )

