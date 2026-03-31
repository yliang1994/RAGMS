from __future__ import annotations

import pytest

from ragms.libs.factories.embedding_factory import EmbeddingFactory
from ragms.libs.providers.embeddings.openai_embedding import OpenAIEmbedding


def test_embedding_factory_creates_configured_provider() -> None:
    embedding = EmbeddingFactory.create({"provider": "openai", "model": "text-embedding-3-large", "batch_size": 32})
    assert isinstance(embedding, OpenAIEmbedding)
    assert embedding.batch_size == 32


def test_embedding_factory_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="requires provider and model"):
        EmbeddingFactory.create({"provider": "openai"})

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingFactory.create({"provider": "missing", "model": "demo"})

