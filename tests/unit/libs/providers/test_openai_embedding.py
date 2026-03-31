from __future__ import annotations

import pytest

from ragms.libs.providers.embeddings.openai_embedding import OpenAIEmbedding


def test_openai_embedding_generates_stable_vectors_for_documents_and_query() -> None:
    embedding = OpenAIEmbedding(model="text-embedding-3-large", api_key="test-key", batch_size=2)

    documents = embedding.embed_documents(["alpha", "beta"])
    query = embedding.embed_query("alpha")

    assert len(documents) == 2
    assert len(documents[0]) == 3
    assert len(query) == 3
    assert documents[0] != documents[1]


def test_openai_embedding_rejects_empty_input_and_invalid_config() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        OpenAIEmbedding(model="text-embedding-3-large", api_key=None)

    with pytest.raises(ValueError, match="Unsupported OpenAI embedding model"):
        OpenAIEmbedding(model="bad-model", api_key="test-key")

    embedding = OpenAIEmbedding(model="text-embedding-3-large", api_key="test-key")
    with pytest.raises(ValueError, match="texts must not be empty"):
        embedding.embed_documents([])

    with pytest.raises(ValueError, match="query text must not be empty"):
        embedding.embed_query("   ")


def test_openai_embedding_detects_dimension_mismatch() -> None:
    embedding = OpenAIEmbedding(model="text-embedding-3-large", api_key="test-key", dimensions=2)

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        embedding._validate_dimensions([[1.0, 2.0], [1.0, 2.0, 3.0]])
