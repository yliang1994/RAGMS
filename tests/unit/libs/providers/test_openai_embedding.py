from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from ragms.libs.providers.embeddings.openai_embedding import (
    EmbeddingProviderError,
    OpenAIEmbedding,
)


class FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.response: object | None = None
        self.error: Exception | None = None

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if self.error:
            raise self.error
        if self.response is None:
            raise AssertionError("No fake response queued")
        return self.response


class FakeOpenAIClient:
    def __init__(self, embeddings: FakeEmbeddingsAPI) -> None:
        self.embeddings = embeddings


def _build_response(vectors: list[list[float]]) -> object:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=vector) for vector in vectors]
    )


def _build_bad_request_error(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body=None)


def test_openai_embedding_embeds_documents() -> None:
    embeddings = FakeEmbeddingsAPI()
    embeddings.response = _build_response([[0.1, 0.2], [0.3, 0.4]])
    provider = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key="test-key",
        client=FakeOpenAIClient(embeddings),
    )

    vectors = provider.embed_documents(["alpha", "beta"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert embeddings.calls[0]["input"] == ["alpha", "beta"]
    assert embeddings.calls[0]["model"] == "text-embedding-3-small"


def test_openai_embedding_embeds_single_query() -> None:
    embeddings = FakeEmbeddingsAPI()
    embeddings.response = _build_response([[0.5, 0.6, 0.7]])
    provider = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key="test-key",
        client=FakeOpenAIClient(embeddings),
    )

    vector = provider.embed_query("hello")

    assert vector == [0.5, 0.6, 0.7]
    assert embeddings.calls[0]["input"] == "hello"


def test_openai_embedding_returns_empty_list_for_empty_documents() -> None:
    provider = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")

    assert provider.embed_documents([]) == []


def test_openai_embedding_rejects_empty_query() -> None:
    provider = OpenAIEmbedding(model="text-embedding-3-small", api_key="test-key")

    with pytest.raises(EmbeddingProviderError, match="OpenAI embedding input must not be empty"):
        provider.embed_query("")


def test_openai_embedding_rejects_inconsistent_dimensions() -> None:
    embeddings = FakeEmbeddingsAPI()
    embeddings.response = _build_response([[0.1, 0.2], [0.3]])
    provider = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key="test-key",
        client=FakeOpenAIClient(embeddings),
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="OpenAI returned embeddings with inconsistent dimensions",
    ):
        provider.embed_documents(["alpha", "beta"])


def test_openai_embedding_rejects_unexpected_dimension() -> None:
    embeddings = FakeEmbeddingsAPI()
    embeddings.response = _build_response([[0.1, 0.2, 0.3]])
    provider = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key="test-key",
        dimension=2,
        client=FakeOpenAIClient(embeddings),
    )

    with pytest.raises(
        EmbeddingProviderError,
        match="OpenAI returned embedding dimension 3, expected 2",
    ):
        provider.embed_query("hello")


def test_openai_embedding_maps_bad_request_failures() -> None:
    embeddings = FakeEmbeddingsAPI()
    embeddings.error = _build_bad_request_error("bad request")
    provider = OpenAIEmbedding(
        model="bad-model",
        api_key="test-key",
        client=FakeOpenAIClient(embeddings),
    )

    with pytest.raises(EmbeddingProviderError, match="OpenAI rejected embedding request: bad-model"):
        provider.embed_documents(["alpha"])
