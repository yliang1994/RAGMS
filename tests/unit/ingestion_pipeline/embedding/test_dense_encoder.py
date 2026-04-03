from __future__ import annotations

import pytest

from ragms.ingestion_pipeline.embedding import DenseEncoder, DenseEncodingError


class RecordingEmbedding:
    def __init__(self, *, dimension: int = 3) -> None:
        self.dimension = dimension
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [[float(len(text)), float(index), float(self.dimension)] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [float(len(text)), 1.0, float(self.dimension)]


class FlakyEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class WrongCountEmbedding:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class WrongDimensionEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == 1:
            return [[0.1, 0.2] for _ in texts]
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_dense_encoder_batches_and_deduplicates_document_encoding() -> None:
    embedding = RecordingEmbedding()
    encoder = DenseEncoder(embedding, batch_size=2)

    vectors = encoder.encode_documents(
        [
            {"content": "alpha"},
            {"content": "beta"},
            {"content": "alpha"},
        ]
    )

    assert len(vectors) == 3
    assert vectors[0] == vectors[2]
    assert embedding.document_calls == [["alpha", "beta"]]


def test_dense_encoder_preserves_input_order_across_batches() -> None:
    embedding = RecordingEmbedding()
    encoder = DenseEncoder(embedding, batch_size=2)

    vectors = encoder.encode_documents(["one", "two", "three"])

    assert embedding.document_calls == [["one", "two"], ["three"]]
    assert vectors == [
        [3.0, 0.0, 3.0],
        [3.0, 1.0, 3.0],
        [5.0, 0.0, 3.0],
    ]


def test_dense_encoder_retries_transient_batch_failures() -> None:
    embedding = FlakyEmbedding()
    encoder = DenseEncoder(embedding, max_retries=2)

    vectors = encoder.encode_documents(["alpha"])

    assert vectors == [[0.1, 0.2]]
    assert embedding.calls == 2


def test_dense_encoder_rejects_unexpected_embedding_count() -> None:
    encoder = DenseEncoder(WrongCountEmbedding())

    with pytest.raises(
        DenseEncodingError,
        match="Dense encoder returned an unexpected embedding count",
    ):
        encoder.encode_documents(["alpha", "beta"])


def test_dense_encoder_rejects_dimension_changes_between_batches() -> None:
    encoder = DenseEncoder(WrongDimensionEmbedding(), batch_size=1)

    with pytest.raises(
        DenseEncodingError,
        match="Dense encoder returned embedding dimension 3, expected 2",
    ):
        encoder.encode_documents(["alpha", "beta"])


def test_dense_encoder_rejects_empty_document_content() -> None:
    encoder = DenseEncoder(RecordingEmbedding())

    with pytest.raises(DenseEncodingError, match="Document content must not be empty"):
        encoder.encode_documents([{"content": "   "}])


def test_dense_encoder_returns_empty_list_for_empty_input() -> None:
    encoder = DenseEncoder(RecordingEmbedding())

    assert encoder.encode_documents([]) == []


def test_dense_encoder_encodes_query() -> None:
    embedding = RecordingEmbedding()
    encoder = DenseEncoder(embedding)

    vector = encoder.encode_query("search me")

    assert vector == [9.0, 1.0, 3.0]
    assert embedding.query_calls == ["search me"]


def test_dense_encoder_rejects_empty_query() -> None:
    encoder = DenseEncoder(RecordingEmbedding())

    with pytest.raises(DenseEncodingError, match="Query text must not be empty"):
        encoder.encode_query(" ")
