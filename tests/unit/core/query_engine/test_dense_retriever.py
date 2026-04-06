from __future__ import annotations

import pytest

from ragms.core.query_engine import QueryProcessor
from ragms.core.query_engine.retrievers import DenseRetriever, DenseRetrieverError


class RecordingEmbedding:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.9, 0.1, 0.3]


class RecordingVectorStore:
    def __init__(self, matches: list[dict[str, object]] | None = None) -> None:
        self.matches = list(matches or [])
        self.calls: list[dict[str, object]] = []

    def query(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append(
            {
                "query_vector": list(query_vector),
                "top_k": top_k,
                "filters": dict(filters or {}),
            }
        )
        return self.matches[:top_k]


class CollectionScopedVectorStore(RecordingVectorStore):
    def __init__(self, matches: list[dict[str, object]] | None = None) -> None:
        super().__init__(matches=matches)
        self.collection = "docs"


class FailingEmbedding:
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("boom")


class FailingVectorStore:
    def query(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        raise RuntimeError("boom")


def test_dense_retriever_calls_embedding_once_and_queries_vector_store() -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process(
        "  retrieval augmented generation  ",
        top_k=2,
        filters={"document_id": "doc-1", "owner": "platform"},
    )
    embedding = RecordingEmbedding()
    vector_store = RecordingVectorStore(
        matches=[
            {
                "id": "chunk-1",
                "document": "RAG overview",
                "score": 0.83,
                "metadata": {
                    "document_id": "doc-1",
                    "page": 3,
                    "collection": "docs",
                },
            },
            {
                "id": "chunk-2",
                "document": "Dense retrieval details",
                "score": 0.71,
                "metadata": {
                    "document_id": "doc-1",
                    "page": 4,
                    "collection": "docs",
                },
            },
        ]
    )

    retriever = DenseRetriever(embedding, vector_store)
    candidates = retriever.retrieve(processed)

    assert embedding.calls == ["retrieval augmented generation"]
    assert vector_store.calls == [
        {
            "query_vector": [0.9, 0.1, 0.3],
            "top_k": 2,
            "filters": {
                "collection": "docs",
                "document_id": "doc-1",
            },
        }
    ]
    assert [candidate.chunk_id for candidate in candidates] == ["chunk-1", "chunk-2"]
    assert [candidate.dense_rank for candidate in candidates] == [1, 2]
    assert all(candidate.source_route == "dense" for candidate in candidates)
    assert candidates[0].dense_score == 0.83


def test_dense_retriever_omits_collection_filter_for_collection_scoped_vector_store() -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process("retrieval", top_k=1, filters={"document_id": "doc-1"})
    embedding = RecordingEmbedding()
    vector_store = CollectionScopedVectorStore(
        matches=[
            {
                "id": "chunk-1",
                "document": "RAG overview",
                "score": 0.83,
                "metadata": {"document_id": "doc-1"},
            }
        ]
    )

    retriever = DenseRetriever(embedding, vector_store)
    retriever.retrieve(processed)

    assert vector_store.calls == [
        {
            "query_vector": [0.9, 0.1, 0.3],
            "top_k": 1,
            "filters": {"document_id": "doc-1"},
        }
    ]


def test_dense_retriever_returns_empty_list_for_empty_matches() -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process("retrieval", top_k=3)
    retriever = DenseRetriever(RecordingEmbedding(), RecordingVectorStore(matches=[]))

    assert retriever.retrieve(processed) == []


@pytest.mark.parametrize(
    ("embedding", "vector_store", "message"),
    [
        (FailingEmbedding(), RecordingVectorStore(), "Dense retriever failed to encode query"),
        (RecordingEmbedding(), FailingVectorStore(), "Dense retriever vector-store query failed"),
        (
            RecordingEmbedding(),
            RecordingVectorStore(
                matches=[
                    {
                        "id": "chunk-1",
                        "document": "missing metadata",
                        "score": 0.5,
                        "metadata": {},
                    }
                ]
            ),
            "Dense retriever returned an invalid match payload",
        ),
    ],
)
def test_dense_retriever_wraps_failures(embedding, vector_store, message: str) -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process("retrieval", top_k=1)
    retriever = DenseRetriever(embedding, vector_store)

    with pytest.raises(DenseRetrieverError, match=message):
        retriever.retrieve(processed)
