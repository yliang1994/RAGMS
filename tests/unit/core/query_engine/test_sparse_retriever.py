from __future__ import annotations

from pathlib import Path

import pytest

from ragms.core.query_engine import QueryProcessor
from ragms.core.query_engine.retrievers import SparseRetriever, SparseRetrieverError
from ragms.ingestion_pipeline.storage import ChunkRecord
from ragms.storage.indexes import BM25Indexer


def _record(
    *,
    chunk_id: str,
    document_id: str,
    content: str,
    source_path: str,
    metadata: dict[str, object],
    tokens: list[str],
    term_frequencies: dict[str, int],
) -> ChunkRecord:
    document_length = sum(term_frequencies.values())
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        metadata=metadata,
        dense_vector=[0.1, 0.2],
        sparse_vector={
            "tokens": tokens,
            "term_frequencies": term_frequencies,
            "term_weights": {
                token: round(count / document_length, 6)
                for token, count in term_frequencies.items()
            },
            "document_length": document_length,
            "unique_terms": len(term_frequencies),
        },
        content_hash=f"hash-{chunk_id}",
        source_path=source_path,
        chunk_index=int(metadata.get("chunk_index", 0) or 0),
        image_refs=[],
    )


class RecordingIndexer:
    def __init__(self, *, results: list[dict[str, object]]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query_terms: list[str] | str,
        *,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        captured_terms = list(query_terms) if not isinstance(query_terms, str) else query_terms.split()
        self.calls.append(
            {
                "query_terms": captured_terms,
                "top_k": top_k,
                "filters": dict(filters or {}),
            }
        )
        return self.results[:top_k]


class FailingIndexer:
    def search(
        self,
        query_terms: list[str] | str,
        *,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        raise RuntimeError("boom")


def test_bm25_indexer_search_respects_collection_and_metadata_filters(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path / "indexes", collection="docs")
    indexer.index_document(
        _record(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="RAG uses BM25 sparse retrieval",
            source_path="docs/rag.pdf",
            metadata={"doc_type": "pdf", "section": "overview"},
            tokens=["rag", "bm25", "sparse", "retrieval"],
            term_frequencies={"rag": 1, "bm25": 1, "sparse": 1, "retrieval": 1},
        )
    )
    indexer.index_document(
        _record(
            chunk_id="chunk-2",
            document_id="doc-2",
            content="Dense retrieval relies on embeddings",
            source_path="docs/dense.pdf",
            metadata={"doc_type": "md", "section": "dense"},
            tokens=["dense", "retrieval", "embeddings"],
            term_frequencies={"dense": 1, "retrieval": 1, "embeddings": 1},
        )
    )

    matches = indexer.search(
        ["rag", "retrieval"],
        top_k=5,
        filters={"collection": "docs", "doc_type": "pdf"},
    )

    assert [match["id"] for match in matches] == ["chunk-1"]
    assert matches[0]["document_id"] == "doc-1"
    assert matches[0]["metadata"]["collection"] == "docs"
    assert matches[0]["metadata"]["source_path"] == "docs/rag.pdf"


def test_sparse_retriever_runs_one_bm25_query_with_combined_terms(tmp_path: Path) -> None:
    indexer = RecordingIndexer(
        results=[
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "document": "RAG overview",
                "score": 1.73,
                "metadata": {"document_id": "doc-1", "doc_type": "pdf"},
            },
            {
                "id": "chunk-2",
                "document_id": "doc-2",
                "document": "LLM retrieval",
                "score": 1.02,
                "metadata": {"document_id": "doc-2", "doc_type": "pdf"},
            },
        ]
    )
    processor = QueryProcessor(
        default_collection="docs",
        synonym_map={"rag": ["retrieval augmented generation"]},
        pre_filter_fields={"doc_type"},
    )
    processed = processor.process(
        "RAG retrieval",
        top_k=2,
        filters={"doc_type": "pdf", "owner": "platform"},
    )

    retriever = SparseRetriever(indexer)
    candidates = retriever.retrieve(processed)

    assert indexer.calls == [
        {
            "query_terms": ["rag", "retrieval", "retrieval augmented generation"],
            "top_k": 2,
            "filters": {"collection": "docs", "doc_type": "pdf"},
        }
    ]
    assert [candidate.chunk_id for candidate in candidates] == ["chunk-1", "chunk-2"]
    assert [candidate.sparse_rank for candidate in candidates] == [1, 2]
    assert all(candidate.source_route == "sparse" for candidate in candidates)
    assert candidates[0].sparse_score == 1.73


def test_sparse_retriever_returns_empty_list_when_bm25_has_no_hits(tmp_path: Path) -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process("retrieval", top_k=3)
    retriever = SparseRetriever(RecordingIndexer(results=[]))

    assert retriever.retrieve(processed) == []


@pytest.mark.parametrize(
    ("indexer", "message"),
    [
        (FailingIndexer(), "Sparse retriever BM25 query failed"),
        (
            RecordingIndexer(
                results=[
                    {
                        "id": "chunk-1",
                        "document": "missing document id",
                        "score": 0.8,
                        "metadata": {},
                    }
                ]
            ),
            "Sparse retriever returned an invalid match payload",
        ),
    ],
)
def test_sparse_retriever_wraps_failures(indexer, message: str) -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process("retrieval", top_k=1)
    retriever = SparseRetriever(indexer)

    with pytest.raises(SparseRetrieverError, match=message):
        retriever.retrieve(processed)
