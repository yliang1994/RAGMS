from __future__ import annotations

import time

import pytest

from ragms.core.models import RetrievalCandidate
from ragms.core.query_engine import QueryProcessor
from ragms.core.query_engine.hybrid_search import HybridSearch, HybridSearchError


class RecordingRetriever:
    def __init__(
        self,
        *,
        results: list[RetrievalCandidate],
        delay_seconds: float = 0.0,
        fail: bool = False,
    ) -> None:
        self.results = list(results)
        self.delay_seconds = delay_seconds
        self.fail = fail
        self.calls: list[object] = []

    def retrieve(self, processed_query) -> list[RetrievalCandidate]:
        self.calls.append(processed_query)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.fail:
            raise RuntimeError("boom")
        return list(self.results)


def _candidate(
    *,
    chunk_id: str,
    document_id: str,
    source_route: str,
    score: float,
    metadata: dict[str, object] | None = None,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {chunk_id}",
        metadata=dict(metadata or {"document_id": document_id}),
        score=score,
        source_route=source_route,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
    )


def test_hybrid_search_runs_dense_and_sparse_in_parallel() -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process("hybrid retrieval", top_k=5)
    dense = RecordingRetriever(
        results=[_candidate(chunk_id="dense-1", document_id="doc-1", source_route="dense", score=0.8, dense_rank=1)],
        delay_seconds=0.15,
    )
    sparse = RecordingRetriever(
        results=[_candidate(chunk_id="sparse-1", document_id="doc-2", source_route="sparse", score=0.7, sparse_rank=1)],
        delay_seconds=0.15,
    )

    hybrid = HybridSearch(dense, sparse, candidate_top_n=5)
    started_at = time.monotonic()
    result = hybrid.search(processed)
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.27
    assert len(dense.calls) == 1
    assert len(sparse.calls) == 1
    assert result.fused_count == 2


def test_hybrid_search_applies_post_filters_before_and_after_fusion() -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process(
        "hybrid retrieval",
        top_k=5,
        filters={"owner": "team-a", "tags": {"$contains": "rag"}},
    )
    dense = RecordingRetriever(
        results=[
            _candidate(
                chunk_id="shared",
                document_id="doc-1",
                source_route="dense",
                score=0.8,
                dense_rank=1,
                metadata={"document_id": "doc-1", "owner": "team-a", "tags": ["rag", "ops"]},
            ),
            _candidate(
                chunk_id="drop-dense",
                document_id="doc-2",
                source_route="dense",
                score=0.7,
                dense_rank=2,
                metadata={"document_id": "doc-2", "owner": "team-b", "tags": ["rag"]},
            ),
        ]
    )
    sparse = RecordingRetriever(
        results=[
            _candidate(
                chunk_id="shared",
                document_id="doc-1",
                source_route="sparse",
                score=0.5,
                sparse_rank=1,
                metadata={"document_id": "doc-1", "tags": ["rag", "ops"]},
            ),
            _candidate(
                chunk_id="drop-sparse",
                document_id="doc-3",
                source_route="sparse",
                score=0.4,
                sparse_rank=2,
                metadata={"document_id": "doc-3", "owner": "team-a", "tags": ["other"]},
            ),
        ]
    )

    result = HybridSearch(dense, sparse, candidate_top_n=5).search(processed)

    assert [candidate.chunk_id for candidate in result.candidates] == ["shared"]
    assert result.filtered_out_count == 2
    assert result.debug_info["filters"]["dense_removed"] == 1
    assert result.debug_info["filters"]["sparse_removed"] == 1
    assert result.debug_info["filters"]["fused_removed"] == 0
    assert result.candidates[0].source_route == "hybrid"


def test_hybrid_search_keeps_missing_post_filter_fields_by_default() -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process("hybrid retrieval", top_k=5, filters={"owner": "team-a"})
    dense = RecordingRetriever(
        results=[
            _candidate(
                chunk_id="candidate-1",
                document_id="doc-1",
                source_route="dense",
                score=0.8,
                dense_rank=1,
                metadata={"document_id": "doc-1"},
            )
        ]
    )
    sparse = RecordingRetriever(results=[])

    result = HybridSearch(dense, sparse, candidate_top_n=5).search(processed)

    assert [candidate.chunk_id for candidate in result.candidates] == ["candidate-1"]
    assert result.filtered_out_count == 0


def test_hybrid_search_truncates_to_candidate_top_n_and_records_debug_counts() -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process("hybrid retrieval", top_k=5)
    dense = RecordingRetriever(
        results=[
            _candidate(chunk_id="a", document_id="doc-a", source_route="dense", score=0.8, dense_rank=1),
            _candidate(chunk_id="b", document_id="doc-b", source_route="dense", score=0.7, dense_rank=2),
        ]
    )
    sparse = RecordingRetriever(
        results=[
            _candidate(chunk_id="c", document_id="doc-c", source_route="sparse", score=0.9, sparse_rank=1),
            _candidate(chunk_id="d", document_id="doc-d", source_route="sparse", score=0.6, sparse_rank=2),
        ]
    )

    result = HybridSearch(dense, sparse, candidate_top_n=2, rrf_k=60).search(processed)

    assert len(result.candidates) == 2
    assert result.candidate_top_n == 2
    assert result.debug_info["fusion"]["before_post_filter"] == 4
    assert result.debug_info["fusion"]["returned_count"] == 2
    assert result.debug_info["fusion"]["truncated_count"] == 2


def test_hybrid_search_wraps_retriever_failures() -> None:
    processor = QueryProcessor(default_collection="docs")
    processed = processor.process("hybrid retrieval", top_k=5)
    dense = RecordingRetriever(results=[], fail=True)
    sparse = RecordingRetriever(results=[])

    with pytest.raises(HybridSearchError, match="Hybrid search retrieval failed"):
        HybridSearch(dense, sparse, candidate_top_n=5).search(processed)
