from __future__ import annotations

import pytest

from ragms.core.models import RetrievalCandidate
from ragms.core.query_engine.hybrid_search import HybridSearchError, reciprocal_rank_fusion


def _candidate(
    *,
    chunk_id: str,
    document_id: str,
    source_route: str,
    score: float,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {chunk_id}",
        metadata={"document_id": document_id},
        score=score,
        source_route=source_route,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
    )


def test_rrf_uses_rank_not_raw_scores() -> None:
    dense = [
        _candidate(chunk_id="chunk-a", document_id="doc-a", source_route="dense", score=0.10, dense_rank=1),
        _candidate(chunk_id="chunk-b", document_id="doc-b", source_route="dense", score=999.0, dense_rank=2),
    ]
    sparse: list[RetrievalCandidate] = []

    results = reciprocal_rank_fusion(dense, sparse, k=60)

    assert [candidate.chunk_id for candidate in results] == ["chunk-a", "chunk-b"]
    assert results[0].rrf_score > results[1].rrf_score


def test_rrf_deduplicates_overlap_and_naturally_boosts_shared_hits() -> None:
    dense = [
        _candidate(chunk_id="shared", document_id="doc-1", source_route="dense", score=0.3, dense_rank=2),
        _candidate(chunk_id="dense-only", document_id="doc-2", source_route="dense", score=0.9, dense_rank=1),
    ]
    sparse = [
        _candidate(chunk_id="shared", document_id="doc-1", source_route="sparse", score=0.2, sparse_rank=1),
        _candidate(chunk_id="sparse-only", document_id="doc-3", source_route="sparse", score=0.8, sparse_rank=2),
    ]

    results = reciprocal_rank_fusion(dense, sparse, k=60)

    assert [candidate.chunk_id for candidate in results] == ["shared", "dense-only", "sparse-only"]
    assert results[0].source_route == "hybrid"
    assert results[0].dense_rank == 2
    assert results[0].sparse_rank == 1


def test_rrf_keeps_single_route_hits() -> None:
    dense = [
        _candidate(chunk_id="dense-only", document_id="doc-1", source_route="dense", score=0.4, dense_rank=1)
    ]
    sparse = [
        _candidate(chunk_id="sparse-only", document_id="doc-2", source_route="sparse", score=0.5, sparse_rank=1)
    ]

    results = reciprocal_rank_fusion(dense, sparse, k=60)

    assert {candidate.chunk_id for candidate in results} == {"dense-only", "sparse-only"}
    assert {candidate.source_route for candidate in results} == {"dense", "sparse"}


def test_rrf_is_stable_even_if_input_lists_are_not_pre_sorted() -> None:
    dense_unsorted = [
        _candidate(chunk_id="chunk-b", document_id="doc-b", source_route="dense", score=0.1, dense_rank=2),
        _candidate(chunk_id="chunk-a", document_id="doc-a", source_route="dense", score=0.9, dense_rank=1),
    ]
    sparse_unsorted = [
        _candidate(chunk_id="chunk-c", document_id="doc-c", source_route="sparse", score=0.2, sparse_rank=2),
        _candidate(chunk_id="chunk-a", document_id="doc-a", source_route="sparse", score=0.7, sparse_rank=1),
    ]

    results = reciprocal_rank_fusion(dense_unsorted, sparse_unsorted, k=60)

    assert [candidate.chunk_id for candidate in results] == ["chunk-a", "chunk-b", "chunk-c"]


def test_rrf_respects_configurable_smoothing_k() -> None:
    dense = [
        _candidate(chunk_id="chunk-a", document_id="doc-a", source_route="dense", score=0.1, dense_rank=1),
        _candidate(chunk_id="chunk-b", document_id="doc-b", source_route="dense", score=0.2, dense_rank=2),
    ]

    low_k = reciprocal_rank_fusion(dense, [], k=1)
    high_k = reciprocal_rank_fusion(dense, [], k=100)

    assert low_k[0].rrf_score - low_k[1].rrf_score > high_k[0].rrf_score - high_k[1].rrf_score


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(HybridSearchError, match="k must be greater than zero"):
        reciprocal_rank_fusion([], [], k=0)
