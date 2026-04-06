from __future__ import annotations

import pytest

from ragms.core.models import HybridSearchResult, RetrievalCandidate, RetrievalModelError


def test_retrieval_candidate_prefers_rerank_then_rrf_for_final_score() -> None:
    candidate = RetrievalCandidate(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="retrieval augmented generation",
        metadata={"page": 3},
        score=0.42,
        source_route="hybrid",
        rrf_score=0.88,
        rerank_score=0.97,
    )

    assert candidate.final_score == 0.97
    assert candidate.to_dict()["final_score"] == 0.97


def test_retrieval_candidate_supports_match_payload_normalization() -> None:
    candidate = RetrievalCandidate.from_match(
        {
            "id": "chunk-7",
            "document": "hello world",
            "score": 0.61,
            "metadata": {
                "document_id": "doc-7",
                "page": 2,
            },
            "dense_rank": 1,
        },
        source_route="dense",
    )

    assert candidate.chunk_id == "chunk-7"
    assert candidate.document_id == "doc-7"
    assert candidate.content == "hello world"
    assert candidate.metadata == {"document_id": "doc-7", "page": 2}
    assert candidate.dense_rank == 1
    assert candidate.source_route == "dense"


def test_retrieval_candidate_ordering_is_stable_on_score_ties() -> None:
    first = RetrievalCandidate(
        chunk_id="chunk-a",
        document_id="doc-1",
        content="alpha",
        metadata={},
        score=0.7,
        source_route="dense",
        rrf_score=0.9,
    )
    second = RetrievalCandidate(
        chunk_id="chunk-b",
        document_id="doc-2",
        content="beta",
        metadata={},
        score=0.7,
        source_route="dense",
        rrf_score=0.9,
    )
    third = RetrievalCandidate(
        chunk_id="chunk-c",
        document_id="doc-3",
        content="gamma",
        metadata={},
        score=0.5,
        source_route="sparse",
    )

    assert [candidate.chunk_id for candidate in sorted([third, second, first])] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]


def test_hybrid_search_result_sorts_candidates_and_serializes_counts() -> None:
    low = RetrievalCandidate(
        chunk_id="chunk-low",
        document_id="doc-low",
        content="low",
        metadata={},
        score=0.2,
        source_route="dense",
    )
    high = RetrievalCandidate(
        chunk_id="chunk-high",
        document_id="doc-high",
        content="high",
        metadata={},
        score=0.3,
        source_route="sparse",
        rrf_score=0.8,
        fallback_applied=True,
        fallback_reason="reranker_timeout",
    )

    result = HybridSearchResult(
        query="what is rag",
        collection="docs",
        candidates=(low, high),
        dense_count=5,
        sparse_count=4,
        filtered_out_count=2,
        candidate_top_n=10,
        fallback_applied=True,
        fallback_reason="reranker_timeout",
        debug_info={"query_normalized": "what is rag"},
    )

    payload = result.to_dict()

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-high", "chunk-low"]
    assert result.top_candidates(1)[0].chunk_id == "chunk-high"
    assert payload["fused_count"] == 2
    assert payload["dense_count"] == 5
    assert payload["sparse_count"] == 4
    assert payload["filtered_out_count"] == 2
    assert payload["fallback_applied"] is True
    assert payload["candidates"][0]["fallback_reason"] == "reranker_timeout"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: RetrievalCandidate(
                chunk_id="",
                document_id="doc",
                content="text",
                metadata={},
                score=0.1,
                source_route="dense",
            ),
            "chunk_id must not be empty",
        ),
        (
            lambda: RetrievalCandidate(
                chunk_id="chunk",
                document_id="doc",
                content="text",
                metadata={},
                score=0.1,
                source_route="dense",
                dense_rank=0,
            ),
            "dense_rank must be greater than zero",
        ),
        (
            lambda: HybridSearchResult(
                query=" ",
                collection="docs",
                candidates=(),
            ),
            "query must not be empty",
        ),
        (
            lambda: HybridSearchResult(
                query="hello",
                collection="docs",
                candidates=(),
                candidate_top_n=0,
            ),
            "candidate_top_n must be greater than zero",
        ),
    ],
)
def test_retrieval_models_validate_invalid_inputs(factory, message: str) -> None:
    with pytest.raises(RetrievalModelError, match=message):
        factory()
