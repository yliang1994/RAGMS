from __future__ import annotations

import pytest

from ragms.core.models import HybridSearchResult, RetrievalCandidate
from ragms.core.query_engine.reranker import Reranker, RerankerError


class RecordingProvider:
    def __init__(
        self,
        *,
        ranked_items: list[dict[str, object]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.ranked_items = list(ranked_items or [])
        self.error = error
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append(
            {
                "query": query,
                "candidates": list(candidates),
                "top_k": top_k,
            }
        )
        if self.error is not None:
            raise self.error
        return list(self.ranked_items)


def _candidate(
    *,
    chunk_id: str,
    document_id: str,
    rrf_score: float,
    source_route: str = "hybrid",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {chunk_id}",
        metadata={"document_id": document_id},
        score=rrf_score,
        source_route=source_route,
        rrf_score=rrf_score,
    )


def _result(*candidates: RetrievalCandidate, candidate_top_n: int = 2) -> HybridSearchResult:
    return HybridSearchResult(
        query="what is rag",
        collection="docs",
        candidates=tuple(candidates),
        dense_count=3,
        sparse_count=3,
        filtered_out_count=1,
        candidate_top_n=candidate_top_n,
        debug_info={"fusion": {"candidate_top_n": candidate_top_n}},
    )


def test_reranker_runs_provider_and_respects_candidate_top_n() -> None:
    first = _candidate(chunk_id="chunk-1", document_id="doc-1", rrf_score=0.8)
    second = _candidate(chunk_id="chunk-2", document_id="doc-2", rrf_score=0.7)
    third = _candidate(chunk_id="chunk-3", document_id="doc-3", rrf_score=0.6)
    provider = RecordingProvider(
        ranked_items=[
            {"document": second.to_dict(), "score": 0.91},
            {"document": first.to_dict(), "score": 0.73},
        ]
    )
    reranker = Reranker(backend="cross_encoder", provider=provider, final_top_k=2)

    result = reranker.run(_result(first, second, third, candidate_top_n=2))

    assert len(provider.calls) == 1
    assert [item["chunk_id"] for item in provider.calls[0]["candidates"]] == ["chunk-1", "chunk-2"]
    assert provider.calls[0]["top_k"] == 2
    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-2", "chunk-1"]
    assert result.candidates[0].rerank_score == 0.91
    assert result.fallback_applied is False
    assert result.debug_info["reranker"]["backend"] == "cross_encoder"


def test_reranker_supports_disabled_backend_via_factory() -> None:
    first = _candidate(chunk_id="chunk-1", document_id="doc-1", rrf_score=0.8)
    second = _candidate(chunk_id="chunk-2", document_id="doc-2", rrf_score=0.7)
    reranker = Reranker(backend="disabled", final_top_k=2)

    result = reranker.run(_result(first, second, candidate_top_n=2))

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-1", "chunk-2"]
    assert all(candidate.rerank_score is not None for candidate in result.candidates)
    assert result.debug_info["reranker"]["backend"] == "disabled"


def test_reranker_run_with_fallback_handles_runtime_failures() -> None:
    first = _candidate(chunk_id="chunk-1", document_id="doc-1", rrf_score=0.8)
    second = _candidate(chunk_id="chunk-2", document_id="doc-2", rrf_score=0.7)
    provider = RecordingProvider(error=TimeoutError("provider timed out"))
    reranker = Reranker(backend="llm_reranker", provider=provider, final_top_k=2)

    result = reranker.run_with_fallback(_result(first, second, candidate_top_n=2))

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-1", "chunk-2"]
    assert all(candidate.fallback_applied is True for candidate in result.candidates)
    assert result.fallback_applied is True
    assert "Reranker execution failed" in str(result.fallback_reason)
    assert result.debug_info["reranker"]["fallback_applied"] is True


def test_reranker_run_with_fallback_handles_initialization_failures() -> None:
    first = _candidate(chunk_id="chunk-1", document_id="doc-1", rrf_score=0.8)

    def failing_factory(config):
        raise RuntimeError("init failed")

    reranker = Reranker(
        backend="cross_encoder",
        provider_factory=failing_factory,
        final_top_k=1,
    )

    result = reranker.run_with_fallback(_result(first, candidate_top_n=1))

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-1"]
    assert result.fallback_applied is True
    assert "Reranker initialization failed" in str(result.fallback_reason)


def test_reranker_rejects_invalid_provider_payloads() -> None:
    first = _candidate(chunk_id="chunk-1", document_id="doc-1", rrf_score=0.8)
    provider = RecordingProvider(
        ranked_items=[{"document": {"chunk_id": ""}, "score": 0.9}]
    )
    reranker = Reranker(backend="cross_encoder", provider=provider, final_top_k=1)

    with pytest.raises(RerankerError, match="missing chunk_id"):
        reranker.run(_result(first, candidate_top_n=1))
