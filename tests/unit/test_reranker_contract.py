from __future__ import annotations

import pytest

from ragms.core.models import HybridSearchResult, RetrievalCandidate
from ragms.core.query_engine.reranker import Reranker


class _RecordingProvider:
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
        self.calls.append({"query": query, "candidates": list(candidates), "top_k": top_k})
        if self.error is not None:
            raise self.error
        return list(self.ranked_items)


def _candidate(chunk_id: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"content for {chunk_id}",
        metadata={"document_id": f"doc-{chunk_id}"},
        score=score,
        source_route="hybrid",
        rrf_score=score,
    )


def _result(*candidates: RetrievalCandidate) -> HybridSearchResult:
    return HybridSearchResult(
        query="what is rag",
        collection="docs",
        candidates=tuple(candidates),
        dense_count=2,
        sparse_count=2,
        filtered_out_count=0,
        candidate_top_n=2,
        debug_info={},
    )


@pytest.mark.unit
def test_reranker_contract_disabled_backend_preserves_order_and_shape() -> None:
    first = _candidate("chunk-a", 0.9)
    second = _candidate("chunk-b", 0.8)

    result = Reranker(backend="disabled", final_top_k=2).run(_result(first, second))

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-a", "chunk-b"]
    assert all(candidate.rerank_score is not None for candidate in result.candidates)
    assert result.debug_info["reranker"] == {
        "backend": "disabled",
        "input_count": 2,
        "output_count": 2,
        "fallback_applied": False,
        "fallback_reason": None,
    }


@pytest.mark.unit
def test_reranker_contract_enabled_backend_returns_ranked_candidates() -> None:
    first = _candidate("chunk-a", 0.9)
    second = _candidate("chunk-b", 0.8)
    provider = _RecordingProvider(
        ranked_items=[
            {"document": second.to_dict(), "score": 0.95},
            {"document": first.to_dict(), "score": 0.71},
        ]
    )

    result = Reranker(backend="cross_encoder", final_top_k=2, provider=provider).run(_result(first, second))

    assert len(provider.calls) == 1
    assert provider.calls[0]["query"] == "what is rag"
    assert provider.calls[0]["top_k"] == 2
    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-b", "chunk-a"]
    assert result.candidates[0].rerank_score == 0.95
    assert result.fallback_applied is False


@pytest.mark.unit
def test_reranker_contract_run_with_fallback_converges_runtime_failures() -> None:
    first = _candidate("chunk-a", 0.9)
    second = _candidate("chunk-b", 0.8)
    provider = _RecordingProvider(error=TimeoutError("provider timed out"))

    result = Reranker(backend="llm_reranker", final_top_k=2, provider=provider).run_with_fallback(
        _result(first, second)
    )

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-a", "chunk-b"]
    assert result.fallback_applied is True
    assert all(candidate.fallback_applied is True for candidate in result.candidates)
    assert "Reranker execution failed" in str(result.fallback_reason)
    assert result.debug_info["reranker"]["fallback_applied"] is True


@pytest.mark.unit
def test_reranker_contract_run_with_fallback_converges_initialization_failures() -> None:
    first = _candidate("chunk-a", 0.9)

    def _failing_factory(_config):
        raise RuntimeError("init failed")

    result = Reranker(
        backend="cross_encoder",
        final_top_k=1,
        provider_factory=_failing_factory,
    ).run_with_fallback(_result(first))

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk-a"]
    assert result.fallback_applied is True
    assert "Reranker initialization failed" in str(result.fallback_reason)
