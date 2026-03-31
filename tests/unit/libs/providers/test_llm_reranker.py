from __future__ import annotations

import pytest

from ragms.libs.providers.rerankers.llm_reranker import LLMReranker


def test_llm_reranker_performs_stable_semantic_reranking() -> None:
    reranker = LLMReranker(model="gpt-4.1-mini")

    ranked = reranker.rerank(
        "ragms semantic retrieval",
        [
            {"id": "a", "text": "ragms semantic retrieval pipeline", "score": 0.1},
            {"id": "b", "text": "unrelated content", "score": 0.8},
            {"id": "c", "text": "semantic retrieval for ragms and ranking", "score": 0.2},
        ],
    )

    assert ranked[0]["id"] == "c"
    assert ranked[1]["id"] == "a"
    assert "rerank_score" in ranked[0]


def test_llm_reranker_handles_empty_candidates() -> None:
    reranker = LLMReranker(model="gpt-4.1-mini")

    assert reranker.rerank("query", []) == []


def test_llm_reranker_raises_for_timeout_and_format_failures() -> None:
    reranker = LLMReranker(model="gpt-4.1-mini")

    with pytest.raises(TimeoutError, match="timed out"):
        reranker.rerank("query", [{"text": "candidate", "simulate_timeout": True}])

    with pytest.raises(RuntimeError, match="formatting failed"):
        reranker.rerank("query", [{"id": "missing-text"}])
