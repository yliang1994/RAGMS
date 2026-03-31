from __future__ import annotations

from ragms.libs.providers.rerankers.cross_encoder_reranker import CrossEncoderReranker


def test_cross_encoder_reranker_sorts_candidates_by_relevance() -> None:
    reranker = CrossEncoderReranker(model="bge-reranker-base")

    ranked = reranker.rerank(
        "ragms retrieval",
        [
            {"id": "a", "text": "ragms retrieval pipeline", "score": 0.1},
            {"id": "b", "text": "other text", "score": 0.9},
            {"id": "c", "text": "ragms retrieval and ranking", "score": 0.2},
        ],
    )

    assert ranked[0]["id"] == "c"
    assert ranked[1]["id"] == "a"
    assert "rerank_score" in ranked[0]


def test_cross_encoder_reranker_handles_empty_and_missing_score_candidates() -> None:
    reranker = CrossEncoderReranker(model="bge-reranker-base")

    assert reranker.rerank("query", []) == []

    ranked = reranker.rerank(
        "query",
        [
            {"id": "a", "text": "query match"},
            {"id": "b", "text": "other"},
        ],
    )

    assert ranked[0]["id"] == "a"
    assert ranked[0]["score"] == 0.0


def test_cross_encoder_reranker_truncates_long_candidate_lists_and_respects_top_k() -> None:
    reranker = CrossEncoderReranker(model="bge-reranker-base")
    candidates = [{"id": f"doc-{index}", "text": f"query {index}", "score": float(index)} for index in range(60)]

    ranked = reranker.rerank("query", candidates, top_k=5)

    assert len(ranked) == 5
    assert all(item["id"] != "doc-59" for item in ranked)
