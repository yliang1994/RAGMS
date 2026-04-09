from __future__ import annotations

from ragms.libs.providers.evaluators.custom_metrics_evaluator import CustomMetricsEvaluator


def test_custom_metrics_evaluator_returns_deterministic_retrieval_and_answer_metrics() -> None:
    evaluator = CustomMetricsEvaluator()

    metrics = evaluator.evaluate(
        ["RAG combines retrieval and generation [1]."],
        ["RAG combines retrieval and generation [1]."],
        metadata={
            "retrieved_ids": ["chunk-2", "chunk-1"],
            "expected_ids": ["chunk-1"],
            "citations": [{"index": 1}],
            "answer": "RAG combines retrieval and generation [1].",
        },
    )

    assert metrics["score"] == 1.0
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["citation_coverage"] == 1.0
    assert metrics["answer_structure_score"] == 1.0
