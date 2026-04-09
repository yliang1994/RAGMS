"""Lightweight built-in evaluator that reports deterministic local metrics."""

from __future__ import annotations

from typing import Any

from ragms.core.evaluation.metrics.answer_metrics import (
    compute_answer_structure_score,
    compute_citation_coverage,
)
from ragms.core.evaluation.metrics.retrieval_metrics import (
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
)
from ragms.libs.abstractions import BaseEvaluator


def _normalize_texts(values: list[str]) -> list[str]:
    """Normalize texts before simple exact-match scoring."""

    return [value.strip().casefold() for value in values]


class CustomMetricsEvaluator(BaseEvaluator):
    """Compute deterministic local metrics without external dependencies."""

    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Return stable metrics from prediction/reference overlap."""

        metadata = dict(metadata or {})
        normalized_predictions = _normalize_texts(predictions)
        normalized_references = _normalize_texts(references or [])
        pair_count = min(len(normalized_predictions), len(normalized_references))
        exact_matches = sum(
            1
            for index in range(pair_count)
            if normalized_predictions[index] == normalized_references[index]
        )
        exact_match_rate = float(exact_matches / pair_count) if pair_count else 0.0
        coverage = float(pair_count / len(normalized_predictions)) if normalized_predictions else 0.0
        retrieved_ids = [str(item) for item in metadata.get("retrieved_ids") or []]
        expected_ids = [str(item) for item in metadata.get("expected_ids") or []]
        citations = list(metadata.get("citations") or [])
        answer_text = str(metadata.get("answer") or (predictions[0] if predictions else ""))
        metrics = {
            "score": exact_match_rate,
            "coverage": coverage,
            "prediction_count": float(len(predictions)),
            "reference_count": float(len(normalized_references)),
        }
        if metadata:
            metrics.update(
                {
                    "hit_rate_at_k": compute_hit_rate_at_k(retrieved_ids, expected_ids, k=max(1, len(retrieved_ids))) if retrieved_ids else 0.0,
                    "mrr": compute_mrr(retrieved_ids, expected_ids) if retrieved_ids else 0.0,
                    "ndcg_at_k": compute_ndcg_at_k(retrieved_ids, expected_ids, k=max(1, len(retrieved_ids))) if retrieved_ids else 0.0,
                    "citation_coverage": compute_citation_coverage(answer_text, citations),
                    "answer_structure_score": compute_answer_structure_score(answer_text),
                }
            )
        return metrics
