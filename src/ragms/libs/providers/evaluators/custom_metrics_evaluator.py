"""Lightweight built-in evaluator that reports deterministic local metrics."""

from __future__ import annotations

from typing import Any

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

        del metadata
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
        return {
            "score": exact_match_rate,
            "coverage": coverage,
            "prediction_count": float(len(predictions)),
            "reference_count": float(len(normalized_references)),
        }
