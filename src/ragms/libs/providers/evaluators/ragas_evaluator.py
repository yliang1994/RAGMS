"""Lightweight Ragas-compatible evaluator stub for factory wiring."""

from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseEvaluator


class RagasEvaluator(BaseEvaluator):
    """Return standardized Ragas-style metrics without importing ragas yet."""

    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Return deterministic placeholder metrics for local assembly tests."""

        del metadata
        references = references or []
        pair_count = min(len(predictions), len(references))
        answer_relevancy = float(pair_count / len(predictions)) if predictions else 0.0
        faithfulness = float(
            sum(
                1
                for index in range(pair_count)
                if predictions[index].strip().casefold() == references[index].strip().casefold()
            )
            / pair_count
        ) if pair_count else 0.0
        return {
            "context_precision": answer_relevancy,
            "answer_relevancy": answer_relevancy,
            "faithfulness": faithfulness,
        }
