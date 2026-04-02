"""Lightweight DeepEval-compatible evaluator stub for factory wiring."""

from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseEvaluator


class DeepEvalEvaluator(BaseEvaluator):
    """Return standardized DeepEval-style metrics without importing deepeval yet."""

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
        correctness = float(
            sum(
                1
                for index in range(pair_count)
                if predictions[index].strip().casefold() == references[index].strip().casefold()
            )
            / pair_count
        ) if pair_count else 0.0
        answer_relevancy = float(pair_count / len(predictions)) if predictions else 0.0
        return {
            "answer_relevancy": answer_relevancy,
            "faithfulness": correctness,
            "correctness": correctness,
        }
