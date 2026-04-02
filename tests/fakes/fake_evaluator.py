"""Stable fake evaluator for local scoring tests."""

from __future__ import annotations

from typing import Any


class FakeEvaluator:
    """Return deterministic metric dictionaries without external services."""

    def __init__(self, *, base_score: float = 1.0) -> None:
        self.base_score = base_score
        self.calls: list[dict[str, object]] = []

    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Return simple metrics based on prediction/reference counts."""

        references = references or []
        self.calls.append(
            {
                "method": "evaluate",
                "prediction_count": len(predictions),
                "reference_count": len(references),
                "metadata": metadata or {},
            }
        )
        coverage = len(predictions) / max(len(references), 1)
        return {
            "score": self.base_score,
            "coverage": round(min(coverage, 1.0), 4),
            "prediction_count": float(len(predictions)),
        }
