from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseEvaluator


class RagasEvaluator(BaseEvaluator):
    def __init__(self, *, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["context_precision", "answer_relevancy"]

    def evaluate(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(samples)
        score = 0.0 if total == 0 else min(1.0, 0.5 + (total / 100.0))
        return {
            "backend": "ragas",
            "metrics": {metric: score for metric in self.metrics},
            "sample_count": total,
        }

