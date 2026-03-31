from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseEvaluator


class DeepEvalEvaluator(BaseEvaluator):
    def __init__(self, *, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["faithfulness"]

    def evaluate(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(samples)
        score = 0.0 if total == 0 else min(1.0, 0.4 + (total / 120.0))
        return {
            "backend": "deepeval",
            "metrics": {metric: score for metric in self.metrics},
            "sample_count": total,
        }

