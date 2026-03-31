from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseEvaluator


class CustomMetricsEvaluator(BaseEvaluator):
    def __init__(self, *, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["hit_rate", "mrr"]

    def evaluate(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(samples)
        passed = sum(1 for sample in samples if sample.get("expected") == sample.get("actual"))
        score = 0.0 if total == 0 else passed / total
        return {
            "backend": "custom_metrics",
            "metrics": {metric: score for metric in self.metrics},
            "sample_count": total,
        }

