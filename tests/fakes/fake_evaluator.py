from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragms.libs.abstractions.base_evaluator import BaseEvaluator


@dataclass
class FakeEvaluator(BaseEvaluator):
    calls: list[dict[str, Any]] = field(default_factory=list)

    def evaluate(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append({"samples": samples})
        total = len(samples)
        passed = sum(1 for sample in samples if sample.get("expected") == sample.get("actual"))
        score = 0.0 if total == 0 else passed / total
        return {
            "total": total,
            "passed": passed,
            "score": score,
        }

