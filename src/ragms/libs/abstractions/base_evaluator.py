from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate samples and return structured metrics."""

