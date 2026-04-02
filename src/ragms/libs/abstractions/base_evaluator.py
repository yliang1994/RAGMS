"""Abstract evaluator contract for retrieval and answer quality metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """Evaluate model predictions and return a stable metrics dictionary."""

    @abstractmethod
    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Evaluate predictions against references and return metrics."""

