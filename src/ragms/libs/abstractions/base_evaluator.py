"""Abstract evaluator contract for retrieval and answer quality metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """Evaluate predictions and return normalized backend results."""

    @abstractmethod
    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate predictions against references and return standardized metrics."""


def normalize_backend_metrics(
    *,
    status: str,
    metrics: dict[str, float] | None = None,
    errors: list[dict[str, Any]] | None = None,
    raw_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a normalized backend result payload."""

    return {
        "status": status,
        "metrics": dict(metrics or {}),
        "errors": list(errors or []),
        "raw_summary": dict(raw_summary or {}),
    }


def serialize_backend_failure(
    backend_name: str,
    *,
    message: str,
    failure_type: str = "backend_failure",
) -> dict[str, str]:
    """Serialize a backend error or skip reason into a stable payload."""

    return {
        "backend": backend_name,
        "type": failure_type,
        "message": message,
    }
