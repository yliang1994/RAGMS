"""Abstract transform contract for chunk enrichment stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTransform(ABC):
    """Transform base chunks into enriched smart chunks."""

    @abstractmethod
    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Transform chunks with optional stage context."""

