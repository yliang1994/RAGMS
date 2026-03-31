from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTransform(ABC):
    @abstractmethod
    def transform(self, chunks: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        """Apply enrichment or cleanup to chunks."""

