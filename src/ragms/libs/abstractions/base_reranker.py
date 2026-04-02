"""Abstract reranker contract for retrieval candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """Rerank retrieval candidates for a given query."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[str | dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates sorted by descending relevance."""

