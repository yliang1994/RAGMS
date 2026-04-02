"""Abstract vector store contract for persistence and similarity search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """Persist vectorized documents and query them by similarity."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add or update vector entries and return their ids."""

    @abstractmethod
    def query(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k nearest matches for a query vector."""

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete entries by id and return the deletion count."""

