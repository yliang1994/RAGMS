from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, items: list[dict[str, Any]]) -> int:
        """Insert or update vector store records."""

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Query records from the vector store."""

