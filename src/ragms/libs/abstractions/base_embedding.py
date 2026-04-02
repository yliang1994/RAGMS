"""Abstract embedding model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Embed documents and queries into numeric vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents into vectors."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query into a vector."""

