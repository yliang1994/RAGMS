"""Lightweight OpenAI embedding provider placeholder."""

from __future__ import annotations

import hashlib

from ragms.libs.abstractions import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """Produce deterministic vectors for the configured embedding model."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimension: int = 8,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using a deterministic hash projection."""

        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query using the same deterministic projection."""

        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(f"{self.model}:{text}".encode("utf-8")).digest()
        return [round(byte / 255, 6) for byte in digest[: self.dimension]]
