"""Deterministic fake embedding provider for local tests."""

from __future__ import annotations

import hashlib


class FakeEmbedding:
    """Produce stable numeric vectors from input text."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls: list[dict[str, object]] = []

    def _embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [round(byte / 255, 6) for byte in digest[: self.dimension]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts with a deterministic hash-based projection."""

        self.calls.append({"method": "embed_documents", "count": len(texts)})
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

        self.calls.append({"method": "embed_query", "text": text})
        return self._embed_text(text)
