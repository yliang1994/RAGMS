"""In-memory fake vector store for retrieval and storage tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StoredVector:
    """Single vector-store entry kept fully in memory."""

    id: str
    vector: list[float]
    document: str
    metadata: dict[str, Any]


class FakeVectorStore:
    """Simple in-memory vector store with deterministic ranking."""

    def __init__(self) -> None:
        self._items: dict[str, StoredVector] = {}

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Store vectors and optional payloads in memory."""

        documents = documents or [""] * len(ids)
        metadatas = metadatas or [{} for _ in ids]
        for index, item_id in enumerate(ids):
            self._items[item_id] = StoredVector(
                id=item_id,
                vector=list(vectors[index]),
                document=documents[index],
                metadata=dict(metadatas[index]),
            )
        return ids

    def query(self, query_vector: list[float], *, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k most similar stored vectors."""

        matches = sorted(
            (
                {
                    "id": item.id,
                    "score": _dot_product(query_vector, item.vector),
                    "document": item.document,
                    "metadata": item.metadata,
                }
                for item in self._items.values()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return matches[:top_k]

    def delete(self, ids: list[str]) -> int:
        """Delete stored vectors by id and return the deletion count."""

        deleted = 0
        for item_id in ids:
            if item_id in self._items:
                deleted += 1
                del self._items[item_id]
        return deleted


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(lhs * rhs for lhs, rhs in zip(left, right, strict=False))
