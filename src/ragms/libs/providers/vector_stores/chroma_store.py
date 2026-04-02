"""In-memory stand-in for the Chroma vector store provider."""

from __future__ import annotations

from math import sqrt
from typing import Any

from ragms.libs.abstractions import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Persist vector records with a Chroma-compatible interface surface."""

    def __init__(self, *, collection: str = "default") -> None:
        self.collection = collection
        self._items: dict[str, dict[str, Any]] = {}

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add or update vector entries and return their ids."""

        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")

        resolved_documents = documents or [""] * len(ids)
        resolved_metadatas = metadatas or [{} for _ in ids]
        if len(resolved_documents) != len(ids) or len(resolved_metadatas) != len(ids):
            raise ValueError("documents and metadatas must align with ids")

        for index, item_id in enumerate(ids):
            self._items[item_id] = {
                "id": item_id,
                "vector": list(vectors[index]),
                "document": resolved_documents[index],
                "metadata": dict(resolved_metadatas[index]),
            }

        return ids

    def query(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the closest stored entries using Euclidean distance."""

        matches = list(self._items.values())
        if filters:
            matches = [
                item
                for item in matches
                if all(item["metadata"].get(key) == value for key, value in filters.items())
            ]

        ranked = sorted(
            matches,
            key=lambda item: self._distance(query_vector, item["vector"]),
        )
        return [
            {
                "id": item["id"],
                "score": 1.0 / (1.0 + self._distance(query_vector, item["vector"])),
                "document": item["document"],
                "metadata": dict(item["metadata"]),
            }
            for item in ranked[:top_k]
        ]

    def delete(self, ids: list[str]) -> int:
        """Delete entries by id and return the deletion count."""

        deleted = 0
        for item_id in ids:
            if item_id in self._items:
                deleted += 1
                del self._items[item_id]
        return deleted

    @staticmethod
    def _distance(left: list[float], right: list[float]) -> float:
        size = min(len(left), len(right))
        if size == 0:
            return 0.0
        return sqrt(sum((left[index] - right[index]) ** 2 for index in range(size)))
