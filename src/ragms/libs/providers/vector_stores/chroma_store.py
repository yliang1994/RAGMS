from __future__ import annotations

from math import sqrt
from typing import Any

from ragms.libs.abstractions import BaseVectorStore


class ChromaStore(BaseVectorStore):
    def __init__(self, *, path: str, collection_prefix: str = "ragms_", collection_name: str = "default") -> None:
        self.path = path
        self.collection_prefix = collection_prefix
        self.collection_name = collection_name
        self._records: dict[str, dict[str, Any]] = {}

    def upsert(self, items: list[dict[str, Any]]) -> int:
        return self.add(items)

    def add(self, items: list[dict[str, Any]]) -> int:
        added = 0
        for item in items:
            item_id = str(item["id"])
            self._records[item_id] = {
                **item,
                "collection": item.get("collection", self.full_collection_name),
            }
            added += 1
        return added

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._records:
            return []

        ranked = sorted(
            self._records.values(),
            key=lambda item: self._score(query_text=query_text, item=item),
            reverse=True,
        )
        return ranked[: max(top_k, 0)]

    def delete(self, ids: list[str]) -> int:
        deleted = 0
        for item_id in ids:
            if self._records.pop(item_id, None) is not None:
                deleted += 1
        return deleted

    @property
    def full_collection_name(self) -> str:
        return f"{self.collection_prefix}{self.collection_name}"

    def _score(self, *, query_text: str, item: dict[str, Any]) -> float:
        if "embedding" in item:
            query_vector = self._embed_query(query_text, len(item["embedding"]))
            return self._cosine_similarity(query_vector, item["embedding"])
        return float(len(set(query_text.lower().split()) & set(str(item.get("text", "")).lower().split())))

    def _embed_query(self, query_text: str, dimensions: int) -> list[float]:
        base = [float(len(query_text)), float(sum(ord(char) for char in query_text) % 97), 0.0]
        return base[:dimensions] + [0.0] * max(0, dimensions - len(base))

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
