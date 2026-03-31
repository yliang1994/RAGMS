from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseVectorStore


class ChromaStore(BaseVectorStore):
    def __init__(self, *, path: str, collection_prefix: str = "ragms_") -> None:
        self.path = path
        self.collection_prefix = collection_prefix
        self._records: list[dict[str, Any]] = []

    def upsert(self, items: list[dict[str, Any]]) -> int:
        self._records.extend(items)
        return len(items)

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        ranked = sorted(
            self._records,
            key=lambda item: float(len(set(query_text.lower().split()) & set(str(item.get("text", "")).lower().split()))),
            reverse=True,
        )
        return ranked[:top_k]

