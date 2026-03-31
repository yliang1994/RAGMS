from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeVectorStore:
    records: list[dict[str, Any]] = field(default_factory=list)

    def upsert(self, items: list[dict[str, Any]]) -> int:
        self.records.extend(items)
        return len(items)

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for item in self.records:
            text = str(item.get("text", ""))
            score = float(len(set(query_text.lower().split()) & set(text.lower().split())))
            scored.append({**item, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

