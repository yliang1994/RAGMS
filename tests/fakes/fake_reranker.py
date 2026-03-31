from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragms.libs.abstractions.base_reranker import BaseReranker


@dataclass
class FakeReranker(BaseReranker):
    calls: list[dict[str, Any]] = field(default_factory=list)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "candidates": candidates, "top_k": top_k})
        ranked = sorted(
            candidates,
            key=lambda item: (float(item.get("score", 0.0)), len(str(item.get("text", "")))),
            reverse=True,
        )
        if top_k is None:
            return ranked
        return ranked[:top_k]
