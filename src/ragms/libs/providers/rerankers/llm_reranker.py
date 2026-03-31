from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseReranker


class LLMReranker(BaseReranker):
    def __init__(self, *, model: str) -> None:
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        ranked = sorted(candidates, key=lambda item: len(str(item.get("text", ""))), reverse=True)
        return ranked if top_k is None else ranked[:top_k]

