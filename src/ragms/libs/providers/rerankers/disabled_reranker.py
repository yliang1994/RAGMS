"""No-op reranker used when reranking is disabled."""

from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseReranker


class DisabledReranker(BaseReranker):
    """Return candidates unchanged with deterministic descending scores."""

    def rerank(
        self,
        query: str,
        candidates: list[str | dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return original candidates in order without semantic reranking."""

        del query
        ranked = [
            {"document": candidate, "score": float(len(candidates) - index)}
            for index, candidate in enumerate(candidates)
        ]
        return ranked[:top_k] if top_k is not None else ranked
