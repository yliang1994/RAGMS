"""Lightweight cross-encoder reranker placeholder."""

from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Rank candidates by lexical overlap with the query."""

    def __init__(self, *, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[str | dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates sorted by descending lexical overlap score."""

        ranked = sorted(
            (
                {
                    "document": candidate,
                    "score": _score_candidate(query, _coerce_candidate_text(candidate)),
                }
                for candidate in candidates
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked[:top_k] if top_k is not None else ranked


def _coerce_candidate_text(candidate: str | dict[str, Any]) -> str:
    if isinstance(candidate, str):
        return candidate
    if "text" in candidate:
        return str(candidate["text"])
    if "document" in candidate:
        return str(candidate["document"])
    return str(candidate)


def _score_candidate(query: str, candidate_text: str) -> float:
    query_terms = set(query.lower().split())
    candidate_terms = set(candidate_text.lower().split())
    overlap = len(query_terms & candidate_terms)
    density_bonus = len(candidate_terms) / 100 if candidate_terms else 0.0
    return round(overlap + density_bonus, 4)
