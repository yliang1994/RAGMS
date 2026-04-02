"""Deterministic fake reranker for retrieval pipeline tests."""

from __future__ import annotations

from typing import Any


class FakeReranker:
    """Score candidates by simple lexical overlap with the query."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: str,
        candidates: list[str | dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates sorted by overlap-derived score."""

        self.calls.append({"method": "rerank", "query": query, "count": len(candidates)})
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
        if top_k is None:
            return ranked
        return ranked[:top_k]


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
