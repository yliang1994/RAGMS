"""Lightweight LLM reranker placeholder."""

from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseReranker


class LLMReranker(BaseReranker):
    """Rank candidates by prompt-aware heuristics without external model calls."""

    def __init__(self, *, model: str = "gpt-4.1-mini") -> None:
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[str | dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates sorted by descending prompt-aware score."""

        query_terms = query.lower().split()
        ranked = sorted(
            (
                {
                    "document": candidate,
                    "score": _score_candidate(query_terms, _coerce_candidate_text(candidate)),
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


def _score_candidate(query_terms: list[str], candidate_text: str) -> float:
    lowered = candidate_text.lower()
    positional_bonus = sum(
        1.0 / (index + 1)
        for index, term in enumerate(query_terms)
        if term and term in lowered
    )
    return round(positional_bonus + len(candidate_text) / 1000, 4)
