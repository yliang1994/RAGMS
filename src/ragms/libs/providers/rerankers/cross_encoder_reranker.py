from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseReranker


class CrossEncoderReranker(BaseReranker):
    MAX_CANDIDATES = 50

    def __init__(self, *, model: str) -> None:
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        normalized = [self._score_candidate(query, candidate, index) for index, candidate in enumerate(candidates[: self.MAX_CANDIDATES])]
        ranked = sorted(
            normalized,
            key=lambda item: (float(item["rerank_score"]), float(item.get("score", 0.0)), -int(item["_input_index"])),
            reverse=True,
        )
        cleaned = [{key: value for key, value in item.items() if key != "_input_index"} for item in ranked]
        if top_k is None:
            return cleaned
        if top_k <= 0:
            return []
        return cleaned[:top_k]

    def _score_candidate(self, query: str, candidate: dict[str, Any], index: int) -> dict[str, Any]:
        text = str(candidate.get("text", ""))
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())
        overlap = len(query_terms & text_terms)
        base_score = float(candidate.get("score", 0.0))
        rerank_score = base_score + overlap + (len(text) / 1000.0)
        return {
            **candidate,
            "score": base_score,
            "rerank_score": rerank_score,
            "_input_index": index,
        }
