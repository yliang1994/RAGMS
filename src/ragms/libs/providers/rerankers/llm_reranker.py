"""LLM-backed reranker provider."""

from __future__ import annotations

import json
from typing import Any

from ragms.libs.abstractions import BaseLLM, BaseReranker
from ragms.libs.providers.llm.openai_llm import OpenAILLM
from ragms.runtime.exceptions import RagMSError


class LLMRerankerError(RagMSError):
    """Raised when the LLM reranker cannot produce a valid ranking."""


class LLMReranker(BaseReranker):
    """Rank candidates by asking an LLM to score semantic relevance."""

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.model = model.strip()
        self.api_key = api_key
        self.base_url = base_url
        self._llm = llm

    def rerank(
        self,
        query: str,
        candidates: list[str | dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return candidates sorted by descending LLM-assigned relevance score."""

        if not candidates:
            return []

        prompt = self._build_prompt(query=query, candidates=candidates)
        try:
            response = self._get_llm().generate(
                prompt,
                system_prompt=(
                    "You are a reranking engine. "
                    "Return strict JSON only: a list of objects with keys index and score."
                ),
            )
        except TimeoutError as exc:
            raise LLMRerankerError("LLM reranker timed out") from exc
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise LLMRerankerError("LLM reranker request failed") from exc

        ranked_items = self._parse_response(response=response, candidates=candidates)
        return ranked_items[:top_k] if top_k is not None else ranked_items

    def _get_llm(self) -> BaseLLM:
        """Return the configured LLM dependency, lazily constructing one if needed."""

        if self._llm is None:
            self._llm = OpenAILLM(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._llm

    @staticmethod
    def _build_prompt(query: str, candidates: list[str | dict[str, Any]]) -> str:
        """Build a deterministic JSON prompt for semantic reranking."""

        serialized_candidates = [
            {
                "index": index,
                "text": _coerce_candidate_text(candidate),
            }
            for index, candidate in enumerate(candidates)
        ]
        payload = {
            "query": query,
            "candidates": serialized_candidates,
            "instruction": (
                "Score each candidate for semantic relevance to the query. "
                "Use higher scores for more relevant candidates."
            ),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _parse_response(
        self,
        *,
        response: str,
        candidates: list[str | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Parse the LLM JSON response into ranked candidate objects."""

        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise LLMRerankerError("LLM reranker returned invalid JSON") from exc

        if isinstance(payload, dict):
            payload = payload.get("ranked", payload.get("results"))
        if not isinstance(payload, list):
            raise LLMRerankerError("LLM reranker returned an invalid ranking payload")

        ranked: list[dict[str, Any]] = []
        seen_indexes: set[int] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise LLMRerankerError("LLM reranker returned an invalid ranking item")
            if "index" not in item or "score" not in item:
                raise LLMRerankerError("LLM reranker response is missing required fields")
            index = item["index"]
            score = item["score"]
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                raise LLMRerankerError("LLM reranker returned an invalid candidate index")
            if index in seen_indexes:
                continue
            seen_indexes.add(index)
            ranked.append(
                {
                    "document": candidates[index],
                    "score": float(score),
                    "index": index,
                }
            )

        if not ranked:
            raise LLMRerankerError("LLM reranker returned no usable ranking items")

        # Preserve determinism on equal scores by falling back to source order.
        ranked.sort(key=lambda item: (-item["score"], item["index"]))
        return [
            {"document": item["document"], "score": item["score"]}
            for item in ranked
        ]


def _coerce_candidate_text(candidate: str | dict[str, Any]) -> str:
    if isinstance(candidate, str):
        return candidate
    if "text" in candidate:
        return str(candidate["text"])
    if "document" in candidate:
        value = candidate["document"]
        if isinstance(value, dict) and "text" in value:
            return str(value["text"])
        return str(value)
    if "content" in candidate:
        return str(candidate["content"])
    return str(candidate)
