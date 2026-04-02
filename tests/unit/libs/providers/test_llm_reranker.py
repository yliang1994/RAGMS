from __future__ import annotations

import json

import pytest

from ragms.libs.providers.rerankers.llm_reranker import (
    LLMReranker,
    LLMRerankerError,
)


class FakeLLM:
    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.error:
            raise self.error
        if self.response is None:
            raise AssertionError("No fake response configured")
        return self.response


def test_llm_reranker_sorts_candidates_by_llm_scores() -> None:
    fake_llm = FakeLLM(
        response=json.dumps(
            [
                {"index": 2, "score": 0.95},
                {"index": 0, "score": 0.8},
                {"index": 1, "score": 0.3},
            ]
        )
    )
    reranker = LLMReranker(llm=fake_llm)

    ranked = reranker.rerank(
        "what is retrieval augmented generation",
        ["candidate a", "candidate b", "candidate c"],
    )

    assert ranked == [
        {"document": "candidate c", "score": 0.95},
        {"document": "candidate a", "score": 0.8},
        {"document": "candidate b", "score": 0.3},
    ]
    assert fake_llm.calls


def test_llm_reranker_returns_empty_list_for_empty_candidates() -> None:
    reranker = LLMReranker(llm=FakeLLM(response="[]"))

    assert reranker.rerank("query", []) == []


def test_llm_reranker_respects_top_k() -> None:
    fake_llm = FakeLLM(
        response=json.dumps(
            [
                {"index": 1, "score": 0.9},
                {"index": 0, "score": 0.5},
            ]
        )
    )
    reranker = LLMReranker(llm=fake_llm)

    ranked = reranker.rerank("query", ["first", "second"], top_k=1)

    assert ranked == [{"document": "second", "score": 0.9}]


def test_llm_reranker_maps_timeout_failures() -> None:
    reranker = LLMReranker(llm=FakeLLM(error=TimeoutError("slow model")))

    with pytest.raises(LLMRerankerError, match="LLM reranker timed out"):
        reranker.rerank("query", ["candidate"])


def test_llm_reranker_rejects_invalid_json() -> None:
    reranker = LLMReranker(llm=FakeLLM(response="not json"))

    with pytest.raises(LLMRerankerError, match="invalid JSON"):
        reranker.rerank("query", ["candidate"])


def test_llm_reranker_rejects_missing_score_fields() -> None:
    reranker = LLMReranker(llm=FakeLLM(response=json.dumps([{"index": 0}])))

    with pytest.raises(LLMRerankerError, match="missing required fields"):
        reranker.rerank("query", ["candidate"])


def test_llm_reranker_rejects_invalid_indexes() -> None:
    reranker = LLMReranker(llm=FakeLLM(response=json.dumps([{"index": 4, "score": 0.8}])))

    with pytest.raises(LLMRerankerError, match="invalid candidate index"):
        reranker.rerank("query", ["candidate"])
