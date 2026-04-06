from __future__ import annotations

import pytest

from ragms.core.models import RetrievalCandidate
from ragms.core.query_engine import AnswerGenerationError, AnswerGenerator
from tests.fakes import FakeLLM


def test_answer_generator_returns_no_answer_without_candidates(fake_llm: FakeLLM) -> None:
    generator = AnswerGenerator(fake_llm)

    answer = generator.generate(query="what is rag", candidates=[], citations=[])

    assert answer == "No relevant context found for the query."
    assert fake_llm.calls == []


def test_answer_generator_builds_prompt_from_cited_candidates() -> None:
    llm = FakeLLM(["RAG combines retrieval and generation [1]."])
    generator = AnswerGenerator(llm)
    candidates = [
        RetrievalCandidate(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="RAG combines retrieval and generation.",
            metadata={"document_id": "doc-1"},
            score=0.9,
            source_route="hybrid",
        )
    ]
    citations = [{"index": 1, "marker": "[1]", "chunk_id": "chunk-1", "document_id": "doc-1"}]

    answer = generator.generate(query="what is rag", candidates=candidates, citations=citations)

    assert answer == "RAG combines retrieval and generation [1]."
    assert "Question:\nwhat is rag" in llm.calls[0]["prompt"]
    assert "[1] chunk_id=chunk-1" in llm.calls[0]["prompt"]


def test_answer_generator_wraps_provider_failures() -> None:
    class FailingLLM:
        def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
            raise RuntimeError("boom")

    generator = AnswerGenerator(FailingLLM())
    candidates = [
        RetrievalCandidate(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="content",
            metadata={},
            score=0.1,
            source_route="dense",
        )
    ]

    with pytest.raises(AnswerGenerationError, match="Answer generation failed"):
        generator.generate(query="q", candidates=candidates, citations=[{"marker": "[1]"}])
