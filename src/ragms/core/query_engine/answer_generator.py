"""Answer generation from retrieved evidence chunks."""

from __future__ import annotations

from typing import Any

from ragms.core.models import RetrievalCandidate
from ragms.libs.abstractions import BaseLLM
from ragms.runtime.exceptions import RagMSError


class AnswerGenerationError(RagMSError):
    """Raised when answer generation cannot complete successfully."""


class AnswerGenerator:
    """Generate a citation-grounded answer from retrieved candidates."""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        no_answer_text: str = "No relevant context found for the query.",
    ) -> None:
        self.llm = llm
        self.no_answer_text = no_answer_text

    def generate(
        self,
        *,
        query: str,
        candidates: list[RetrievalCandidate],
        citations: list[dict[str, Any]],
    ) -> str:
        """Return a grounded answer or a no-result fallback."""

        if not candidates:
            return self.no_answer_text

        prompt = self._build_prompt(query=query, candidates=candidates, citations=citations)
        try:
            response = self.llm.generate(
                prompt,
                system_prompt=(
                    "Answer only from the provided evidence. "
                    "Use citation markers like [1] when you rely on a source."
                ),
            )
        except Exception as exc:
            raise AnswerGenerationError("Answer generation failed") from exc

        answer = response.strip()
        return answer or self.no_answer_text

    @staticmethod
    def _build_prompt(
        *,
        query: str,
        candidates: list[RetrievalCandidate],
        citations: list[dict[str, Any]],
    ) -> str:
        sections = []
        for citation, candidate in zip(citations, candidates, strict=True):
            sections.append(
                "\n".join(
                    [
                        f"{citation['marker']} chunk_id={candidate.chunk_id}",
                        candidate.content,
                    ]
                )
            )
        evidence = "\n\n".join(sections)
        return (
            f"Question:\n{query}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Write a concise answer grounded in the evidence."
        )
