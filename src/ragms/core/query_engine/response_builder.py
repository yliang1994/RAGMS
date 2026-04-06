"""Build stable query-engine response payloads for CLI and later MCP usage."""

from __future__ import annotations

from typing import Any

from ragms.core.models import HybridSearchResult, RetrievalCandidate


class ResponseBuilder:
    """Assemble final structured responses from answer and retrieval outputs."""

    def build(
        self,
        *,
        query: str,
        answer: str,
        result: HybridSearchResult,
        citations: list[dict[str, Any]],
        retrieved_candidates: list[RetrievalCandidate],
        trace_context: dict[str, Any] | None = None,
        return_debug: bool = False,
    ) -> dict[str, Any]:
        """Return a stable response payload."""

        citation_indexes = {
            citation["chunk_id"]: citation["index"]
            for citation in citations
        }
        payload = {
            "query": query,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": [
                self._serialize_candidate(
                    candidate,
                    citation_index=citation_indexes.get(candidate.chunk_id),
                )
                for candidate in retrieved_candidates
            ],
            "trace_id": None if trace_context is None else trace_context.get("trace_id"),
            "fallback_applied": result.fallback_applied,
            "fallback_reason": result.fallback_reason,
        }
        if return_debug:
            payload["debug_info"] = dict(result.debug_info)
        return payload

    @staticmethod
    def _serialize_candidate(
        candidate: RetrievalCandidate,
        *,
        citation_index: int | None,
    ) -> dict[str, Any]:
        return {
            "chunk_id": candidate.chunk_id,
            "document_id": candidate.document_id,
            "content": candidate.content,
            "metadata": dict(candidate.metadata),
            "source_route": candidate.source_route,
            "score": candidate.score,
            "rrf_score": candidate.rrf_score,
            "rerank_score": candidate.rerank_score,
            "citation_index": citation_index,
            "fallback_applied": candidate.fallback_applied,
            "fallback_reason": candidate.fallback_reason,
        }
