"""Build stable query-engine response payloads for CLI and MCP usage."""

from __future__ import annotations

import mimetypes
from typing import Any

from mcp import types

from ragms.core.models import HybridSearchResult, RetrievalCandidate
from ragms.storage.images.image_storage import ImageStorage
from ragms.storage.sqlite.repositories.images import ImagesRepository


class ResponseBuilder:
    """Assemble final structured responses from answer and retrieval outputs."""

    def __init__(
        self,
        *,
        images_repository: ImagesRepository | None = None,
        image_storage: ImageStorage | None = None,
    ) -> None:
        self.images_repository = images_repository
        self.image_storage = image_storage

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

        retrieved_chunks = self.serialize_retrieved_chunks(
            retrieved_candidates,
            citations=citations,
        )
        payload = {
            "query": query,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "trace_id": None if trace_context is None else trace_context.get("trace_id"),
            "fallback_applied": result.fallback_applied,
            "fallback_reason": result.fallback_reason,
        }
        payload["markdown"] = self.build_query_markdown_content(
            answer=answer,
            citations=citations,
            fallback_applied=result.fallback_applied,
            fallback_reason=result.fallback_reason,
        )
        payload["structured_content"] = self.build_query_structured_content(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=payload["retrieved_chunks"],
            trace_id=payload["trace_id"],
            fallback_applied=result.fallback_applied,
            fallback_reason=result.fallback_reason,
            debug_info=dict(result.debug_info) if return_debug else None,
        )
        payload["content"] = self.build_multimodal_contents(
            markdown=payload["markdown"],
            retrieved_chunks=payload["retrieved_chunks"],
        )
        if return_debug:
            payload["debug_info"] = dict(result.debug_info)
        return payload

    def serialize_retrieved_chunks(
        self,
        retrieved_candidates: list[RetrievalCandidate],
        *,
        citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Serialize retrieved candidates with citation indexes for response reuse."""

        citation_indexes = {
            citation["chunk_id"]: citation["index"]
            for citation in citations
        }
        return [
            self._serialize_candidate(
                candidate,
                citation_index=citation_indexes.get(candidate.chunk_id),
            )
            for candidate in retrieved_candidates
        ]

    def build_query_structured_content(
        self,
        *,
        query: str,
        answer: str,
        citations: list[dict[str, Any]],
        retrieved_chunks: list[dict[str, Any]],
        trace_id: str | None,
        fallback_applied: bool,
        fallback_reason: str | None,
        debug_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build MCP-friendly structured content for query responses."""

        payload = {
            "query": query,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "trace_id": trace_id,
            "fallback_applied": fallback_applied,
            "fallback_reason": fallback_reason,
        }
        if debug_info is not None:
            payload["debug"] = debug_info
        return payload

    @staticmethod
    def build_query_markdown_content(
        *,
        answer: str,
        citations: list[dict[str, Any]],
        fallback_applied: bool,
        fallback_reason: str | None,
    ) -> str:
        """Build a readable markdown answer with citation references."""

        lines = [answer.strip() or "No relevant context found for the query."]
        if fallback_applied and fallback_reason:
            lines.append(f"\nFallback: {fallback_reason}")
        if citations:
            lines.append("\nSources:")
            for citation in citations:
                source_path = citation.get("source_path") or "unknown"
                page_range = citation.get("page_range")
                if page_range:
                    source_line = (
                        f"- [{citation['index']}] {source_path} "
                        f"(pages {page_range['start']}-{page_range['end']})"
                    )
                else:
                    source_line = f"- [{citation['index']}] {source_path}"
                lines.append(source_line)
        return "\n".join(lines)

    def build_multimodal_contents(
        self,
        *,
        markdown: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[types.TextContent | types.ImageContent]:
        """Build the final MCP content array with optional image payloads."""

        contents: list[types.TextContent | types.ImageContent] = [
            types.TextContent(type="text", text=markdown)
        ]
        for image_payload in self.load_image_payloads(retrieved_chunks):
            contents.append(
                types.ImageContent(
                    type="image",
                    data=image_payload["data"],
                    mimeType=image_payload["mime_type"],
                )
            )
        return contents

    def load_image_payloads(self, retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Load image payloads referenced by retrieved chunks with graceful fallback."""

        if self.images_repository is None or self.image_storage is None:
            return []

        chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        chunk_image_refs = {
            chunk["chunk_id"]: list((chunk.get("metadata") or {}).get("image_refs") or [])
            for chunk in retrieved_chunks
        }
        records = self.images_repository.list_by_chunk_ids(chunk_ids)
        payloads: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        for record in records:
            if record["image_id"] not in chunk_image_refs.get(record["chunk_id"], []):
                continue
            file_path = str(record["file_path"])
            if file_path in seen_paths:
                continue
            try:
                payloads.append(
                    {
                        "image_id": str(record["image_id"]),
                        "mime_type": self._guess_mime_type(file_path),
                        "data": self.encode_image_as_base64(file_path),
                    }
                )
                seen_paths.add(file_path)
            except FileNotFoundError:
                continue
        return payloads

    def encode_image_as_base64(self, file_path: str) -> str:
        """Encode a single image file into a base64 string."""

        if self.image_storage is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Image storage is not configured")
        return self.image_storage.encode_image_as_base64(file_path)

    @staticmethod
    def _guess_mime_type(file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

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
