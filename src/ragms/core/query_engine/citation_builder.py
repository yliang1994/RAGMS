"""Build stable citation payloads from retrieved candidates."""

from __future__ import annotations

from typing import Any

from ragms.core.models import RetrievalCandidate


class CitationBuilder:
    """Convert retrieval candidates into stable citation metadata."""

    def build(self, candidates: list[RetrievalCandidate]) -> list[dict[str, Any]]:
        """Return numbered citations aligned with the candidate order."""

        citations: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            metadata = dict(candidate.metadata)
            citations.append(
                {
                    "index": index,
                    "marker": f"[{index}]",
                    "chunk_id": candidate.chunk_id,
                    "document_id": candidate.document_id,
                    "source_path": metadata.get("source_path"),
                    "page": metadata.get("page"),
                    "source_ref": metadata.get("source_ref"),
                    "snippet": candidate.content[:240],
                }
            )
        return citations
