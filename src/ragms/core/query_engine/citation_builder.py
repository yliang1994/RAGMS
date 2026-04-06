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
                    "page_range": _page_range_from_metadata(metadata),
                    "section_title": metadata.get("section_title") or metadata.get("title"),
                    "source_ref": metadata.get("source_ref"),
                    "score": candidate.final_score,
                    "snippet": candidate.content[:240],
                }
            )
        return citations


def _page_range_from_metadata(metadata: dict[str, Any]) -> dict[str, int] | None:
    start = metadata.get("page_start", metadata.get("page"))
    end = metadata.get("page_end", metadata.get("page"))
    if start is None and end is None:
        return None
    start_value = int(start if start is not None else end)
    end_value = int(end if end is not None else start)
    return {"start": start_value, "end": end_value}
