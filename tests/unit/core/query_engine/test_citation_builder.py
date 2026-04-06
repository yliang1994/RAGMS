from __future__ import annotations

from ragms.core.models import RetrievalCandidate
from ragms.core.query_engine import CitationBuilder


def test_citation_builder_numbers_candidates_and_preserves_source_details() -> None:
    builder = CitationBuilder()
    candidates = [
        RetrievalCandidate(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="RAG overview",
            metadata={"source_path": "docs/rag.pdf", "page": 3, "source_ref": "§1.2"},
            score=0.9,
            source_route="hybrid",
        )
    ]

    citations = builder.build(candidates)

    assert citations == [
        {
            "index": 1,
            "marker": "[1]",
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "source_path": "docs/rag.pdf",
            "page_range": {"start": 3, "end": 3},
            "section_title": None,
            "source_ref": "§1.2",
            "score": 0.9,
            "snippet": "RAG overview",
        }
    ]
