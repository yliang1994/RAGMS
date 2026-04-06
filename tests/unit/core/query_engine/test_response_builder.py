from __future__ import annotations

import sqlite3
from pathlib import Path

from ragms.core.models import HybridSearchResult, RetrievalCandidate
from ragms.core.query_engine import ResponseBuilder
from ragms.storage.images.image_storage import ImageStorage
from ragms.storage.sqlite.repositories.images import ImagesRepository


def test_response_builder_returns_structured_payload_with_optional_debug() -> None:
    builder = ResponseBuilder()
    candidate = RetrievalCandidate(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="RAG overview",
        metadata={"source_path": "docs/rag.pdf"},
        score=0.9,
        source_route="hybrid",
        rrf_score=0.7,
        rerank_score=0.95,
    )
    result = HybridSearchResult(
        query="what is rag",
        collection="docs",
        candidates=(candidate,),
        dense_count=1,
        sparse_count=1,
        filtered_out_count=0,
        candidate_top_n=5,
        fallback_applied=True,
        fallback_reason="reranker timeout",
        debug_info={"fusion": {"returned_count": 1}},
    )

    payload = builder.build(
        query="what is rag",
        answer="RAG is retrieval augmented generation [1].",
        result=result,
        citations=[{"index": 1, "chunk_id": "chunk-1", "marker": "[1]"}],
        retrieved_candidates=[candidate],
        trace_context={"trace_id": "trace-123"},
        return_debug=True,
    )

    assert payload["answer"] == "RAG is retrieval augmented generation [1]."
    assert payload["trace_id"] == "trace-123"
    assert payload["fallback_applied"] is True
    assert payload["retrieved_chunks"][0]["citation_index"] == 1
    assert payload["debug_info"] == {"fusion": {"returned_count": 1}}
    assert payload["structured_content"]["trace_id"] == "trace-123"
    assert payload["content"][0].type == "text"
    assert "Sources:" in payload["markdown"]


def test_response_builder_adds_multimodal_image_content_when_image_refs_exist(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png-bytes")

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    repository = ImagesRepository(connection)
    repository.upsert_image(
        image_id="img-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        file_path=str(image_path),
        source_path="docs/rag.pdf",
        image_hash="hash",
        page=2,
        position={"x": 1},
    )
    builder = ResponseBuilder(
        images_repository=repository,
        image_storage=ImageStorage(root_dir=tmp_path),
    )
    candidate = RetrievalCandidate(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Chunk with image",
        metadata={"source_path": "docs/rag.pdf", "image_refs": ["img-1"], "page": 2},
        score=0.8,
        source_route="hybrid",
    )
    result = HybridSearchResult(
        query="what is rag",
        collection="docs",
        candidates=(candidate,),
        dense_count=1,
        sparse_count=0,
    )

    payload = builder.build(
        query="what is rag",
        answer="RAG architecture [1].",
        result=result,
        citations=[
            {
                "index": 1,
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "marker": "[1]",
                "source_path": "docs/rag.pdf",
                "page_range": {"start": 2, "end": 2},
                "section_title": None,
                "snippet": "Chunk with image",
            }
        ],
        retrieved_candidates=[candidate],
    )

    assert payload["content"][1].type == "image"
    assert payload["content"][1].mimeType == "image/png"
    assert payload["content"][1].data
