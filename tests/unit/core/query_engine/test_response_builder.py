from __future__ import annotations

from ragms.core.models import HybridSearchResult, RetrievalCandidate
from ragms.core.query_engine import ResponseBuilder


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
