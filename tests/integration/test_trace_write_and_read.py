from __future__ import annotations

from pathlib import Path

import pytest

from ragms.core.management import TraceService
from ragms.core.trace_collector import TraceManager
from ragms.storage.traces import TraceRepository


@pytest.mark.integration
def test_trace_service_lists_and_reads_trace_details(tmp_path: Path) -> None:
    repository = TraceRepository(tmp_path / "logs" / "traces.jsonl")
    manager = TraceManager()

    ingestion_trace = manager.start_trace(
        "ingestion",
        collection="demo",
        source_path="docs/a.pdf",
        document_id="doc-a",
    )
    manager.start_stage(ingestion_trace, "file_integrity", input_payload={"source_path": "docs/a.pdf"})
    manager.finish_stage(
        ingestion_trace,
        "file_integrity",
        output_payload={"should_skip": False},
        metadata={"method": "sha256"},
    )
    repository.append(
        manager.finish_trace(
            ingestion_trace,
            status="succeeded",
            total_chunks=2,
            total_images=0,
            skipped=False,
        )
    )

    query_trace = manager.start_trace(
        "query",
        collection="demo",
        query="what is rag",
    )
    manager.start_stage(query_trace, "query_processing", input_payload={"query": "what is rag"})
    manager.finish_stage(
        query_trace,
        "query_processing",
        output_payload={"normalized_query": "what is rag"},
        metadata={"method": "normalize"},
    )
    repository.append(
        manager.finish_trace(
            query_trace,
            status="succeeded",
            top_k_results=["chunk-1"],
        )
    )

    service = TraceService(repository=repository)
    summaries = service.list_traces(collection="demo")
    ingestion_only = service.list_traces(trace_type="ingestion", status="succeeded")
    detail = service.get_trace_detail(ingestion_trace.trace_id)

    assert len(summaries) == 2
    assert [item["trace_id"] for item in ingestion_only] == [ingestion_trace.trace_id]
    assert detail["trace_id"] == ingestion_trace.trace_id
    assert detail["trace_type"] == "ingestion"
    assert detail["stages"][0]["stage_name"] == "file_integrity"
