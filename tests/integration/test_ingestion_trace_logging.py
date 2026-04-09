from __future__ import annotations

from pathlib import Path

import pytest

from ragms.ingestion_pipeline.pipeline import IngestionPipeline
from ragms.storage.traces import TraceRepository
from tests.unit.ingestion_pipeline.test_pipeline_callbacks import (
    FakeDocumentRegistry,
    FakeFileIntegrity,
    RecordingEmbedding,
    RecordingLoader,
    RecordingSplitter,
    RecordingTransform,
    RecordingVectorStore,
)


@pytest.mark.integration
def test_ingestion_pipeline_writes_success_trace_and_returns_trace_id(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_text("sample payload", encoding="utf-8")
    repository = TraceRepository(tmp_path / "logs" / "traces.jsonl")
    pipeline = IngestionPipeline(
        loader=RecordingLoader(),
        splitter=RecordingSplitter(),
        transform=RecordingTransform(),
        embedding=RecordingEmbedding(),
        vector_store=RecordingVectorStore(),
        file_integrity=FakeFileIntegrity(source_sha256="sha-success"),
        document_registry=FakeDocumentRegistry(),
        trace_repository=repository,
    )

    result = pipeline.run(source, metadata={"collection": "demo"})

    trace = repository.get_by_trace_id(result["trace_id"])

    assert result["trace_id"]
    assert trace is not None
    assert trace["trace_type"] == "ingestion"
    assert trace["collection"] == "demo"
    assert trace["status"] == "succeeded"
    assert trace["document_id"] == result["document_id"]
    assert [stage["stage_name"] for stage in trace["stages"]] == [
        "file_integrity",
        "load",
        "chunking",
        "transform",
        "embedding",
        "storage",
        "lifecycle_finalize",
    ]


@pytest.mark.integration
def test_ingestion_pipeline_writes_skipped_trace_with_explanatory_stages(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_text("sample payload", encoding="utf-8")
    repository = TraceRepository(tmp_path / "logs" / "traces.jsonl")
    pipeline = IngestionPipeline(
        loader=RecordingLoader(),
        splitter=RecordingSplitter(),
        transform=RecordingTransform(),
        embedding=RecordingEmbedding(),
        vector_store=RecordingVectorStore(),
        file_integrity=FakeFileIntegrity(source_sha256="sha-skip", should_skip=True),
        document_registry=FakeDocumentRegistry(),
        trace_repository=repository,
    )

    result = pipeline.run(source, metadata={"collection": "demo"})

    trace = repository.get_by_trace_id(result["trace_id"])

    assert result["status"] == "skipped"
    assert trace is not None
    assert trace["status"] == "skipped"
    assert trace["skipped"] == "content_unchanged"
    assert [stage["stage_name"] for stage in trace["stages"]] == [
        "file_integrity",
        "lifecycle_finalize",
    ]
