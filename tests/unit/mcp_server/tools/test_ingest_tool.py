from __future__ import annotations

from pathlib import Path

from ragms.mcp_server.tools.ingest import (
    handle_ingest_documents,
    normalize_ingest_request,
    serialize_ingestion_result,
)
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings


def _runtime(tmp_path: Path) -> ServiceContainer:
    settings = AppSettings.model_validate({"environment": "test"})
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.storage.sqlite.path = tmp_path / "data" / "metadata" / "ragms.db"
    return ServiceContainer(settings=settings, services={})


def test_normalize_ingest_request_deduplicates_paths_and_keeps_options() -> None:
    request = normalize_ingest_request(
        paths=["docs/a.md", "docs/a.md", " docs/b.md "],
        collection="demo",
        force_rebuild=True,
        options={"dry_run": False},
    )

    assert request.paths == ["docs/a.md", "docs/b.md"]
    assert request.collection == "demo"
    assert request.force_rebuild is True
    assert request.options == {"dry_run": False}


def test_serialize_ingestion_result_preserves_document_status() -> None:
    serialized = serialize_ingestion_result(
        "docs/a.md",
        {
            "trace_id": "trace-a",
            "document_id": "doc-a",
            "status": "completed",
            "current_stage": "lifecycle",
            "smart_chunks": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
            "stored_ids": ["c1"],
            "lifecycle": {"final_status": "indexed"},
        },
        collection="demo",
    )

    assert serialized["trace_id"] == "trace-a"
    assert serialized["document_id"] == "doc-a"
    assert serialized["status"] == "indexed"
    assert serialized["chunk_count"] == 2
    assert serialized["stored_count"] == 1
    assert serialized["skipped"] is False


def test_handle_ingest_documents_wraps_partial_failures_without_error(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    pipeline = object()

    def pipeline_builder(settings, *, collection=None):
        assert settings is runtime.settings
        assert collection == "demo"
        return pipeline

    def source_discovery(paths: list[str]):
        assert paths == ["docs", "missing.md"]
        return [tmp_path / "docs" / "a.md", tmp_path / "docs" / "b.md"], [
            {"path": "missing.md", "message": "Ingestion path does not exist: missing.md"}
        ]

    def batch_runner(pipeline_arg, *, sources, collection, force_rebuild):
        assert pipeline_arg is pipeline
        assert collection == "demo"
        assert force_rebuild is True
        assert [str(item) for item in sources] == [str(tmp_path / "docs" / "a.md"), str(tmp_path / "docs" / "b.md")]
        return [
            {
                "source_path": str(tmp_path / "docs" / "a.md"),
                "result": {
                    "trace_id": "trace-a",
                    "document_id": "doc-a",
                    "status": "completed",
                    "smart_chunks": [{"chunk_id": "c1"}],
                    "stored_ids": ["c1"],
                    "lifecycle": {"final_status": "indexed"},
                },
            },
            {
                "source_path": str(tmp_path / "docs" / "b.md"),
                "result": {
                    "trace_id": "trace-b",
                    "document_id": "doc-b",
                    "status": "skipped",
                    "smart_chunks": [],
                    "stored_ids": [],
                    "lifecycle": {"final_status": "skipped"},
                },
            },
        ]

    result = handle_ingest_documents(
        paths=["docs", "missing.md"],
        collection="demo",
        force_rebuild=True,
        options={"source": "mcp"},
        runtime=runtime,
        pipeline_builder=pipeline_builder,
        source_discovery=source_discovery,
        batch_runner=batch_runner,
    )

    assert result.isError is False
    assert result.structuredContent["collection"] == "demo"
    assert result.structuredContent["summary"] == {
        "requested_path_count": 2,
        "resolved_source_count": 2,
        "accepted_count": 2,
        "document_count": 3,
        "indexed_count": 1,
        "skipped_count": 1,
        "failed_count": 1,
    }
    assert result.structuredContent["failure_summary"]["count"] == 1
    assert result.structuredContent["failure_summary"]["documents"][0]["source_path"] == "missing.md"
    assert result.content[0].text == "Ingestion accepted 2 source(s): indexed 1, skipped 1, failed 1."


def test_handle_ingest_documents_returns_error_for_builder_failures(tmp_path: Path) -> None:
    def failing_builder(settings, *, collection=None):
        raise RuntimeError("boom")

    result = handle_ingest_documents(
        paths=["docs"],
        runtime=_runtime(tmp_path),
        pipeline_builder=failing_builder,
    )

    assert result.isError is True
    assert result.structuredContent["error"]["code"] == -32603
    assert result.content[0].text == "Internal error"
