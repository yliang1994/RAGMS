from __future__ import annotations

from pathlib import Path

import anyio

from ragms.mcp_server.server import create_server
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


def test_mcp_server_ingest_tool_returns_document_level_contract(tmp_path: Path, monkeypatch) -> None:
    server = create_server(_runtime(tmp_path))

    monkeypatch.setattr(
        "ragms.mcp_server.tools.ingest.build_ingestion_pipeline",
        lambda settings, *, collection=None: object(),
    )
    monkeypatch.setattr(
        "ragms.mcp_server.tools.ingest.discover_ingestion_sources",
        lambda paths: ([tmp_path / "docs" / "a.md"], []),
    )
    monkeypatch.setattr(
        "ragms.mcp_server.tools.ingest.run_ingestion_batch",
        lambda pipeline, *, sources, collection, force_rebuild: [
            {
                "source_path": str(tmp_path / "docs" / "a.md"),
                "result": {
                    "trace_id": "trace-ingest-1",
                    "document_id": "doc-1",
                    "status": "completed",
                    "smart_chunks": [{"chunk_id": "chunk-1"}],
                    "stored_ids": ["chunk-1"],
                    "lifecycle": {"final_status": "indexed"},
                },
            }
        ],
    )

    result = anyio.run(
        server.call_tool,
        "ingest_documents",
        {"paths": ["docs"], "collection": "demo", "force_rebuild": True},
    )

    assert result.isError is False
    assert result.structuredContent["trace_id"] == "trace-ingest-1"
    assert result.structuredContent["summary"]["indexed_count"] == 1
    assert result.structuredContent["documents"][0]["document_id"] == "doc-1"
    assert result.structuredContent["documents"][0]["status"] == "indexed"


def test_mcp_server_ingest_tool_returns_error_contract(tmp_path: Path, monkeypatch) -> None:
    server = create_server(_runtime(tmp_path))

    def failing_builder(settings, *, collection=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("ragms.mcp_server.tools.ingest.build_ingestion_pipeline", failing_builder)

    result = anyio.run(server.call_tool, "ingest_documents", {"paths": ["docs"]})

    assert result.isError is True
    assert result.structuredContent["error"]["code"] == -32603
