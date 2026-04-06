from __future__ import annotations

import anyio

from ragms.core.management.data_service import DocumentSummaryNotFoundError
from ragms.mcp_server.server import create_server
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings


def _runtime() -> ServiceContainer:
    settings = AppSettings.model_validate({"environment": "test"})
    return ServiceContainer(settings=settings, services={})


def test_mcp_server_list_collections_tool_returns_summary(monkeypatch) -> None:
    server = create_server(_runtime())

    class StubCollectionsService:
        def __init__(self, settings) -> None:
            self.settings = settings

        def list_collections(self, *, filters=None, page=None, page_size=None):
            return {
                "collections": [
                    {
                        "name": "alpha",
                        "document_count": 2,
                        "chunk_count": 5,
                        "image_count": 1,
                        "latest_updated_at": "2026-04-07T00:00:00+00:00",
                    }
                ],
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": 1,
                    "returned_count": 1,
                    "has_more": False,
                },
            }

    monkeypatch.setattr("ragms.mcp_server.tools.collections.DataService", StubCollectionsService)

    result = anyio.run(server.call_tool, "list_collections", {"filters": {"collection": "alpha"}, "page": 1})

    assert result.isError is False
    assert result.structuredContent["collections"][0]["name"] == "alpha"
    assert result.structuredContent["summary"]["collection_count"] == 1


def test_mcp_server_get_document_summary_tool_returns_summary_and_error(monkeypatch) -> None:
    server = create_server(_runtime())

    class StubDocumentsService:
        def __init__(self, settings) -> None:
            self.settings = settings

        def get_document_summary(self, document_id: str):
            if document_id == "missing":
                raise DocumentSummaryNotFoundError("Document not found: missing")
            return {
                "document_id": document_id,
                "source_path": "docs/a.md",
                "primary_collection": "alpha",
                "collections": ["alpha"],
                "summary": "Document summary",
                "structure_outline": ["Intro"],
                "key_metadata": {"title": "Alpha"},
                "ingestion_status": {"status": "indexed"},
                "page_summary": {"pages": [1], "page_count": 1},
                "image_summary": {"image_count": 0, "images": []},
                "chunk_count": 1,
            }

    monkeypatch.setattr("ragms.mcp_server.tools.documents.DataService", StubDocumentsService)

    success = anyio.run(server.call_tool, "get_document_summary", {"document_id": "doc-1"})
    missing = anyio.run(server.call_tool, "get_document_summary", {"document_id": "missing"})

    assert success.isError is False
    assert success.structuredContent["document_id"] == "doc-1"
    assert success.structuredContent["primary_collection"] == "alpha"
    assert missing.isError is True
    assert missing.structuredContent["error"]["code"] == -32602
