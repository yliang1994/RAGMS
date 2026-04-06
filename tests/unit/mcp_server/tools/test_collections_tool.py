from __future__ import annotations

import json
from pathlib import Path

from ragms.core.management import DataService
from ragms.mcp_server.tools.collections import (
    handle_list_collections,
    serialize_collection_summary,
)
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings
from ragms.storage.sqlite.repositories import DocumentsRepository, ImagesRepository
from ragms.storage.sqlite.schema import initialize_metadata_schema


def _runtime(tmp_path: Path) -> ServiceContainer:
    settings = AppSettings.model_validate({"environment": "test"})
    settings = settings.model_copy(deep=True)
    settings.vector_store.collection = "default"
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.storage.sqlite.path = tmp_path / "data" / "metadata" / "ragms.db"
    return ServiceContainer(settings=settings, services={})


def _write_bm25_snapshot(path: Path, *, collection: str, documents: list[dict[str, object]]) -> None:
    payload = {
        "collection": collection,
        "documents": {str(item["chunk_id"]): item for item in documents},
        "idf": {},
        "inverted_index": {},
        "document_count": len(documents),
        "average_document_length": 12.0 if documents else 0.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_serialize_collection_summary_preserves_core_fields() -> None:
    assert serialize_collection_summary(
        {
            "name": "demo",
            "document_count": 2,
            "chunk_count": 5,
            "image_count": 1,
            "latest_updated_at": "2026-04-07T00:00:00+00:00",
        }
    ) == {
        "name": "demo",
        "document_count": 2,
        "chunk_count": 5,
        "image_count": 1,
        "latest_updated_at": "2026-04-07T00:00:00+00:00",
    }


def test_data_service_list_collections_reads_bm25_and_sqlite_metadata(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    bm25_dir = runtime.settings.paths.data_dir / "indexes" / "sparse"
    _write_bm25_snapshot(
        bm25_dir / "alpha.json",
        collection="alpha",
        documents=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source_path": str(tmp_path / "docs" / "a.md"),
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-1",
                "source_path": str(tmp_path / "docs" / "a.md"),
            },
            {
                "chunk_id": "chunk-3",
                "document_id": "doc-2",
                "source_path": str(tmp_path / "docs" / "b.md"),
            },
        ],
    )
    _write_bm25_snapshot(bm25_dir / "beta.json", collection="beta", documents=[])

    connection = initialize_metadata_schema(runtime.settings.storage.sqlite.path)
    documents_repo = DocumentsRepository(connection)
    images_repo = ImagesRepository(connection)
    documents_repo.upsert_document(
        document_id="doc-1",
        source_path=str(tmp_path / "docs" / "a.md"),
        source_sha256="sha-a",
        status="indexed",
        current_stage="lifecycle",
        last_ingested_at="2026-04-06T00:00:00+00:00",
    )
    documents_repo.upsert_document(
        document_id="doc-2",
        source_path=str(tmp_path / "docs" / "b.md"),
        source_sha256="sha-b",
        status="indexed",
        current_stage="lifecycle",
        last_ingested_at="2026-04-07T00:00:00+00:00",
    )
    alpha_image = runtime.settings.paths.data_dir / "images" / "alpha" / "img-1.png"
    images_repo.upsert_image(
        image_id="img-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        file_path=str(alpha_image),
        source_path=str(tmp_path / "docs" / "img-1.png"),
        image_hash="hash-1",
    )

    service = DataService(runtime.settings, connection=connection)
    payload = service.list_collections(filters={"name": "alpha"}, page=1, page_size=10)

    assert payload["pagination"] == {
        "page": 1,
        "page_size": 10,
        "total_count": 1,
        "returned_count": 1,
        "has_more": False,
    }
    assert payload["collections"] == [
        {
            "name": "alpha",
            "document_count": 2,
            "chunk_count": 3,
            "image_count": 1,
            "latest_updated_at": "2026-04-07T00:00:00+00:00",
        }
    ]


def test_handle_list_collections_wraps_empty_and_non_empty_results(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    class StubService:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def list_collections(self, *, filters=None, page=None, page_size=None):
            self.calls.append(
                {
                    "filters": filters,
                    "page": page,
                    "page_size": page_size,
                }
            )
            return {
                "collections": [
                    {
                        "name": "alpha",
                        "document_count": 2,
                        "chunk_count": 3,
                        "image_count": 1,
                        "latest_updated_at": "2026-04-07T00:00:00+00:00",
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "total_count": 1,
                    "returned_count": 1,
                    "has_more": False,
                },
            }

    service = StubService()
    result = handle_list_collections(
        filters={"collection": "alpha"},
        page=1,
        page_size=10,
        runtime=runtime,
        data_service=service,  # type: ignore[arg-type]
    )

    assert service.calls == [{"filters": {"collection": "alpha"}, "page": 1, "page_size": 10}]
    assert result.isError is False
    assert result.content[0].text == "Found 1 collection(s)."
    assert result.structuredContent["collections"][0]["name"] == "alpha"
    assert result.structuredContent["summary"]["collection_count"] == 1

    class EmptyService:
        def list_collections(self, *, filters=None, page=None, page_size=None):
            return {
                "collections": [],
                "pagination": {
                    "page": None,
                    "page_size": None,
                    "total_count": 0,
                    "returned_count": 0,
                    "has_more": False,
                },
            }

    empty = handle_list_collections(
        runtime=runtime,
        data_service=EmptyService(),  # type: ignore[arg-type]
    )

    assert empty.isError is False
    assert empty.content[0].text == "No collections found."
    assert empty.structuredContent["collections"] == []
