from __future__ import annotations

import json
from pathlib import Path

from ragms.core.management import DataService
from ragms.mcp_server.tools.documents import (
    handle_get_document_summary,
    serialize_document_summary,
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
        "average_document_length": 8.0 if documents else 0.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_serialize_document_summary_preserves_contract_fields() -> None:
    payload = serialize_document_summary(
        {
            "document_id": "doc-1",
            "source_path": "docs/a.md",
            "primary_collection": "alpha",
            "collections": ["alpha"],
            "summary": "overview",
            "structure_outline": ["Intro"],
            "key_metadata": {"title": "Alpha"},
            "ingestion_status": {"status": "indexed"},
            "page_summary": {"pages": [1], "page_count": 1},
            "image_summary": {"image_count": 1, "images": []},
            "chunk_count": 2,
        }
    )

    assert payload["document_id"] == "doc-1"
    assert payload["primary_collection"] == "alpha"
    assert payload["page_summary"]["page_count"] == 1
    assert payload["chunk_count"] == 2


def test_data_service_get_document_summary_assembles_registry_index_and_images(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    connection = initialize_metadata_schema(runtime.settings.storage.sqlite.path)
    documents_repo = DocumentsRepository(connection)
    images_repo = ImagesRepository(connection)

    documents_repo.upsert_document(
        document_id="doc-1",
        source_path=str(tmp_path / "docs" / "alpha.md"),
        source_sha256="sha-alpha",
        status="indexed",
        current_stage="lifecycle",
        last_ingested_at="2026-04-07T00:00:00+00:00",
    )
    images_repo.upsert_image(
        image_id="img-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        file_path=str(runtime.settings.paths.data_dir / "images" / "alpha" / "img-1.png"),
        source_path=str(tmp_path / "docs" / "img-1.png"),
        image_hash="hash-1",
        page=2,
    )

    _write_bm25_snapshot(
        runtime.settings.paths.data_dir / "indexes" / "sparse" / "alpha.json",
        collection="alpha",
        documents=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source_path": str(tmp_path / "docs" / "alpha.md"),
                "chunk_index": 0,
                "content": "Alpha intro",
                "metadata": {
                    "title": "Alpha",
                    "doc_type": "md",
                    "page": 1,
                    "section_path": ["Intro"],
                    "chunk_title": "Introduction",
                    "chunk_summary": "Alpha introduction summary",
                    "chunk_tags": ["ops"],
                },
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-1",
                "source_path": str(tmp_path / "docs" / "alpha.md"),
                "chunk_index": 1,
                "content": "Alpha details",
                "metadata": {
                    "page": 2,
                    "section_path": ["Usage", "CLI"],
                },
            },
        ],
    )

    service = DataService(runtime.settings, connection=connection)
    summary = service.get_document_summary("doc-1")

    assert summary["document_id"] == "doc-1"
    assert summary["primary_collection"] == "alpha"
    assert summary["collections"] == ["alpha"]
    assert summary["summary"] == "Alpha introduction summary"
    assert summary["structure_outline"] == ["Intro", "Usage / CLI"]
    assert summary["key_metadata"] == {
        "source_sha256": "sha-alpha",
        "doc_type": "md",
        "title": "Alpha",
        "chunk_title": "Introduction",
        "chunk_tags": ["ops"],
    }
    assert summary["ingestion_status"]["status"] == "indexed"
    assert summary["page_summary"] == {"pages": [1, 2], "page_count": 2}
    assert summary["image_summary"]["image_count"] == 1
    assert summary["chunk_count"] == 2


def test_handle_get_document_summary_returns_error_when_document_is_missing(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    result = handle_get_document_summary(document_id="missing-doc", runtime=runtime)

    assert result.isError is True
    assert result.structuredContent["error"]["code"] == -32602
    assert result.structuredContent["error"]["data"] == {"document_id": "missing-doc"}
    assert result.content[0].text == "Document not found: missing-doc"


def test_handle_get_document_summary_returns_empty_index_summary_without_error(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    connection = initialize_metadata_schema(runtime.settings.storage.sqlite.path)
    documents_repo = DocumentsRepository(connection)
    documents_repo.upsert_document(
        document_id="doc-empty",
        source_path=str(tmp_path / "docs" / "empty.md"),
        source_sha256="sha-empty",
        status="pending",
        current_stage="registered",
    )

    result = handle_get_document_summary(
        document_id="doc-empty",
        runtime=runtime,
        data_service=DataService(runtime.settings, connection=connection),
    )

    assert result.isError is False
    assert result.structuredContent["document_id"] == "doc-empty"
    assert result.structuredContent["collections"] == []
    assert result.structuredContent["structure_outline"] == []
    assert result.structuredContent["image_summary"]["image_count"] == 0
    assert result.structuredContent["chunk_count"] == 0
