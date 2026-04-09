from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragms.core.management.document_admin_service import DocumentAdminService
from ragms.ingestion_pipeline.callbacks import ProgressEvent
from ragms.ingestion_pipeline.lifecycle import IngestionLifecycleError
from ragms.runtime.settings_models import AppSettings
from ragms.storage.sqlite.repositories import DocumentsRepository, ImagesRepository, IngestionHistoryRepository
from ragms.storage.sqlite.schema import initialize_metadata_schema


class _FakePipeline:
    def __init__(self, callbacks: list[object]) -> None:
        self.callbacks = callbacks


def _build_settings(tmp_path: Path) -> AppSettings:
    settings = AppSettings()
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.storage.sqlite.path = tmp_path / "data" / "metadata" / "ragms.db"
    settings.vector_store.collection = "contract-demo"
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _build_service(tmp_path: Path) -> tuple[DocumentAdminService, Any]:
    settings = _build_settings(tmp_path)
    connection = initialize_metadata_schema(settings.storage.sqlite.path)
    capture: dict[str, Any] = {}

    def pipeline_factory(_settings, *, collection=None, callbacks=None):  # noqa: ANN001
        capture["collection"] = collection
        capture["callbacks"] = list(callbacks or [])
        return _FakePipeline(list(callbacks or []))

    def batch_runner(pipeline, *, sources, collection, force_rebuild):  # noqa: ANN001
        capture["sources"] = list(sources)
        capture["force_rebuild"] = force_rebuild
        for callback in pipeline.callbacks:
            callback.on_progress(
                ProgressEvent(
                    trace_id="trace-contract",
                    source_path=str(sources[0]),
                    document_id="doc-contract",
                    completed_stages=1,
                    total_stages=7,
                    current_stage="load",
                    status="running",
                    elapsed_ms=10.0,
                    metadata={"collection": collection},
                )
            )
            callback.on_progress(
                ProgressEvent(
                    trace_id="trace-contract",
                    source_path=str(sources[0]),
                    document_id="doc-contract",
                    completed_stages=7,
                    total_stages=7,
                    current_stage="lifecycle_finalize",
                    status="completed",
                    elapsed_ms=30.0,
                    metadata={"force_rebuild": force_rebuild},
                )
            )
        return [{"source_path": str(sources[0]), "result": {"status": "completed", "document_id": "doc-contract"}}]

    service = DocumentAdminService(
        settings,
        connection=connection,
        pipeline_factory=pipeline_factory,
        batch_runner=batch_runner,
    )
    return service, capture


def _seed_document_state(service: DocumentAdminService, *, image: bool = True) -> Path:
    service.documents_repository.upsert_document(
        document_id="doc-contract",
        source_path="docs/a.pdf",
        source_sha256="sha-contract",
        status="indexed",
        current_stage="lifecycle_finalize",
    )
    service.ingestion_history.record_processed(
        source_path="docs/a.pdf",
        source_sha256="sha-contract",
        document_id="doc-contract",
    )
    image_path = service.settings.paths.data_dir / "images" / service.settings.vector_store.collection / "img-contract.png"
    if image:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")
        ImagesRepository(service.connection).upsert_image(
            image_id="img-contract",
            document_id="doc-contract",
            chunk_id="chunk-1",
            file_path=str(image_path),
            source_path="docs/a.pdf",
            image_hash="image-hash",
            page=1,
        )
    bm25_dir = service.settings.paths.data_dir / "indexes" / "sparse"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    (bm25_dir / f"{service.settings.vector_store.collection}.json").write_text(
        json.dumps(
            {
                "documents": {
                    "chunk-1": {
                        "document_id": "doc-contract",
                        "source_path": "docs/a.pdf",
                    },
                    "chunk-2": {
                        "document_id": "other-doc",
                        "source_path": "docs/b.pdf",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return image_path


@pytest.mark.unit
def test_document_admin_service_contract_ingest_triggers_pipeline_and_returns_progress(tmp_path: Path) -> None:
    service, capture = _build_service(tmp_path)
    progress_updates: list[dict[str, Any]] = []

    result = service.ingest_documents(
        [tmp_path / "docs" / "a.pdf"],
        collection="override-demo",
        force_rebuild=True,
        on_progress=progress_updates.append,
    )

    assert capture["collection"] == "override-demo"
    assert capture["sources"] == [str(tmp_path / "docs" / "a.pdf")]
    assert capture["force_rebuild"] is True
    assert result["collection"] == "override-demo"
    assert result["results"][0]["result"]["document_id"] == "doc-contract"
    assert [event["current_stage"] for event in result["progress_events"]] == ["load", "lifecycle_finalize"]
    assert progress_updates[-1]["metadata"]["force_rebuild"] is True


@pytest.mark.unit
def test_document_admin_service_contract_delete_and_rebuild_return_stable_payloads(tmp_path: Path) -> None:
    service, _capture = _build_service(tmp_path)
    image_path = _seed_document_state(service)

    deleted = service.delete_document("doc-contract")
    rebuilt = service.rebuild_document("doc-contract")
    bm25_snapshot = json.loads(
        (
            service.settings.paths.data_dir
            / "indexes"
            / "sparse"
            / f"{service.settings.vector_store.collection}.json"
        ).read_text(encoding="utf-8")
    )

    assert deleted["action"] == "delete"
    assert deleted["document"]["status"] == "deleted"
    assert deleted["cleanup"]["ingestion_history"] == 1
    assert deleted["cleanup"]["bm25_entries"] == 1
    assert deleted["cleanup"]["image_entries"] == 1
    assert rebuilt["action"] == "rebuild"
    assert rebuilt["document"]["status"] == "pending"
    assert rebuilt["document"]["current_stage"] == "rebuild_requested"
    assert "chunk-1" not in (bm25_snapshot.get("documents") or {})
    assert not image_path.exists()


@pytest.mark.unit
def test_document_admin_service_contract_converges_unknown_document_and_cleanup_failures(tmp_path: Path) -> None:
    service, _capture = _build_service(tmp_path)
    _seed_document_state(service, image=False)
    service._delete_bm25_entries = lambda _record: (_ for _ in ()).throw(RuntimeError("broken"))  # type: ignore[method-assign]

    with pytest.raises(IngestionLifecycleError, match="Unknown document: missing"):
        service.delete_document("missing")

    with pytest.raises(IngestionLifecycleError, match="Lifecycle cleanup failed in bm25_cleanup"):
        service.delete_document("doc-contract")

    assert IngestionHistoryRepository(service.connection).get_by_sha256("sha-contract") is not None
    assert DocumentsRepository(service.connection).get_by_document_id("doc-contract") is not None
