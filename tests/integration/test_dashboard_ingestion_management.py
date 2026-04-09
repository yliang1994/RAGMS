from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ragms.core.management import DataService, DocumentAdminService, TraceService
from ragms.ingestion_pipeline.callbacks import ProgressEvent
from ragms.observability.dashboard import build_dashboard_context, render_app_shell
from ragms.observability.dashboard.pages import render_ingestion_management
from ragms.runtime.config import load_settings
from ragms.storage.sqlite.repositories import DocumentsRepository
from ragms.storage.traces import TraceRepository
from tests.integration.test_dashboard_data_access import _seed_metadata


def _write_settings(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """
            app_name: ragms
            environment: test
            paths:
              project_root: .
              data_dir: data
              logs_dir: logs
            llm:
              provider: openai
              model: gpt-4.1-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_store:
              backend: chroma
              collection: dashboard-demo
            retrieval:
              strategy: hybrid
              fusion_algorithm: rrf
              rerank_backend: disabled
            evaluation:
              backends: [custom_metrics]
            observability:
              enabled: true
              log_file: logs/traces.jsonl
              log_level: INFO
            dashboard:
              enabled: true
              port: 8501
              traces_file: logs/traces.jsonl
              auto_refresh: true
              refresh_interval: 5
              title: RagMS Dashboard
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


class _FakePipeline:
    def __init__(self, callbacks: list[object]) -> None:
        self.callbacks = callbacks


def _build_admin_service(
    settings_path: Path,
    data_service: DataService,
) -> DocumentAdminService:
    settings = load_settings(settings_path)
    documents = DocumentsRepository(data_service.connection)
    traces = TraceRepository(settings.dashboard.traces_file)

    def pipeline_factory(_settings, *, collection=None, callbacks=None):  # noqa: ANN001
        del collection
        return _FakePipeline(list(callbacks or []))

    def batch_runner(pipeline, *, sources, collection, force_rebuild):  # noqa: ANN001
        for callback in pipeline.callbacks:
            callback.on_progress(
                ProgressEvent(
                    trace_id="trace-manage-upload",
                    source_path=str(sources[0]),
                    document_id="doc-upload",
                    completed_stages=1,
                    total_stages=7,
                    current_stage="load",
                    status="running",
                    elapsed_ms=12.0,
                    metadata={"collection": collection},
                )
            )
            callback.on_progress(
                ProgressEvent(
                    trace_id="trace-manage-upload",
                    source_path=str(sources[0]),
                    document_id="doc-upload",
                    completed_stages=7,
                    total_stages=7,
                    current_stage="lifecycle_finalize",
                    status="completed",
                    elapsed_ms=42.0,
                    metadata={"force_rebuild": force_rebuild},
                )
            )
        documents.upsert_document(
            document_id="doc-upload",
            source_path=str(sources[0]),
            source_sha256="sha-upload",
            status="indexed",
            current_stage="lifecycle_finalize",
            last_ingested_at="2026-04-09T12:00:00Z",
        )
        traces.append(
            {
                "trace_id": "trace-manage-upload",
                "trace_type": "ingestion",
                "status": "succeeded",
                "started_at": "2026-04-09T12:00:00Z",
                "finished_at": "2026-04-09T12:00:01Z",
                "duration_ms": 1000,
                "collection": collection,
                "metadata": {},
                "error": None,
                "stages": [],
                "source_path": str(sources[0]),
                "document_id": "doc-upload",
                "total_chunks": 0,
                "total_images": 0,
                "skipped": False,
            }
        )
        return [{"source_path": str(sources[0]), "result": {"status": "completed", "document_id": "doc-upload"}}]

    return DocumentAdminService(
        settings,
        connection=data_service.connection,
        pipeline_factory=pipeline_factory,
        batch_runner=batch_runner,
    )


@pytest.mark.integration
def test_dashboard_ingestion_management_page_supports_upload_and_progress(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _seed_metadata(settings_path)
    settings = load_settings(settings_path)
    admin_service = _build_admin_service(settings_path, data_service)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
        document_admin_service=admin_service,
    )

    shell = render_app_shell(context, selected_page="ingestion_management")
    page = render_ingestion_management(
        context,
        action="ingest",
        uploads=[{"name": "new.pdf", "content": b"pdf-bytes"}],
        force_rebuild=True,
    )

    assert shell["page"]["kind"] == "ingestion_management"
    assert page["documents"]["row_count"] == 3
    assert page["action_result"]["results"][0]["result"]["document_id"] == "doc-upload"
    assert page["progress"]["kind"] == "progress"
    assert page["progress"]["latest"]["current_stage"] == "lifecycle_finalize"
    assert page["progress"]["latest"]["metadata"]["force_rebuild"] is True


@pytest.mark.integration
def test_dashboard_ingestion_management_page_delete_and_rebuild_refresh_state(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _seed_metadata(settings_path)
    settings = load_settings(settings_path)
    admin_service = _build_admin_service(settings_path, data_service)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
        document_admin_service=admin_service,
    )

    image_path = settings.paths.data_dir / "images" / "dashboard-demo" / "img-1.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"image")

    delete_page = render_ingestion_management(
        context,
        action="delete",
        document_id="doc-1",
    )
    rebuilt_page = render_ingestion_management(
        context,
        action="rebuild",
        document_id="doc-1",
    )

    bm25_snapshot = json.loads((settings.paths.data_dir / "indexes" / "sparse" / "dashboard-demo.json").read_text(encoding="utf-8"))
    doc_summary = next(row for row in (rebuilt_page["documents"]["rows"] or []) if row["document_id"] == "doc-1")

    assert delete_page["action_result"]["document"]["status"] == "deleted"
    assert delete_page["progress"]["kind"] == "summary"
    assert rebuilt_page["action_result"]["document"]["current_stage"] == "rebuild_requested"
    assert doc_summary["status"] == "pending"
    assert doc_summary["current_stage"] == "rebuild_requested"
    assert "chunk-1" not in (bm25_snapshot.get("documents") or {})
    assert not image_path.exists()
