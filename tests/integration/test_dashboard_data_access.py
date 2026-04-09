from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ragms.core.evaluation import ReportService
from ragms.core.management import DataService, TraceService
from ragms.core.trace_collector import TraceManager
from ragms.runtime.config import load_settings
from ragms.storage.sqlite.repositories import DocumentsRepository, ImagesRepository
from ragms.storage.sqlite.schema import initialize_metadata_schema
from ragms.storage.traces import TraceRepository


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


def _seed_metadata(settings_path: Path) -> tuple[DataService, TraceService, ReportService]:
    settings = load_settings(settings_path)
    connection = initialize_metadata_schema(settings.storage.sqlite.path)
    documents = DocumentsRepository(connection)
    images = ImagesRepository(connection)
    documents.upsert_document(
        document_id="doc-1",
        source_path="docs/a.pdf",
        source_sha256="sha-a",
        status="indexed",
        current_stage="lifecycle_finalize",
        last_ingested_at="2026-04-09T10:00:00Z",
    )
    documents.upsert_document(
        document_id="doc-2",
        source_path="docs/b.pdf",
        source_sha256="sha-b",
        status="failed",
        current_stage="transform",
        failure_reason="parse failed",
    )
    images.upsert_image(
        image_id="img-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        file_path=str(settings.paths.data_dir / "images" / "dashboard-demo" / "img-1.png"),
        source_path="docs/a.pdf",
        image_hash="hash-1",
        page=1,
        position={"x": 1},
    )
    bm25_dir = settings.paths.data_dir / "indexes" / "sparse"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    (bm25_dir / "dashboard-demo.json").write_text(
        json.dumps(
            {
                "collection": "dashboard-demo",
                "documents": {
                    "chunk-1": {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "source_path": "docs/a.pdf",
                        "chunk_index": 0,
                        "content": "alpha",
                        "metadata": {"page": 1, "chunk_summary": "alpha summary"},
                    },
                    "chunk-2": {
                        "chunk_id": "chunk-2",
                        "document_id": "doc-2",
                        "source_path": "docs/b.pdf",
                        "chunk_index": 0,
                        "content": "beta",
                        "metadata": {"page": 2},
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    traces = TraceRepository(settings.dashboard.traces_file)
    manager = TraceManager()
    left = manager.start_trace("query", trace_id="trace-left", collection="dashboard-demo", query="what is rag")
    manager.start_stage(left, "query_processing", input_payload={"query": "what is rag"})
    manager.finish_stage(left, "query_processing", output_payload={"normalized_query": "what is rag"}, metadata={"method": "normalize"})
    manager.start_stage(left, "dense_retrieval", input_payload={"query": "what is rag"})
    manager.finish_stage(left, "dense_retrieval", output_payload={"retrieved_count": 2}, metadata={"provider": "chroma"})
    traces.append(manager.finish_trace(left, status="succeeded", top_k_results=["chunk-1"]))

    right = manager.start_trace("query", trace_id="trace-right", collection="dashboard-demo", query="what is rag")
    manager.start_stage(right, "query_processing", input_payload={"query": "what is rag"})
    manager.finish_stage(right, "query_processing", output_payload={"normalized_query": "what is rag"}, metadata={"method": "normalize"})
    manager.start_stage(right, "dense_retrieval", input_payload={"query": "what is rag"})
    manager.finish_stage(right, "dense_retrieval", output_payload={"retrieved_count": 1}, metadata={"provider": "chroma", "fallback_reason": "timeout"})
    traces.append(manager.finish_trace(right, status="failed", error=RuntimeError("boom"), top_k_results=[]))

    reports_dir = settings.paths.data_dir / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "run-1.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "dataset_version": "v1",
                "collection": "dashboard-demo",
                "metrics_summary": {"hit_rate": 0.9},
                "quality_gate_status": "passed",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return DataService(settings, connection=connection), TraceService(settings, repository=traces), ReportService(settings)


@pytest.mark.integration
def test_dashboard_data_services_expose_unified_read_models(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _seed_metadata(settings_path)

    overview = data_service.get_system_overview_metrics()
    failures = trace_service.get_recent_failures(limit=5)
    comparison = trace_service.compare_traces("trace-left", "trace-right")
    reports = report_service.list_evaluation_runs()

    assert overview["collection_count"] >= 1
    assert overview["document_count"] == 2
    assert overview["chunk_count"] == 2
    assert overview["image_count"] == 1
    assert overview["status_counts"]["indexed"] == 1
    assert overview["status_counts"]["failed"] == 1
    assert failures[0]["trace_id"] == "trace-right"
    assert comparison["left_trace_id"] == "trace-left"
    assert comparison["right_trace_id"] == "trace-right"
    assert comparison["stage_comparisons"][1]["stage_name"] == "dense_retrieval"
    assert reports[0]["run_id"] == "run-1"
