from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ragms.core.trace_collector import TraceManager
from ragms.observability.dashboard import (
    build_dashboard_context,
    render_app_shell,
    render_ingestion_trace,
)
from ragms.runtime.config import load_settings
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


def _append_ingestion_variants(settings_path: Path) -> tuple[object, object, object]:
    data_service, trace_service, report_service = _seed_metadata(settings_path)
    manager = TraceManager()

    legacy = manager.start_trace(
        "ingestion",
        trace_id="trace-ingest-legacy",
        collection="dashboard-demo",
        document_id="doc-1",
        source_path="docs/a.pdf",
    )
    manager.start_stage(legacy, "file_integrity", input_payload={"source_path": "docs/a.pdf"})
    manager.finish_stage(legacy, "file_integrity", output_payload={"should_skip": False}, metadata={"method": "sha256"})
    manager.start_stage(legacy, "load", input_payload={"source_path": "docs/a.pdf"})
    manager.finish_stage(legacy, "load", output_payload={"document_id": "doc-1"}, metadata={"provider": "markitdown"})
    manager.start_stage(legacy, "split", input_payload={"document_id": "doc-1"})
    manager.finish_stage(legacy, "split", output_payload={"chunk_count": 2}, metadata={"provider": "recursive_character", "chunk_count": 2})
    manager.start_stage(legacy, "transform", input_payload={"chunk_count": 2})
    manager.finish_stage(
        legacy,
        "transform",
        status="partial_success",
        output_payload={"enriched_count": 2},
        metadata={"provider": "rule+llm", "fallback_reason": "vision_disabled"},
    )
    manager.start_stage(legacy, "embed", input_payload={"chunk_count": 2})
    manager.finish_stage(legacy, "embed", output_payload={"vector_count": 2}, metadata={"provider": "openai", "batch_count": 1})
    manager.start_stage(legacy, "upsert", input_payload={"vector_count": 2})
    manager.finish_stage(legacy, "upsert", output_payload={"written": 2}, metadata={"provider": "chroma", "upsert_count": 2})
    manager.start_stage(legacy, "lifecycle_finalize", input_payload={"document_id": "doc-1"})
    manager.finish_stage(
        legacy,
        "lifecycle_finalize",
        status="partial_success",
        output_payload={"status": "indexed"},
        metadata={"provider": "document_registry"},
    )
    trace_service.repository.append(
        manager.finish_trace(
            legacy,
            status="partial_success",
            total_chunks=2,
            total_images=1,
        )
    )

    skipped = manager.start_trace(
        "ingestion",
        trace_id="trace-ingest-skip",
        collection="dashboard-demo",
        document_id="doc-2",
        source_path="docs/b.pdf",
    )
    manager.start_stage(skipped, "file_integrity", input_payload={"source_path": "docs/b.pdf"})
    manager.finish_stage(
        skipped,
        "file_integrity",
        status="skipped",
        output_payload={"should_skip": True},
        metadata={"method": "sha256", "skip_reason": "content_unchanged"},
    )
    manager.start_stage(skipped, "lifecycle_finalize", input_payload={"document_id": "doc-2"})
    manager.finish_stage(
        skipped,
        "lifecycle_finalize",
        status="skipped",
        output_payload={"status": "skipped"},
        metadata={"skip_reason": "content_unchanged"},
    )
    trace_service.repository.append(
        manager.finish_trace(
            skipped,
            status="skipped",
            total_chunks=0,
            total_images=0,
            skipped="content_unchanged",
        )
    )
    return data_service, trace_service, report_service


@pytest.mark.integration
def test_dashboard_ingestion_trace_page_filters_and_normalizes_stages(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _append_ingestion_variants(settings_path)
    settings = load_settings(settings_path)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
    )

    shell = render_app_shell(context, selected_page="ingestion_trace")
    filtered = render_ingestion_trace(
        context,
        status="partial_success",
        collection="dashboard-demo",
        trace_id="legacy",
    )
    service_rows = trace_service.list_traces(trace_type="ingestion", trace_id="legacy")

    assert shell["page"]["kind"] == "ingestion_trace"
    assert "partial_success" in shell["page"]["filter_model"]["statuses"]
    assert shell["page"]["filter_model"]["collections"] == ["dashboard-demo"]
    assert filtered["trace_list"]["row_count"] == 1
    assert filtered["selected_trace"]["summary"]["trace_id"] == "trace-ingest-legacy"
    assert filtered["selected_trace"]["navigation"][1]["target_page"] == "data_browser"
    assert filtered["timeline"]["timeline"][2]["stage_name"] == "chunking"
    assert filtered["timeline"]["timeline"][2]["raw_stage_name"] == "split"
    assert filtered["timeline"]["timeline"][4]["stage_name"] == "embedding"
    assert filtered["timeline"]["timeline"][4]["provider"] == "openai"
    assert filtered["timeline"]["timeline"][5]["stage_name"] == "storage"
    assert filtered["timeline"]["timeline"][5]["progress"]["upsert_count"] == 2
    assert filtered["duration_chart"]["point_count"] == 7
    assert service_rows[0]["trace_id"] == "trace-ingest-legacy"


@pytest.mark.integration
def test_dashboard_ingestion_trace_page_explains_skipped_stages(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _append_ingestion_variants(settings_path)
    settings = load_settings(settings_path)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
    )

    page = render_ingestion_trace(context, status="skipped", trace_id="trace-ingest-skip")

    assert page["trace_list"]["row_count"] == 1
    assert page["selected_trace"]["summary"]["status"] == "skipped"
    assert page["timeline"]["timeline"][0]["stage_name"] == "file_integrity"
    assert page["timeline"]["timeline"][1]["stage_name"] == "lifecycle_finalize"
    assert page["timeline"]["omitted_stage_names"] == [
        "load",
        "chunking",
        "transform",
        "embedding",
        "storage",
    ]
    assert page["timeline"]["progress_summary"] == {"completed": 2, "expected": 7}
