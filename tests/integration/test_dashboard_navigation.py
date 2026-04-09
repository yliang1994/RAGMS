from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ragms.observability.dashboard import (
    build_dashboard_context,
    resolve_dashboard_navigation_target,
)
from ragms.runtime.config import load_settings
from tests.integration.test_dashboard_data_access import _seed_metadata
from tests.integration.test_dashboard_ingestion_trace import _append_ingestion_variants
from tests.integration.test_dashboard_query_trace import _append_query_variants


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


@pytest.mark.integration
def test_dashboard_navigation_resolves_document_trace_and_report_targets(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _append_ingestion_variants(settings_path)
    settings = load_settings(settings_path)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
    )

    browser = resolve_dashboard_navigation_target(
        context,
        {"target_page": "data_browser", "document_id": "doc-1"},
    )
    ingestion = resolve_dashboard_navigation_target(
        context,
        {"target_page": "ingestion_trace", "trace_ids": ["trace-ingest-legacy"]},
    )
    evaluation = resolve_dashboard_navigation_target(
        context,
        {"target_page": "evaluation_panel", "run_id": "run-1"},
    )

    assert browser["kind"] == "data_browser"
    assert browser["selected_document"]["document_id"] == "doc-1"
    assert ingestion["kind"] == "ingestion_trace"
    assert ingestion["selected_trace"]["summary"]["trace_id"] == "trace-ingest-legacy"
    assert evaluation["kind"] == "evaluation_panel"
    assert evaluation["selected_report"]["run_id"] == "run-1"


@pytest.mark.integration
def test_dashboard_navigation_resolves_query_trace_comparison_targets(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _append_query_variants(settings_path)
    settings = load_settings(settings_path)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
    )

    query_page = resolve_dashboard_navigation_target(
        context,
        {
            "target_page": "query_trace",
            "left_trace_id": "trace-query-full",
            "right_trace_id": "trace-query-retrieval-only",
        },
    )

    assert query_page["kind"] == "query_trace"
    assert query_page["comparison_selection"]["left_trace_id"] == "trace-query-full"
    assert query_page["comparison_selection"]["right_trace_id"] == "trace-query-retrieval-only"
