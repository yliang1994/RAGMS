from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ragms.observability.dashboard import build_dashboard_context, render_app_shell
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


@pytest.mark.integration
def test_dashboard_context_exposes_system_overview_ready_services(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _seed_metadata(settings_path)
    settings = load_settings(settings_path)

    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
    )
    shell = render_app_shell(context, selected_page="system_overview")
    metrics = context.data_service.get_system_overview_metrics()
    recent_failures = context.trace_service.get_recent_failures(limit=5)
    evaluation_runs = context.report_service.list_evaluation_runs(limit=5)

    assert shell["selected_page"] == "system_overview"
    assert shell["service_snapshot"]["report_service"] == "ReportService"
    assert metrics["config_summary"]["default_collection"] == "dashboard-demo"
    assert recent_failures[0]["trace_id"] == "trace-right"
    assert evaluation_runs[0]["collection"] == "dashboard-demo"
