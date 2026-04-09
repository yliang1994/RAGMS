from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ragms.observability.dashboard import build_dashboard_context, render_app_shell
from ragms.runtime.config import load_settings
from tests.integration.test_dashboard_data_access import _seed_metadata
from tests.integration.test_dashboard_ingestion_management import _build_admin_service
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


@pytest.mark.e2e
def test_dashboard_smoke_renders_all_g_stage_pages(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _append_query_variants(settings_path)
    settings = load_settings(settings_path)
    admin_service = _build_admin_service(settings_path, data_service)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
        document_admin_service=admin_service,
    )

    rendered = {
        page: render_app_shell(context, selected_page=page)["page"]["kind"]
        for page in [
            "system_overview",
            "data_browser",
            "ingestion_management",
            "ingestion_trace",
            "query_trace",
            "evaluation_panel",
        ]
    }

    assert rendered == {
        "system_overview": "system_overview",
        "data_browser": "data_browser",
        "ingestion_management": "ingestion_management",
        "ingestion_trace": "ingestion_trace",
        "query_trace": "query_trace",
        "evaluation_panel": "evaluation_panel",
    }
