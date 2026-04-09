from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ragms.core.management import DocumentAdminService
from ragms.observability.dashboard import build_dashboard_context, render_app_shell
from ragms.runtime.config import load_settings
from ragms.runtime.container import build_container, PlaceholderService
from ragms.core.evaluation import ReportService
from scripts.run_dashboard import run_dashboard_main


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
              collection: dashboard-tests
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
              port: 8601
              traces_file: logs/traces.jsonl
              auto_refresh: true
              refresh_interval: 7
              title: RagMS Local Dashboard
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.integration
def test_dashboard_shell_registers_all_placeholder_pages_and_context(tmp_path: Path) -> None:
    settings = load_settings(_write_settings(tmp_path / "settings.yaml"))
    runtime = build_container(settings)

    context = build_dashboard_context(settings, runtime=runtime)
    shell = render_app_shell(context, selected_page="query_trace")

    assert [page.key for page in context.pages] == [
        "system_overview",
        "data_browser",
        "ingestion_management",
        "ingestion_trace",
        "query_trace",
        "evaluation_panel",
    ]
    assert shell["selected_page"] == "query_trace"
    assert shell["auto_refresh"] is True
    assert shell["refresh_interval"] == 7
    assert shell["title"] == "RagMS Local Dashboard"
    assert isinstance(context.document_admin_service, DocumentAdminService)
    assert isinstance(context.report_service, ReportService)
    assert context.service_snapshot["trace_service"] == "TraceService"


@pytest.mark.integration
def test_run_dashboard_main_validates_shell_without_starting_streamlit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")

    assert run_dashboard_main(["--settings", str(settings_path)]) == 0

    output = capsys.readouterr().out
    assert "Dashboard bootstrap ready" in output
    assert "pages=6" in output
    assert "port=8601" in output
