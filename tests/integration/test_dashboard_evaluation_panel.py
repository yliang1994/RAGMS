from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ragms.core.evaluation import ReportService
from ragms.observability.dashboard import build_dashboard_context, render_app_shell
from ragms.observability.dashboard.pages import render_evaluation_panel
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
def test_evaluation_panel_renders_empty_state_without_reports(tmp_path: Path) -> None:
    settings = load_settings(_write_settings(tmp_path / "settings.yaml"))
    context = build_dashboard_context(settings, report_service=ReportService(settings))

    page = render_evaluation_panel(context)

    assert page["kind"] == "evaluation_panel"
    assert page["reports"]["kind"] == "empty"
    assert page["report_empty_state"]["title"] == "暂无报告详情"


@pytest.mark.integration
def test_evaluation_panel_renders_report_list_and_detail_entry(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _seed_metadata(settings_path)
    settings = load_settings(settings_path)

    reports_dir = settings.paths.data_dir / "evaluation" / "reports"
    (reports_dir / "sample-report.json").write_text(
        json.dumps(
            {
                "run_id": "sample-report",
                "dataset_version": "sample-v1",
                "collection": "dashboard-demo",
                "metrics_summary": {"hit_rate": 0.95, "mrr": 0.88},
                "quality_gate_status": "passed",
                "config_summary": {"strategy": "hybrid"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
    )

    shell = render_app_shell(context, selected_page="evaluation_panel")
    page = render_evaluation_panel(context, run_id="sample-report")

    assert shell["page"]["kind"] == "evaluation_panel"
    assert page["reports"]["row_count"] >= 2
    assert page["selected_report"]["run_id"] == "sample-report"
    assert page["metric_cards"][0]["label"] in {"hit_rate", "mrr"}
    assert page["selected_report"]["navigation"][1]["target_page"] == "data_browser"
