from __future__ import annotations

from pathlib import Path

import pytest

from ragms.core.evaluation import ReportService
from ragms.observability.dashboard import build_dashboard_context
from ragms.observability.dashboard.pages import render_evaluation_panel
from ragms.runtime.config import load_settings
from tests.integration.test_dashboard_evaluation_panel import _write_settings


class _StubDashboardEvalRunner:
    def __init__(self, report_service: ReportService) -> None:
        self.report_service = report_service

    def run(self, **kwargs):
        payload = self.report_service.write_report(
            {
                "run_id": "dashboard-run",
                "trace_id": "dashboard-trace",
                "collection": kwargs.get("collection") or "dashboard-demo",
                "dataset_name": kwargs.get("dataset_name") or "golden",
                "dataset_version": kwargs.get("dataset_version") or "v1",
                "backend_set": list(kwargs.get("backend_set") or ["custom_metrics"]),
                "aggregate_metrics": {"hit_rate": 0.93, "mrr": 0.81},
                "quality_gate_status": "passed",
                "config_snapshot": {"evaluation": {"backends": list(kwargs.get("backend_set") or ["custom_metrics"])}},
                "samples": [{"sample_id": "sample-1", "metrics_summary": {"hit_rate": 0.93}}],
                "failed_samples": [{"sample_id": "sample-2", "stage": "sample_build", "error": {"message": "missing citation"}}],
            }
        )
        return {
            "run_id": payload["run_id"],
            "trace_id": payload["trace_id"],
            "path": payload["path"],
        }

def assert_dashboard_regression_flow(tmp_path: Path) -> dict[str, object]:
    """Run the final dashboard evaluation flow and return key page payloads."""

    settings = load_settings(_write_settings(tmp_path / "settings.yaml"))
    dataset_dir = settings.paths.data_dir / "evaluation" / "datasets" / "golden"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "v1.json").write_text(
        '{"dataset_name":"golden","dataset_version":"v1","collection":"dashboard-demo","samples":[{"sample_id":"sample-1","query":"what is rag","evaluation_modes":["retrieval"]}]}\n',
        encoding="utf-8",
    )
    report_service = ReportService(settings)
    report_service.write_report(
        {
            "run_id": "baseline-run",
            "trace_id": "baseline-trace",
            "collection": "dashboard-demo",
            "dataset_name": "golden",
            "dataset_version": "v1",
            "backend_set": ["custom_metrics"],
            "aggregate_metrics": {"hit_rate": 0.80, "mrr": 0.70},
            "quality_gate_status": "passed",
            "config_snapshot": {"evaluation": {"backends": ["custom_metrics"]}},
            "samples": [{"sample_id": "sample-1", "metrics_summary": {"hit_rate": 0.80}}],
            "failed_samples": [],
        }
    )
    report_service.set_baseline("baseline-run")
    context = build_dashboard_context(
        settings,
        report_service=report_service,
        eval_runner=_StubDashboardEvalRunner(report_service),
    )

    page = render_evaluation_panel(
        context,
        run_request={
            "collection": "dashboard-demo",
            "dataset_name": "golden",
            "dataset_version": "v1",
            "backend_set": ["custom_metrics"],
        },
        run_id="dashboard-run",
    )
    empty_root = tmp_path / "empty"
    empty_root.mkdir(parents=True, exist_ok=True)
    empty_settings = load_settings(_write_settings(empty_root / "settings.yaml"))
    empty_context = build_dashboard_context(
        empty_settings,
        report_service=ReportService(empty_settings),
        eval_runner=_StubDashboardEvalRunner(ReportService(empty_settings)),
    )
    empty_page = render_evaluation_panel(empty_context)
    return {
        "page": page,
        "empty_page": empty_page,
    }


@pytest.mark.e2e
def test_evaluation_run_is_visible_and_comparable_in_dashboard(tmp_path: Path) -> None:
    payloads = assert_dashboard_regression_flow(tmp_path)
    page = payloads["page"]
    empty_page = payloads["empty_page"]

    assert page["run_state"]["status"] == "succeeded"
    assert page["selected_report"]["run_id"] == "dashboard-run"
    assert page["results"]["comparison"]["baseline_run"]["run_id"] == "baseline-run"
    assert page["baseline_actions"]["current_baseline"]["run_id"] == "baseline-run"
    assert page["results"]["failed_samples"]["row_count"] == 1
    assert page["results"]["provider_compare"][0]["run_id"] in {"baseline-run", "dashboard-run"}
    assert empty_page["reports"]["kind"] == "empty"
    assert empty_page["run_form"]["selected_dataset_name"] is None
    assert empty_page["report_empty_state"]["title"] == "暂无报告详情"
