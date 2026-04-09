from __future__ import annotations

from pathlib import Path

import anyio

from ragms.core.evaluation import ReportService
from ragms.mcp_server.server import create_server
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings


class StubEvalRunner:
    def __init__(self, payload: dict[str, object] | None = None, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def run(self, **kwargs):
        if self.error is not None:
            raise self.error
        assert self.payload is not None
        return dict(self.payload)


def _runtime(tmp_path: Path) -> tuple[ServiceContainer, ReportService]:
    settings = AppSettings.model_validate({"environment": "test"})
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.storage.sqlite.path = tmp_path / "data" / "metadata" / "ragms.db"
    report_service = ReportService(settings)
    runtime = ServiceContainer(settings=settings, services={"report_service": report_service})
    return runtime, report_service


def test_mcp_server_evaluation_tool_returns_real_structured_payload(tmp_path: Path) -> None:
    runtime, report_service = _runtime(tmp_path)
    report_service.write_report(
        {
            "run_id": "baseline-run",
            "trace_id": "trace-baseline",
            "collection": "demo",
            "dataset_name": "golden",
            "dataset_version": "v1",
            "backend_set": ["custom_metrics"],
            "aggregate_metrics": {"hit_rate": 0.80},
            "quality_gate_status": "passed",
            "config_snapshot": {},
            "samples": [{"sample_id": "sample-1", "metrics_summary": {"hit_rate": 0.80}}],
            "failed_samples": [],
        }
    )
    report_service.set_baseline("baseline-run")
    report_service.write_report(
        {
            "run_id": "run-1",
            "trace_id": "trace-run-1",
            "collection": "demo",
            "dataset_name": "golden",
            "dataset_version": "v1",
            "backend_set": ["custom_metrics"],
            "aggregate_metrics": {"hit_rate": 0.95},
            "quality_gate_status": "passed",
            "config_snapshot": {},
            "samples": [{"sample_id": "sample-1", "metrics_summary": {"hit_rate": 0.95}}],
            "failed_samples": [],
        }
    )
    runtime.services["eval_runner"] = StubEvalRunner(
        {
            "run_id": "run-1",
            "trace_id": "trace-run-1",
            "collection": "demo",
            "dataset_name": "golden",
            "dataset_version": "v1",
            "backend_set": ["custom_metrics"],
            "aggregate_metrics": {"hit_rate": 0.95},
            "quality_gate_status": "passed",
            "failed_samples": [],
            "path": str(tmp_path / "data/evaluation/reports/run-1.json"),
        }
    )

    server = create_server(runtime)
    result = anyio.run(
        server.call_tool,
        "evaluate_collection",
        {
            "collection": "demo",
            "dataset": "golden",
            "eval_options": {"dataset_version": "v1", "backend_set": ["custom_metrics"]},
        },
    )

    assert result.isError is False
    assert result.structuredContent["run_id"] == "run-1"
    assert result.structuredContent["baseline_delta"]["hit_rate"] == 0.15


def test_mcp_server_evaluation_tool_returns_structured_error_for_invalid_request(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    server = create_server(runtime)

    result = anyio.run(server.call_tool, "evaluate_collection", {"collection": ""})

    assert result.isError is True
    assert result.structuredContent["error"]["code"] == -32602
