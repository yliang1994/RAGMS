from __future__ import annotations

from pathlib import Path

from ragms.core.evaluation import ReportService
from ragms.mcp_server.tools.evaluation import (
    handle_evaluate_collection,
    normalize_evaluation_request,
    serialize_evaluation_result,
)
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings


class StubEvalRunner:
    def __init__(self, payload: dict[str, object] | None = None, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        assert self.payload is not None
        return dict(self.payload)


def _runtime(tmp_path: Path) -> ServiceContainer:
    settings = AppSettings.model_validate({"environment": "test"})
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.storage.sqlite.path = tmp_path / "data" / "metadata" / "ragms.db"
    return ServiceContainer(settings=settings, services={})


def test_normalize_evaluation_request_maps_options_to_runner_contract() -> None:
    request = normalize_evaluation_request(
        collection="demo",
        dataset="golden",
        metrics=["hit_rate"],
        eval_options={"dataset_version": "v2", "backend_set": ["custom_metrics", "custom_metrics"], "top_k": 3},
        baseline_mode="compare",
    )

    assert request["collection"] == "demo"
    assert request["dataset_name"] == "golden"
    assert request["dataset_version"] == "v2"
    assert request["backend_set"] == ["custom_metrics"]
    assert request["top_k"] == 3


def test_handle_evaluate_collection_wraps_runner_and_baseline_delta(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    report_service = ReportService(runtime.settings)
    report_service.write_report(
        {
            "run_id": "baseline-run",
            "trace_id": "trace-baseline",
            "collection": "demo",
            "dataset_name": "golden",
            "dataset_version": "v1",
            "backend_set": ["custom_metrics"],
            "aggregate_metrics": {"hit_rate": 0.8},
            "quality_gate_status": "passed",
            "config_snapshot": {},
            "samples": [{"sample_id": "sample-1", "metrics_summary": {"hit_rate": 0.8}}],
            "failed_samples": [],
        }
    )
    report_service.set_baseline("baseline-run")
    report_service.write_report(
        {
            "run_id": "run-1",
            "trace_id": "trace-1",
            "collection": "demo",
            "dataset_name": "golden",
            "dataset_version": "v1",
            "backend_set": ["custom_metrics"],
            "aggregate_metrics": {"hit_rate": 0.95},
            "quality_gate_status": "passed",
            "config_snapshot": {},
            "samples": [{"sample_id": "sample-1", "metrics_summary": {"hit_rate": 0.95}}],
            "failed_samples": [{"sample_id": "sample-2", "stage": "sample_build", "error": {"message": "broken"}}],
        }
    )
    runner = StubEvalRunner(
        {
            "run_id": "run-1",
            "trace_id": "trace-1",
            "collection": "demo",
            "dataset_name": "golden",
            "dataset_version": "v1",
            "backend_set": ["custom_metrics"],
            "aggregate_metrics": {"hit_rate": 0.95},
            "quality_gate_status": "passed",
            "failed_samples": [{"sample_id": "sample-2"}],
            "path": str(tmp_path / "data/evaluation/reports/run-1.json"),
        }
    )

    result = handle_evaluate_collection(
        collection="demo",
        dataset="golden",
        eval_options={"dataset_version": "v1", "backend_set": ["custom_metrics"]},
        runtime=runtime,
        eval_runner=runner,
        report_service=report_service,
    )

    assert result.isError is False
    assert runner.calls[0]["dataset_name"] == "golden"
    assert result.structuredContent["run_id"] == "run-1"
    assert result.structuredContent["baseline_delta"]["hit_rate"] == 0.15
    assert result.structuredContent["failed_samples_count"] == 1


def test_handle_evaluate_collection_returns_structured_error_on_failure(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    result = handle_evaluate_collection(
        collection="demo",
        dataset="golden",
        runtime=runtime,
        eval_runner=StubEvalRunner(error=RuntimeError("backend exploded")),
        report_service=ReportService(runtime.settings),
    )

    assert result.isError is True
    assert result.structuredContent["collection"] == "demo"
    assert result.structuredContent["errors"][0]["message"] == "Internal error"


def test_serialize_evaluation_result_preserves_required_fields() -> None:
    payload = serialize_evaluation_result(
        {
            "run_id": "run-1",
            "trace_id": "trace-1",
            "collection": "demo",
            "dataset_name": "golden",
            "dataset_version": "v1",
            "backend_set": ["custom_metrics"],
            "aggregate_metrics": {"hit_rate": 0.9},
            "quality_gate_status": "passed",
            "failed_samples": [{"sample_id": "sample-1"}],
            "path": "data/evaluation/reports/run-1.json",
        },
        baseline_delta={"hit_rate": 0.1},
        errors=[{"message": "warn"}],
    )

    assert payload["run_id"] == "run-1"
    assert payload["baseline_delta"]["hit_rate"] == 0.1
    assert payload["failed_samples_count"] == 1
