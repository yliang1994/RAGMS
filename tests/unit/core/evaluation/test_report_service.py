from __future__ import annotations

from pathlib import Path

import pytest

from ragms.core.evaluation import ReportService
from ragms.core.evaluation.report_service import ReportServiceError
from ragms.core.models import EvaluationRunSummary, build_baseline_scope
from ragms.runtime.settings_models import AppSettings


def _build_settings(tmp_path: Path) -> AppSettings:
    settings = AppSettings()
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.storage.sqlite.path = tmp_path / "data" / "metadata" / "ragms.db"
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _write_report(
    service: ReportService,
    *,
    run_id: str,
    dataset_version: str = "v1",
    backend_set: list[str] | None = None,
    aggregate_metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    return service.write_report(
        EvaluationRunSummary(
            run_id=run_id,
            trace_id=f"trace-{run_id}",
            collection="docs",
            dataset_name="golden",
            dataset_version=dataset_version,
            backend_set=backend_set or ["custom_metrics"],
            config_snapshot={"evaluation": {"backends": backend_set or ["custom_metrics"]}},
            started_at="2026-04-09T00:00:00Z",
            finished_at="2026-04-09T00:01:00Z",
            aggregate_metrics=aggregate_metrics or {"hit_rate": 0.9, "mrr": 0.8},
            quality_gate_status="passed",
            samples=[
                {"sample_id": "sample-1", "metrics_summary": {"hit_rate": aggregate_metrics.get("hit_rate", 0.9) if aggregate_metrics else 0.9}}
            ],
        )
    )


def test_report_service_sets_switches_and_clears_baseline(tmp_path: Path) -> None:
    service = ReportService(_build_settings(tmp_path))
    _write_report(service, run_id="run-a")
    _write_report(service, run_id="run-b", aggregate_metrics={"hit_rate": 0.95, "mrr": 0.82})

    first = service.set_baseline("run-a")
    assert first["baseline"]["run_id"] == "run-a"
    assert service.get_baseline(collection="docs", dataset_version="v1", backend_set=["custom_metrics"])["run_id"] == "run-a"

    switched = service.set_baseline("run-b")
    assert switched["baseline"]["run_id"] == "run-b"
    assert service.get_baseline(collection="docs", dataset_version="v1", backend_set=["custom_metrics"])["run_id"] == "run-b"

    cleared = service.set_baseline(
        collection="docs",
        dataset_version="v1",
        backend_set=["custom_metrics"],
    )
    assert cleared["cleared"] is True
    assert service.get_baseline(collection="docs", dataset_version="v1", backend_set=["custom_metrics"]) is None


def test_report_service_compare_contract_is_stable_and_scope_checked(tmp_path: Path) -> None:
    service = ReportService(_build_settings(tmp_path))
    _write_report(service, run_id="baseline", aggregate_metrics={"hit_rate": 0.80, "mrr": 0.70})
    _write_report(service, run_id="current", aggregate_metrics={"hit_rate": 0.95, "mrr": 0.84})
    _write_report(service, run_id="mismatch", dataset_version="v2")

    comparison = service.compare_runs("current", "baseline")

    assert comparison["current_run"]["run_id"] == "current"
    assert comparison["baseline_run"]["run_id"] == "baseline"
    assert comparison["metric_deltas"]["hit_rate"] == pytest.approx(0.15)
    assert comparison["sample_deltas"]["shared_sample_count"] == 1
    assert "quality_gate_delta" in comparison
    assert "config_diff_summary" in comparison

    with pytest.raises(ReportServiceError, match="different baseline scopes"):
        service.compare_runs("current", "mismatch")


def test_report_service_compare_against_baseline_uses_scope_binding(tmp_path: Path) -> None:
    service = ReportService(_build_settings(tmp_path))
    _write_report(service, run_id="baseline", aggregate_metrics={"hit_rate": 0.80, "mrr": 0.70})
    _write_report(service, run_id="current", aggregate_metrics={"hit_rate": 0.95, "mrr": 0.84})
    service.set_baseline("baseline")

    comparison = service.compare_against_baseline("current")

    assert comparison is not None
    assert comparison["baseline_run"]["run_id"] == "baseline"
    assert comparison["metric_deltas"]["mrr"] == pytest.approx(0.14)
    assert build_baseline_scope(collection="docs", dataset_version="v1", backend_set=["custom_metrics"]) == comparison["current_run"]["baseline_scope"]
