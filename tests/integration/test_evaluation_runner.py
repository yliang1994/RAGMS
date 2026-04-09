from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragms.core.evaluation import EvalRunner, ReportService
from ragms.runtime.settings_models import AppSettings


def _build_settings(tmp_path: Path) -> AppSettings:
    settings = AppSettings()
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.storage.sqlite.path = tmp_path / "data" / "metadata" / "ragms.db"
    settings.vector_store.collection = "docs"
    settings.evaluation.backends = ["custom_metrics"]
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.observability.log_file = settings.paths.logs_dir / "traces.jsonl"
    settings.dashboard.traces_file = settings.paths.logs_dir / "traces.jsonl"
    return settings


def _write_dataset(settings: AppSettings) -> None:
    dataset_dir = settings.paths.data_dir / "evaluation" / "datasets" / "golden"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "v1.json").write_text(
        json.dumps(
            {
                "dataset_name": "golden",
                "dataset_version": "v1",
                "collection": "docs",
                "samples": [
                    {
                        "sample_id": "sample-1",
                        "query": "what is rag",
                        "evaluation_modes": ["retrieval", "answer"],
                        "ground_truth_answer": "RAG combines retrieval and generation [1].",
                        "expected_chunk_ids": ["chunk-1"],
                    },
                    {
                        "sample_id": "sample-2",
                        "query": "broken sample",
                        "evaluation_modes": ["retrieval"],
                        "expected_chunk_ids": ["chunk-x"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _sample_executor(sample) -> dict[str, object]:
    if sample.sample_id == "sample-2":
        raise RuntimeError("mock query failure")
    return {
        "query": sample.query,
        "answer": "RAG combines retrieval and generation [1].",
        "citations": [{"citation_id": "c1", "chunk_id": "chunk-1"}],
        "retrieved_chunks": [{"chunk_id": "chunk-1", "content": "RAG combines retrieval and generation."}],
        "trace_id": f"query-{sample.sample_id}",
        "config_snapshot": {"strategy": "hybrid"},
    }


@pytest.mark.integration
def test_eval_runner_persists_report_artifact_and_sqlite_summary(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    _write_dataset(settings)
    report_service = ReportService(settings)
    runner = EvalRunner(
        settings=settings,
        report_service=report_service,
        sample_executor=_sample_executor,
    )

    result = runner.run(dataset_name="golden", dataset_version="v1")

    assert result["status"] == "partial_success"
    assert result["run_id"]
    assert result["trace_id"]
    assert result["aggregate_metrics"]["sample_count"] == 2
    assert result["aggregate_metrics"]["failed_sample_count"] == 1
    assert Path(result["path"]).is_file()

    listing = report_service.list_runs()
    assert listing[0]["run_id"] == result["run_id"]
    detail = report_service.load_report_detail(result["run_id"])
    assert detail is not None
    assert detail["report"]["samples"][0]["sample_id"] == "sample-1"
    assert detail["report"]["failed_samples"][0]["sample_id"] == "sample-2"
    assert detail["metrics_summary"]["failed_sample_count"] == 1
