from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragms.core.evaluation import EvalRunner
from ragms.core.management.trace_service import TraceService
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
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.integration
def test_evaluation_trace_is_logged_with_stage_order_and_readable_by_trace_service(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    _write_dataset(settings)
    runner = EvalRunner(
        settings=settings,
        sample_executor=lambda sample: {
            "query": sample.query,
            "answer": "RAG combines retrieval and generation [1].",
            "citations": [{"citation_id": "c1", "chunk_id": "chunk-1"}],
            "retrieved_chunks": [{"chunk_id": "chunk-1", "content": "RAG combines retrieval and generation."}],
            "trace_id": "query-trace-1",
            "config_snapshot": {"strategy": "hybrid"},
        },
    )

    result = runner.run(dataset_name="golden", dataset_version="v1")
    service = TraceService(settings)
    detail = service.get_trace_detail(result["trace_id"])

    assert detail["trace_type"] == "evaluation"
    assert detail["run_id"] == result["run_id"]
    assert detail["dataset_version"] == "v1"
    assert detail["summary"]["target_page"] == "evaluation_panel"
    assert [stage["stage_name"] for stage in detail["stages"]] == [
        "dataset_load",
        "sample_build",
        "evaluator_execute",
        "metrics_aggregate",
        "report_persist",
    ]
    assert detail["metrics_summary"]["sample_count"] == 1
