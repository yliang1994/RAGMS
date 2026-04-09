from __future__ import annotations

import json

import pytest

from ragms.core.models import (
    EvaluationModelError,
    EvaluationRunSummary,
    QueryResponsePayload,
    build_evaluation_input,
    normalize_evaluation_sample,
)
from ragms.runtime.settings_models import AppSettings, snapshot_runtime_config


def test_normalize_evaluation_sample_applies_defaults_and_answer_requirements() -> None:
    sample = normalize_evaluation_sample(
        {
            "sample_id": "sample-1",
            "query": "what is rag",
            "collection": "docs",
            "evaluation_modes": ["retrieval", "answer"],
            "ground_truth_answer": "RAG combines retrieval and generation.",
            "expected_chunk_ids": ["chunk-1"],
        },
        defaults={
            "dataset_version": "v1",
            "filters": {"doc_type": "pdf"},
            "config_snapshot": {"retrieval": {"strategy": "hybrid"}},
        },
    )

    payload = sample.to_dict()

    assert sample.dataset_version == "v1"
    assert sample.filters == {"doc_type": "pdf"}
    assert payload["expected_chunk_ids"] == ["chunk-1"]
    json.dumps(payload)


def test_build_evaluation_input_attaches_query_response_fields() -> None:
    sample = normalize_evaluation_sample(
        {
            "sample_id": "sample-2",
            "query": "what is rag",
            "collection": "docs",
            "evaluation_modes": ["retrieval"],
        }
    )
    response = QueryResponsePayload(
        query="what is rag",
        answer="RAG combines retrieval and generation [1].",
        citations=[{"index": 1, "chunk_id": "chunk-1"}],
        retrieved_chunks=[{"chunk_id": "chunk-1"}],
        trace_id="trace-query-1",
        config_snapshot={"collection": "docs"},
    )

    hydrated = build_evaluation_input(sample, response, backend_results={"custom_metrics": {"hit_rate": 1.0}})

    assert hydrated.generated_answer == "RAG combines retrieval and generation [1]."
    assert hydrated.trace_id == "trace-query-1"
    assert hydrated.backend_results["custom_metrics"]["hit_rate"] == 1.0


def test_evaluation_run_summary_and_config_snapshot_are_json_safe() -> None:
    settings = AppSettings()
    snapshot = snapshot_runtime_config(settings)
    report = EvaluationRunSummary(
        run_id="run-1",
        trace_id="trace-eval-1",
        collection="default",
        dataset_name="golden",
        dataset_version="v1",
        backend_set=["custom_metrics"],
        config_snapshot=snapshot,
        aggregate_metrics={"hit_rate": 0.95},
        samples=[{"sample_id": "sample-1"}],
    )

    payload = report.to_dict()

    assert payload["config_snapshot"]["retrieval"]["strategy"] == "hybrid"
    json.dumps(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "sample_id": "",
                "query": "what is rag",
                "collection": "docs",
            },
            "sample_id must not be empty",
        ),
        (
            {
                "sample_id": "sample-3",
                "query": "what is rag",
                "collection": "docs",
                "evaluation_modes": ["answer"],
            },
            "answer mode requires ground_truth_answer or ground_truth_citations",
        ),
        (
            {
                "sample_id": "sample-4",
                "query": "what is rag",
                "collection": "docs",
                "evaluation_modes": ["broken"],
            },
            "unsupported evaluation_modes",
        ),
    ],
)
def test_evaluation_models_validate_invalid_inputs(payload: dict[str, object], message: str) -> None:
    with pytest.raises(EvaluationModelError, match=message):
        normalize_evaluation_sample(payload)
