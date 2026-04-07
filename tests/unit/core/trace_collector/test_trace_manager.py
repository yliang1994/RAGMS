from __future__ import annotations

import json

import pytest

from ragms.core.trace_collector import TraceLifecycleError, TraceManager


def test_trace_manager_records_query_trace_with_json_safe_stage_payloads() -> None:
    manager = TraceManager()
    trace = manager.start_trace(
        "query",
        collection="knowledge-hub",
        metadata={"provider": "openai", "api_key": "sk-secret"},
        query="what is rag?",
    )

    manager.start_stage(
        trace,
        "query_processing",
        input_payload={
            "query": "what is rag?",
            "headers": {"Authorization": "Bearer super-secret"},
        },
        metadata={"method": "keyword_extract"},
    )
    manager.finish_stage(
        trace,
        "query_processing",
        output_payload={
            "normalized_query": "what is rag?",
            "keywords": ["rag", "retrieval", "generation"],
        },
        metadata={"keyword_count": 3},
    )
    manager.finish_trace(
        trace,
        status="succeeded",
        top_k_results=["chunk-1", "chunk-2"],
        evaluation_metrics={"context_relevance": 0.98},
    )

    payload = trace.to_dict()

    json.dumps(payload)
    assert payload["trace_type"] == "query"
    assert payload["status"] == "succeeded"
    assert payload["query"] == "what is rag?"
    assert payload["top_k_results"] == ["chunk-1", "chunk-2"]
    assert payload["metadata"]["api_key"] == "[REDACTED]"
    assert payload["finished_at"] is not None
    assert payload["duration_ms"] is not None
    assert len(payload["stages"]) == 1
    stage = payload["stages"][0]
    assert stage["stage_name"] == "query_processing"
    assert stage["metadata"]["method"] == "keyword_extract"
    assert stage["metadata"]["keyword_count"] == 3
    assert stage["input_summary"]["kind"] == "mapping"
    assert stage["input_summary"]["preview"]["headers"]["kind"] == "mapping"
    assert stage["input_summary"]["preview"]["headers"]["preview"]["Authorization"] == "[REDACTED]"
    assert stage["output_summary"]["preview"]["keywords"]["count"] == 3
    assert stage["status"] == "succeeded"


def test_trace_manager_rejects_duplicate_stage_names() -> None:
    manager = TraceManager()
    trace = manager.start_trace("ingestion", source_path="docs/sample.pdf")

    manager.start_stage(trace, "load")

    with pytest.raises(TraceLifecycleError, match="duplicate stage name: load"):
        manager.start_stage(trace, "load")

    manager.finish_stage(trace, "load")

    with pytest.raises(TraceLifecycleError, match="duplicate stage name: load"):
        manager.start_stage(trace, "load")


def test_finish_trace_auto_converges_open_stages_and_normalizes_failure() -> None:
    manager = TraceManager()
    trace = manager.start_trace(
        "ingestion",
        collection="docs",
        source_path="docs/report.pdf",
        document_id=None,
    )

    manager.start_stage(
        trace,
        "file_integrity",
        input_payload={"source_path": "docs/report.pdf", "token": "abc123"},
    )
    manager.start_stage(trace, "load", input_payload={"prompt": "Summarize this document."})
    manager.finish_trace(
        trace,
        error=ValueError("Authorization: Bearer top-secret"),
        total_chunks=0,
        total_images=0,
    )

    payload = trace.to_dict()

    json.dumps(payload)
    assert payload["trace_type"] == "ingestion"
    assert payload["status"] == "failed"
    assert payload["error"]["message"] == "Authorization: Bearer [REDACTED]"
    assert len(payload["stages"]) == 2
    assert {stage["stage_name"] for stage in payload["stages"]} == {"file_integrity", "load"}
    assert all(stage["status"] == "failed" for stage in payload["stages"])
    assert all(stage["finished_at"] is not None for stage in payload["stages"])
    assert payload["document_id"] is None
    assert payload["total_chunks"] == 0
    assert payload["total_images"] == 0


def test_trace_manager_supports_evaluation_trace_contract() -> None:
    manager = TraceManager()
    trace = manager.start_trace(
        "evaluation",
        collection="docs",
        run_id="run-1",
        dataset_version="v1",
        backends=["ragas", "deepeval"],
    )

    manager.finish_trace(
        trace,
        status="skipped",
        metrics_summary={"hit_rate": 0.9},
        quality_gate_status="not_run",
        baseline_delta={"hit_rate": -0.02},
    )

    payload = trace.to_dict()

    assert payload["trace_type"] == "evaluation"
    assert payload["status"] == "skipped"
    assert payload["run_id"] == "run-1"
    assert payload["dataset_version"] == "v1"
    assert payload["backends"] == ["ragas", "deepeval"]
    assert payload["metrics_summary"] == {"hit_rate": 0.9}
    assert payload["baseline_delta"] == {"hit_rate": -0.02}
