from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ragms.core.trace_collector import TraceManager
from ragms.observability.dashboard import (
    build_dashboard_context,
    render_app_shell,
    render_query_trace,
)
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


def _append_query_variants(settings_path: Path) -> tuple[object, object, object]:
    data_service, trace_service, report_service = _seed_metadata(settings_path)
    manager = TraceManager()

    full = manager.start_trace(
        "query",
        trace_id="trace-query-full",
        collection="dashboard-demo",
        query="what is rag advanced",
    )
    manager.start_stage(full, "query_processing", input_payload={"query": "what is rag advanced"})
    manager.finish_stage(
        full,
        "query_processing",
        output_payload={"normalized_query": "what is rag advanced", "keywords": ["rag", "advanced"]},
        metadata={"method": "normalize", "keywords": ["rag", "advanced"]},
    )
    manager.start_stage(full, "dense_retrieval", input_payload={"query": "what is rag advanced"})
    manager.finish_stage(full, "dense_retrieval", output_payload={"retrieved_count": 3}, metadata={"provider": "chroma", "retrieved_count": 3})
    manager.start_stage(full, "sparse_retrieval", input_payload={"query": "what is rag advanced"})
    manager.finish_stage(full, "sparse_retrieval", output_payload={"retrieved_count": 2}, metadata={"method": "bm25", "retrieved_count": 2})
    manager.start_stage(full, "fusion", input_payload={"dense": 3, "sparse": 2})
    manager.finish_stage(full, "fusion", output_payload={"fused_count": 3}, metadata={"method": "rrf"})
    manager.start_stage(full, "rerank", input_payload={"candidate_count": 3})
    manager.finish_stage(full, "rerank", output_payload={"top_chunk_ids": ["chunk-1", "chunk-2"]}, metadata={"backend": "disabled"})
    manager.start_stage(full, "response_build", input_payload={"candidate_count": 2})
    manager.finish_stage(full, "response_build", output_payload={"citation_count": 2}, metadata={"citation_count": 2, "image_count": 1})
    manager.start_stage(full, "answer_generation", input_payload={"query": "what is rag advanced"})
    manager.finish_stage(full, "answer_generation", output_payload={"answer_preview": "RAG is ..."}, metadata={"provider": "openai", "model": "gpt-4.1-mini"})
    trace_service.repository.append(manager.finish_trace(full, status="succeeded", top_k_results=["chunk-1", "chunk-2"]))

    retrieval_only = manager.start_trace(
        "query",
        trace_id="trace-query-retrieval-only",
        collection="dashboard-demo",
        query="what is rag retrieval",
    )
    manager.start_stage(retrieval_only, "query_processing", input_payload={"query": "what is rag retrieval"})
    manager.finish_stage(
        retrieval_only,
        "query_processing",
        output_payload={"normalized_query": "what is rag retrieval"},
        metadata={"method": "normalize", "keywords": ["rag", "retrieval"]},
    )
    manager.start_stage(retrieval_only, "dense_retrieval", input_payload={"query": "what is rag retrieval"})
    manager.finish_stage(retrieval_only, "dense_retrieval", output_payload={"retrieved_count": 2}, metadata={"provider": "chroma", "retrieved_count": 2})
    manager.start_stage(retrieval_only, "sparse_retrieval", input_payload={"query": "what is rag retrieval"})
    manager.finish_stage(retrieval_only, "sparse_retrieval", output_payload={"retrieved_count": 1}, metadata={"method": "bm25", "retrieved_count": 1})
    manager.start_stage(retrieval_only, "fusion", input_payload={"dense": 2, "sparse": 1})
    manager.finish_stage(retrieval_only, "fusion", output_payload={"fused_count": 2}, metadata={"method": "rrf"})
    manager.start_stage(retrieval_only, "response_build", input_payload={"candidate_count": 2})
    manager.finish_stage(retrieval_only, "response_build", output_payload={"citation_count": 1}, metadata={"citation_count": 1, "image_count": 0})
    trace_service.repository.append(manager.finish_trace(retrieval_only, status="succeeded", top_k_results=["chunk-1"]))

    return data_service, trace_service, report_service


@pytest.mark.integration
def test_dashboard_query_trace_page_renders_timeline_and_comparison(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _append_query_variants(settings_path)
    settings = load_settings(settings_path)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
    )

    shell = render_app_shell(context, selected_page="query_trace")
    page = render_query_trace(
        context,
        collection="dashboard-demo",
        left_trace_id="trace-query-full",
        right_trace_id="trace-query-retrieval-only",
    )

    assert shell["page"]["kind"] == "query_trace"
    assert "dashboard-demo" in page["filter_model"]["collections"]
    assert page["trace_list"]["row_count"] >= 2
    assert page["selected_trace"]["summary"]["trace_id"] == "trace-query-full"
    assert page["timeline"]["timeline"][0]["stage_name"] == "query_processing"
    assert page["timeline"]["timeline"][-1]["stage_name"] == "answer_generation"
    assert page["timeline"]["timeline"][1]["progress"]["retrieved_count"] == 3
    assert page["comparison"]["metric_deltas"]["citation_count"] == 1
    assert page["comparison"]["query_differences"]["response_build"]["left"] == 2
    assert page["comparison"]["query_differences"]["answer_generation"]["right"] is None
    assert page["comparison"]["stage_chart"]["point_count"] >= 4


@pytest.mark.integration
def test_dashboard_query_trace_page_handles_traces_without_generation_stage(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _append_query_variants(settings_path)
    settings = load_settings(settings_path)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
    )

    page = render_query_trace(
        context,
        trace_id="trace-query-retrieval-only",
        left_trace_id="trace-query-retrieval-only",
        right_trace_id="trace-left",
    )

    stage_names = [row["stage_name"] for row in page["timeline"]["timeline"]]
    assert page["selected_trace"]["summary"]["trace_id"] == "trace-query-retrieval-only"
    assert "answer_generation" not in stage_names
    assert page["timeline_table"]["row_count"] == len(stage_names)
    assert page["comparison"]["query_differences"]["retrieval"]["left"] == 2
