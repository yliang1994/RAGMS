from __future__ import annotations

from pathlib import Path

import pytest

from ragms.core.query_engine import build_query_engine
from ragms.storage.traces import TraceRepository
from tests.fakes import FakeLLM
from tests.integration.test_query_engine import _build_runtime


@pytest.mark.integration
def test_query_engine_writes_query_trace_and_returns_trace_id(tmp_path: Path) -> None:
    settings, container = _build_runtime(tmp_path, reranker_provider=None, llm=FakeLLM(["RAG [1]."]))
    settings.observability.log_file = tmp_path / "logs" / "traces.jsonl"
    engine = build_query_engine(container, settings=settings)

    response = engine.run(query="what is rag", top_k=1)

    repository = TraceRepository(settings.observability.log_file)
    trace = repository.get_by_trace_id(response["trace_id"])

    assert response["trace_id"]
    assert trace is not None
    assert trace["trace_type"] == "query"
    assert trace["status"] == "succeeded"
    assert trace["query"] == "what is rag"
    assert trace["top_k_results"] == ["chunk-1"]
    assert [stage["stage_name"] for stage in trace["stages"]] == [
        "query_processing",
        "dense_retrieval",
        "sparse_retrieval",
        "fusion",
        "rerank",
        "response_build",
        "answer_generation",
    ]
