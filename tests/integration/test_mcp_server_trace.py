from __future__ import annotations

from pathlib import Path

import anyio

from ragms.core.trace_collector import TraceManager
from ragms.mcp_server.server import create_server
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings
from ragms.storage.traces import TraceRepository


def _runtime(tmp_path: Path) -> ServiceContainer:
    settings = AppSettings.model_validate({"environment": "test"})
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.observability.log_file = tmp_path / "logs" / "traces.jsonl"
    settings.dashboard.traces_file = tmp_path / "logs" / "traces.jsonl"
    return ServiceContainer(settings=settings, services={})


def test_mcp_server_trace_tool_returns_real_trace_detail(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    repository = TraceRepository(runtime.settings.dashboard.traces_file)
    manager = TraceManager()
    trace = manager.start_trace(
        "query",
        trace_id="trace-query-1",
        collection="demo",
        query="what is rag",
    )
    manager.start_stage(trace, "query_processing", input_payload={"query": "what is rag"})
    manager.finish_stage(
        trace,
        "query_processing",
        output_payload={"normalized_query": "what is rag"},
        metadata={"method": "normalize"},
    )
    repository.append(manager.finish_trace(trace, status="succeeded", top_k_results=["chunk-1"]))

    server = create_server(runtime)
    result = anyio.run(server.call_tool, "get_trace_detail", {"trace_id": "trace-query-1"})

    assert result.isError is False
    assert result.structuredContent["trace_id"] == "trace-query-1"
    assert result.structuredContent["trace_type"] == "query"
    assert result.structuredContent["stages"][0]["stage_name"] == "query_processing"


def test_mcp_server_trace_tool_returns_error_for_unknown_trace(tmp_path: Path) -> None:
    server = create_server(_runtime(tmp_path))

    result = anyio.run(server.call_tool, "get_trace_detail", {"trace_id": "missing"})

    assert result.isError is True
    assert result.structuredContent["error"]["code"] == -32602
    assert result.structuredContent["error"]["message"] == "Trace not found: missing"
