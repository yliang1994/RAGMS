from __future__ import annotations

import anyio
from mcp import types

from ragms.mcp_server.server import create_server
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings


def _runtime() -> ServiceContainer:
    settings = AppSettings.model_validate(
        {
            "environment": "test",
            "retrieval": {
                "strategy": "hybrid",
                "fusion_algorithm": "rrf",
                "rerank_backend": "disabled",
            },
        }
    )
    return ServiceContainer(settings=settings, services={})


def test_mcp_server_query_tool_returns_contract_with_citations(monkeypatch) -> None:
    server = create_server(_runtime())

    async def fake_call_tool(name: str, arguments: dict[str, object], context=None, convert_result=False):
        if name != "query_knowledge_hub":
            raise AssertionError(name)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="RAG combines retrieval and generation [1].")],
            structuredContent={
                "citations": [{"index": 1, "chunk_id": "chunk-1", "source_path": "docs/rag.pdf"}],
                "retrieved_chunks": [{"chunk_id": "chunk-1"}],
                "trace_id": "trace-query-1",
            },
            isError=False,
        )

    monkeypatch.setattr(server._tool_manager, "call_tool", fake_call_tool)
    result = anyio.run(server.call_tool, "query_knowledge_hub", {"query": "what is rag"})

    assert result.isError is False
    assert result.structuredContent["trace_id"] == "trace-query-1"
    assert result.structuredContent["citations"][0]["chunk_id"] == "chunk-1"


def test_mcp_server_query_tool_returns_error_contract(monkeypatch) -> None:
    server = create_server(_runtime())

    async def fake_call_tool(name: str, arguments: dict[str, object], context=None, convert_result=False):
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="Internal error")],
            structuredContent={"error": {"code": -32603, "message": "Internal error"}},
            isError=True,
        )

    monkeypatch.setattr(server._tool_manager, "call_tool", fake_call_tool)
    result = anyio.run(server.call_tool, "query_knowledge_hub", {"query": "what is rag"})

    assert result.isError is True
    assert result.structuredContent["error"]["code"] == -32603
