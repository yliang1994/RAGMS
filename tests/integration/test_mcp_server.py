from __future__ import annotations

import anyio
import pytest

from ragms.mcp_server.server import create_server, handle_initialize
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


def test_mcp_server_initialize_and_list_tools_cover_core_contract() -> None:
    server = create_server(_runtime())
    init_options = handle_initialize(server)
    tools = anyio.run(server.list_tools)

    assert init_options.server_name == "ragms"
    assert init_options.capabilities.tools is not None
    assert [tool.name for tool in tools] == [
        "query_knowledge_hub",
        "list_collections",
        "get_document_summary",
        "ingest_documents",
        "get_trace_detail",
        "evaluate_collection",
    ]


def test_mcp_server_invalid_tool_arguments_do_not_break_followup_calls() -> None:
    server = create_server(_runtime())

    with pytest.raises(Exception, match="Invalid arguments for query_knowledge_hub.top_k"):
        anyio.run(server.call_tool, "query_knowledge_hub", {"query": "what is rag", "top_k": 0})

    result = anyio.run(server.call_tool, "list_collections", {})

    assert result.isError is False
    assert result.structuredContent["collections"]
    assert result.structuredContent["collections"][0]["name"] == "default"
