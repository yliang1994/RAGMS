from __future__ import annotations

import anyio
import pytest

from ragms.mcp_server.server import create_server
from ragms.mcp_server.tool_registry import (
    ToolDefinition,
    ToolRegistryError,
    build_tool_registry,
    list_tool_definitions,
    register_tools,
)
from ragms.runtime.container import build_container
from ragms.runtime.settings_models import AppSettings


def test_build_tool_registry_declares_core_and_future_tools() -> None:
    registry = build_tool_registry()

    assert list(registry) == [
        "query_knowledge_hub",
        "list_collections",
        "get_document_summary",
        "ingest_documents",
        "get_trace_detail",
        "evaluate_collection",
    ]
    assert registry["query_knowledge_hub"].input_schema["type"] == "object"
    assert registry["ingest_documents"].input_schema["required"] == ["paths"]


def test_build_tool_registry_rejects_duplicate_names() -> None:
    duplicate_definitions = [
        ToolDefinition(name="dup_tool", description="first", handler=lambda: "ok"),
        ToolDefinition(name="dup_tool", description="second", handler=lambda: "ok"),
    ]

    with pytest.raises(ToolRegistryError, match="Duplicate tool definition: dup_tool"):
        build_tool_registry(duplicate_definitions)


def test_tool_definition_rejects_invalid_handler_or_schema() -> None:
    with pytest.raises(ToolRegistryError, match="Tool handler must be callable: broken_tool"):
        ToolDefinition(name="broken_tool", description="desc", handler=None)  # type: ignore[arg-type]

    with pytest.raises(ToolRegistryError, match="Tool schema must be an object schema: schema_tool"):
        ToolDefinition(
            name="schema_tool",
            description="desc",
            handler=lambda query: query,
            input_schema={"type": "string"},
        )


def test_register_tools_uses_registry_as_single_source_of_truth() -> None:
    registry = build_tool_registry()
    runtime = build_container(
        AppSettings.model_validate(
            {
                "environment": "test",
                "retrieval": {
                    "strategy": "hybrid",
                    "fusion_algorithm": "rrf",
                    "rerank_backend": "disabled",
                },
            }
        )
    )
    server = create_server(runtime)

    assert getattr(server, "tool_registry") == registry
    mcp_tools = anyio.run(server.list_tools)
    listed_names = [tool.name for tool in mcp_tools]

    assert listed_names == [definition.name for definition in list_tool_definitions(registry)]
    query_tool = next(tool for tool in mcp_tools if tool.name == "query_knowledge_hub")
    assert query_tool.description == registry["query_knowledge_hub"].description
    assert query_tool.inputSchema == registry["query_knowledge_hub"].input_schema


def test_register_tools_keeps_existing_duplicate_registration_stable() -> None:
    registry = build_tool_registry()
    runtime = build_container(
        AppSettings.model_validate(
            {
                "environment": "test",
                "retrieval": {
                    "strategy": "hybrid",
                    "fusion_algorithm": "rrf",
                    "rerank_backend": "disabled",
                },
            }
        )
    )
    server = create_server(runtime)

    register_tools(server, registry)

    listed_names = [tool.name for tool in anyio.run(server.list_tools)]
    assert listed_names.count("query_knowledge_hub") == 1
