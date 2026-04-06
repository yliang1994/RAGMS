"""Central MCP tool registry definitions and server registration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool

from ragms.mcp_server.schemas import get_input_schema
from ragms.runtime.exceptions import RagMSError


ToolHandler = Callable[..., Any]


class ToolRegistryError(RagMSError):
    """Raised when MCP tool declarations are invalid or inconsistent."""


@dataclass(frozen=True)
class ToolDefinition:
    """Single-source declaration for an externally visible MCP tool."""

    name: str
    description: str
    handler: ToolHandler
    title: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ToolRegistryError("Tool name must not be empty")
        if not self.description.strip():
            raise ToolRegistryError(f"Tool description must not be empty: {self.name}")
        if not callable(self.handler):
            raise ToolRegistryError(f"Tool handler must be callable: {self.name}")

        schema = self.input_schema or self._derive_input_schema()
        if schema.get("type") != "object":
            raise ToolRegistryError(f"Tool schema must be an object schema: {self.name}")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "input_schema", schema)

    def _derive_input_schema(self) -> dict[str, Any]:
        derived = Tool.from_function(
            self.handler,
            name=self.name,
            title=self.title,
            description=self.description,
        ).parameters
        if not derived:
            raise ToolRegistryError(f"Tool schema derivation failed: {self.name}")
        return derived


def _not_implemented_message(tool_name: str) -> str:
    return f"{tool_name} is not implemented yet"


def _query_knowledge_hub(
    query: str,
    collection: str | None = None,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    return_debug: bool = False,
) -> str:
    """Execute a knowledge-hub query against the retrieval pipeline."""

    return _not_implemented_message("query_knowledge_hub")


def _ingest_documents(
    paths: list[str],
    collection: str | None = None,
    force_rebuild: bool = False,
    options: dict[str, Any] | None = None,
) -> str:
    """Ingest documents into a target collection."""

    return _not_implemented_message("ingest_documents")


def _list_collections(
    filters: dict[str, Any] | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> str:
    """List available document collections and summary statistics."""

    return _not_implemented_message("list_collections")


def _get_document_summary(document_id: str) -> str:
    """Return the summary and status for a single ingested document."""

    return _not_implemented_message("get_document_summary")


def _get_trace_detail(trace_id: str) -> str:
    """Return detailed observability data for a recorded trace."""

    return _not_implemented_message("get_trace_detail")


def _evaluate_collection(
    collection: str,
    dataset: str | None = None,
    metrics: list[str] | None = None,
    eval_options: dict[str, Any] | None = None,
) -> str:
    """Run evaluation against a collection and return structured metrics."""

    return _not_implemented_message("evaluate_collection")


def build_tool_registry(
    definitions: list[ToolDefinition] | None = None,
) -> dict[str, ToolDefinition]:
    """Build the MCP tool registry as the single source of truth for tool metadata."""

    tool_definitions = definitions or [
        ToolDefinition(
            name="query_knowledge_hub",
            description="执行知识库检索、融合、重排与回答生成，是面向 Agent 的核心查询入口。",
            handler=_query_knowledge_hub,
            input_schema=get_input_schema("query_knowledge_hub"),
        ),
        ToolDefinition(
            name="list_collections",
            description="列出当前可用知识库集合及其基础统计信息，供 Agent 做集合发现与选择。",
            handler=_list_collections,
            input_schema=get_input_schema("list_collections"),
        ),
        ToolDefinition(
            name="get_document_summary",
            description="查看指定文档的摘要、结构概览与最新摄取状态。",
            handler=_get_document_summary,
            input_schema=get_input_schema("get_document_summary"),
        ),
        ToolDefinition(
            name="ingest_documents",
            description="触发文档摄取任务，支持新文档入库、增量跳过或强制重建。",
            handler=_ingest_documents,
            input_schema=get_input_schema("ingest_documents"),
        ),
        ToolDefinition(
            name="get_trace_detail",
            description="查询指定 trace 的详细执行过程，用于排障和性能分析。",
            handler=_get_trace_detail,
            input_schema=get_input_schema("get_trace_detail"),
        ),
        ToolDefinition(
            name="evaluate_collection",
            description="对指定集合执行检索或问答评估，产出结构化评估结果。",
            handler=_evaluate_collection,
            input_schema=get_input_schema("evaluate_collection"),
        ),
    ]

    registry: dict[str, ToolDefinition] = {}
    for definition in tool_definitions:
        if definition.name in registry:
            raise ToolRegistryError(f"Duplicate tool definition: {definition.name}")
        registry[definition.name] = definition
    return registry


def list_tool_definitions(registry: dict[str, ToolDefinition]) -> list[ToolDefinition]:
    """Return registry declarations in insertion order for deterministic registration."""

    return list(registry.values())


def register_tools(server: FastMCP, registry: dict[str, ToolDefinition]) -> None:
    """Register every declared tool from the registry onto the MCP server."""

    for definition in list_tool_definitions(registry):
        registered_tool = server._tool_manager.add_tool(
            definition.handler,
            name=definition.name,
            title=definition.title,
            description=definition.description,
        )
        registered_tool.parameters = definition.input_schema
