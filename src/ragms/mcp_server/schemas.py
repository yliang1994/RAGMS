"""Pydantic request schemas and protocol payload helpers for MCP tools."""

from __future__ import annotations

from typing import Any, TypeAlias

from mcp import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragms.runtime.exceptions import RagMSError


NonEmptyString: TypeAlias = str
JSONMapping: TypeAlias = dict[str, Any]


class SchemaValidationError(RagMSError):
    """Raised when MCP tool arguments fail schema validation."""


class ProtocolError(BaseModel):
    """Structured JSON-RPC style error payload for tool responses."""

    code: int
    message: str
    data: JSONMapping | None = None


class BaseToolRequest(BaseModel):
    """Base request schema that forbids undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class QueryToolRequest(BaseToolRequest):
    """Validated request payload for `query_knowledge_hub`."""

    query: NonEmptyString = Field(min_length=1)
    collection: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    filters: JSONMapping | None = None
    return_debug: bool = False


class IngestToolRequest(BaseToolRequest):
    """Validated request payload for `ingest_documents`."""

    paths: list[NonEmptyString] = Field(min_length=1)
    collection: str | None = None
    force_rebuild: bool = False
    options: JSONMapping | None = None


class CollectionToolRequest(BaseToolRequest):
    """Validated request payload for `list_collections`."""

    filters: JSONMapping | None = None
    page: int | None = Field(default=None, ge=1)
    page_size: int | None = Field(default=None, ge=1, le=100)


class DocumentSummaryRequest(BaseToolRequest):
    """Validated request payload for `get_document_summary`."""

    document_id: NonEmptyString = Field(min_length=1)


class TraceDetailRequest(BaseToolRequest):
    """Validated request payload for a future `get_trace_detail` tool."""

    trace_id: NonEmptyString = Field(min_length=1)


class EvaluationToolRequest(BaseToolRequest):
    """Validated request payload for a future `evaluate_collection` tool."""

    collection: NonEmptyString = Field(min_length=1)
    dataset: str | None = None
    metrics: list[NonEmptyString] | None = None
    eval_options: JSONMapping | None = None
    baseline_mode: str = "compare"


TOOL_REQUEST_MODELS: dict[str, type[BaseToolRequest]] = {
    "query_knowledge_hub": QueryToolRequest,
    "ingest_documents": IngestToolRequest,
    "list_collections": CollectionToolRequest,
    "get_document_summary": DocumentSummaryRequest,
    "get_trace_detail": TraceDetailRequest,
    "evaluate_collection": EvaluationToolRequest,
}


def get_request_model(tool_name: str) -> type[BaseToolRequest]:
    """Return the request schema class for a tool name."""

    try:
        return TOOL_REQUEST_MODELS[tool_name]
    except KeyError as exc:
        raise SchemaValidationError(f"Unknown tool: {tool_name}") from exc


def get_input_schema(tool_name: str) -> JSONMapping:
    """Return JSON schema for a tool request model."""

    return get_request_model(tool_name).model_json_schema()


def validate_tool_arguments(tool_name: str, arguments: JSONMapping | None) -> BaseToolRequest:
    """Validate and normalize arguments for a given tool."""

    model = get_request_model(tool_name)
    try:
        return model.model_validate(arguments or {})
    except ValidationError as exc:
        error = exc.errors()[0]
        location = ".".join(str(item) for item in error.get("loc", ())) or tool_name
        message = error.get("msg", "Invalid parameters")
        raise SchemaValidationError(f"Invalid arguments for {tool_name}.{location}: {message}") from exc


def build_text_content(text: str) -> types.TextContent:
    """Create a plain-text MCP content block."""

    return types.TextContent(type="text", text=text)
