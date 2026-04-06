from __future__ import annotations

import pytest
from mcp import types

from ragms.mcp_server.protocol_handler import (
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_METHOD_NOT_FOUND,
    ProtocolHandler,
)
from ragms.mcp_server.schemas import SchemaValidationError
from ragms.runtime.exceptions import RagMSError


def test_build_success_response_wraps_text_structured_content_and_debug() -> None:
    handler = ProtocolHandler()

    response = handler.build_success_response(
        text="query completed",
        structured_content={"trace_id": "trace-1"},
        debug={"reranker": "disabled"},
    )

    assert response.isError is False
    assert response.content == [types.TextContent(type="text", text="query completed")]
    assert response.structuredContent == {
        "trace_id": "trace-1",
        "debug": {"reranker": "disabled"},
    }


def test_build_success_response_keeps_explicit_content_blocks() -> None:
    handler = ProtocolHandler()
    image = types.ImageContent(type="image", mimeType="image/png", data="YmFzZTY0")

    response = handler.build_success_response(
        text="ignored fallback",
        content=[types.TextContent(type="text", text="query completed"), image],
    )

    assert response.isError is False
    assert response.content[0].text == "query completed"
    assert response.content[1] == image


def test_build_error_response_embeds_protocol_error_payload() -> None:
    handler = ProtocolHandler()

    response = handler.build_error_response(
        code=JSONRPC_INVALID_PARAMS,
        message="Invalid arguments for query_knowledge_hub.query: Field required",
        data={"tool": "query_knowledge_hub"},
        structured_content={"trace_id": "trace-2"},
    )

    assert response.isError is True
    assert response.content == [
        types.TextContent(type="text", text="Invalid arguments for query_knowledge_hub.query: Field required")
    ]
    assert response.structuredContent == {
        "trace_id": "trace-2",
        "error": {
            "code": JSONRPC_INVALID_PARAMS,
            "message": "Invalid arguments for query_knowledge_hub.query: Field required",
            "data": {"tool": "query_knowledge_hub"},
        },
    }


def test_serialize_exception_maps_unknown_tool_to_method_not_found() -> None:
    handler = ProtocolHandler()

    error = handler.serialize_exception(Exception("plain"))
    assert error.code == JSONRPC_INTERNAL_ERROR
    assert error.message == "Internal error"

    with pytest.raises(SchemaValidationError) as exc_info:
        handler.validate_arguments("missing_tool", {})

    unknown_tool = handler.serialize_exception(exc_info.value)

    assert unknown_tool.code == JSONRPC_METHOD_NOT_FOUND


def test_serialize_exception_maps_validation_and_runtime_errors() -> None:
    handler = ProtocolHandler()

    invalid_params = handler.serialize_exception(ValueError("top_k must be positive"))
    assert invalid_params.code == JSONRPC_INVALID_PARAMS
    assert invalid_params.message == "top_k must be positive"

    runtime_error = handler.serialize_exception(RagMSError("vector store unavailable"))
    assert runtime_error.code == JSONRPC_INTERNAL_ERROR
    assert runtime_error.message == "vector store unavailable"
