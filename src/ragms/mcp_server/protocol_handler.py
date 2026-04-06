"""Protocol-layer helpers for MCP tool argument validation and response wrapping."""

from __future__ import annotations

from typing import Any

from mcp import types

from ragms.mcp_server.schemas import (
    BaseToolRequest,
    ProtocolError,
    SchemaValidationError,
    build_text_content,
    validate_tool_arguments,
)
from ragms.runtime.exceptions import RagMSError


JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


class ProtocolHandler:
    """Validate MCP tool calls and normalize result envelopes."""

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> BaseToolRequest:
        """Return the normalized request model for a tool invocation."""

        return validate_tool_arguments(tool_name, arguments)

    def build_success_response(
        self,
        *,
        text: str,
        structured_content: dict[str, Any] | None = None,
        debug: dict[str, Any] | None = None,
        content: list[types.TextContent | types.ImageContent | types.AudioContent | types.ResourceLink | types.EmbeddedResource] | None = None,
    ) -> types.CallToolResult:
        """Build a stable successful MCP tool result envelope."""

        result_content = content or [build_text_content(text)]
        payload = dict(structured_content or {})
        if debug is not None:
            payload["debug"] = debug

        return types.CallToolResult(
            content=result_content,
            structuredContent=payload or None,
            isError=False,
        )

    def build_error_response(
        self,
        *,
        code: int,
        message: str,
        data: dict[str, Any] | None = None,
        structured_content: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        """Build a stable error MCP tool result envelope."""

        error = ProtocolError(code=code, message=message, data=data)
        payload = dict(structured_content or {})
        payload["error"] = error.model_dump(mode="python", exclude_none=True)
        return types.CallToolResult(
            content=[build_text_content(message)],
            structuredContent=payload,
            isError=True,
        )

    def serialize_exception(self, exc: Exception) -> ProtocolError:
        """Map internal exceptions to stable JSON-RPC style error payloads."""

        if isinstance(exc, SchemaValidationError):
            message = str(exc)
            if message.startswith("Unknown tool:"):
                return ProtocolError(code=JSONRPC_METHOD_NOT_FOUND, message=message)
            return ProtocolError(code=JSONRPC_INVALID_PARAMS, message=message)
        if isinstance(exc, (TypeError, ValueError)):
            return ProtocolError(code=JSONRPC_INVALID_PARAMS, message=str(exc) or "Invalid parameters")
        if isinstance(exc, RagMSError):
            return ProtocolError(code=JSONRPC_INTERNAL_ERROR, message=str(exc) or "Internal error")
        return ProtocolError(code=JSONRPC_INTERNAL_ERROR, message="Internal error")
