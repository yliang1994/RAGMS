"""Protocol-layer helpers for MCP tool argument validation."""

from __future__ import annotations

from typing import Any

from ragms.mcp_server.schemas import BaseToolRequest, validate_tool_arguments


class ProtocolHandler:
    """Validate incoming MCP tool arguments before business handlers execute."""

    def validate_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> BaseToolRequest:
        """Return the normalized request model for a tool invocation."""

        return validate_tool_arguments(tool_name, arguments)
