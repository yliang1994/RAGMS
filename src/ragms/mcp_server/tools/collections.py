"""`list_collections` MCP tool adapter."""

from __future__ import annotations

from typing import Any, Callable

from mcp import types

from ragms.core.management import DataService
from ragms.mcp_server.protocol_handler import ProtocolHandler
from ragms.runtime.container import ServiceContainer


def serialize_collection_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Normalize one collection summary for MCP clients."""

    return {
        "name": str(summary.get("name") or ""),
        "document_count": int(summary.get("document_count") or 0),
        "chunk_count": int(summary.get("chunk_count") or 0),
        "image_count": int(summary.get("image_count") or 0),
        "latest_updated_at": summary.get("latest_updated_at"),
    }


def handle_list_collections(
    filters: dict[str, Any] | None = None,
    page: int | None = None,
    page_size: int | None = None,
    *,
    runtime: ServiceContainer,
    data_service: DataService | None = None,
    protocol_handler: ProtocolHandler | None = None,
) -> types.CallToolResult:
    """Execute the collection listing tool and return a stable MCP result."""

    handler = protocol_handler or ProtocolHandler()
    request = handler.validate_arguments(
        "list_collections",
        {
            "filters": filters,
            "page": page,
            "page_size": page_size,
        },
    )
    service = data_service or DataService(runtime.settings)

    try:
        payload = service.list_collections(
            filters=request.filters,
            page=request.page,
            page_size=request.page_size,
        )
    except Exception as exc:
        error = handler.serialize_exception(exc)
        return handler.build_error_response(
            code=error.code,
            message=error.message,
            data=error.data,
        )

    collections = [serialize_collection_summary(item) for item in payload.get("collections") or []]
    summary = {
        "collection_count": len(collections),
        "filters_applied": dict(request.filters or {}),
    }
    structured_content = {
        "collections": collections,
        "pagination": dict(payload.get("pagination") or {}),
        "summary": summary,
    }
    if request.filters:
        structured_content["filters"] = dict(request.filters)

    text = (
        f"Found {len(collections)} collection(s)."
        if collections
        else "No collections found."
    )
    return handler.build_success_response(
        text=text,
        structured_content=structured_content,
    )


def bind_collections_tool(runtime: ServiceContainer) -> Callable[..., types.CallToolResult]:
    """Bind the collection listing tool to a concrete runtime container."""

    def list_collections(
        filters: dict[str, Any] | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> types.CallToolResult:
        return handle_list_collections(
            filters=filters,
            page=page,
            page_size=page_size,
            runtime=runtime,
        )

    list_collections.__name__ = "list_collections"
    list_collections.__doc__ = "List available collections and summary statistics."
    return list_collections
