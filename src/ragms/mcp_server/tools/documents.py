"""`get_document_summary` MCP tool adapter."""

from __future__ import annotations

from typing import Callable

from mcp import types

from ragms.core.management import DataService
from ragms.core.management.data_service import DocumentSummaryNotFoundError
from ragms.mcp_server.protocol_handler import JSONRPC_INVALID_PARAMS, ProtocolHandler
from ragms.runtime.container import ServiceContainer


def serialize_document_summary(summary: dict[str, object]) -> dict[str, object]:
    """Normalize document summary fields for MCP clients."""

    return {
        "document_id": summary.get("document_id"),
        "source_path": summary.get("source_path"),
        "primary_collection": summary.get("primary_collection"),
        "collections": list(summary.get("collections") or []),
        "summary": summary.get("summary"),
        "structure_outline": list(summary.get("structure_outline") or []),
        "key_metadata": dict(summary.get("key_metadata") or {}),
        "ingestion_status": dict(summary.get("ingestion_status") or {}),
        "page_summary": dict(summary.get("page_summary") or {}),
        "image_summary": dict(summary.get("image_summary") or {}),
        "chunk_count": int(summary.get("chunk_count") or 0),
    }


def handle_get_document_summary(
    document_id: str,
    *,
    runtime: ServiceContainer,
    data_service: DataService | None = None,
    protocol_handler: ProtocolHandler | None = None,
) -> types.CallToolResult:
    """Execute the document summary tool and return a stable MCP result."""

    handler = protocol_handler or ProtocolHandler()
    request = handler.validate_arguments(
        "get_document_summary",
        {
            "document_id": document_id,
        },
    )
    service = data_service or DataService(runtime.settings)

    try:
        payload = service.get_document_summary(request.document_id)
    except DocumentSummaryNotFoundError as exc:
        return handler.build_error_response(
            code=JSONRPC_INVALID_PARAMS,
            message=str(exc),
            data={"document_id": request.document_id},
        )
    except Exception as exc:
        error = handler.serialize_exception(exc)
        return handler.build_error_response(
            code=error.code,
            message=error.message,
            data=error.data,
        )

    summary = serialize_document_summary(payload)
    text = summary.get("summary") or f"Document {request.document_id} summary is available."
    return handler.build_success_response(
        text=str(text),
        structured_content=summary,
    )


def bind_documents_tool(runtime: ServiceContainer) -> Callable[..., types.CallToolResult]:
    """Bind the document summary tool to a concrete runtime container."""

    def get_document_summary(document_id: str) -> types.CallToolResult:
        return handle_get_document_summary(
            document_id=document_id,
            runtime=runtime,
        )

    get_document_summary.__name__ = "get_document_summary"
    get_document_summary.__doc__ = "Return document summary and ingestion status."
    return get_document_summary
