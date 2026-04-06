"""`query_knowledge_hub` MCP tool adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mcp import types

from ragms.core.query_engine import ResponseBuilder, build_query_engine
from ragms.mcp_server.protocol_handler import ProtocolHandler
from ragms.runtime.container import ServiceContainer
from ragms.storage.images.image_storage import ImageStorage
from ragms.storage.sqlite.connection import create_sqlite_connection
from ragms.storage.sqlite.repositories.images import ImagesRepository


def _build_response_builder(runtime: ServiceContainer) -> ResponseBuilder:
    connection = create_sqlite_connection(runtime.settings.storage.sqlite.path)
    images_repository = ImagesRepository(connection)
    image_storage = ImageStorage(root_dir=runtime.settings.paths.data_dir / "images")
    return ResponseBuilder(
        images_repository=images_repository,
        image_storage=image_storage,
    )


def handle_query_knowledge_hub(
    query: str,
    collection: str | None = None,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    return_debug: bool = False,
    *,
    runtime: ServiceContainer,
    query_engine: Any | None = None,
    response_builder: ResponseBuilder | None = None,
    protocol_handler: ProtocolHandler | None = None,
) -> types.CallToolResult:
    """Execute the query tool and return a fully wrapped MCP result."""

    handler = protocol_handler or ProtocolHandler()
    request = handler.validate_arguments(
        "query_knowledge_hub",
        {
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "filters": filters,
            "return_debug": return_debug,
        },
    )

    engine = query_engine or build_query_engine(runtime, settings=runtime.settings)
    builder = response_builder or _build_response_builder(runtime)

    try:
        payload = engine.run(
            query=request.query,
            collection=request.collection,
            top_k=request.top_k,
            filters=request.filters,
            return_debug=request.return_debug,
            trace_context={"trace_id": None},
        )
    except Exception as exc:
        error = handler.serialize_exception(exc)
        return handler.build_error_response(
            code=error.code,
            message=error.message,
            data=error.data,
        )

    structured_content = dict(payload.get("structured_content") or {})
    debug_info = payload.get("debug_info") if request.return_debug else None
    return handler.build_success_response(
        text=payload.get("markdown") or payload.get("answer") or "",
        structured_content=structured_content,
        debug=debug_info,
        content=payload.get("content"),
    )


def bind_query_tool(runtime: ServiceContainer) -> Callable[..., types.CallToolResult]:
    """Bind the query tool to a concrete runtime container for MCP registration."""

    def query_knowledge_hub(
        query: str,
        collection: str | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        return_debug: bool = False,
    ) -> types.CallToolResult:
        return handle_query_knowledge_hub(
            query=query,
            collection=collection,
            top_k=top_k,
            filters=filters,
            return_debug=return_debug,
            runtime=runtime,
        )

    query_knowledge_hub.__name__ = "query_knowledge_hub"
    query_knowledge_hub.__doc__ = "Execute the knowledge-hub query tool."
    return query_knowledge_hub
