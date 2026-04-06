"""MCP tool adapters."""

from .ingest import bind_ingest_tool, handle_ingest_documents, normalize_ingest_request, serialize_ingestion_result
from .query import bind_query_tool, handle_query_knowledge_hub

__all__ = [
    "bind_ingest_tool",
    "bind_query_tool",
    "handle_ingest_documents",
    "handle_query_knowledge_hub",
    "normalize_ingest_request",
    "serialize_ingestion_result",
]
