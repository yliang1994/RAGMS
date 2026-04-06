"""MCP tool adapters."""

from .collections import bind_collections_tool, handle_list_collections, serialize_collection_summary
from .ingest import bind_ingest_tool, handle_ingest_documents, normalize_ingest_request, serialize_ingestion_result
from .query import bind_query_tool, handle_query_knowledge_hub

__all__ = [
    "bind_collections_tool",
    "bind_ingest_tool",
    "bind_query_tool",
    "handle_list_collections",
    "handle_ingest_documents",
    "handle_query_knowledge_hub",
    "normalize_ingest_request",
    "serialize_collection_summary",
    "serialize_ingestion_result",
]
