"""MCP tool adapters."""

from .collections import bind_collections_tool, handle_list_collections, serialize_collection_summary
from .documents import bind_documents_tool, handle_get_document_summary, serialize_document_summary
from .evaluation import bind_evaluation_tool, handle_evaluate_collection, normalize_evaluation_request, serialize_evaluation_result
from .ingest import bind_ingest_tool, handle_ingest_documents, normalize_ingest_request, serialize_ingestion_result
from .query import bind_query_tool, handle_query_knowledge_hub
from .traces import bind_traces_tool, handle_get_trace_detail, serialize_trace_detail

__all__ = [
    "bind_collections_tool",
    "bind_documents_tool",
    "bind_evaluation_tool",
    "bind_ingest_tool",
    "bind_query_tool",
    "bind_traces_tool",
    "handle_evaluate_collection",
    "handle_get_document_summary",
    "handle_get_trace_detail",
    "handle_list_collections",
    "handle_ingest_documents",
    "handle_query_knowledge_hub",
    "normalize_evaluation_request",
    "normalize_ingest_request",
    "serialize_collection_summary",
    "serialize_document_summary",
    "serialize_evaluation_result",
    "serialize_ingestion_result",
    "serialize_trace_detail",
]
