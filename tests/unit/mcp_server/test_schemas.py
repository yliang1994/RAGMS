from __future__ import annotations

import pytest

from ragms.mcp_server.protocol_handler import ProtocolHandler
from ragms.mcp_server.schemas import (
    CollectionToolRequest,
    DocumentSummaryRequest,
    IngestToolRequest,
    QueryToolRequest,
    SchemaValidationError,
    get_input_schema,
)


def test_query_tool_request_applies_defaults_and_types() -> None:
    request = QueryToolRequest.model_validate({"query": "what is ragms"})

    assert request.query == "what is ragms"
    assert request.collection is None
    assert request.top_k == 5
    assert request.filters is None
    assert request.return_debug is False


def test_ingest_tool_request_requires_non_empty_paths() -> None:
    request = IngestToolRequest.model_validate({"paths": ["docs/a.pdf"]})

    assert request.paths == ["docs/a.pdf"]
    assert request.force_rebuild is False
    assert request.options is None

    with pytest.raises(Exception):
        IngestToolRequest.model_validate({"paths": []})


def test_collection_tool_request_supports_optional_paging_filters() -> None:
    request = CollectionToolRequest.model_validate(
        {"filters": {"owner": "team-a"}, "page": 2, "page_size": 20}
    )

    assert request.filters == {"owner": "team-a"}
    assert request.page == 2
    assert request.page_size == 20


def test_document_summary_request_requires_document_id() -> None:
    request = DocumentSummaryRequest.model_validate({"document_id": "doc-123"})

    assert request.document_id == "doc-123"

    with pytest.raises(Exception):
        DocumentSummaryRequest.model_validate({})


def test_protocol_handler_validate_arguments_returns_normalized_models() -> None:
    handler = ProtocolHandler()

    query_request = handler.validate_arguments("query_knowledge_hub", {"query": "ragms"})
    assert isinstance(query_request, QueryToolRequest)
    assert query_request.top_k == 5

    ingest_request = handler.validate_arguments("ingest_documents", {"paths": ["a.pdf"]})
    assert isinstance(ingest_request, IngestToolRequest)
    assert ingest_request.force_rebuild is False


def test_protocol_handler_validate_arguments_rejects_invalid_payloads() -> None:
    handler = ProtocolHandler()

    with pytest.raises(SchemaValidationError, match="query_knowledge_hub.query"):
        handler.validate_arguments("query_knowledge_hub", {})

    with pytest.raises(SchemaValidationError, match="ingest_documents.paths"):
        handler.validate_arguments("ingest_documents", {"paths": []})

    with pytest.raises(SchemaValidationError, match="Unknown tool: missing_tool"):
        handler.validate_arguments("missing_tool", {})


def test_tool_input_schema_matches_request_models() -> None:
    query_schema = get_input_schema("query_knowledge_hub")
    ingest_schema = get_input_schema("ingest_documents")

    assert query_schema["properties"]["query"]["type"] == "string"
    assert query_schema["properties"]["top_k"]["default"] == 5
    assert query_schema["required"] == ["query"]

    assert ingest_schema["properties"]["force_rebuild"]["default"] is False
    assert ingest_schema["required"] == ["paths"]
