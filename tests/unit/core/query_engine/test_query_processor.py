from __future__ import annotations

import pytest

from ragms.core.query_engine import ParsedFilters, QueryProcessor, QueryProcessorError


def test_extract_keywords_normalizes_whitespace_and_filters_stop_words() -> None:
    processor = QueryProcessor(stop_words={"how", "does", "the"})

    keywords = processor.extract_keywords("  How   does   the RAG system handle error-code E42?  ")

    assert keywords == ["rag", "system", "handle", "error-code", "e42"]


def test_parse_filters_splits_pre_and_post_filters() -> None:
    processor = QueryProcessor(pre_filter_fields={"doc_type", "document_id"})

    parsed = processor.parse_filters(
        {
            "doc_type": "pdf",
            "document_id": ["doc-1", "doc-2"],
            "author": "alice",
            "tags": {"$contains": "rag"},
        }
    )

    assert parsed == ParsedFilters(
        pre_filters={
            "doc_type": "pdf",
            "document_id": ["doc-1", "doc-2"],
        },
        post_filters={
            "author": "alice",
            "tags": {"$contains": "rag"},
        },
        collection=None,
    )


def test_parse_filters_accepts_json_and_extracts_collection() -> None:
    processor = QueryProcessor()

    parsed = processor.parse_filters('{"collection":"docs","doc_type":"pdf","topic":"ops"}')

    assert parsed.collection == "docs"
    assert parsed.pre_filters == {"doc_type": "pdf"}
    assert parsed.post_filters == {"topic": "ops"}


def test_process_builds_dense_and_sparse_query_payloads() -> None:
    processor = QueryProcessor(
        default_collection="knowledge-base",
        stop_words={"how", "does"},
        synonym_map={
            "rag": ["retrieval augmented generation"],
            "llm": ["large language model"],
        },
        pre_filter_fields={"doc_type"},
    )

    processed = processor.process(
        "  How does RAG help LLM search?  ",
        top_k="3",
        filters={"doc_type": "pdf", "owner": "platform"},
    )

    assert processed.normalized_query == "How does RAG help LLM search?"
    assert processed.dense_query == "How does RAG help LLM search?"
    assert processed.collection == "knowledge-base"
    assert processed.top_k == 3
    assert processed.keywords == ["rag", "help", "llm", "search"]
    assert processed.expanded_keywords == [
        "retrieval augmented generation",
        "large language model",
    ]
    assert processed.sparse_terms == [
        "rag",
        "help",
        "llm",
        "search",
        "retrieval augmented generation",
        "large language model",
    ]
    assert processed.pre_filters == {"doc_type": "pdf"}
    assert processed.post_filters == {"owner": "platform"}


@pytest.mark.parametrize(
    ("query", "top_k", "filters", "message"),
    [
        ("   ", None, None, "query must not be empty"),
        ("hello", "0", None, "top_k must be greater than zero"),
        ("hello", "abc", None, "top_k must be an integer"),
        ("hello", None, '{"author": null}', "filter 'author' must not be null"),
    ],
)
def test_process_rejects_invalid_inputs(
    query: str,
    top_k: str | None,
    filters: str | None,
    message: str,
) -> None:
    processor = QueryProcessor()

    with pytest.raises(QueryProcessorError, match=message):
        processor.process(query, top_k=top_k, filters=filters)


def test_process_rejects_conflicting_collection_sources() -> None:
    processor = QueryProcessor()

    with pytest.raises(QueryProcessorError, match="collection conflicts with filters.collection"):
        processor.process(
            "retrieval",
            collection="docs-a",
            filters={"collection": "docs-b"},
        )
