"""Query normalization, keyword extraction, and metadata-filter parsing."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ragms.runtime.exceptions import RagMSError

try:  # pragma: no cover - exercised only when jieba is installed
    import jieba
except Exception:  # pragma: no cover - optional dependency boundary
    jieba = None


class QueryProcessorError(RagMSError):
    """Raised when query preprocessing cannot produce a valid retrieval request."""


@dataclass(frozen=True)
class ParsedFilters:
    """Structured filter split between retriever-side and post-retrieval execution."""

    pre_filters: dict[str, Any]
    post_filters: dict[str, Any]
    collection: str | None = None


@dataclass(frozen=True)
class ProcessedQuery:
    """Normalized retrieval request shared by dense and sparse routes."""

    raw_query: str
    normalized_query: str
    dense_query: str
    collection: str
    top_k: int
    keywords: list[str]
    expanded_keywords: list[str]
    sparse_terms: list[str]
    sparse_query: str
    pre_filters: dict[str, Any]
    post_filters: dict[str, Any]


class QueryProcessor:
    """Prepare a user query for dense and sparse retrieval paths."""

    DEFAULT_STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "by",
        "does",
        "for",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "why",
        "with",
    }
    DEFAULT_PRE_FILTER_FIELDS = {
        "chunk_id",
        "collection",
        "doc_type",
        "document_id",
        "source_path",
        "tenant_id",
    }
    DEFAULT_SYNONYM_MAP = {
        "ai": ["artificial intelligence"],
        "bm25": ["okapi bm25"],
        "llm": ["large language model"],
        "rag": ["retrieval augmented generation"],
        "rrf": ["reciprocal rank fusion"],
    }
    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*|[\u4e00-\u9fff]+")

    def __init__(
        self,
        *,
        default_collection: str = "default",
        default_top_k: int = 5,
        max_top_k: int = 50,
        stop_words: Sequence[str] | None = None,
        synonym_map: Mapping[str, Sequence[str]] | None = None,
        pre_filter_fields: Sequence[str] | None = None,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than zero")
        if max_top_k <= 0:
            raise ValueError("max_top_k must be greater than zero")
        if default_top_k > max_top_k:
            raise ValueError("default_top_k must not exceed max_top_k")

        self.default_collection = self._normalize_collection(default_collection)
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k
        self.stop_words = {
            word.strip().casefold()
            for word in (stop_words or self.DEFAULT_STOP_WORDS)
            if str(word).strip()
        }
        self.synonym_map = {
            str(key).strip().casefold(): [
                term.strip().casefold()
                for term in value
                if str(term).strip()
            ]
            for key, value in (synonym_map or self.DEFAULT_SYNONYM_MAP).items()
            if str(key).strip()
        }
        self.pre_filter_fields = {
            field.strip()
            for field in (pre_filter_fields or self.DEFAULT_PRE_FILTER_FIELDS)
            if str(field).strip()
        }

    def extract_keywords(self, query: str) -> list[str]:
        """Extract stable sparse-retrieval keywords from a normalized query."""

        normalized_query = self._normalize_query(query)
        candidates = self._regex_tokens(normalized_query)
        if self._contains_cjk(normalized_query):
            candidates.extend(self._jieba_tokens(normalized_query))

        keywords: list[str] = []
        seen: set[str] = set()
        for token in candidates:
            normalized_token = token.strip().casefold()
            if not normalized_token:
                continue
            if normalized_token in self.stop_words:
                continue
            if not self._is_meaningful_token(normalized_token):
                continue
            if normalized_token in seen:
                continue
            seen.add(normalized_token)
            keywords.append(normalized_token)

        return keywords

    def parse_filters(self, filters: Mapping[str, Any] | str | None) -> ParsedFilters:
        """Validate request filters and split them into pre/post filter groups."""

        if filters is None:
            return ParsedFilters(pre_filters={}, post_filters={})

        if isinstance(filters, str):
            raw_filters = filters.strip()
            if not raw_filters:
                return ParsedFilters(pre_filters={}, post_filters={})
            try:
                filters_payload = json.loads(raw_filters)
            except json.JSONDecodeError as exc:
                raise QueryProcessorError("filters must be valid JSON when provided as a string") from exc
        else:
            filters_payload = dict(filters)

        if not isinstance(filters_payload, Mapping):
            raise QueryProcessorError("filters must be a mapping")

        pre_filters: dict[str, Any] = {}
        post_filters: dict[str, Any] = {}
        collection: str | None = None

        for raw_key, value in filters_payload.items():
            key = str(raw_key).strip()
            if not key:
                raise QueryProcessorError("filter keys must not be empty")
            self._validate_filter_value(key, value)
            if key == "collection":
                collection = self._normalize_collection(str(value))
                continue
            if self._is_pre_filter_candidate(key, value):
                pre_filters[key] = value
            else:
                post_filters[key] = value

        return ParsedFilters(
            pre_filters=pre_filters,
            post_filters=post_filters,
            collection=collection,
        )

    def process(
        self,
        query: str,
        *,
        collection: str | None = None,
        top_k: int | str | None = None,
        filters: Mapping[str, Any] | str | None = None,
    ) -> ProcessedQuery:
        """Return the normalized retrieval request for dense and sparse routes."""

        normalized_query = self._normalize_query(query)
        parsed_filters = self.parse_filters(filters)
        resolved_collection = self._resolve_collection(
            collection=collection,
            filter_collection=parsed_filters.collection,
        )
        resolved_top_k = self._parse_top_k(top_k)
        keywords = self.extract_keywords(normalized_query)
        expanded_keywords = self._expand_keywords(keywords)
        sparse_terms = keywords + [
            term for term in expanded_keywords if term not in set(keywords)
        ]

        return ProcessedQuery(
            raw_query=query,
            normalized_query=normalized_query,
            dense_query=normalized_query,
            collection=resolved_collection,
            top_k=resolved_top_k,
            keywords=keywords,
            expanded_keywords=expanded_keywords,
            sparse_terms=sparse_terms,
            sparse_query=" ".join(sparse_terms),
            pre_filters=parsed_filters.pre_filters,
            post_filters=parsed_filters.post_filters,
        )

    def _expand_keywords(self, keywords: Sequence[str]) -> list[str]:
        """Return deterministic sparse-only expansions for extracted keywords."""

        expansions: list[str] = []
        seen = set(keywords)
        for keyword in keywords:
            for expanded in self.synonym_map.get(keyword, []):
                if expanded not in seen:
                    seen.add(expanded)
                    expansions.append(expanded)
        return expansions

    @classmethod
    def _normalize_query(cls, query: str) -> str:
        normalized = re.sub(r"\s+", " ", str(query)).strip()
        if not normalized:
            raise QueryProcessorError("query must not be empty")
        return normalized

    def _resolve_collection(
        self,
        *,
        collection: str | None,
        filter_collection: str | None,
    ) -> str:
        explicit_collection = None if collection is None else self._normalize_collection(collection)
        if explicit_collection and filter_collection and explicit_collection != filter_collection:
            raise QueryProcessorError("collection conflicts with filters.collection")
        return explicit_collection or filter_collection or self.default_collection

    @staticmethod
    def _normalize_collection(collection: str) -> str:
        normalized = str(collection).strip()
        if not normalized:
            raise QueryProcessorError("collection must not be empty")
        return normalized

    def _parse_top_k(self, top_k: int | str | None) -> int:
        if top_k is None:
            return self.default_top_k
        try:
            value = int(top_k)
        except (TypeError, ValueError) as exc:
            raise QueryProcessorError("top_k must be an integer") from exc
        if value <= 0:
            raise QueryProcessorError("top_k must be greater than zero")
        if value > self.max_top_k:
            raise QueryProcessorError(f"top_k must not exceed {self.max_top_k}")
        return value

    def _is_pre_filter_candidate(self, key: str, value: Any) -> bool:
        if key not in self.pre_filter_fields:
            return False
        return self._is_scalar_filter(value) or self._is_scalar_list_filter(value)

    @staticmethod
    def _is_scalar_filter(value: Any) -> bool:
        return isinstance(value, (str, int, float, bool)) and not isinstance(value, bool) or isinstance(value, bool)

    @classmethod
    def _is_scalar_list_filter(cls, value: Any) -> bool:
        return isinstance(value, list) and value and all(cls._is_scalar_filter(item) for item in value)

    @classmethod
    def _validate_filter_value(cls, key: str, value: Any) -> None:
        if value is None:
            raise QueryProcessorError(f"filter '{key}' must not be null")
        if cls._is_scalar_filter(value):
            return
        if cls._is_scalar_list_filter(value):
            return
        if isinstance(value, Mapping) and value:
            return
        raise QueryProcessorError(f"filter '{key}' has unsupported value type")

    @classmethod
    def _regex_tokens(cls, query: str) -> list[str]:
        return cls._TOKEN_PATTERN.findall(query)

    @staticmethod
    def _contains_cjk(query: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", query))

    @staticmethod
    def _jieba_tokens(query: str) -> list[str]:
        if jieba is None:
            return []
        return [
            token.strip()
            for token in jieba.cut_for_search(query)
            if token.strip()
        ]

    @staticmethod
    def _is_meaningful_token(token: str) -> bool:
        if len(token) > 1:
            return True
        return any(character.isdigit() for character in token)
