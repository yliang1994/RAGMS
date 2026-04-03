"""Sparse BM25-oriented encoding for smart chunks."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from typing import Any

import jieba

from ragms.ingestion_pipeline.embedding.optimization import optimize_embedding_batches
from ragms.runtime.exceptions import RagMSError


_ENGLISH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_CJK_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "into",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


class SparseEncodingError(RagMSError):
    """Raised when sparse encoding cannot produce stable term statistics."""


class SparseEncoder:
    """Build deterministic sparse token statistics suitable for later BM25 indexing."""

    def __init__(
        self,
        *,
        batch_size: int = 32,
        max_workers: int = 1,
        stopwords: set[str] | None = None,
        normalize_case: bool = True,
        enable_jieba: bool = True,
        min_token_length: int = 2,
        cache: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        if min_token_length <= 0:
            raise ValueError("min_token_length must be greater than zero")

        self.batch_size = batch_size
        self.max_workers = max_workers
        self.normalize_case = normalize_case
        self.enable_jieba = enable_jieba
        self.min_token_length = min_token_length
        self.stopwords = {
            (word.lower() if normalize_case else word)
            for word in (_DEFAULT_STOPWORDS | set(stopwords or set()))
        }
        self._cache = cache if cache is not None else {}

    def encode(self, documents: Sequence[object]) -> list[dict[str, Any]]:
        """Encode smart chunks or raw texts into sparse BM25-ready representations."""

        if not documents:
            return []

        texts = [self._extract_text(document) for document in documents]
        return optimize_embedding_batches(
            texts,
            self._encode_batch,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            cache=self._cache,
            cache_key_fn=self._content_hash,
        )

    def _encode_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Encode one batch of texts without changing item order."""

        return [self._encode_text(text) for text in texts]

    def _encode_text(self, text: str) -> dict[str, Any]:
        """Convert one document text into stable sparse term statistics."""

        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return {
                "content_hash": self._content_hash(text),
                "tokens": [],
                "term_frequencies": {},
                "term_weights": {},
                "document_length": 0,
                "unique_terms": 0,
            }

        tokens = self._tokenize(normalized_text)
        frequencies = Counter(tokens)
        document_length = sum(frequencies.values())
        term_weights = (
            {
                token: round(count / document_length, 6)
                for token, count in sorted(frequencies.items())
            }
            if document_length
            else {}
        )
        return {
            "content_hash": self._content_hash(text),
            "tokens": tokens,
            "term_frequencies": dict(sorted(frequencies.items())),
            "term_weights": term_weights,
            "document_length": document_length,
            "unique_terms": len(frequencies),
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse repeated whitespace while preserving semantic token boundaries."""

        return " ".join(str(text).split()).strip()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize mixed-language text with configurable normalization."""

        english_tokens = self._extract_english_tokens(text)
        cjk_tokens = self._extract_cjk_tokens(text)
        merged = english_tokens + cjk_tokens
        return [
            token
            for token in merged
            if len(token) >= self.min_token_length and token not in self.stopwords
        ]

    def _extract_english_tokens(self, text: str) -> list[str]:
        """Extract ASCII-like tokens with optional lower-casing."""

        tokens = _ENGLISH_TOKEN_PATTERN.findall(text)
        if self.normalize_case:
            return [token.lower() for token in tokens]
        return tokens

    def _extract_cjk_tokens(self, text: str) -> list[str]:
        """Extract Chinese tokens using jieba or a regex fallback."""

        if self.enable_jieba:
            raw_tokens = [token.strip() for token in jieba.lcut(text)]
            return [
                token
                for token in raw_tokens
                if token
                and re.search(r"[\u4e00-\u9fff]", token)
                and len(token) >= self.min_token_length
            ]
        return _CJK_TOKEN_PATTERN.findall(text)

    @staticmethod
    def _content_hash(text: str) -> str:
        """Compute a stable content hash for sparse-cache reuse."""

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _extract_text(self, document: object) -> str:
        """Extract text from supported smart chunk inputs."""

        if isinstance(document, str):
            return document
        if isinstance(document, Mapping):
            return str(document.get("content", ""))
        if hasattr(document, "content"):
            return str(getattr(document, "content"))
        if is_dataclass(document) and hasattr(document, "__dict__"):
            return str(document.__dict__.get("content", ""))
        raise SparseEncodingError("Sparse encoder expected strings or chunk-like objects")
