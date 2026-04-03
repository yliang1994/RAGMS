"""Dense embedding orchestration with batching, retries, and content-level deduplication."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from typing import Any

from tenacity import RetryError, retry, stop_after_attempt, wait_fixed

from ragms.libs.abstractions import BaseEmbedding
from ragms.runtime.exceptions import RagMSError


class DenseEncodingError(RagMSError):
    """Raised when dense embedding generation cannot produce stable vectors."""


class DenseEncoder:
    """Encode smart chunks and queries into dense vectors with stable ordering."""

    def __init__(
        self,
        embedding: BaseEmbedding,
        *,
        batch_size: int = 16,
        max_retries: int = 3,
        retry_interval_seconds: float = 0.0,
        min_interval_seconds: float = 0.0,
        expected_dimension: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than zero")
        if retry_interval_seconds < 0:
            raise ValueError("retry_interval_seconds must be non-negative")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        if expected_dimension is not None and expected_dimension <= 0:
            raise ValueError("expected_dimension must be greater than zero")

        self.embedding = embedding
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_interval_seconds = retry_interval_seconds
        self.min_interval_seconds = min_interval_seconds
        self.expected_dimension = expected_dimension
        self._last_request_started_at = 0.0
        self._observed_dimension: int | None = expected_dimension

    def encode_documents(self, documents: Sequence[object]) -> list[list[float]]:
        """Encode smart chunks or raw texts into vectors aligned with the input order."""

        if not documents:
            return []

        texts = [self._extract_text(document) for document in documents]
        unique_texts: list[str] = []
        hash_to_indexes: dict[str, list[int]] = {}

        for index, text in enumerate(texts):
            content_hash = self._content_hash(text)
            hash_to_indexes.setdefault(content_hash, []).append(index)
            if len(hash_to_indexes[content_hash]) == 1:
                unique_texts.append(text)

        vectors_by_hash: dict[str, list[float]] = {}
        for batch in self._iter_batches(unique_texts):
            batch_vectors = self._embed_batch_with_retry(batch)
            self._validate_batch(batch, batch_vectors)
            for text, vector in zip(batch, batch_vectors, strict=True):
                vectors_by_hash[self._content_hash(text)] = vector

        return [
            list(vectors_by_hash[self._content_hash(texts[index])])
            for index in range(len(texts))
        ]

    def encode_query(self, text: str) -> list[float]:
        """Encode a single query string and validate the resulting vector."""

        query = text.strip()
        if not query:
            raise DenseEncodingError("Query text must not be empty")

        self._wait_for_rate_limit()
        try:
            vector = list(self.embedding.embed_query(query))
        except Exception as exc:  # pragma: no cover - defensive provider boundary
            raise DenseEncodingError("Dense query encoding failed") from exc
        self._validate_vector(vector)
        return vector

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Request one embedding batch with bounded retries."""

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_fixed(self.retry_interval_seconds),
            reraise=True,
        )
        def request() -> list[list[float]]:
            self._wait_for_rate_limit()
            try:
                return [list(vector) for vector in self.embedding.embed_documents(texts)]
            except Exception as exc:
                raise DenseEncodingError("Dense document encoding failed") from exc

        try:
            return request()
        except RetryError as exc:  # pragma: no cover - tenacity re-raises when reraise=False
            raise DenseEncodingError("Dense document encoding failed") from exc
        except DenseEncodingError:
            raise

    def _validate_batch(self, texts: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        """Validate count and dimensions for one embedding batch."""

        if len(vectors) != len(texts):
            raise DenseEncodingError("Dense encoder returned an unexpected embedding count")
        for vector in vectors:
            self._validate_vector(vector)

    def _validate_vector(self, vector: Sequence[float]) -> None:
        """Validate that a vector is non-empty and dimensionally consistent."""

        if not vector:
            raise DenseEncodingError("Dense encoder returned an empty embedding vector")

        dimension = len(vector)
        if self._observed_dimension is None:
            self._observed_dimension = dimension
        elif self._observed_dimension != dimension:
            raise DenseEncodingError(
                f"Dense encoder returned embedding dimension {dimension}, expected {self._observed_dimension}"
            )

    def _wait_for_rate_limit(self) -> None:
        """Apply a simple minimum interval between provider requests."""

        if self.min_interval_seconds <= 0:
            self._last_request_started_at = time.monotonic()
            return

        now = time.monotonic()
        elapsed = now - self._last_request_started_at
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_started_at = time.monotonic()

    def _iter_batches(self, texts: Sequence[str]) -> list[list[str]]:
        """Split texts into provider-sized batches while preserving order."""

        return [
            list(texts[index : index + self.batch_size])
            for index in range(0, len(texts), self.batch_size)
        ]

    @staticmethod
    def _content_hash(text: str) -> str:
        """Compute a stable content hash for deduplication."""

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _extract_text(self, document: object) -> str:
        """Extract raw text from a smart chunk payload or raw string."""

        if isinstance(document, str):
            text = document
        elif isinstance(document, Mapping):
            text = str(document.get("content", ""))
        elif hasattr(document, "content"):
            text = str(getattr(document, "content"))
        elif is_dataclass(document) and hasattr(document, "__dict__"):
            text = str(document.__dict__.get("content", ""))
        else:
            raise DenseEncodingError("Dense encoder expected strings or chunk-like objects")

        normalized = text.strip()
        if not normalized:
            raise DenseEncodingError("Document content must not be empty")
        return normalized
