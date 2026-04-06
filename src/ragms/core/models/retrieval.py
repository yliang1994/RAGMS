"""Structured retrieval models shared across dense, sparse, fusion, and rerank stages."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import total_ordering
from typing import Any

from ragms.runtime.exceptions import RagMSError


class RetrievalModelError(RagMSError):
    """Raised when retrieval model payloads cannot be normalized safely."""


@total_ordering
@dataclass(frozen=True)
class RetrievalCandidate:
    """One retrieval candidate with route-specific ranking signals."""

    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    score: float
    source_route: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fallback_applied: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise RetrievalModelError("chunk_id must not be empty")
        if not self.document_id.strip():
            raise RetrievalModelError("document_id must not be empty")
        if not self.source_route.strip():
            raise RetrievalModelError("source_route must not be empty")
        self._validate_optional_rank("dense_rank", self.dense_rank)
        self._validate_optional_rank("sparse_rank", self.sparse_rank)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.rrf_score is not None:
            object.__setattr__(self, "rrf_score", float(self.rrf_score))
        if self.rerank_score is not None:
            object.__setattr__(self, "rerank_score", float(self.rerank_score))
        if self.dense_score is not None:
            object.__setattr__(self, "dense_score", float(self.dense_score))
        if self.sparse_score is not None:
            object.__setattr__(self, "sparse_score", float(self.sparse_score))

    @property
    def final_score(self) -> float:
        """Return the active score after fusion or rerank."""

        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        return self.score

    def sort_key(self) -> tuple[float, float, float, str, str]:
        """Return a deterministic descending sort key for ranking."""

        rrf_score = self.rrf_score if self.rrf_score is not None else float("-inf")
        rerank_score = self.rerank_score if self.rerank_score is not None else float("-inf")
        return (
            -self.final_score,
            -rerank_score,
            -rrf_score,
            self.source_route,
            self.chunk_id,
        )

    def with_updates(self, **changes: Any) -> "RetrievalCandidate":
        """Return an updated immutable candidate."""

        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate into a stable, JSON-friendly structure."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "score": self.score,
            "source_route": self.source_route,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "final_score": self.final_score,
            "fallback_applied": self.fallback_applied,
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_match(
        cls,
        payload: dict[str, Any],
        *,
        source_route: str,
    ) -> "RetrievalCandidate":
        """Build a candidate from a retriever or vector-store match payload."""

        metadata = dict(payload.get("metadata") or {})
        chunk_id = _pick_string(payload, metadata, "chunk_id", "id")
        if not chunk_id:
            raise RetrievalModelError("retrieval payload is missing chunk_id")

        document_id = _pick_string(payload, metadata, "document_id")
        if not document_id:
            raise RetrievalModelError("retrieval payload is missing document_id")

        content = _pick_string(payload, metadata, "content", "document", "text") or ""
        score = payload.get("score", 0.0)

        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            metadata=metadata,
            score=float(score),
            source_route=source_route,
            dense_rank=_optional_int(payload.get("dense_rank")),
            sparse_rank=_optional_int(payload.get("sparse_rank")),
            rrf_score=_optional_float(payload.get("rrf_score")),
            rerank_score=_optional_float(payload.get("rerank_score")),
            dense_score=_optional_float(payload.get("dense_score")),
            sparse_score=_optional_float(payload.get("sparse_score")),
            fallback_applied=bool(payload.get("fallback_applied", False)),
            fallback_reason=_optional_string(payload.get("fallback_reason")),
        )

    @staticmethod
    def _validate_optional_rank(field_name: str, value: int | None) -> None:
        if value is None:
            return
        if value <= 0:
            raise RetrievalModelError(f"{field_name} must be greater than zero")

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, RetrievalCandidate):
            return NotImplemented
        return self.sort_key() < other.sort_key()


@dataclass(frozen=True)
class HybridSearchResult:
    """Top-level retrieval result for hybrid search and optional rerank stages."""

    query: str
    collection: str
    candidates: tuple[RetrievalCandidate, ...]
    dense_count: int = 0
    sparse_count: int = 0
    filtered_out_count: int = 0
    candidate_top_n: int | None = None
    fallback_applied: bool = False
    fallback_reason: str | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        query = self.query.strip()
        collection = self.collection.strip()
        if not query:
            raise RetrievalModelError("query must not be empty")
        if not collection:
            raise RetrievalModelError("collection must not be empty")
        if self.dense_count < 0 or self.sparse_count < 0 or self.filtered_out_count < 0:
            raise RetrievalModelError("result counts must not be negative")
        if self.candidate_top_n is not None and self.candidate_top_n <= 0:
            raise RetrievalModelError("candidate_top_n must be greater than zero")

        ordered_candidates = tuple(sorted(self.candidates))
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "collection", collection)
        object.__setattr__(self, "candidates", ordered_candidates)
        object.__setattr__(self, "debug_info", dict(self.debug_info))

    @property
    def fused_count(self) -> int:
        """Return the current candidate count after fusion and filtering."""

        return len(self.candidates)

    def top_candidates(self, limit: int | None = None) -> tuple[RetrievalCandidate, ...]:
        """Return candidates in deterministic order with an optional cap."""

        if limit is None:
            return self.candidates
        if limit <= 0:
            raise RetrievalModelError("limit must be greater than zero")
        return self.candidates[:limit]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result in a stable structure for CLI, MCP, and tracing."""

        return {
            "query": self.query,
            "collection": self.collection,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "dense_count": self.dense_count,
            "sparse_count": self.sparse_count,
            "filtered_out_count": self.filtered_out_count,
            "fused_count": self.fused_count,
            "candidate_top_n": self.candidate_top_n,
            "fallback_applied": self.fallback_applied,
            "fallback_reason": self.fallback_reason,
            "debug_info": dict(self.debug_info),
        }


def _pick_string(primary: dict[str, Any], secondary: dict[str, Any], *keys: str) -> str | None:
    for source in (primary, secondary):
        for key in keys:
            if key in source and source[key] is not None:
                return str(source[key])
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
