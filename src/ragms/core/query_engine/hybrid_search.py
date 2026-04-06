"""Hybrid retrieval helpers such as reciprocal rank fusion."""

from __future__ import annotations

from typing import Any

from ragms.core.models import RetrievalCandidate
from ragms.runtime.exceptions import RagMSError


class HybridSearchError(RagMSError):
    """Raised when hybrid-search helpers receive invalid inputs."""


def reciprocal_rank_fusion(
    dense_candidates: list[RetrievalCandidate],
    sparse_candidates: list[RetrievalCandidate],
    *,
    k: int = 60,
) -> list[RetrievalCandidate]:
    """Fuse dense and sparse candidates by reciprocal rank only."""

    if k <= 0:
        raise HybridSearchError("k must be greater than zero")

    fused: dict[str, dict[str, Any]] = {}

    for candidate, rank in _iter_ranked_candidates(dense_candidates, route="dense"):
        entry = fused.setdefault(candidate.chunk_id, _seed_entry(candidate))
        entry["candidate"] = _merge_candidate(entry["candidate"], candidate, route="dense")
        entry["rrf_score"] += 1.0 / (k + rank)
        entry["dense_rank"] = rank
        entry["dense_score"] = candidate.score

    for candidate, rank in _iter_ranked_candidates(sparse_candidates, route="sparse"):
        entry = fused.setdefault(candidate.chunk_id, _seed_entry(candidate))
        entry["candidate"] = _merge_candidate(entry["candidate"], candidate, route="sparse")
        entry["rrf_score"] += 1.0 / (k + rank)
        entry["sparse_rank"] = rank
        entry["sparse_score"] = candidate.score

    results = [
        entry["candidate"].with_updates(
            source_route=_resolve_source_route(
                dense_rank=entry["dense_rank"],
                sparse_rank=entry["sparse_rank"],
            ),
            dense_rank=entry["dense_rank"],
            sparse_rank=entry["sparse_rank"],
            dense_score=entry["dense_score"],
            sparse_score=entry["sparse_score"],
            rrf_score=round(entry["rrf_score"], 12),
        )
        for entry in fused.values()
    ]
    return sorted(results)


def _iter_ranked_candidates(
    candidates: list[RetrievalCandidate],
    *,
    route: str,
) -> list[tuple[RetrievalCandidate, int]]:
    """Return candidates paired with stable route-specific ranks."""

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            _resolve_route_rank(candidate, route=route),
            candidate.chunk_id,
        ),
    )
    return [
        (candidate, _resolve_route_rank(candidate, route=route, fallback_rank=index))
        for index, candidate in enumerate(sorted_candidates, start=1)
    ]


def _resolve_route_rank(
    candidate: RetrievalCandidate,
    *,
    route: str,
    fallback_rank: int | None = None,
) -> int:
    if route == "dense" and candidate.dense_rank is not None:
        return candidate.dense_rank
    if route == "sparse" and candidate.sparse_rank is not None:
        return candidate.sparse_rank
    if fallback_rank is None:
        return 10**9
    return fallback_rank


def _seed_entry(candidate: RetrievalCandidate) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "rrf_score": 0.0,
        "dense_rank": None,
        "sparse_rank": None,
        "dense_score": None,
        "sparse_score": None,
    }


def _merge_candidate(
    current: RetrievalCandidate,
    incoming: RetrievalCandidate,
    *,
    route: str,
) -> RetrievalCandidate:
    metadata = dict(current.metadata)
    metadata.update(incoming.metadata)
    content = current.content or incoming.content
    score = max(current.score, incoming.score)
    source_route = current.source_route
    if current.chunk_id == incoming.chunk_id and current.source_route != incoming.source_route:
        source_route = "hybrid"
    if route == "dense":
        return current.with_updates(
            document_id=current.document_id or incoming.document_id,
            content=content,
            metadata=metadata,
            score=score,
            source_route=source_route,
            dense_rank=incoming.dense_rank or current.dense_rank,
            dense_score=incoming.score,
        )
    return current.with_updates(
        document_id=current.document_id or incoming.document_id,
        content=content,
        metadata=metadata,
        score=score,
        source_route=source_route,
        sparse_rank=incoming.sparse_rank or current.sparse_rank,
        sparse_score=incoming.score,
    )


def _resolve_source_route(*, dense_rank: int | None, sparse_rank: int | None) -> str:
    if dense_rank is not None and sparse_rank is not None:
        return "hybrid"
    if dense_rank is not None:
        return "dense"
    if sparse_rank is not None:
        return "sparse"
    raise HybridSearchError("RRF requires at least one route contribution")
