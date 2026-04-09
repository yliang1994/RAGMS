"""Hybrid retrieval helpers and orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ragms.core.models import HybridSearchResult, RetrievalCandidate
from ragms.core.query_engine.query_processor import ProcessedQuery
from ragms.runtime.exceptions import RagMSError


class HybridSearchError(RagMSError):
    """Raised when hybrid-search helpers receive invalid inputs."""


class HybridSearch:
    """Coordinate dense and sparse retrieval, filtering, fusion, and truncation."""

    def __init__(
        self,
        dense_retriever: Any,
        sparse_retriever: Any,
        *,
        rrf_k: int = 60,
        candidate_top_n: int = 20,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if candidate_top_n <= 0:
            raise ValueError("candidate_top_n must be greater than zero")

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k
        self.candidate_top_n = candidate_top_n

    def search(self, processed_query: ProcessedQuery) -> HybridSearchResult:
        """Run dense and sparse retrieval in parallel and return fused candidates."""

        result, _ = self.search_with_trace(processed_query)
        return result

    def search_with_trace(
        self,
        processed_query: ProcessedQuery,
        *,
        stage_callback: Callable[[str, Any, Any, dict[str, Any]], None] | None = None,
    ) -> tuple[HybridSearchResult, dict[str, Any]]:
        """Run hybrid search and expose per-stage summaries for tracing."""

        dense_candidates, sparse_candidates = self._retrieve_candidates(processed_query)
        dense_filtered, dense_removed = self._apply_post_filters(
            dense_candidates,
            post_filters=processed_query.post_filters,
        )
        sparse_filtered, sparse_removed = self._apply_post_filters(
            sparse_candidates,
            post_filters=processed_query.post_filters,
        )
        fused_candidates = reciprocal_rank_fusion(
            dense_filtered,
            sparse_filtered,
            k=self.rrf_k,
        )
        fused_filtered, fused_removed = self._apply_post_filters(
            fused_candidates,
            post_filters=processed_query.post_filters,
        )
        truncated_candidates = tuple(fused_filtered[: self.candidate_top_n])
        trace_payload = {
            "dense_retrieval": {
                "input": {
                    "query": processed_query.dense_query,
                    "filters": dict(processed_query.pre_filters),
                    "top_k": processed_query.top_k,
                },
                "output": {
                    "retrieved_count": len(dense_filtered),
                    "removed_count": dense_removed,
                    "top_chunk_ids": [candidate.chunk_id for candidate in dense_filtered[: processed_query.top_k]],
                },
                "metadata": {
                    "provider": getattr(self.dense_retriever.vector_store, "implementation", self.dense_retriever.vector_store.__class__.__name__),
                    "retry_count": 0,
                },
            },
            "sparse_retrieval": {
                "input": {
                    "terms": list(processed_query.sparse_terms),
                    "filters": dict(processed_query.pre_filters),
                    "top_k": processed_query.top_k,
                },
                "output": {
                    "retrieved_count": len(sparse_filtered),
                    "removed_count": sparse_removed,
                    "top_chunk_ids": [candidate.chunk_id for candidate in sparse_filtered[: processed_query.top_k]],
                },
                "metadata": {
                    "method": "bm25",
                    "retry_count": 0,
                },
            },
            "fusion": {
                "input": {
                    "dense_count": len(dense_filtered),
                    "sparse_count": len(sparse_filtered),
                },
                "output": {
                    "fused_count": len(fused_filtered),
                    "returned_count": len(truncated_candidates),
                    "top_chunk_ids": [candidate.chunk_id for candidate in truncated_candidates],
                },
                "metadata": {
                    "method": "rrf",
                    "rrf_k": self.rrf_k,
                    "candidate_top_n": self.candidate_top_n,
                    "removed_count": fused_removed,
                },
            },
        }
        if stage_callback is not None:
            for stage_name in ("dense_retrieval", "sparse_retrieval", "fusion"):
                payload = trace_payload[stage_name]
                stage_callback(
                    stage_name,
                    payload["input"],
                    payload["output"],
                    payload["metadata"],
                )

        result = HybridSearchResult(
            query=processed_query.normalized_query,
            collection=processed_query.collection,
            candidates=truncated_candidates,
            dense_count=len(dense_candidates),
            sparse_count=len(sparse_candidates),
            filtered_out_count=dense_removed + sparse_removed + fused_removed,
            candidate_top_n=self.candidate_top_n,
            debug_info={
                "query": {
                    "normalized_query": processed_query.normalized_query,
                    "dense_query": processed_query.dense_query,
                    "sparse_terms": list(processed_query.sparse_terms),
                    "pre_filters": dict(processed_query.pre_filters),
                    "post_filters": dict(processed_query.post_filters),
                },
                "retrieval": {
                    "dense_before_post_filter": len(dense_candidates),
                    "dense_after_post_filter": len(dense_filtered),
                    "sparse_before_post_filter": len(sparse_candidates),
                    "sparse_after_post_filter": len(sparse_filtered),
                },
                "fusion": {
                    "rrf_k": self.rrf_k,
                    "before_post_filter": len(fused_candidates),
                    "after_post_filter": len(fused_filtered),
                    "candidate_top_n": self.candidate_top_n,
                    "returned_count": len(truncated_candidates),
                    "truncated_count": max(0, len(fused_filtered) - len(truncated_candidates)),
                },
                "filters": {
                    "dense_removed": dense_removed,
                    "sparse_removed": sparse_removed,
                    "fused_removed": fused_removed,
                    "total_removed": dense_removed + sparse_removed + fused_removed,
                },
            },
        )
        return result, trace_payload

    def _retrieve_candidates(
        self,
        processed_query: ProcessedQuery,
    ) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(self.dense_retriever.retrieve, processed_query)
                sparse_future = executor.submit(self.sparse_retriever.retrieve, processed_query)
                return dense_future.result(), sparse_future.result()
        except Exception as exc:
            raise HybridSearchError("Hybrid search retrieval failed") from exc

    def _apply_post_filters(
        self,
        candidates: list[RetrievalCandidate],
        *,
        post_filters: Mapping[str, Any],
    ) -> tuple[list[RetrievalCandidate], int]:
        if not post_filters:
            return list(candidates), 0

        filtered: list[RetrievalCandidate] = []
        removed = 0
        for candidate in candidates:
            if self._matches_post_filters(candidate, post_filters=post_filters):
                filtered.append(candidate)
            else:
                removed += 1
        return filtered, removed

    def _matches_post_filters(
        self,
        candidate: RetrievalCandidate,
        *,
        post_filters: Mapping[str, Any],
    ) -> bool:
        metadata = dict(candidate.metadata)
        for key, expected in post_filters.items():
            if key not in metadata:
                continue
            actual = metadata.get(key)
            if not _match_filter_value(actual, expected):
                return False
        return True


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


def _match_filter_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return _match_mapping_filter(actual, expected)
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _match_mapping_filter(actual: Any, expected: Mapping[str, Any]) -> bool:
    for operator, operand in expected.items():
        if operator == "$contains":
            if isinstance(actual, str):
                if str(operand) not in actual:
                    return False
                continue
            if isinstance(actual, list):
                if operand not in actual:
                    return False
                continue
            return False
        if operator == "$in":
            if actual not in list(operand):
                return False
            continue
        raise HybridSearchError(f"Unsupported post-filter operator: {operator}")
    return True
