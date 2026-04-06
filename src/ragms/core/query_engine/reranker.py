"""Core reranker orchestration with request-level fallback behavior."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ragms.core.models import HybridSearchResult, RetrievalCandidate
from ragms.libs.abstractions import BaseReranker
from ragms.libs.factories.reranker_factory import RerankerFactory
from ragms.runtime.exceptions import RagMSError


class RerankerError(RagMSError):
    """Raised when reranker orchestration cannot produce a valid ranking."""


class Reranker:
    """Apply an optional reranker backend to hybrid-search candidates."""

    def __init__(
        self,
        *,
        backend: str = "disabled",
        final_top_k: int | None = None,
        provider: BaseReranker | None = None,
        provider_factory: Callable[[Mapping[str, Any] | None], BaseReranker] | None = None,
        provider_config: Mapping[str, Any] | None = None,
    ) -> None:
        if final_top_k is not None and final_top_k <= 0:
            raise ValueError("final_top_k must be greater than zero")

        self.backend = backend.strip().lower() or "disabled"
        self.final_top_k = final_top_k
        self._provider = provider
        self._provider_factory = provider_factory or RerankerFactory.create
        self._provider_config = (
            dict(provider_config)
            if provider_config is not None
            else {"provider": self.backend}
        )

    def run(self, result: HybridSearchResult) -> HybridSearchResult:
        """Apply the configured reranker and return a reranked result."""

        input_candidates = list(result.top_candidates(result.candidate_top_n))
        if not input_candidates:
            return self._build_result(
                result,
                candidates=[],
                backend=self.backend,
                fallback_applied=False,
                fallback_reason=None,
                input_count=0,
            )

        payloads = [candidate.to_dict() for candidate in input_candidates]
        provider = self._get_provider()
        try:
            ranked_items = provider.rerank(
                result.query,
                payloads,
                top_k=self.final_top_k,
            )
        except Exception as exc:
            raise RerankerError("Reranker execution failed") from exc

        reranked_candidates = self._map_ranked_items(
            ranked_items,
            original_candidates=input_candidates,
        )
        return self._build_result(
            result,
            candidates=reranked_candidates,
            backend=self.backend,
            fallback_applied=False,
            fallback_reason=None,
            input_count=len(input_candidates),
        )

    def run_with_fallback(self, result: HybridSearchResult) -> HybridSearchResult:
        """Run reranking and fall back to the RRF order on initialization or runtime failures."""

        try:
            return self.run(result)
        except Exception as exc:
            fallback_reason = str(exc) or "reranker_failed"
            fallback_candidates = [
                candidate.with_updates(
                    fallback_applied=True,
                    fallback_reason=fallback_reason,
                    rerank_score=None,
                )
                for candidate in result.top_candidates(self.final_top_k or result.candidate_top_n)
            ]
            return self._build_result(
                result,
                candidates=fallback_candidates,
                backend=self.backend,
                fallback_applied=True,
                fallback_reason=fallback_reason,
                input_count=len(result.top_candidates(result.candidate_top_n)),
            )

    def _get_provider(self) -> BaseReranker:
        if self._provider is None:
            try:
                self._provider = self._provider_factory(self._provider_config)
            except Exception as exc:
                raise RerankerError("Reranker initialization failed") from exc
        return self._provider

    def _map_ranked_items(
        self,
        ranked_items: list[dict[str, Any]],
        *,
        original_candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        candidates_by_id = {
            candidate.chunk_id: candidate
            for candidate in original_candidates
        }
        reranked: list[RetrievalCandidate] = []
        seen_chunk_ids: set[str] = set()

        for item in ranked_items:
            if not isinstance(item, Mapping):
                raise RerankerError("Reranker returned an invalid ranking item")
            if "document" not in item or "score" not in item:
                raise RerankerError("Reranker response is missing required fields")
            document = item["document"]
            if not isinstance(document, Mapping):
                raise RerankerError("Reranker response must return document payloads")
            chunk_id = str(document.get("chunk_id", "")).strip()
            if not chunk_id:
                raise RerankerError("Reranker response is missing chunk_id")
            if chunk_id in seen_chunk_ids:
                continue
            original = candidates_by_id.get(chunk_id)
            if original is None:
                raise RerankerError("Reranker response referenced an unknown candidate")
            seen_chunk_ids.add(chunk_id)
            reranked.append(
                original.with_updates(
                    rerank_score=float(item["score"]),
                    fallback_applied=False,
                    fallback_reason=None,
                )
            )

        if not reranked and original_candidates:
            raise RerankerError("Reranker returned no usable candidates")
        return reranked

    @staticmethod
    def _build_result(
        result: HybridSearchResult,
        *,
        candidates: list[RetrievalCandidate],
        backend: str,
        fallback_applied: bool,
        fallback_reason: str | None,
        input_count: int,
    ) -> HybridSearchResult:
        debug_info = dict(result.debug_info)
        debug_info["reranker"] = {
            "backend": backend,
            "input_count": input_count,
            "output_count": len(candidates),
            "fallback_applied": fallback_applied,
            "fallback_reason": fallback_reason,
        }
        return HybridSearchResult(
            query=result.query,
            collection=result.collection,
            candidates=tuple(candidates),
            dense_count=result.dense_count,
            sparse_count=result.sparse_count,
            filtered_out_count=result.filtered_out_count,
            candidate_top_n=result.candidate_top_n,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            debug_info=debug_info,
        )
