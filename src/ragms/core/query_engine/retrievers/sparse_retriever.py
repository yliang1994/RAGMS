"""Sparse retrieval route backed by a persistent BM25 index."""

from __future__ import annotations

from ragms.core.models import RetrievalCandidate
from ragms.core.query_engine.query_processor import ProcessedQuery
from ragms.runtime.exceptions import RagMSError
from ragms.storage.indexes import BM25Indexer


class SparseRetrieverError(RagMSError):
    """Raised when sparse retrieval cannot complete or normalize its results."""


class SparseRetriever:
    """Retrieve sparse candidates from the BM25 index using one combined query."""

    def __init__(self, indexer: BM25Indexer) -> None:
        self.indexer = indexer

    def retrieve(self, processed_query: ProcessedQuery) -> list[RetrievalCandidate]:
        """Execute one sparse retrieval call and normalize the results."""

        try:
            matches = self.indexer.search(
                processed_query.sparse_terms,
                top_k=processed_query.top_k,
                filters=self._build_filters(processed_query),
            )
        except Exception as exc:  # pragma: no cover - storage boundary
            raise SparseRetrieverError("Sparse retriever BM25 query failed") from exc

        candidates: list[RetrievalCandidate] = []
        for rank, match in enumerate(matches, start=1):
            payload = dict(match)
            payload.setdefault("sparse_rank", rank)
            payload.setdefault("sparse_score", payload.get("score"))
            try:
                candidates.append(
                    RetrievalCandidate.from_match(
                        payload,
                        source_route="sparse",
                    )
                )
            except Exception as exc:
                raise SparseRetrieverError("Sparse retriever returned an invalid match payload") from exc
        return candidates

    @staticmethod
    def _build_filters(processed_query: ProcessedQuery) -> dict[str, object]:
        filters = dict(processed_query.pre_filters)
        filters["collection"] = processed_query.collection
        return filters
