"""Dense retrieval route backed by one query embedding and one vector-store lookup."""

from __future__ import annotations

from typing import Any

from ragms.core.models import RetrievalCandidate
from ragms.core.query_engine.query_processor import ProcessedQuery
from ragms.libs.abstractions import BaseEmbedding, BaseVectorStore
from ragms.runtime.exceptions import RagMSError


class DenseRetrieverError(RagMSError):
    """Raised when dense retrieval cannot complete or normalize its results."""


class DenseRetriever:
    """Retrieve dense candidates from a vector store using a normalized query."""

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store

    def retrieve(self, processed_query: ProcessedQuery) -> list[RetrievalCandidate]:
        """Execute one dense retrieval call and normalize the results."""

        filters = self._build_filters(processed_query)

        try:
            query_vector = list(self.embedding.embed_query(processed_query.dense_query))
        except Exception as exc:  # pragma: no cover - provider boundary
            raise DenseRetrieverError("Dense retriever failed to encode query") from exc

        try:
            matches = self.vector_store.query(
                query_vector,
                top_k=processed_query.top_k,
                filters=filters,
            )
        except Exception as exc:  # pragma: no cover - provider boundary
            raise DenseRetrieverError("Dense retriever vector-store query failed") from exc

        candidates: list[RetrievalCandidate] = []
        for rank, match in enumerate(matches, start=1):
            payload = dict(match)
            payload.setdefault("dense_rank", rank)
            payload.setdefault("dense_score", payload.get("score"))
            try:
                candidates.append(
                    RetrievalCandidate.from_match(
                        payload,
                        source_route="dense",
                    )
                )
            except Exception as exc:
                raise DenseRetrieverError("Dense retriever returned an invalid match payload") from exc
        return candidates

    def _build_filters(self, processed_query: ProcessedQuery) -> dict[str, Any]:
        filters = dict(processed_query.pre_filters)
        if not hasattr(self.vector_store, "collection"):
            filters["collection"] = processed_query.collection
        return filters
