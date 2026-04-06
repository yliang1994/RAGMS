"""End-to-end query engine orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragms.core.query_engine.answer_generator import AnswerGenerator
from ragms.core.query_engine.citation_builder import CitationBuilder
from ragms.core.query_engine.hybrid_search import HybridSearch
from ragms.core.query_engine.query_processor import QueryProcessor
from ragms.core.query_engine.reranker import Reranker
from ragms.core.query_engine.response_builder import ResponseBuilder
from ragms.core.query_engine.retrievers import DenseRetriever, SparseRetriever
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings
from ragms.storage.indexes import BM25Indexer


class QueryEngine:
    """Run the full query pipeline from preprocessing to final response payload."""

    def __init__(
        self,
        *,
        query_processor: QueryProcessor,
        hybrid_search: HybridSearch,
        reranker: Reranker,
        citation_builder: CitationBuilder,
        answer_generator: AnswerGenerator,
        response_builder: ResponseBuilder,
    ) -> None:
        self.query_processor = query_processor
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.citation_builder = citation_builder
        self.answer_generator = answer_generator
        self.response_builder = response_builder

    def run(
        self,
        *,
        query: str,
        collection: str | None = None,
        top_k: int | str | None = None,
        filters: dict[str, Any] | str | None = None,
        return_debug: bool = False,
        trace_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the query pipeline and return a stable structured response."""

        processed_query = self.query_processor.process(
            query,
            collection=collection,
            top_k=top_k,
            filters=filters,
        )
        hybrid_result = self.hybrid_search.search(processed_query)
        reranked_result = self.reranker.run_with_fallback(hybrid_result)
        final_candidates = list(reranked_result.top_candidates(processed_query.top_k))
        citations = self.citation_builder.build(final_candidates)
        answer = self.answer_generator.generate(
            query=processed_query.normalized_query,
            candidates=final_candidates,
            citations=citations,
        )
        return self.response_builder.build(
            query=processed_query.normalized_query,
            answer=answer,
            result=reranked_result,
            citations=citations,
            retrieved_candidates=final_candidates,
            trace_context=trace_context,
            return_debug=return_debug,
        )


def build_query_engine(
    container: ServiceContainer,
    *,
    settings: AppSettings | None = None,
    bm25_index_dir: str | Path | None = None,
    rrf_k: int = 60,
    candidate_top_n: int = 20,
) -> QueryEngine:
    """Build a query engine from the runtime container and settings."""

    resolved_settings = settings or container.settings
    collection = resolved_settings.vector_store.collection
    bm25_indexer = BM25Indexer(
        index_dir=(
            Path(bm25_index_dir)
            if bm25_index_dir is not None
            else resolved_settings.paths.data_dir / "indexes" / "sparse"
        ),
        collection=collection,
    )
    query_processor = QueryProcessor(default_collection=collection)
    hybrid_search = HybridSearch(
        DenseRetriever(container.get("embedding"), container.get("vector_store")),
        SparseRetriever(bm25_indexer),
        rrf_k=rrf_k,
        candidate_top_n=candidate_top_n,
    )
    reranker = Reranker(
        backend=resolved_settings.retrieval.rerank_backend,
        provider=container.get("reranker"),
        final_top_k=None,
    )
    return QueryEngine(
        query_processor=query_processor,
        hybrid_search=hybrid_search,
        reranker=reranker,
        citation_builder=CitationBuilder(),
        answer_generator=AnswerGenerator(container.get("llm")),
        response_builder=ResponseBuilder(),
    )
