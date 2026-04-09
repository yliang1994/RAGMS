"""End-to-end query engine orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ragms.core.query_engine.answer_generator import AnswerGenerationError, AnswerGenerator
from ragms.core.query_engine.citation_builder import CitationBuilder
from ragms.core.query_engine.hybrid_search import HybridSearch
from ragms.core.query_engine.query_processor import QueryProcessor
from ragms.core.query_engine.reranker import Reranker
from ragms.core.query_engine.response_builder import ResponseBuilder
from ragms.core.query_engine.retrievers import DenseRetriever, SparseRetriever
from ragms.core.trace_collector import TraceManager
from ragms.core.trace_collector.trace_schema import BaseTrace
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings
from ragms.storage.indexes import BM25Indexer
from ragms.storage.traces import TraceRepository


def attach_query_trace(
    trace_context: dict[str, Any] | None,
    *,
    trace_id: str,
) -> dict[str, Any]:
    """Attach the active query trace id onto the outgoing response context."""

    attached = dict(trace_context or {})
    attached["trace_id"] = trace_id
    return attached


def record_query_stage(
    trace_manager: TraceManager,
    trace: BaseTrace,
    *,
    stage_name: str,
    input_payload: Any = None,
    metadata: dict[str, Any] | None = None,
    operation: Callable[[], Any],
    output_builder: Callable[[Any], Any] | None = None,
    metadata_builder: Callable[[Any], dict[str, Any]] | None = None,
) -> Any:
    """Execute one query stage and record a stable trace entry."""

    trace_manager.start_stage(
        trace,
        stage_name,
        input_payload=input_payload,
        metadata=metadata,
    )
    try:
        result = operation()
    except Exception as exc:
        trace_manager.finish_stage(
            trace,
            stage_name,
            status="failed",
            output_payload=None,
            metadata=metadata,
            error=exc,
        )
        raise

    stage_metadata = dict(metadata or {})
    if metadata_builder is not None:
        stage_metadata.update(metadata_builder(result))
    trace_manager.finish_stage(
        trace,
        stage_name,
        output_payload=result if output_builder is None else output_builder(result),
        metadata=stage_metadata,
    )
    return result


class QueryEngine:
    """Run the full query pipeline from preprocessing to final response payload."""

    ANSWER_FALLBACK_TEXT = "Answer generation unavailable; inspect the retrieved chunks below."

    def __init__(
        self,
        *,
        query_processor: QueryProcessor,
        hybrid_search: HybridSearch,
        reranker: Reranker,
        citation_builder: CitationBuilder,
        answer_generator: AnswerGenerator,
        response_builder: ResponseBuilder,
        trace_manager: TraceManager | None = None,
        trace_repository: TraceRepository | None = None,
    ) -> None:
        self.query_processor = query_processor
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.citation_builder = citation_builder
        self.answer_generator = answer_generator
        self.response_builder = response_builder
        self.trace_manager = trace_manager or TraceManager()
        self.trace_repository = trace_repository

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

        trace = self.trace_manager.start_trace(
            "query",
            trace_id=str((trace_context or {}).get("trace_id") or "").strip() or None,
            collection=collection or self.query_processor.default_collection,
            metadata={
                "strategy": getattr(self.hybrid_search, "__class__", type("HybridSearch", (), {})).__name__,
                "reranker_backend": self.reranker.backend,
            },
            query=query,
        )

        try:
            processed_query = record_query_stage(
                self.trace_manager,
                trace,
                stage_name="query_processing",
                input_payload={
                    "query": query,
                    "collection": collection,
                    "top_k": top_k,
                    "filters": filters,
                },
                metadata={"method": "normalize_and_extract", "retry_count": 0},
                operation=lambda: self.query_processor.process(
                    query,
                    collection=collection,
                    top_k=top_k,
                    filters=filters,
                ),
            )
            hybrid_result, _ = self.hybrid_search.search_with_trace(
                processed_query,
                stage_callback=lambda stage_name, input_payload, output_payload, metadata: self._record_hybrid_stage(
                    trace,
                    stage_name=stage_name,
                    input_payload=input_payload,
                    output_payload=output_payload,
                    metadata=metadata,
                ),
            )
            reranked_result = record_query_stage(
                self.trace_manager,
                trace,
                stage_name="rerank",
                input_payload={
                    "candidate_count": hybrid_result.fused_count,
                    "top_chunk_ids": [candidate.chunk_id for candidate in hybrid_result.top_candidates(processed_query.top_k)],
                },
                metadata={"backend": self.reranker.backend, "retry_count": 0},
                operation=lambda: self.reranker.run_with_fallback(hybrid_result),
                output_builder=lambda result: {
                    "candidate_count": result.fused_count,
                    "top_chunk_ids": [candidate.chunk_id for candidate in result.top_candidates(processed_query.top_k)],
                },
                metadata_builder=lambda result: dict(result.debug_info.get("reranker") or {}),
            )
        except Exception as exc:
            self._finish_failed_trace(trace, error=exc)
            raise

        final_candidates = list(reranked_result.top_candidates(processed_query.top_k))
        citations = self.citation_builder.build(final_candidates)
        retrieved_chunks = record_query_stage(
            self.trace_manager,
            trace,
            stage_name="response_build",
            input_payload={
                "candidate_count": len(final_candidates),
                "citation_count": len(citations),
            },
            metadata={"retry_count": 0},
            operation=lambda: self.response_builder.serialize_retrieved_chunks(
                final_candidates,
                citations=citations,
            ),
            output_builder=lambda chunks: {
                "retrieved_chunk_count": len(chunks),
                "citation_count": len(citations),
            },
            metadata_builder=lambda chunks: {
                "citation_count": len(citations),
                "image_count": sum(
                    len((chunk.get("metadata") or {}).get("image_refs") or [])
                    for chunk in chunks
                ),
            },
        )
        try:
            answer = record_query_stage(
                self.trace_manager,
                trace,
                stage_name="answer_generation",
                input_payload={
                    "query": processed_query.normalized_query,
                    "candidate_count": len(final_candidates),
                    "citation_count": len(citations),
                },
                metadata={
                    "provider": getattr(self.answer_generator.llm, "implementation", self.answer_generator.llm.__class__.__name__),
                    "model": getattr(self.answer_generator.llm, "model", None),
                    "retry_count": 0,
                },
                operation=lambda: self.answer_generator.generate(
                    query=processed_query.normalized_query,
                    candidates=final_candidates,
                    citations=citations,
                ),
                output_builder=lambda generated_answer: {"answer_preview": generated_answer},
            )
        except AnswerGenerationError:
            answer = self.ANSWER_FALLBACK_TEXT
        payload = self.response_builder.build(
            query=processed_query.normalized_query,
            answer=answer,
            result=reranked_result,
            citations=citations,
            retrieved_candidates=final_candidates,
            trace_context=attach_query_trace(trace_context, trace_id=trace.trace_id),
            return_debug=return_debug,
        )
        payload["retrieved_chunks"] = retrieved_chunks
        payload["structured_content"]["retrieved_chunks"] = retrieved_chunks
        payload["content"] = self.response_builder.build_multimodal_contents(
            markdown=payload["markdown"],
            retrieved_chunks=retrieved_chunks,
        )
        self._finish_succeeded_trace(
            trace,
            processed_query=processed_query,
            final_candidates=final_candidates,
        )
        return payload

    def _record_hybrid_stage(
        self,
        trace: BaseTrace,
        *,
        stage_name: str,
        input_payload: Any,
        output_payload: Any,
        metadata: dict[str, Any],
    ) -> None:
        self.trace_manager.start_stage(
            trace,
            stage_name,
            input_payload=input_payload,
            metadata=metadata,
        )
        self.trace_manager.finish_stage(
            trace,
            stage_name,
            output_payload=output_payload,
            metadata=metadata,
        )

    def _finish_succeeded_trace(
        self,
        trace: BaseTrace,
        *,
        processed_query: Any,
        final_candidates: list[Any],
    ) -> None:
        finished_trace = self.trace_manager.finish_trace(
            trace,
            status="succeeded",
            collection=processed_query.collection,
            top_k_results=[candidate.chunk_id for candidate in final_candidates],
        )
        if self.trace_repository is not None:
            self.trace_repository.append(finished_trace)

    def _finish_failed_trace(self, trace: BaseTrace, *, error: BaseException) -> None:
        finished_trace = self.trace_manager.finish_trace(
            trace,
            status="failed",
            error=error,
        )
        if self.trace_repository is not None:
            self.trace_repository.append(finished_trace)


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
        trace_repository=TraceRepository(resolved_settings.observability.log_file),
    )
