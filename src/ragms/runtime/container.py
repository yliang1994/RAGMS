from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragms.libs.factories import (
    EmbeddingFactory,
    EvaluatorFactory,
    LLMFactory,
    LoaderFactory,
    RerankerFactory,
    SplitterFactory,
    VectorStoreFactory,
    VisionLLMFactory,
)
from ragms.runtime.config import load_settings
from ragms.runtime.exceptions import DependencyAssemblyError
from ragms.runtime.settings_models import AppSettings


@dataclass(slots=True)
class RuntimeComponent:
    name: str
    implementation: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServiceContainer:
    settings: AppSettings
    loader: Any
    splitter: Any
    llm: Any
    vision_llm: Any
    embedding: Any
    reranker: Any
    vector_store: Any
    evaluator: Any
    mcp_server: RuntimeComponent
    query_engine: RuntimeComponent
    ingestion_pipeline: RuntimeComponent
    trace_manager: RuntimeComponent

    def get(self, service_name: str) -> Any:
        try:
            return getattr(self, service_name)
        except AttributeError as exc:
            raise KeyError(f"Unknown service: {service_name}") from exc


def build_container(settings_path: str | Path | None = None) -> ServiceContainer:
    try:
        settings = load_settings(settings_path)
        loader = _build_loader(settings)
        splitter = _build_splitter(settings)
        llm = _build_llm(settings)
        vision_llm = _build_vision_llm(settings)
        embedding = _build_embedding(settings)
        reranker = _build_reranker(settings)
        vector_store = _build_vector_store(settings)
        evaluator = _build_evaluator(settings)
        return ServiceContainer(
            settings=settings,
            loader=loader,
            splitter=splitter,
            llm=llm,
            vision_llm=vision_llm,
            embedding=embedding,
            reranker=reranker,
            vector_store=vector_store,
            evaluator=evaluator,
            mcp_server=_build_mcp_server(settings),
            query_engine=_build_query_engine(settings, llm=llm, reranker=reranker),
            ingestion_pipeline=_build_ingestion_pipeline(settings, loader=loader, splitter=splitter, embedding=embedding, vector_store=vector_store),
            trace_manager=_build_trace_manager(settings),
        )
    except DependencyAssemblyError:
        raise
    except Exception as exc:
        raise DependencyAssemblyError("Failed to assemble runtime dependencies") from exc


def _build_loader(settings: AppSettings) -> Any:
    return LoaderFactory.create(
        {
            "provider": "markitdown",
            "extract_images": True,
            "output_format": "markdown",
        }
    )


def _build_splitter(settings: AppSettings) -> Any:
    return SplitterFactory.create(
        {
            "provider": "recursive_character",
            "chunk_size": 900,
            "chunk_overlap": 150,
            "separators": ["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
        }
    )


def _build_llm(settings: AppSettings) -> Any:
    return LLMFactory.create(settings.models.llm.model_dump())


def _build_vision_llm(settings: AppSettings) -> Any:
    return VisionLLMFactory.create(settings.models.vision_llm.model_dump())


def _build_embedding(settings: AppSettings) -> Any:
    return EmbeddingFactory.create(settings.models.embedding.dense.model_dump())


def _build_reranker(settings: AppSettings) -> Any:
    return RerankerFactory.create(settings.models.reranker.model_dump())


def _build_vector_store(settings: AppSettings) -> Any:
    return VectorStoreFactory.create(
        {
            "backend": "chroma",
            "path": settings.storage.chroma.path,
            "collection_prefix": settings.storage.chroma.collection_prefix,
        }
    )


def _build_evaluator(settings: AppSettings) -> Any:
    return EvaluatorFactory.create(
        {
            "type": "custom_metrics",
            "metrics": ["hit_rate", "mrr"],
        }
    )


def _build_mcp_server(settings: AppSettings) -> RuntimeComponent:
    return RuntimeComponent(
        name="mcp_server",
        implementation="ragms.mcp_server.server:bootstrap_server",
        config={
            "transport": settings.mcp.transport,
            "server_name": settings.mcp.server_name,
            "tools": list(settings.mcp.tools),
        },
    )


def _build_query_engine(settings: AppSettings, *, llm: Any, reranker: Any) -> RuntimeComponent:
    return RuntimeComponent(
        name="query_engine",
        implementation="ragms.core.query_engine.engine:QueryEngine",
        config={
            "default_collection": settings.app.default_collection,
            "llm_provider": settings.models.llm.provider,
            "reranker_mode": settings.models.reranker.mode,
            "llm": llm,
            "reranker": reranker,
        },
    )


def _build_ingestion_pipeline(settings: AppSettings, *, loader: Any, splitter: Any, embedding: Any, vector_store: Any) -> RuntimeComponent:
    return RuntimeComponent(
        name="ingestion_pipeline",
        implementation="ragms.ingestion_pipeline.pipeline:IngestionPipeline",
        config={
            "dense_embedding_provider": settings.models.embedding.dense.provider,
            "sparse_embedding_provider": settings.models.embedding.sparse.provider,
            "chroma_path": str(settings.storage.chroma.path),
            "loader": loader,
            "splitter": splitter,
            "embedding": embedding,
            "vector_store": vector_store,
        },
    )


def _build_trace_manager(settings: AppSettings) -> RuntimeComponent:
    return RuntimeComponent(
        name="trace_manager",
        implementation="ragms.core.trace_collector.trace_manager:TraceManager",
        config={
            "trace_file": str(settings.storage.traces.file),
            "app_log_dir": str(settings.storage.app_logs.dir),
        },
    )
