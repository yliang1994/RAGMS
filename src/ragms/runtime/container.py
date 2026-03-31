from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        return ServiceContainer(
            settings=settings,
            mcp_server=_build_mcp_server(settings),
            query_engine=_build_query_engine(settings),
            ingestion_pipeline=_build_ingestion_pipeline(settings),
            trace_manager=_build_trace_manager(settings),
        )
    except DependencyAssemblyError:
        raise
    except Exception as exc:
        raise DependencyAssemblyError("Failed to assemble runtime dependencies") from exc


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


def _build_query_engine(settings: AppSettings) -> RuntimeComponent:
    return RuntimeComponent(
        name="query_engine",
        implementation="ragms.core.query_engine.engine:QueryEngine",
        config={
            "default_collection": settings.app.default_collection,
            "llm_provider": settings.models.llm.provider,
            "reranker_mode": settings.models.reranker.mode,
        },
    )


def _build_ingestion_pipeline(settings: AppSettings) -> RuntimeComponent:
    return RuntimeComponent(
        name="ingestion_pipeline",
        implementation="ragms.ingestion_pipeline.pipeline:IngestionPipeline",
        config={
            "dense_embedding_provider": settings.models.embedding.dense.provider,
            "sparse_embedding_provider": settings.models.embedding.sparse.provider,
            "chroma_path": str(settings.storage.chroma.path),
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

