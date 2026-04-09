"""Runtime dependency container with real factory-backed service assembly."""

from __future__ import annotations

from dataclasses import dataclass
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

from .config import load_settings
from .exceptions import RagMSError, RuntimeAssemblyError, ServiceNotFoundError
from .settings_models import AppSettings


@dataclass(frozen=True)
class PlaceholderService:
    """Metadata-only runtime dependency used for services not yet implemented."""

    name: str
    implementation: str
    config: dict[str, Any]


@dataclass
class ServiceContainer:
    """Simple runtime registry for factory-assembled core services."""

    settings: AppSettings
    services: dict[str, Any]

    def get(self, service_name: str) -> Any:
        """Return a named service or raise a unified lookup exception."""

        if service_name not in self.services:
            raise ServiceNotFoundError(f"Unknown service requested: {service_name}")
        return self.services[service_name]


def _attach_service_metadata(
    instance: Any,
    *,
    name: str,
    implementation: str,
    config: dict[str, Any],
) -> Any:
    """Attach lightweight service metadata for bootstrap scripts and diagnostics."""

    setattr(instance, "name", name)
    setattr(instance, "implementation", implementation)
    setattr(instance, "config", config)
    return instance


def _vector_store_config(settings: AppSettings) -> dict[str, Any]:
    """Build vector store config with a stable default persistence path."""

    options = settings.vector_store.model_dump(mode="python")
    options.setdefault(
        "persist_directory",
        str(settings.paths.data_dir / "vector_store" / "chroma"),
    )
    return options


def _build_services(settings: AppSettings) -> dict[str, Any]:
    """Build the runtime service registry from real provider factories."""

    llm = _attach_service_metadata(
        LLMFactory.create(settings),
        name="llm",
        implementation=str(settings.llm.provider),
        config=settings.llm.model_dump(mode="python"),
    )
    vision_llm = VisionLLMFactory.create(
        settings,
        deployment_environment=settings.environment,
    )
    vision_llm = _attach_service_metadata(
        vision_llm,
        name="vision_llm",
        implementation=str(getattr(vision_llm, "provider_name", settings.vision_llm.provider)),
        config=settings.vision_llm.model_dump(mode="python"),
    )
    embedding = _attach_service_metadata(
        EmbeddingFactory.create(settings),
        name="embedding",
        implementation=str(settings.embedding.provider),
        config=settings.embedding.model_dump(mode="python"),
    )
    vector_store_config = _vector_store_config(settings)
    vector_store = _attach_service_metadata(
        VectorStoreFactory.create(vector_store_config),
        name="vector_store",
        implementation=str(settings.vector_store.backend),
        config=vector_store_config,
    )
    reranker = _attach_service_metadata(
        RerankerFactory.create(settings),
        name="reranker",
        implementation=str(settings.retrieval.rerank_backend),
        config=settings.retrieval.model_dump(mode="python"),
    )
    evaluator = _attach_service_metadata(
        EvaluatorFactory.create(settings),
        name="evaluator",
        implementation=str((settings.evaluation.backends or ["custom_metrics"])[0]),
        config=settings.evaluation.model_dump(mode="python"),
    )
    loader = _attach_service_metadata(
        LoaderFactory.create(),
        name="loader",
        implementation="markitdown",
        config={"provider": "markitdown"},
    )
    splitter = _attach_service_metadata(
        SplitterFactory.create(),
        name="splitter",
        implementation="recursive_character",
        config={"provider": "recursive_character"},
    )

    return {
        "settings": settings,
        "loader": loader,
        "splitter": splitter,
        "llm": llm,
        "vision_llm": vision_llm,
        "embedding": embedding,
        "reranker": reranker,
        "vector_store": vector_store,
        "evaluator": evaluator,
        "retrieval": PlaceholderService(
            name="retrieval",
            implementation=settings.retrieval.strategy,
            config=settings.retrieval.model_dump(mode="python"),
        ),
        "document_admin_service": PlaceholderService(
            name="document_admin_service",
            implementation="pending",
            config={},
        ),
        "report_service": PlaceholderService(
            name="report_service",
            implementation="pending",
            config={},
        ),
    }


def build_container(
    settings: AppSettings | None = None,
    *,
    settings_path: str | Path = "settings.yaml",
) -> ServiceContainer:
    """Build the runtime service container from explicit or file-backed settings."""

    try:
        resolved_settings = settings or load_settings(settings_path)
        services = _build_services(resolved_settings)
        return ServiceContainer(settings=resolved_settings, services=services)
    except RagMSError as exc:
        raise RuntimeAssemblyError(f"Failed to assemble runtime container: {exc}") from exc
    except Exception as exc:  # pragma: no cover - unified error boundary
        if isinstance(exc, RuntimeAssemblyError):
            raise
        raise RuntimeAssemblyError(f"Failed to assemble runtime container: {exc}") from exc


def bootstrap_mcp_runtime(settings_path: str | Path = "settings.yaml") -> ServiceContainer:
    """Build the runtime container used by the MCP server bootstrap."""

    return build_container(settings_path=settings_path)
