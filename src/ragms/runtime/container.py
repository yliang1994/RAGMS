"""Runtime dependency container with placeholder service assembly for Stage A3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_settings
from .exceptions import RuntimeAssemblyError, ServiceNotFoundError
from .settings_models import AppSettings


@dataclass(frozen=True)
class PlaceholderService:
    """Placeholder runtime dependency built from configuration metadata."""

    name: str
    implementation: str
    config: dict[str, Any]


@dataclass
class ServiceContainer:
    """Simple runtime registry for lazily assembled core placeholders."""

    settings: AppSettings
    services: dict[str, Any]

    def get(self, service_name: str) -> Any:
        """Return a named service or raise a unified lookup exception."""

        if service_name not in self.services:
            raise ServiceNotFoundError(f"Unknown service requested: {service_name}")
        return self.services[service_name]


def _build_placeholder_services(settings: AppSettings) -> dict[str, Any]:
    return {
        "settings": settings,
        "llm": PlaceholderService(
            name="llm",
            implementation=settings.llm.provider,
            config=settings.llm.model_dump(mode="python"),
        ),
        "embedding": PlaceholderService(
            name="embedding",
            implementation=settings.embedding.provider,
            config=settings.embedding.model_dump(mode="python"),
        ),
        "vector_store": PlaceholderService(
            name="vector_store",
            implementation=settings.vector_store.backend,
            config=settings.vector_store.model_dump(mode="python"),
        ),
        "retrieval": PlaceholderService(
            name="retrieval",
            implementation=settings.retrieval.strategy,
            config=settings.retrieval.model_dump(mode="python"),
        ),
        "evaluation": PlaceholderService(
            name="evaluation",
            implementation="composite",
            config=settings.evaluation.model_dump(mode="python"),
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
        services = _build_placeholder_services(resolved_settings)
        return ServiceContainer(settings=resolved_settings, services=services)
    except Exception as exc:  # pragma: no cover - unified error boundary
        if isinstance(exc, RuntimeAssemblyError):
            raise
        raise RuntimeAssemblyError(f"Failed to assemble runtime container: {exc}") from exc

