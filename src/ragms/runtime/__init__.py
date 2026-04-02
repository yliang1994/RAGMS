"""Runtime utilities for configuration and container bootstrap."""

from __future__ import annotations

from typing import Any

from .config import load_settings
from .exceptions import RagMSError, RuntimeAssemblyError, ServiceNotFoundError
from .settings_models import AppSettings

__all__ = [
    "AppSettings",
    "RagMSError",
    "RuntimeAssemblyError",
    "ServiceContainer",
    "ServiceNotFoundError",
    "build_container",
    "load_settings",
]


def __getattr__(name: str) -> Any:
    """Lazily expose container symbols to avoid package import cycles."""

    if name in {"ServiceContainer", "build_container"}:
        from .container import ServiceContainer, build_container

        exports = {
            "ServiceContainer": ServiceContainer,
            "build_container": build_container,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
