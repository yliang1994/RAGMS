"""Factory for loader provider instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragms.libs.abstractions import BaseLoader
from ragms.libs.providers.loaders.markitdown_loader import MarkItDownLoader
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings


def _coerce_loader_config(config: AppSettings | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None or isinstance(config, AppSettings):
        return {}
    return dict(config)


class LoaderFactory:
    """Create loader implementations from provider configuration."""

    _PROVIDERS = {
        "markitdown": MarkItDownLoader,
    }

    @staticmethod
    def create(config: AppSettings | Mapping[str, Any] | None = None) -> BaseLoader:
        """Return a loader implementation for the configured provider."""

        options = _coerce_loader_config(config)
        provider = str(options.pop("provider", "markitdown")).strip().lower() or "markitdown"
        provider_class = LoaderFactory._PROVIDERS.get(provider)
        if provider_class is None:
            raise RagMSError(f"Unknown loader provider: {provider}")
        return provider_class(**options)
