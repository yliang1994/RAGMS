"""Factory for splitter provider instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragms.libs.abstractions import BaseSplitter
from ragms.libs.providers.splitters.recursive_character_splitter import (
    RecursiveCharacterSplitter,
)
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings


def _coerce_splitter_config(config: AppSettings | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None or isinstance(config, AppSettings):
        return {}
    return dict(config)


class SplitterFactory:
    """Create splitter implementations from provider configuration."""

    _PROVIDERS = {
        "recursive_character": RecursiveCharacterSplitter,
        "recursive_character_text_splitter": RecursiveCharacterSplitter,
    }

    @staticmethod
    def create(config: AppSettings | Mapping[str, Any] | None = None) -> BaseSplitter:
        """Return a splitter implementation for the configured provider."""

        options = _coerce_splitter_config(config)
        provider = (
            str(options.pop("provider", "recursive_character")).strip().lower()
            or "recursive_character"
        )
        provider_class = SplitterFactory._PROVIDERS.get(provider)
        if provider_class is None:
            raise RagMSError(f"Unknown splitter provider: {provider}")
        return provider_class(**options)
