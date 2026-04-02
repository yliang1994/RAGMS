from __future__ import annotations

import pytest

from ragms.libs.factories.loader_factory import LoaderFactory
from ragms.libs.providers.loaders.markitdown_loader import MarkItDownLoader
from ragms.runtime.exceptions import RagMSError


def test_loader_factory_returns_default_markitdown_loader() -> None:
    loader = LoaderFactory.create()

    assert isinstance(loader, MarkItDownLoader)


def test_loader_factory_accepts_explicit_provider_config() -> None:
    loader = LoaderFactory.create({"provider": "markitdown", "encoding": "utf-8"})

    assert isinstance(loader, MarkItDownLoader)
    assert loader.encoding == "utf-8"


def test_loader_factory_rejects_unknown_provider() -> None:
    with pytest.raises(RagMSError, match="Unknown loader provider: unknown"):
        LoaderFactory.create({"provider": "unknown"})
