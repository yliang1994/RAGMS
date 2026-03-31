from __future__ import annotations

import pytest

from ragms.libs.factories.loader_factory import LoaderFactory
from ragms.libs.providers.loaders.markitdown_loader import MarkItDownLoader


def test_loader_factory_creates_default_loader() -> None:
    loader = LoaderFactory.create(
        {
            "provider": "markitdown",
            "extract_images": False,
            "output_format": "markdown",
        }
    )

    assert isinstance(loader, MarkItDownLoader)
    assert loader.extract_images is False


def test_loader_factory_raises_on_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown loader provider"):
        LoaderFactory.create({"provider": "missing"})

