from __future__ import annotations

import pytest

from ragms.libs.factories.splitter_factory import SplitterFactory
from ragms.libs.providers.splitters.recursive_character_splitter import RecursiveCharacterSplitter


def test_splitter_factory_creates_default_splitter() -> None:
    splitter = SplitterFactory.create(
        {
            "provider": "recursive_character",
            "chunk_size": 128,
            "chunk_overlap": 16,
            "separators": ["\n\n", "\n", " ", ""],
        }
    )

    assert isinstance(splitter, RecursiveCharacterSplitter)
    assert splitter.chunk_size == 128
    assert splitter.chunk_overlap == 16


def test_splitter_factory_raises_on_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown splitter provider"):
        SplitterFactory.create({"provider": "missing"})

