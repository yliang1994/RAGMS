from __future__ import annotations

import pytest

from ragms.libs.factories.splitter_factory import SplitterFactory
from ragms.libs.providers.splitters.recursive_character_splitter import (
    RecursiveCharacterSplitter,
)
from ragms.runtime.exceptions import RagMSError


def test_splitter_factory_returns_default_recursive_splitter() -> None:
    splitter = SplitterFactory.create()

    assert isinstance(splitter, RecursiveCharacterSplitter)


def test_splitter_factory_accepts_explicit_provider_config() -> None:
    splitter = SplitterFactory.create(
        {"provider": "recursive_character", "chunk_size": 128, "chunk_overlap": 16}
    )

    assert isinstance(splitter, RecursiveCharacterSplitter)
    assert splitter.default_chunk_size == 128
    assert splitter.default_chunk_overlap == 16


def test_splitter_factory_rejects_unknown_provider() -> None:
    with pytest.raises(RagMSError, match="Unknown splitter provider: unknown"):
        SplitterFactory.create({"provider": "unknown"})
