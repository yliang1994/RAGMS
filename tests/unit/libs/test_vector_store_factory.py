from __future__ import annotations

import pytest

from ragms.libs.factories.vector_store_factory import VectorStoreFactory
from ragms.libs.providers.vector_stores.chroma_store import ChromaStore


def test_vector_store_factory_creates_default_store() -> None:
    vector_store = VectorStoreFactory.create(
        {
            "provider": "chroma",
            "path": "data/vector_store/chroma",
            "collection_prefix": "ragms_",
        }
    )

    assert isinstance(vector_store, ChromaStore)
    assert vector_store.path == "data/vector_store/chroma"


def test_vector_store_factory_supports_backend_alias() -> None:
    vector_store = VectorStoreFactory.create(
        {
            "backend": "chroma",
            "path": "data/vector_store/chroma",
        }
    )

    assert isinstance(vector_store, ChromaStore)


def test_vector_store_factory_raises_on_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown vector store provider"):
        VectorStoreFactory.create({"provider": "missing", "path": "data/vector_store/chroma"})
