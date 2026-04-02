from __future__ import annotations

import pytest

from ragms.libs.factories.vector_store_factory import VectorStoreFactory
from ragms.libs.providers.vector_stores.chroma_store import ChromaStore
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings, VectorStoreSettings


def test_vector_store_factory_returns_default_chroma_store() -> None:
    store = VectorStoreFactory.create()

    assert isinstance(store, ChromaStore)
    assert store.collection == "default"


def test_vector_store_factory_reads_backend_from_app_settings() -> None:
    settings = AppSettings(vector_store=VectorStoreSettings(backend="chroma", collection="docs"))

    store = VectorStoreFactory.create(settings)

    assert isinstance(store, ChromaStore)
    assert store.collection == "docs"


def test_vector_store_factory_rejects_unknown_backend() -> None:
    with pytest.raises(RagMSError, match="Unknown vector store provider: memory"):
        VectorStoreFactory.create({"backend": "memory"})
