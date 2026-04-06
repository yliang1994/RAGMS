from __future__ import annotations

import pytest

from ragms.libs.factories.embedding_factory import EmbeddingFactory
from ragms.libs.providers.embeddings.openai_embedding import OpenAIEmbedding
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings, EmbeddingSettings


def test_embedding_factory_uses_provider_from_app_settings() -> None:
    settings = AppSettings(
        embedding=EmbeddingSettings(
            provider="openai",
            model="text-embedding-test",
            api_key="secret",
            base_url="https://embeddings.example/v1",
            batch_size=10,
        )
    )

    embedding = EmbeddingFactory.create(settings)

    assert isinstance(embedding, OpenAIEmbedding)
    assert embedding.model == "text-embedding-test"
    assert embedding.api_key == "secret"
    assert embedding.base_url == "https://embeddings.example/v1"


def test_embedding_factory_accepts_qwen_as_openai_compatible_alias() -> None:
    embedding = EmbeddingFactory.create(
        {
            "provider": "qwen",
            "model": "qwen-text-embedding-v1",
            "api_key": "secret",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
    )

    assert isinstance(embedding, OpenAIEmbedding)
    assert embedding.model == "qwen-text-embedding-v1"
    assert embedding.api_key == "secret"
    assert embedding.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_embedding_factory_rejects_missing_provider_in_mapping() -> None:
    with pytest.raises(RagMSError, match="Missing embedding provider in configuration"):
        EmbeddingFactory.create({"model": "text-embedding-test"})


def test_embedding_factory_rejects_unknown_provider() -> None:
    with pytest.raises(RagMSError, match="Unknown embedding provider: custom"):
        EmbeddingFactory.create({"provider": "custom", "model": "text-embedding-test"})
