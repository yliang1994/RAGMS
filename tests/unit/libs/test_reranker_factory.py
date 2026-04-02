from __future__ import annotations

import pytest

from ragms.libs.factories.reranker_factory import RerankerFactory
from ragms.libs.providers.rerankers.cross_encoder_reranker import CrossEncoderReranker
from ragms.libs.providers.rerankers.disabled_reranker import DisabledReranker
from ragms.libs.providers.rerankers.llm_reranker import LLMReranker
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings


def test_reranker_factory_uses_default_disabled_backend_from_settings() -> None:
    reranker = RerankerFactory.create(AppSettings())

    assert isinstance(reranker, DisabledReranker)


def test_reranker_factory_accepts_cross_encoder_backend() -> None:
    reranker = RerankerFactory.create({"provider": "cross_encoder", "model": "cross-mini"})

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model == "cross-mini"


def test_reranker_factory_accepts_llm_backend() -> None:
    reranker = RerankerFactory.create({"provider": "llm_reranker", "model": "gpt-rerank"})

    assert isinstance(reranker, LLMReranker)
    assert reranker.model == "gpt-rerank"


def test_reranker_factory_rejects_missing_provider_in_mapping() -> None:
    with pytest.raises(RagMSError, match="Missing reranker provider in configuration"):
        RerankerFactory.create({"model": "cross-mini"})


def test_reranker_factory_rejects_unknown_provider() -> None:
    with pytest.raises(RagMSError, match="Unknown reranker provider: custom"):
        RerankerFactory.create({"provider": "custom"})
