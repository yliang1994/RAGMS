from __future__ import annotations

import pytest

from ragms.libs.factories.reranker_factory import RerankerFactory
from ragms.libs.providers.rerankers.cross_encoder_reranker import CrossEncoderReranker


def test_reranker_factory_creates_configured_provider() -> None:
    reranker = RerankerFactory.create({"enabled": True, "mode": "cross_encoder", "model": "bge-reranker"})
    assert isinstance(reranker, CrossEncoderReranker)


def test_reranker_factory_returns_none_when_disabled() -> None:
    assert RerankerFactory.create({"enabled": False, "mode": "none"}) is None


def test_reranker_factory_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="requires mode and model"):
        RerankerFactory.create({"enabled": True, "mode": "cross_encoder"})

    with pytest.raises(ValueError, match="Unknown reranker mode"):
        RerankerFactory.create({"enabled": True, "mode": "missing", "model": "demo"})
