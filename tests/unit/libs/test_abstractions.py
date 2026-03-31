from __future__ import annotations

import inspect
from abc import ABC

import pytest

from ragms.libs.abstractions import (
    BaseEmbedding,
    BaseEvaluator,
    BaseLLM,
    BaseLoader,
    BaseReranker,
    BaseSplitter,
    BaseTransform,
    BaseVectorStore,
    BaseVisionLLM,
)
from tests.fakes.fake_embedding import FakeEmbedding
from tests.fakes.fake_evaluator import FakeEvaluator
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_reranker import FakeReranker
from tests.fakes.fake_vector_store import FakeVectorStore
from tests.fakes.fake_vision_llm import FakeVisionLLM


@pytest.mark.parametrize(
    ("base_class", "abstract_methods"),
    [
        (BaseLoader, {"load"}),
        (BaseSplitter, {"split"}),
        (BaseTransform, {"transform"}),
        (BaseLLM, {"chat"}),
        (BaseVisionLLM, {"caption"}),
        (BaseEmbedding, {"embed"}),
        (BaseReranker, {"rerank"}),
        (BaseVectorStore, {"upsert", "query"}),
        (BaseEvaluator, {"evaluate"}),
    ],
)
def test_base_classes_define_expected_abstract_methods(base_class: type[ABC], abstract_methods: set[str]) -> None:
    assert inspect.isabstract(base_class)
    assert abstract_methods == set(base_class.__abstractmethods__)


@pytest.mark.parametrize(
    "base_class",
    [
        BaseLoader,
        BaseSplitter,
        BaseTransform,
        BaseLLM,
        BaseVisionLLM,
        BaseEmbedding,
        BaseReranker,
        BaseVectorStore,
        BaseEvaluator,
    ],
)
def test_abstract_base_classes_cannot_be_instantiated(base_class: type[ABC]) -> None:
    with pytest.raises(TypeError):
        base_class()


def test_fake_llm_implements_base_contract(fake_llm: FakeLLM) -> None:
    assert isinstance(fake_llm, BaseLLM)
    assert "messages" in inspect.signature(fake_llm.chat).parameters


def test_fake_vision_llm_implements_base_contract(fake_vision_llm: FakeVisionLLM) -> None:
    assert isinstance(fake_vision_llm, BaseVisionLLM)
    assert "image_ref" in inspect.signature(fake_vision_llm.caption).parameters


def test_fake_embedding_implements_base_contract(fake_embedding: FakeEmbedding) -> None:
    assert isinstance(fake_embedding, BaseEmbedding)
    assert "texts" in inspect.signature(fake_embedding.embed).parameters


def test_fake_reranker_implements_base_contract(fake_reranker: FakeReranker) -> None:
    assert isinstance(fake_reranker, BaseReranker)
    assert "query" in inspect.signature(fake_reranker.rerank).parameters


def test_fake_vector_store_implements_base_contract(fake_vector_store: FakeVectorStore) -> None:
    assert isinstance(fake_vector_store, BaseVectorStore)
    assert "query_text" in inspect.signature(fake_vector_store.query).parameters


def test_fake_evaluator_implements_base_contract(fake_evaluator: FakeEvaluator) -> None:
    assert isinstance(fake_evaluator, BaseEvaluator)
    assert "samples" in inspect.signature(fake_evaluator.evaluate).parameters
