from __future__ import annotations

from pathlib import Path

import pytest

from ragms import get_project_root
from tests.fakes.fake_embedding import FakeEmbedding
from tests.fakes.fake_evaluator import FakeEvaluator
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_reranker import FakeReranker
from tests.fakes.fake_vector_store import FakeVectorStore
from tests.fakes.fake_vision_llm import FakeVisionLLM


@pytest.fixture(scope="session")
def project_root() -> Path:
    return get_project_root()


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def fake_vision_llm() -> FakeVisionLLM:
    return FakeVisionLLM()


@pytest.fixture()
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture()
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture()
def fake_evaluator() -> FakeEvaluator:
    return FakeEvaluator()


@pytest.fixture()
def sample_documents() -> list[dict[str, object]]:
    return [
        {
            "id": "doc-1",
            "text": "RAGMS uses local-first modular retrieval.",
            "metadata": {"source": "spec", "lang": "en"},
        },
        {
            "id": "doc-2",
            "text": "Qwen provider can be configured without code changes.",
            "metadata": {"source": "notes", "lang": "en"},
        },
    ]

