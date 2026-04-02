"""Shared pytest configuration, runtime isolation, and fake providers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.fakes import (
    FakeEmbedding,
    FakeEvaluator,
    FakeLLM,
    FakeReranker,
    FakeVectorStore,
    FakeVisionLLM,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class RuntimeLayout:
    """Temporary runtime directories used by tests."""

    runtime_root: Path
    data_dir: Path
    logs_dir: Path


@pytest.fixture
def runtime_layout(tmp_path: Path) -> RuntimeLayout:
    """Create isolated runtime paths for a single test run."""

    runtime_root = tmp_path / "runtime"
    data_dir = runtime_root / "data"
    logs_dir = runtime_root / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return RuntimeLayout(
        runtime_root=runtime_root,
        data_dir=data_dir,
        logs_dir=logs_dir,
    )


@pytest.fixture(autouse=True)
def isolated_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
    runtime_layout: RuntimeLayout,
) -> None:
    """Force tests to use isolated runtime directories."""

    monkeypatch.setenv("RAGMS_RUNTIME_ROOT", str(runtime_layout.runtime_root))
    monkeypatch.setenv("RAGMS_DATA_DIR", str(runtime_layout.data_dir))
    monkeypatch.setenv("RAGMS_LOG_DIR", str(runtime_layout.logs_dir))
    monkeypatch.setenv("RAGMS_TEST_MODE", "1")


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Return a deterministic fake text-generation model."""

    return FakeLLM()


@pytest.fixture
def fake_vision_llm() -> FakeVisionLLM:
    """Return a deterministic fake vision-language model."""

    return FakeVisionLLM()


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    """Return a deterministic fake embedding model."""

    return FakeEmbedding()


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    """Return an in-memory fake vector store."""

    return FakeVectorStore()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    """Return a deterministic fake reranker."""

    return FakeReranker()


@pytest.fixture
def fake_evaluator() -> FakeEvaluator:
    """Return a fake evaluator with stable metrics."""

    return FakeEvaluator()
