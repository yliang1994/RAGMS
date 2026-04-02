from __future__ import annotations

from pathlib import Path

import pytest

from ragms.libs.providers.embeddings.openai_embedding import OpenAIEmbedding
from ragms.libs.providers.evaluators.custom_metrics_evaluator import CustomMetricsEvaluator
from ragms.libs.providers.llm.openai_llm import OpenAILLM
from ragms.libs.providers.loaders.markitdown_loader import MarkItDownLoader
from ragms.libs.providers.rerankers.disabled_reranker import DisabledReranker
from ragms.libs.providers.splitters.recursive_character_splitter import RecursiveCharacterSplitter
from ragms.libs.providers.vector_stores.chroma_store import ChromaStore
from ragms.libs.providers.vision_llms.qwen_vl_llm import QwenVLLLM
from ragms.runtime.container import PlaceholderService, ServiceContainer, build_container
from ragms.runtime.exceptions import RuntimeAssemblyError, ServiceNotFoundError
from ragms.runtime.settings_models import AppSettings


def write_settings(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_build_container_returns_factory_backed_services_from_explicit_settings(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"

    container = build_container(settings)

    assert isinstance(container, ServiceContainer)
    llm = container.get("llm")
    assert isinstance(llm, OpenAILLM)
    assert llm.implementation == "openai"
    assert llm.config["model"] == "gpt-4.1-mini"
    assert isinstance(container.get("loader"), MarkItDownLoader)
    assert isinstance(container.get("splitter"), RecursiveCharacterSplitter)
    assert isinstance(container.get("vision_llm"), QwenVLLLM)
    assert isinstance(container.get("embedding"), OpenAIEmbedding)
    assert isinstance(container.get("reranker"), DisabledReranker)
    assert isinstance(container.get("vector_store"), ChromaStore)
    assert isinstance(container.get("evaluator"), CustomMetricsEvaluator)
    assert container.get("vector_store").implementation == "chroma"
    assert container.get("retrieval").implementation == "hybrid"


def test_build_container_can_load_settings_from_file(tmp_path: Path) -> None:
    settings_path = write_settings(
        tmp_path / "settings.yaml",
        """
app_name: ragms
environment: development
llm:
  provider: openai
  model: gpt-4.1-mini
embedding:
  provider: openai
  model: text-embedding-3-small
vector_store:
  backend: chroma
  collection: docs
retrieval:
  strategy: hybrid
  fusion_algorithm: rrf
  rerank_backend: disabled
evaluation:
  backends: [custom_metrics]
observability:
  enabled: true
  log_file: logs/traces.jsonl
  log_level: INFO
dashboard:
  enabled: true
  port: 8501
  traces_file: logs/traces.jsonl
        """,
    )

    container = build_container(settings_path=settings_path)

    assert container.settings.vector_store.collection == "docs"
    assert container.get("embedding").implementation == "openai"
    assert container.get("vector_store").collection == "docs"


def test_unknown_service_raises_unified_exception() -> None:
    container = build_container(AppSettings())

    with pytest.raises(ServiceNotFoundError):
        container.get("missing")


def test_assembly_failures_raise_runtime_assembly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_settings: AppSettings) -> dict[str, object]:
        raise ValueError("bad wiring")

    monkeypatch.setattr("ragms.runtime.container._build_services", boom)

    with pytest.raises(RuntimeAssemblyError, match="Failed to assemble runtime container"):
        build_container(AppSettings())
