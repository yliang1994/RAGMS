from __future__ import annotations

import textwrap
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
from ragms.runtime.container import PlaceholderService, build_container
from ragms.runtime.exceptions import RuntimeAssemblyError
from ragms.runtime.settings_models import AppSettings, EvaluationSettings


def write_settings(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


@pytest.mark.integration
def test_build_container_wires_default_factory_services(tmp_path: Path) -> None:
    settings_path = write_settings(
        tmp_path / "settings.yaml",
        """
        app_name: ragms
        environment: test
        paths:
          project_root: .
          data_dir: data
          logs_dir: logs
        llm:
          provider: openai
          model: gpt-4.1-mini
          api_key: null
        embedding:
          provider: openai
          model: text-embedding-3-small
          api_key: null
        vision_llm:
          provider: auto
          model: gpt-4.1-mini
          language_providers:
            zh: qwen_vl
            en: gpt4o
          environment_providers:
            development: qwen_vl
            test: qwen_vl
            production: gpt4o
        vector_store:
          backend: chroma
          collection: factory-tests
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

    assert isinstance(container.get("loader"), MarkItDownLoader)
    assert container.get("loader").implementation == "markitdown"
    assert isinstance(container.get("splitter"), RecursiveCharacterSplitter)
    assert container.get("splitter").implementation == "recursive_character"
    assert isinstance(container.get("llm"), OpenAILLM)
    assert container.get("llm").implementation == "openai"
    assert isinstance(container.get("vision_llm"), QwenVLLLM)
    assert container.get("vision_llm").implementation == "qwen_vl"
    assert isinstance(container.get("embedding"), OpenAIEmbedding)
    assert container.get("embedding").implementation == "openai"
    assert isinstance(container.get("reranker"), DisabledReranker)
    assert container.get("reranker").implementation == "disabled"
    assert isinstance(container.get("vector_store"), ChromaStore)
    assert container.get("vector_store").implementation == "chroma"
    assert container.get("vector_store").collection == "factory-tests"
    assert container.get("vector_store").persist_directory == str(
        (tmp_path / "data" / "vector_store" / "chroma").resolve()
    )
    assert isinstance(container.get("evaluator"), CustomMetricsEvaluator)
    assert container.get("evaluator").implementation == "custom_metrics"
    assert isinstance(container.get("retrieval"), PlaceholderService)
    assert container.get("retrieval").implementation == "hybrid"


def test_build_container_wraps_invalid_evaluator_configuration() -> None:
    settings = AppSettings(evaluation=EvaluationSettings(backends=["custom"]))

    with pytest.raises(RuntimeAssemblyError, match="Unknown evaluator provider: custom"):
        build_container(settings)


def test_build_container_wraps_missing_evaluator_configuration() -> None:
    settings = AppSettings(evaluation=EvaluationSettings(backends=[""]))

    with pytest.raises(RuntimeAssemblyError, match="Missing evaluator provider in configuration"):
        build_container(settings)
