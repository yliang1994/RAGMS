from __future__ import annotations

from pathlib import Path

import pytest

from ragms.runtime.container import build_container
from ragms.runtime.exceptions import DependencyAssemblyError


def test_build_container_wires_stage_b_default_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    container = build_container()

    assert container.loader.__class__.__name__ == "MarkItDownLoader"
    assert container.splitter.__class__.__name__ == "RecursiveCharacterSplitter"
    assert container.llm.__class__.__name__ == "OpenAILLM"
    assert container.vision_llm.__class__.__name__ == "GPT4OVisionLLM"
    assert container.embedding.__class__.__name__ == "OpenAIEmbedding"
    assert container.reranker.__class__.__name__ == "CrossEncoderReranker"
    assert container.vector_store.__class__.__name__ == "ChromaStore"
    assert container.evaluator.__class__.__name__ == "CustomMetricsEvaluator"
    assert container.query_engine.config["llm"] is container.llm
    assert container.ingestion_pipeline.config["vector_store"] is container.vector_store


def test_build_container_wraps_invalid_provider_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        """
app:
  name: ragms
  env: local
  log_level: INFO
  default_collection: knowledge_hub
runtime:
  settings_file: settings.yaml
  env_file: .env
  fail_fast_on_invalid_config: true
mcp:
  transport: stdio
  server_name: ragms-mcp-server
  tools: []
models:
  llm:
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
  transform_llm:
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
  vision_llm:
    provider: missing
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
  embedding:
    dense:
      provider: openai
      model: text-embedding-3-large
      api_key_env: OPENAI_API_KEY
      batch_size: 64
    sparse:
      provider: bm25
      tokenizer: default
  reranker:
    mode: cross_encoder
    model: BAAI/bge-reranker-base
    enabled: true
storage:
  sqlite:
    path: data/metadata/ragms.db
  chroma:
    path: data/vector_store/chroma
    collection_prefix: ragms_
  bm25:
    index_dir: data/indexes/sparse
  images:
    dir: data/images
  traces:
    file: logs/traces.jsonl
  app_logs:
    dir: logs/app
dashboard:
  enabled: true
  title: RAGMS Local Dashboard
  auto_refresh: true
""",
        encoding="utf-8",
    )

    with pytest.raises(DependencyAssemblyError):
        build_container(settings_path)
