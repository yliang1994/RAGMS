from __future__ import annotations

from pathlib import Path

import pytest

from ragms.runtime.container import ServiceContainer, build_container
from ragms.runtime.exceptions import DependencyAssemblyError


def test_build_container_returns_placeholder_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    container = build_container()

    assert isinstance(container, ServiceContainer)
    assert container.settings.app.name == "ragms"
    assert container.mcp_server.config["transport"] == "stdio"
    assert container.query_engine.config["llm_provider"] == "openai"
    assert container.ingestion_pipeline.config["dense_embedding_provider"] == "openai"
    assert container.trace_manager.config["trace_file"].endswith("logs/traces.jsonl")
    assert container.get("mcp_server") is container.mcp_server


def test_build_container_wraps_dependency_assembly_failures(tmp_path: Path) -> None:
    invalid_settings = tmp_path / "settings.yaml"
    invalid_settings.write_text(
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
    provider: invalid-provider
    model: demo
  transform_llm:
    provider: openai
    model: demo
  vision_llm:
    provider: openai
    model: demo
  embedding:
    dense:
      provider: openai
      model: demo
    sparse:
      provider: bm25
      tokenizer: default
  reranker:
    mode: none
    enabled: false
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
        build_container(invalid_settings)


def test_service_container_get_raises_key_error_for_unknown_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    container = build_container()

    with pytest.raises(KeyError):
        container.get("missing_service")
