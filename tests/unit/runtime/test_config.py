from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ragms.runtime.config import load_settings


def test_load_settings_parses_yaml_and_normalizes_paths() -> None:
    settings = load_settings()

    assert settings.app.name == "ragms"
    assert settings.models.llm.provider == "openai"
    assert settings.storage.sqlite.path.name == "ragms.db"
    assert settings.storage.sqlite.path.is_absolute()
    assert settings.storage.chroma.path.is_absolute()


def test_load_settings_env_overrides_api_key_and_storage_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-env-file\n", encoding="utf-8")

    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        """
app:
  name: ragms
  env: test
  log_level: DEBUG
  default_collection: sandbox
runtime:
  settings_file: settings.yaml
  env_file: .env
  fail_fast_on_invalid_config: true
mcp:
  transport: stdio
  server_name: ragms-mcp-server
  tools: [query_knowledge_hub]
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
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
  embedding:
    dense:
      provider: openai
      model: text-embedding-3-large
      api_key_env: OPENAI_API_KEY
      batch_size: 16
    sparse:
      provider: bm25
      tokenizer: default
  reranker:
    mode: none
    enabled: false
storage:
  sqlite:
    path: data/metadata/test.db
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
  title: Test Dashboard
  auto_refresh: false
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("RAGMS_STORAGE_SQLITE_PATH", "runtime/sqlite/override.db")
    monkeypatch.setenv("OPENAI_API_KEY", "from-process-env")

    settings = load_settings(settings_file)

    assert settings.models.llm.api_key == "from-process-env"
    assert settings.models.embedding.dense.api_key == "from-process-env"
    assert settings.storage.sqlite.path == (Path.cwd() / "runtime/sqlite/override.db").resolve()


def test_load_settings_fails_fast_on_invalid_provider(
    tmp_path: Path,
) -> None:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
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
    model: gpt-4.1-mini
  transform_llm:
    provider: openai
    model: gpt-4.1-mini
  vision_llm:
    provider: openai
    model: gpt-4.1-mini
  embedding:
    dense:
      provider: openai
      model: text-embedding-3-large
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

    with pytest.raises(ValidationError):
        load_settings(settings_file)

