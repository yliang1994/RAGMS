from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragms.runtime.config import load_settings


def write_settings(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_load_settings_parses_yaml_and_normalizes_paths(tmp_path: Path) -> None:
    settings_path = write_settings(
        tmp_path / "settings.yaml",
        """
        app_name: ragms
        environment: development
        paths:
          project_root: .
          data_dir: data
          logs_dir: logs
        llm:
          provider: openai
          model: gpt-4.1-mini
        embedding:
          provider: openai
          model: text-embedding-3-small
        vector_store:
          backend: chroma
          collection: default
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

    settings = load_settings(settings_path)

    assert settings.llm.provider == "openai"
    assert settings.paths.project_root == tmp_path.resolve()
    assert settings.paths.data_dir == (tmp_path / "data").resolve()
    assert settings.observability.log_file == (tmp_path / "logs/traces.jsonl").resolve()
    assert settings.dashboard.traces_file == (tmp_path / "logs/traces.jsonl").resolve()


def test_environment_variables_override_keys_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_path = write_settings(
        tmp_path / "settings.yaml",
        """
        llm:
          provider: openai
          model: gpt-4.1-mini
          api_key: null
        embedding:
          provider: openai
          model: text-embedding-3-small
        vector_store:
          backend: chroma
          collection: default
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
    monkeypatch.setenv("RAGMS_LLM__API_KEY", "secret-key")
    monkeypatch.setenv("RAGMS_PATHS__DATA_DIR", "custom-data")

    settings = load_settings(settings_path)

    assert settings.llm.api_key == "secret-key"
    assert settings.paths.data_dir == (tmp_path / "custom-data").resolve()


def test_invalid_provider_fails_fast(tmp_path: Path) -> None:
    settings_path = write_settings(
        tmp_path / "settings.yaml",
        """
        llm:
          provider: invalid-provider
          model: gpt-4.1-mini
        embedding:
          provider: openai
          model: text-embedding-3-small
        vector_store:
          backend: chroma
          collection: default
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

    with pytest.raises(ValidationError):
        load_settings(settings_path)
