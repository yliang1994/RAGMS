from __future__ import annotations

from pathlib import Path

import pytest

from ragms.runtime.container import PlaceholderService, ServiceContainer, build_container
from ragms.runtime.exceptions import RuntimeAssemblyError, ServiceNotFoundError
from ragms.runtime.settings_models import AppSettings


def write_settings(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_build_container_returns_placeholder_services_from_explicit_settings() -> None:
    settings = AppSettings()

    container = build_container(settings)

    assert isinstance(container, ServiceContainer)
    llm = container.get("llm")
    assert isinstance(llm, PlaceholderService)
    assert llm.implementation == "openai"
    assert llm.config["model"] == "gpt-4.1-mini"
    assert container.get("vector_store").implementation == "chroma"


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


def test_unknown_service_raises_unified_exception() -> None:
    container = build_container(AppSettings())

    with pytest.raises(ServiceNotFoundError):
        container.get("missing")


def test_assembly_failures_raise_runtime_assembly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_settings: AppSettings) -> dict[str, object]:
        raise ValueError("bad wiring")

    monkeypatch.setattr("ragms.runtime.container._build_placeholder_services", boom)

    with pytest.raises(RuntimeAssemblyError, match="Failed to assemble runtime container"):
        build_container(AppSettings())
