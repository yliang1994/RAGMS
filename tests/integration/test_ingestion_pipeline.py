from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ragms.libs.providers.loaders.markitdown_loader import MarkItDownLoader
from ragms.libs.providers.splitters.recursive_character_splitter import RecursiveCharacterSplitter
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings
from scripts.ingest_documents import ingest_documents_main
from tests.fakes import FakeEmbedding, FakeVectorStore, FakeVisionLLM


def _write_settings(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
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
              api_key: null
              language_providers:
                zh: qwen_vl
                en: gpt4o
              environment_providers:
                development: qwen_vl
                test: qwen_vl
                production: gpt4o
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
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _service_container(settings: AppSettings) -> ServiceContainer:
    return ServiceContainer(
        settings=settings,
        services={
            "settings": settings,
            "loader": MarkItDownLoader(),
            "splitter": RecursiveCharacterSplitter(chunk_size=64, chunk_overlap=0),
            "embedding": FakeEmbedding(dimension=4),
            "vector_store": FakeVectorStore(),
            "vision_llm": FakeVisionLLM(),
        },
    )


@pytest.mark.integration
def test_ingest_documents_cli_supports_batch_skip_and_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.md").write_text("# Alpha\n\nFirst document body.", encoding="utf-8")
    (docs_dir / "beta.md").write_text("# Beta\n\nSecond document body.", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "scripts.ingest_documents.build_container",
        lambda settings: _service_container(settings),
    )

    first_rc = ingest_documents_main(
        ["--settings", str(settings_path), "--path", str(docs_dir), "--collection", "demo"]
    )
    first_output = capsys.readouterr().out
    second_rc = ingest_documents_main(
        ["--settings", str(settings_path), "--path", str(docs_dir), "--collection", "demo"]
    )
    second_output = capsys.readouterr().out
    third_rc = ingest_documents_main(
        [
            "--settings",
            str(settings_path),
            "--path",
            str(docs_dir),
            "--collection",
            "demo",
            "--force",
        ]
    )
    third_output = capsys.readouterr().out

    assert first_rc == 0
    assert second_rc == 0
    assert third_rc == 0
    assert "stage=store status=completed" in first_output
    assert first_output.count("status=indexed") == 2
    assert second_output.count("status=skipped") == 2
    assert third_output.count("status=indexed") == 2
    assert "force=true" in third_output
