from __future__ import annotations

from pathlib import Path

from mcp import types

from ragms.mcp_server.tools import query as query_tool_module
from ragms.mcp_server.tools.query import handle_query_knowledge_hub
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings


class StubQueryEngine:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        return dict(self.payload)


def _runtime(tmp_path: Path) -> ServiceContainer:
    settings = AppSettings.model_validate({"environment": "test"})
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.storage.sqlite.path = tmp_path / "data" / "metadata" / "ragms.db"
    return ServiceContainer(settings=settings, services={})


def test_handle_query_knowledge_hub_wraps_query_engine_payload(tmp_path: Path) -> None:
    engine = StubQueryEngine(
        {
            "answer": "RAG combines retrieval and generation [1].",
            "markdown": "RAG combines retrieval and generation [1].\n\nSources:\n- [1] docs/rag.pdf",
            "content": [types.TextContent(type="text", text="RAG combines retrieval and generation [1].")],
            "structured_content": {
                "citations": [{"index": 1, "chunk_id": "chunk-1"}],
                "retrieved_chunks": [{"chunk_id": "chunk-1"}],
                "trace_id": "trace-1",
            },
            "debug_info": {"reranker": "disabled"},
        }
    )

    result = handle_query_knowledge_hub(
        query="what is rag",
        collection="docs",
        top_k=3,
        filters={"owner": "team-a"},
        return_debug=True,
        runtime=_runtime(tmp_path),
        query_engine=engine,
    )

    assert engine.calls[0]["query"] == "what is rag"
    assert engine.calls[0]["collection"] == "docs"
    assert engine.calls[0]["top_k"] == 3
    assert engine.calls[0]["filters"] == {"owner": "team-a"}
    assert result.isError is False
    assert result.structuredContent["trace_id"] == "trace-1"
    assert result.structuredContent["debug"] == {"reranker": "disabled"}


def test_handle_query_knowledge_hub_returns_error_wrapper_for_failures(tmp_path: Path) -> None:
    class FailingQueryEngine:
        def run(self, **kwargs):
            raise RuntimeError("boom")

    result = handle_query_knowledge_hub(
        query="what is rag",
        runtime=_runtime(tmp_path),
        query_engine=FailingQueryEngine(),
    )

    assert result.isError is True
    assert result.structuredContent["error"]["code"] == -32603
    assert result.content[0].text == "Internal error"


def test_handle_query_knowledge_hub_rebinds_runtime_for_collection_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime(tmp_path)
    engine = StubQueryEngine(
        {
            "answer": "matched",
            "markdown": "matched",
            "structured_content": {
                "citations": [],
                "retrieved_chunks": [{"chunk_id": "chunk-1", "metadata": {}}],
                "trace_id": "trace-override",
            },
        }
    )
    build_container_calls: list[str] = []
    build_query_engine_calls: list[str] = []
    response_builder = type(
        "StubResponseBuilder",
        (),
        {
            "build_multimodal_contents": staticmethod(
                lambda *, markdown, retrieved_chunks: [types.TextContent(type="text", text=markdown)]
            )
        },
    )()

    def fake_build_container(*, settings):
        build_container_calls.append(settings.vector_store.collection)
        return ServiceContainer(settings=settings, services={})

    def fake_build_query_engine(runtime_arg, *, settings=None):
        build_query_engine_calls.append(runtime_arg.settings.vector_store.collection)
        assert settings is not None
        assert settings.vector_store.collection == "docs"
        return engine

    monkeypatch.setattr(query_tool_module, "build_container", fake_build_container)
    monkeypatch.setattr(query_tool_module, "build_query_engine", fake_build_query_engine)

    result = handle_query_knowledge_hub(
        query="what is rag",
        collection="docs",
        runtime=runtime,
        response_builder=response_builder,
    )

    assert build_container_calls == ["docs"]
    assert build_query_engine_calls == ["docs"]
    assert engine.calls[0]["collection"] == "docs"
    assert result.isError is False
