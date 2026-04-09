from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.query_cli import run_cli
from ragms.core.query_engine import build_query_engine
from ragms.ingestion_pipeline.storage import ChunkRecord
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings
from ragms.storage.indexes import BM25Indexer
from tests.fakes import FakeLLM


class FilterableVectorStore:
    def __init__(self) -> None:
        self._items: list[dict[str, object]] = []

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        documents: list[str] | None = None,
        metadatas: list[dict[str, object]] | None = None,
    ) -> list[str]:
        documents = documents or [""] * len(ids)
        metadatas = metadatas or [{} for _ in ids]
        for item_id, vector, document, metadata in zip(ids, vectors, documents, metadatas, strict=True):
            self._items.append(
                {
                    "id": item_id,
                    "vector": list(vector),
                    "document": document,
                    "metadata": dict(metadata),
                }
            )
        return ids

    def query(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        filters = dict(filters or {})
        requested_collection = filters.pop("collection", None)
        matches = []
        for item in self._items:
            metadata = dict(item["metadata"])
            if requested_collection is not None and metadata.get("collection") != requested_collection:
                continue
            if any(metadata.get(key) != value for key, value in filters.items()):
                continue
            score = sum(a * b for a, b in zip(query_vector, item["vector"], strict=False))
            matches.append(
                {
                    "id": item["id"],
                    "score": score,
                    "document": item["document"],
                    "metadata": metadata,
                }
            )
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:top_k]


class FailingRerankerProvider:
    def rerank(self, query, candidates, *, top_k=None):
        raise TimeoutError("provider timeout")


class QueryAwareEmbedding:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        normalized = text.lower()
        if "rag" in normalized:
            return [1.0, 0.0, 0.0, 0.0]
        if "dense" in normalized or "embedding" in normalized:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]


def _record(
    *,
    chunk_id: str,
    document_id: str,
    content: str,
    source_path: str,
    metadata: dict[str, object],
    tokens: list[str],
    term_frequencies: dict[str, int],
) -> ChunkRecord:
    document_length = sum(term_frequencies.values())
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        metadata=metadata,
        dense_vector=[0.0, 0.0],
        sparse_vector={
            "tokens": tokens,
            "term_frequencies": term_frequencies,
            "term_weights": {
                token: round(count / document_length, 6)
                for token, count in term_frequencies.items()
            },
            "document_length": document_length,
            "unique_terms": len(term_frequencies),
        },
        content_hash=f"hash-{chunk_id}",
        source_path=source_path,
        chunk_index=int(metadata.get("chunk_index", 0) or 0),
        image_refs=[],
    )


def _build_runtime(tmp_path: Path, *, reranker_provider, llm: FakeLLM) -> tuple[AppSettings, ServiceContainer]:
    settings = AppSettings()
    settings = settings.model_copy(deep=True)
    settings.paths.project_root = tmp_path
    settings.paths.data_dir = tmp_path / "data"
    settings.paths.logs_dir = tmp_path / "logs"
    settings.vector_store.collection = "docs"
    settings.retrieval.rerank_backend = "llm_reranker" if reranker_provider else "disabled"
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.observability.log_file = settings.paths.logs_dir / "traces.jsonl"
    settings.dashboard.traces_file = settings.paths.logs_dir / "traces.jsonl"

    embedding = QueryAwareEmbedding()
    vector_store = FilterableVectorStore()
    dense_docs = [
        (
            "chunk-1",
            "RAG combines retrieval and generation.",
            {
                "document_id": "doc-1",
                "collection": "docs",
                "doc_type": "pdf",
                "owner": "team-a",
                "source_path": "docs/rag.pdf",
                "page": 2,
            },
        ),
        (
            "chunk-2",
            "Dense retrieval uses embeddings.",
            {
                "document_id": "doc-2",
                "collection": "docs",
                "doc_type": "pdf",
                "owner": "team-b",
                "source_path": "docs/dense.pdf",
                "page": 1,
            },
        ),
    ]
    ids = [item[0] for item in dense_docs]
    texts = [item[1] for item in dense_docs]
    metadatas = [item[2] for item in dense_docs]
    vectors = embedding.embed_documents(texts)
    vector_store.add(ids, vectors, documents=texts, metadatas=metadatas)

    indexer = BM25Indexer(index_dir=settings.paths.data_dir / "indexes" / "sparse", collection="docs")
    indexer.index_document(
        _record(
            chunk_id="chunk-1",
            document_id="doc-1",
            content=texts[0],
            source_path="docs/rag.pdf",
            metadata=metadatas[0],
            tokens=["rag", "retrieval", "generation"],
            term_frequencies={"rag": 1, "retrieval": 1, "generation": 1},
        )
    )
    indexer.index_document(
        _record(
            chunk_id="chunk-2",
            document_id="doc-2",
            content=texts[1],
            source_path="docs/dense.pdf",
            metadata=metadatas[1],
            tokens=["dense", "retrieval", "embeddings"],
            term_frequencies={"dense": 1, "retrieval": 1, "embeddings": 1},
        )
    )

    container = ServiceContainer(
        settings=settings,
        services={
            "llm": llm,
            "embedding": embedding,
            "vector_store": vector_store,
            "reranker": reranker_provider,
        },
    )
    return settings, container


@pytest.mark.integration
def test_query_engine_runs_end_to_end_and_cli_uses_real_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm = FakeLLM(
        [
            "RAG combines retrieval and generation [1].",
            "RAG combines retrieval and generation [1].",
        ]
    )
    settings, container = _build_runtime(tmp_path, reranker_provider=None, llm=llm)
    engine = build_query_engine(container, settings=settings)

    response = engine.run(
        query="what is rag",
        top_k=1,
        filters={"doc_type": "pdf"},
        return_debug=True,
        trace_context={"trace_id": "trace-query-1"},
    )

    assert response["answer"] == "RAG combines retrieval and generation [1]."
    assert response["citations"][0]["document_id"] == "doc-1"
    assert response["retrieved_chunks"][0]["chunk_id"] == "chunk-1"
    assert response["trace_id"] == "trace-query-1"

    monkeypatch.setattr("scripts.query_cli.load_settings", lambda path: settings)
    monkeypatch.setattr("scripts.query_cli.build_container", lambda loaded: container)
    monkeypatch.setattr(
        "scripts.query_cli.build_query_engine",
        lambda built_container, settings=None: build_query_engine(built_container, settings=settings),
    )

    assert (
        run_cli(
            [
                "--settings",
                str(tmp_path / "settings.yaml"),
                "--top-k",
                "1",
                "--print-top-chunks",
                "1",
                "what is rag",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Query CLI ready" in output
    assert "Answer: RAG combines retrieval and generation [1]." in output
    assert "Top Chunks:" in output
    assert "RAG combines retrieval and generation." in output


@pytest.mark.integration
def test_query_engine_handles_no_result_queries(tmp_path: Path) -> None:
    llm = FakeLLM()
    settings, container = _build_runtime(tmp_path, reranker_provider=None, llm=llm)
    engine = build_query_engine(container, settings=settings)

    response = engine.run(query="unknown term", collection="empty", top_k=2)

    assert response["answer"] == "No relevant context found for the query."
    assert response["citations"] == []
    assert response["retrieved_chunks"] == []


@pytest.mark.integration
def test_query_engine_returns_retrieved_chunks_when_answer_generation_fails(tmp_path: Path) -> None:
    class FailingLLM:
        def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
            raise RuntimeError("provider blocked request")

    settings, container = _build_runtime(tmp_path, reranker_provider=None, llm=FailingLLM())
    engine = build_query_engine(container, settings=settings)

    response = engine.run(query="rag", top_k=2)

    assert response["answer"] == "Answer generation unavailable; inspect the retrieved chunks below."
    assert len(response["retrieved_chunks"]) == 2
    assert len(response["citations"]) == 2


@pytest.mark.integration
def test_query_cli_applies_collection_override_before_runtime_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm = FakeLLM(["No relevant context found for the query."])
    settings, container = _build_runtime(tmp_path, reranker_provider=None, llm=llm)
    captured: dict[str, object] = {}

    class StubEngine:
        def run(self, **kwargs):
            captured["run_kwargs"] = dict(kwargs)
            return {
                "answer": "No relevant context found for the query.",
                "citations": [],
                "retrieved_chunks": [],
                "debug_info": {},
            }

    monkeypatch.setattr("scripts.query_cli.load_settings", lambda path: settings)

    def fake_build_container(loaded):
        captured["container_collection"] = loaded.vector_store.collection
        return container

    def fake_build_query_engine(_container, settings=None):
        captured["engine_collection"] = None if settings is None else settings.vector_store.collection
        return StubEngine()

    monkeypatch.setattr("scripts.query_cli.build_container", fake_build_container)
    monkeypatch.setattr("scripts.query_cli.build_query_engine", fake_build_query_engine)

    assert (
        run_cli(
            [
                "--settings",
                str(tmp_path / "settings.yaml"),
                "--collection",
                "real_c_ingestion_test",
                "query",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "collection=real_c_ingestion_test" in output
    assert captured["container_collection"] == "real_c_ingestion_test"
    assert captured["engine_collection"] == "real_c_ingestion_test"
    assert captured["run_kwargs"]["collection"] == "real_c_ingestion_test"


@pytest.mark.integration
def test_query_engine_reports_reranker_fallback_and_filter_effect(tmp_path: Path) -> None:
    llm = FakeLLM(["RAG combines retrieval and generation [1]."])
    settings, container = _build_runtime(
        tmp_path,
        reranker_provider=FailingRerankerProvider(),
        llm=llm,
    )
    engine = build_query_engine(container, settings=settings)

    response = engine.run(
        query="rag retrieval",
        top_k=2,
        filters=json.dumps({"doc_type": "pdf", "owner": "team-a"}),
        return_debug=True,
    )

    assert response["fallback_applied"] is True
    assert "Reranker execution failed" in str(response["fallback_reason"])
    assert response["retrieved_chunks"][0]["chunk_id"] == "chunk-1"
    assert response["debug_info"]["filters"]["total_removed"] >= 1
