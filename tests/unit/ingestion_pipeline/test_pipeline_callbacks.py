from __future__ import annotations

from pathlib import Path
from typing import Any

from ragms.ingestion_pipeline.callbacks import PipelineCallback, ProgressEvent, StageEvent
from ragms.ingestion_pipeline.pipeline import IngestionPipeline


class RecordingLoader:
    def load(self, source_path: str | Path, *, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [
            {
                "content": "alpha beta gamma",
                "source_path": str(source_path),
                "metadata": {"document_id": "doc-1", **dict(metadata or {})},
            }
        ]


class RecordingSplitter:
    def split(
        self,
        document: dict[str, Any],
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[dict[str, Any]]:
        del chunk_size, chunk_overlap
        return [
            {
                "content": document["content"],
                "chunk_index": 0,
                "source": document["source_path"],
                "metadata": dict(document["metadata"]),
            }
        ]


class RecordingTransform:
    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        transformed = [dict(chunk) for chunk in chunks]
        for chunk in transformed:
            chunk["metadata"] = {**dict(chunk.get("metadata") or {}), **dict(context or {})}
            chunk["content"] = f"smart:{chunk['content']}"
        return transformed


class RecordingEmbedding:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text))] for text in texts]


class RecordingVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        self.calls.append(
            {
                "ids": list(ids),
                "vectors": [list(vector) for vector in vectors],
                "documents": list(documents or []),
                "metadatas": list(metadatas or []),
            }
        )
        return ids


class ExplodingTransform:
    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del chunks, context
        raise ValueError("transform blew up")


class RecordingCallback(PipelineCallback):
    def __init__(self) -> None:
        self.starts: list[StageEvent] = []
        self.ends: list[StageEvent] = []
        self.progress: list[ProgressEvent] = []

    def on_stage_start(self, event: StageEvent) -> None:
        self.starts.append(event)

    def on_stage_end(self, event: StageEvent) -> None:
        self.ends.append(event)

    def on_progress(self, event: ProgressEvent) -> None:
        self.progress.append(event)


def test_ingestion_pipeline_runs_all_stages_and_returns_unified_state(tmp_path: Path) -> None:
    embedding = RecordingEmbedding()
    vector_store = RecordingVectorStore()
    pipeline = IngestionPipeline(
        loader=RecordingLoader(),
        splitter=RecordingSplitter(),
        transform=RecordingTransform(),
        embedding=embedding,
        vector_store=vector_store,
    )

    result = pipeline.run(tmp_path / "sample.pdf", metadata={"tenant": "acme"})

    assert result["status"] == "completed"
    assert len(result["documents"]) == 1
    assert len(result["chunks"]) == 1
    assert len(result["smart_chunks"]) == 1
    assert result["smart_chunks"][0]["content"] == "smart:alpha beta gamma"
    assert result["vectors"] == [[22.0]]
    assert len(result["stored_ids"]) == 1
    assert embedding.calls == [["smart:alpha beta gamma"]]
    assert vector_store.calls[0]["documents"] == ["smart:alpha beta gamma"]
    assert vector_store.calls[0]["metadatas"][0]["tenant"] == "acme"


def test_ingestion_pipeline_emits_stage_boundary_and_progress_callbacks(tmp_path: Path) -> None:
    callback = RecordingCallback()
    pipeline = IngestionPipeline(
        loader=RecordingLoader(),
        splitter=RecordingSplitter(),
        transform=RecordingTransform(),
        embedding=RecordingEmbedding(),
        vector_store=RecordingVectorStore(),
        callbacks=[callback],
    )

    result = pipeline.run(tmp_path / "sample.pdf")

    assert result["status"] == "completed"
    assert [event.stage for event in callback.starts] == ["load", "split", "transform", "embed", "store"]
    assert [event.status for event in callback.ends] == ["completed"] * 5
    assert [event.current_stage for event in callback.progress] == ["load", "split", "transform", "embed", "store"]
    assert callback.progress[-1].status == "completed"
    assert callback.progress[-1].completed_stages == 5


def test_ingestion_pipeline_returns_unified_failure_context_and_failed_callback(tmp_path: Path) -> None:
    callback = RecordingCallback()
    pipeline = IngestionPipeline(
        loader=RecordingLoader(),
        splitter=RecordingSplitter(),
        transform=ExplodingTransform(),
        embedding=RecordingEmbedding(),
        vector_store=RecordingVectorStore(),
        callbacks=[callback],
    )

    result = pipeline.run(tmp_path / "sample.pdf")

    assert result["status"] == "failed"
    assert result["stage"] == "transform"
    assert result["source_path"].endswith("sample.pdf")
    assert result["error"] == {"type": "ValueError", "message": "transform blew up"}
    assert len(result["documents"]) == 1
    assert len(result["chunks"]) == 1
    assert result["smart_chunks"] == []
    assert callback.ends[-1].stage == "transform"
    assert callback.ends[-1].status == "failed"
    assert callback.progress[-1].status == "failed"
    assert callback.progress[-1].completed_stages == 2
