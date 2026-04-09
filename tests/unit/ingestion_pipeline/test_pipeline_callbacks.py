from __future__ import annotations

from pathlib import Path
from typing import Any

from ragms.ingestion_pipeline.callbacks import (
    ErrorEvent,
    PipelineCallback,
    PipelineEvent,
    ProgressEvent,
    StageEvent,
)
from ragms.ingestion_pipeline.pipeline import IngestionPipeline


class RecordingLoader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def load(self, source_path: str | Path, *, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        payload = {
            "source_path": str(source_path),
            "metadata": dict(metadata or {}),
        }
        self.calls.append(payload)
        return [
            {
                "content": "alpha beta gamma",
                "source_path": str(source_path),
                "metadata": {"document_id": payload["metadata"].get("document_id"), **payload["metadata"]},
            }
        ]


class ExplodingLoader:
    def load(self, source_path: str | Path, *, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        del source_path, metadata
        raise ValueError("load blew up")


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


class ExplodingTransform:
    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del chunks, context
        raise ValueError("transform blew up")


class RecordingEmbedding:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


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

    def query(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del query_vector, top_k, filters
        return []

    def delete(self, ids: list[str]) -> int:
        return len(ids)


class FakeFileIntegrity:
    def __init__(self, *, source_sha256: str = "sha-123", should_skip: bool = False) -> None:
        self.source_sha256 = source_sha256
        self.skip = should_skip
        self.calls: list[dict[str, Any]] = []

    def compute_sha256(self, source_path: str | Path) -> str:
        self.calls.append({"action": "compute", "source_path": str(source_path)})
        return self.source_sha256

    def should_skip(self, source_path: str | Path, *, force_rebuild: bool = False) -> bool:
        self.calls.append(
            {
                "action": "should_skip",
                "source_path": str(source_path),
                "force_rebuild": force_rebuild,
            }
        )
        return False if force_rebuild else self.skip


class FakeDocumentRegistry:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.register_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []

    def register(
        self,
        *,
        source_path: str,
        source_sha256: str,
        document_id: str | None = None,
        status: str = "pending",
        current_stage: str = "registered",
    ) -> dict[str, Any]:
        resolved_document_id = document_id or f"doc-{source_sha256[:8]}"
        record = {
            "document_id": resolved_document_id,
            "source_path": source_path,
            "source_sha256": source_sha256,
            "status": status,
            "current_stage": current_stage,
        }
        self.records[resolved_document_id] = dict(record)
        self.register_calls.append(dict(record))
        return dict(record)

    def update_status(
        self,
        document_id: str,
        *,
        status: str,
        current_stage: str | None = None,
    ) -> dict[str, Any]:
        record = dict(self.records[document_id])
        record["status"] = status
        if current_stage is not None:
            record["current_stage"] = current_stage
        self.records[document_id] = dict(record)
        self.update_calls.append(
            {
                "document_id": document_id,
                "status": status,
                "current_stage": current_stage,
            }
        )
        return dict(record)

    def get(self, document_id: str) -> dict[str, Any] | None:
        record = self.records.get(document_id)
        return None if record is None else dict(record)


class RecordingCallback(PipelineCallback):
    def __init__(self) -> None:
        self.pipeline_starts: list[PipelineEvent] = []
        self.starts: list[StageEvent] = []
        self.ends: list[StageEvent] = []
        self.progress: list[ProgressEvent] = []
        self.errors: list[ErrorEvent] = []

    def on_pipeline_start(self, event: PipelineEvent) -> None:
        self.pipeline_starts.append(event)

    def on_stage_start(self, event: StageEvent) -> None:
        self.starts.append(event)

    def on_stage_end(self, event: StageEvent) -> None:
        self.ends.append(event)

    def on_progress(self, event: ProgressEvent) -> None:
        self.progress.append(event)

    def on_error(self, event: ErrorEvent) -> None:
        self.errors.append(event)


def test_ingestion_pipeline_runs_all_stages_with_trace_context_and_lifecycle(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_text("sample payload", encoding="utf-8")
    callback = RecordingCallback()
    loader = RecordingLoader()
    embedding = RecordingEmbedding()
    vector_store = RecordingVectorStore()
    registry = FakeDocumentRegistry()
    pipeline = IngestionPipeline(
        loader=loader,
        splitter=RecordingSplitter(),
        transform=RecordingTransform(),
        embedding=embedding,
        vector_store=vector_store,
        file_integrity=FakeFileIntegrity(source_sha256="sha-success"),
        document_registry=registry,
        callbacks=[callback],
    )

    result = pipeline.run(source, metadata={"tenant": "acme"})

    assert result["status"] == "completed"
    assert result["source_sha256"] == "sha-success"
    assert result["document_id"] == IngestionPipeline._resolve_document_id(
        source_path=str(source),
        source_sha256="sha-success",
    )
    assert result["current_stage"] == "lifecycle_finalize"
    assert len(result["documents"]) == 1
    assert len(result["chunks"]) == 1
    assert len(result["smart_chunks"]) == 1
    assert result["smart_chunks"][0]["content"] == "smart:alpha beta gamma"
    assert result["vectors"] == [[22.0]]
    assert len(result["stored_ids"]) == 1
    assert result["lifecycle"]["registry_status"] == "indexed"
    assert embedding.calls == [["smart:alpha beta gamma"]]
    assert vector_store.calls[0]["documents"] == ["smart:alpha beta gamma"]
    assert vector_store.calls[0]["metadatas"][0]["tenant"] == "acme"
    assert loader.calls[0]["metadata"]["trace_id"] == result["trace_id"]
    assert loader.calls[0]["metadata"]["document_id"] == result["document_id"]
    assert loader.calls[0]["metadata"]["source_sha256"] == "sha-success"
    assert len(callback.pipeline_starts) == 1
    assert callback.pipeline_starts[0].trace_id == result["trace_id"]
    assert [event.stage for event in callback.starts] == [
        "file_integrity",
        "load",
        "chunking",
        "transform",
        "embedding",
        "storage",
        "lifecycle_finalize",
    ]
    assert [event.status for event in callback.ends] == ["completed"] * 7
    assert callback.ends[0].payload["source_sha256"] == "sha-success"
    assert callback.ends[-1].payload["final_status"] == "indexed"
    assert callback.progress[0].current_stage == "pipeline_start"
    assert callback.progress[0].elapsed_ms == 0.0
    assert callback.progress[-1].current_stage == "lifecycle_finalize"
    assert callback.progress[-1].status == "completed"
    assert callback.progress[-1].completed_stages == 7
    assert callback.progress[-1].metadata["collection"] is None
    assert callback.errors == []
    assert registry.register_calls[0]["status"] == "pending"
    assert registry.update_calls == [
        {
            "document_id": result["document_id"],
            "status": "processing",
            "current_stage": "load",
        },
        {
            "document_id": result["document_id"],
            "status": "indexed",
            "current_stage": "lifecycle_finalize",
        },
    ]


def test_ingestion_pipeline_emits_error_and_marks_lifecycle_failed(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_text("sample payload", encoding="utf-8")
    callback = RecordingCallback()
    registry = FakeDocumentRegistry()
    pipeline = IngestionPipeline(
        loader=RecordingLoader(),
        splitter=RecordingSplitter(),
        transform=ExplodingTransform(),
        embedding=RecordingEmbedding(),
        vector_store=RecordingVectorStore(),
        file_integrity=FakeFileIntegrity(source_sha256="sha-failure"),
        document_registry=registry,
        callbacks=[callback],
    )

    result = pipeline.run(source)

    assert result["status"] == "failed"
    assert result["stage"] == "transform"
    assert result["source_path"].endswith("sample.pdf")
    assert result["source_sha256"] == "sha-failure"
    assert result["error"] == {"type": "ValueError", "message": "transform blew up"}
    assert len(result["documents"]) == 1
    assert len(result["chunks"]) == 1
    assert result["smart_chunks"] == []
    assert result["current_stage"] == "lifecycle_finalize"
    assert result["lifecycle"]["registry_status"] == "failed"
    assert [event.stage for event in callback.starts] == [
        "file_integrity",
        "load",
        "chunking",
        "transform",
        "lifecycle_finalize",
    ]
    assert callback.ends[-2].stage == "transform"
    assert callback.ends[-2].status == "failed"
    assert callback.ends[-1].stage == "lifecycle_finalize"
    assert callback.ends[-1].status == "completed"
    assert callback.errors[-1].stage == "transform"
    assert callback.errors[-1].error == {"type": "ValueError", "message": "transform blew up"}
    assert callback.progress[-2].current_stage == "transform"
    assert callback.progress[-2].status == "failed"
    assert callback.progress[-1].current_stage == "lifecycle_finalize"
    assert callback.progress[-1].status == "failed"
    assert registry.update_calls == [
        {
            "document_id": result["document_id"],
            "status": "processing",
            "current_stage": "load",
        },
        {
            "document_id": result["document_id"],
            "status": "failed",
            "current_stage": "lifecycle_finalize",
        },
    ]


def test_ingestion_pipeline_short_circuits_to_skipped_lifecycle_when_integrity_hits(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_text("sample payload", encoding="utf-8")
    callback = RecordingCallback()
    loader = RecordingLoader()
    registry = FakeDocumentRegistry()
    pipeline = IngestionPipeline(
        loader=loader,
        splitter=RecordingSplitter(),
        transform=RecordingTransform(),
        embedding=RecordingEmbedding(),
        vector_store=RecordingVectorStore(),
        file_integrity=FakeFileIntegrity(source_sha256="sha-skip", should_skip=True),
        document_registry=registry,
        callbacks=[callback],
    )

    result = pipeline.run(source)

    assert result["status"] == "skipped"
    assert result["document_id"] == IngestionPipeline._resolve_document_id(
        source_path=str(source),
        source_sha256="sha-skip",
    )
    assert result["source_sha256"] == "sha-skip"
    assert result["documents"] == []
    assert result["stored_ids"] == []
    assert result["lifecycle"]["registry_status"] == "skipped"
    assert loader.calls == []
    assert [event.stage for event in callback.starts] == ["file_integrity", "lifecycle_finalize"]
    assert [event.status for event in callback.ends] == ["completed", "completed"]
    assert callback.progress[-1].current_stage == "lifecycle_finalize"
    assert callback.progress[-1].status == "skipped"
    assert callback.progress[-1].completed_stages == 2
    assert callback.errors == []
    assert registry.update_calls == [
        {
            "document_id": result["document_id"],
            "status": "skipped",
            "current_stage": "lifecycle_finalize",
        }
    ]
