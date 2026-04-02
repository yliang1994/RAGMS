"""Core ingestion pipeline orchestration with stage callbacks and unified failures."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ragms.ingestion_pipeline.callbacks import PipelineCallback, ProgressEvent, StageEvent
from ragms.libs.abstractions import (
    BaseEmbedding,
    BaseLoader,
    BaseSplitter,
    BaseTransform,
    BaseVectorStore,
)


class IngestionPipeline:
    """Orchestrate the ingestion stages from loading to vector-store persistence."""

    STAGE_ORDER = ("load", "split", "transform", "embed", "store")

    def __init__(
        self,
        *,
        loader: BaseLoader,
        splitter: BaseSplitter,
        transform: BaseTransform | None = None,
        embedding: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
        callbacks: list[PipelineCallback] | None = None,
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.transform = transform
        self.embedding = embedding
        self.vector_store = vector_store
        self.callbacks = list(callbacks or [])

    def run(
        self,
        source_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the ingestion stages and return a unified success or failure payload."""

        normalized_source = str(Path(source_path))
        state: dict[str, Any] = {
            "status": "running",
            "source_path": normalized_source,
            "metadata": dict(metadata or {}),
            "documents": [],
            "chunks": [],
            "smart_chunks": [],
            "vectors": [],
            "stored_ids": [],
        }

        for index, stage in enumerate(self.STAGE_ORDER, start=1):
            self._emit_stage_start(stage=stage, source_path=normalized_source, index=index)
            try:
                if stage == "load":
                    state["documents"] = self.loader.load(normalized_source, metadata=state["metadata"])
                    payload = {"document_count": len(state["documents"])}
                elif stage == "split":
                    chunks: list[dict[str, Any]] = []
                    for document in state["documents"]:
                        chunks.extend(self.splitter.split(document))
                    state["chunks"] = chunks
                    payload = {"chunk_count": len(chunks)}
                elif stage == "transform":
                    state["smart_chunks"] = self._run_transform_stage(
                        state["chunks"],
                        source_path=normalized_source,
                    )
                    payload = {"smart_chunk_count": len(state["smart_chunks"])}
                elif stage == "embed":
                    state["vectors"] = self._run_embedding_stage(state["smart_chunks"])
                    payload = {"vector_count": len(state["vectors"])}
                else:
                    state["stored_ids"] = self._run_store_stage(
                        state["smart_chunks"],
                        state["vectors"],
                        source_path=normalized_source,
                    )
                    payload = {"stored_count": len(state["stored_ids"])}
            except Exception as exc:
                failure = self._build_failure_result(
                    stage=stage,
                    source_path=normalized_source,
                    error=exc,
                    state=state,
                )
                self._emit_stage_end(
                    stage=stage,
                    source_path=normalized_source,
                    index=index,
                    status="failed",
                    payload=failure["error"],
                )
                self._emit_progress(
                    source_path=normalized_source,
                    completed_stages=index - 1,
                    current_stage=stage,
                    status="failed",
                )
                return failure

            self._emit_stage_end(
                stage=stage,
                source_path=normalized_source,
                index=index,
                status="completed",
                payload=payload,
            )
            self._emit_progress(
                source_path=normalized_source,
                completed_stages=index,
                current_stage=stage,
                status="running" if index < len(self.STAGE_ORDER) else "completed",
            )

        state["status"] = "completed"
        return state

    def _run_transform_stage(
        self,
        chunks: list[dict[str, Any]],
        *,
        source_path: str,
    ) -> list[dict[str, Any]]:
        """Run the transform stage or pass chunks through unchanged."""

        if self.transform is None:
            return [dict(chunk) for chunk in chunks]
        return self.transform.transform(chunks, context={"source_path": source_path})

    def _run_embedding_stage(self, smart_chunks: list[dict[str, Any]]) -> list[list[float]]:
        """Run the embedding stage or skip it when no embedding backend is configured."""

        if self.embedding is None or not smart_chunks:
            return []
        return self.embedding.embed_documents([str(chunk.get("content", "")) for chunk in smart_chunks])

    def _run_store_stage(
        self,
        smart_chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        *,
        source_path: str,
    ) -> list[str]:
        """Persist vectorized chunks or skip persistence when backend is absent."""

        if self.vector_store is None:
            return []
        if smart_chunks and not vectors:
            raise ValueError("Embedding vectors are required before vector-store persistence")
        if len(smart_chunks) != len(vectors):
            raise ValueError("Chunk count and vector count must match before storage")
        if not smart_chunks:
            return []

        ids = [self._build_chunk_id(source_path, chunk, index) for index, chunk in enumerate(smart_chunks)]
        documents = [str(chunk.get("content", "")) for chunk in smart_chunks]
        metadatas = [dict(chunk.get("metadata") or {}) for chunk in smart_chunks]
        return self.vector_store.add(ids, vectors, documents=documents, metadatas=metadatas)

    def _emit_stage_start(self, *, stage: str, source_path: str, index: int) -> None:
        """Emit a stage start event to all registered callbacks."""

        event = StageEvent(
            stage=stage,
            source_path=source_path,
            index=index,
            total=len(self.STAGE_ORDER),
            status="started",
        )
        for callback in self.callbacks:
            callback.on_stage_start(event)

    def _emit_stage_end(
        self,
        *,
        stage: str,
        source_path: str,
        index: int,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit a stage end event to all registered callbacks."""

        event = StageEvent(
            stage=stage,
            source_path=source_path,
            index=index,
            total=len(self.STAGE_ORDER),
            status=status,
            payload=payload,
        )
        for callback in self.callbacks:
            callback.on_stage_end(event)

    def _emit_progress(
        self,
        *,
        source_path: str,
        completed_stages: int,
        current_stage: str,
        status: str,
    ) -> None:
        """Emit a progress update to all registered callbacks."""

        event = ProgressEvent(
            source_path=source_path,
            completed_stages=completed_stages,
            total_stages=len(self.STAGE_ORDER),
            current_stage=current_stage,
            status=status,
        )
        for callback in self.callbacks:
            callback.on_progress(event)

    @staticmethod
    def _build_failure_result(
        *,
        stage: str,
        source_path: str,
        error: Exception,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize stage failures into a stable pipeline error payload."""

        return {
            "status": "failed",
            "stage": stage,
            "source_path": source_path,
            "documents": list(state.get("documents", [])),
            "chunks": list(state.get("chunks", [])),
            "smart_chunks": list(state.get("smart_chunks", [])),
            "vectors": list(state.get("vectors", [])),
            "stored_ids": list(state.get("stored_ids", [])),
            "error": {
                "type": error.__class__.__name__,
                "message": str(error),
            },
        }

    @staticmethod
    def _build_chunk_id(source_path: str, chunk: dict[str, Any], index: int) -> str:
        """Build a deterministic chunk id from source path, chunk content, and position."""

        metadata = dict(chunk.get("metadata") or {})
        explicit_id = metadata.get("chunk_id") or chunk.get("chunk_id")
        if explicit_id:
            return str(explicit_id)
        content = str(chunk.get("content", ""))
        digest = hashlib.sha256(f"{source_path}:{index}:{content}".encode("utf-8")).hexdigest()
        return f"chunk_{digest[:16]}"
