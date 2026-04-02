"""Core ingestion pipeline orchestration with stage callbacks and unified failures."""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

from ragms.ingestion_pipeline.callbacks import (
    ErrorEvent,
    PipelineCallback,
    PipelineEvent,
    ProgressEvent,
    StageEvent,
)
from ragms.ingestion_pipeline.file_integrity import FileIntegrity
from ragms.ingestion_pipeline.lifecycle.document_registry import DocumentRegistry, DocumentRegistryError
from ragms.libs.abstractions import (
    BaseEmbedding,
    BaseLoader,
    BaseSplitter,
    BaseTransform,
    BaseVectorStore,
)


class IngestionPipeline:
    """Orchestrate the ingestion stages from integrity checks to lifecycle finalization."""

    STAGE_ORDER = ("file_integrity", "load", "split", "transform", "embed", "store", "lifecycle")

    def __init__(
        self,
        *,
        loader: BaseLoader,
        splitter: BaseSplitter,
        transform: BaseTransform | None = None,
        embedding: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
        file_integrity: FileIntegrity | None = None,
        document_registry: DocumentRegistry | None = None,
        callbacks: list[PipelineCallback] | None = None,
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.transform = transform
        self.embedding = embedding
        self.vector_store = vector_store
        self.file_integrity = file_integrity
        self.document_registry = document_registry
        self.callbacks = list(callbacks or [])

    def run(
        self,
        source_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the ingestion stages and return a unified success, skipped, or failure payload."""

        normalized_source = str(Path(source_path))
        trace_id = uuid.uuid4().hex
        state: dict[str, Any] = {
            "status": "running",
            "trace_id": trace_id,
            "source_path": normalized_source,
            "document_id": None,
            "source_sha256": None,
            "current_stage": "pipeline_start",
            "metadata": dict(metadata or {}),
            "documents": [],
            "chunks": [],
            "smart_chunks": [],
            "vectors": [],
            "stored_ids": [],
        }
        self._emit_pipeline_start(state)

        file_integrity_stage = 1
        try:
            file_integrity_payload = self._run_file_integrity_stage(state)
        except Exception as exc:
            return self._handle_failure(
                stage="file_integrity",
                index=file_integrity_stage,
                error=exc,
                state=state,
                completed_stages=0,
            )

        if file_integrity_payload["should_skip"]:
            lifecycle_payload = self._run_lifecycle_stage(
                state,
                index=len(self.STAGE_ORDER),
                final_status="skipped",
                completed_before=1,
            )
            state["status"] = "skipped"
            state["current_stage"] = "lifecycle"
            state["lifecycle"] = lifecycle_payload
            return state

        completed_stages = 1
        if self.document_registry is not None and state["document_id"] is not None:
            self._promote_registry_to_processing(state["document_id"])

        stage_handlers: list[tuple[str, Any]] = [
            ("load", self._run_load_stage),
            ("split", self._run_split_stage),
            ("transform", self._run_transform_stage),
            ("embed", self._run_embedding_stage),
            ("store", self._run_store_stage),
        ]

        for index, (stage, handler) in enumerate(stage_handlers, start=2):
            try:
                handler(state, index)
            except Exception as exc:
                return self._handle_failure(
                    stage=stage,
                    index=index,
                    error=exc,
                    state=state,
                    completed_stages=completed_stages,
                )
            completed_stages += 1

        lifecycle_payload = self._run_lifecycle_stage(
            state,
            index=len(self.STAGE_ORDER),
            final_status="indexed",
            completed_before=completed_stages,
        )
        state["status"] = "completed"
        state["current_stage"] = "lifecycle"
        state["lifecycle"] = lifecycle_payload
        return state

    def _run_file_integrity_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the file integrity gate, register the document, and decide skip behavior."""

        stage = "file_integrity"
        index = 1
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()

        force_rebuild = bool(state["metadata"].get("force_rebuild", False))
        if self.file_integrity is not None:
            source_sha256 = self.file_integrity.compute_sha256(state["source_path"])
            should_skip = self.file_integrity.should_skip(
                state["source_path"],
                force_rebuild=force_rebuild,
            )
        else:
            source_sha256 = None
            should_skip = False

        state["source_sha256"] = source_sha256
        state["document_id"] = self._resolve_document_id(
            source_path=state["source_path"],
            source_sha256=source_sha256,
        )
        state["metadata"].update(
            {
                "trace_id": state["trace_id"],
                "document_id": state["document_id"],
            }
        )
        if source_sha256 is not None:
            state["metadata"]["source_sha256"] = source_sha256

        registry_record = None
        if self.document_registry is not None:
            registry_record = self.document_registry.register(
                source_path=state["source_path"],
                source_sha256=source_sha256 or self._hash_text(state["source_path"]),
                document_id=state["document_id"],
                status="pending",
                current_stage=stage,
            )
            state["document_id"] = str(registry_record["document_id"])
            state["metadata"]["document_id"] = state["document_id"]

        payload = {
            "enabled": self.file_integrity is not None,
            "source_sha256": source_sha256,
            "document_id": state["document_id"],
            "should_skip": should_skip,
            "force_rebuild": force_rebuild,
            "registry_enabled": self.document_registry is not None,
        }
        if registry_record is not None:
            payload["registry_status"] = str(registry_record["status"])

        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        self._emit_progress(
            state=state,
            completed_stages=index,
            current_stage=stage,
            status="running",
        )
        return payload

    def _run_load_stage(self, state: dict[str, Any], index: int) -> None:
        """Run the document loading stage."""

        stage = "load"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()

        documents = self.loader.load(state["source_path"], metadata=state["metadata"])
        state["documents"] = list(documents)
        inferred_document_id = self._infer_document_id_from_documents(state["documents"])
        if inferred_document_id and state["document_id"] is None:
            state["document_id"] = inferred_document_id
            state["metadata"]["document_id"] = inferred_document_id

        payload = {"document_count": len(state["documents"])}
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        self._emit_progress(state=state, completed_stages=index, current_stage=stage, status="running")

    def _run_split_stage(self, state: dict[str, Any], index: int) -> None:
        """Split canonical documents into chunk dictionaries."""

        stage = "split"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()

        chunks: list[dict[str, Any]] = []
        for document in state["documents"]:
            chunks.extend(self.splitter.split(document))
        state["chunks"] = chunks

        payload = {"chunk_count": len(chunks)}
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        self._emit_progress(state=state, completed_stages=index, current_stage=stage, status="running")

    def _run_transform_stage(self, state: dict[str, Any], index: int) -> None:
        """Run the transform stage or pass chunks through unchanged."""

        stage = "transform"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()

        if self.transform is None:
            state["smart_chunks"] = [dict(chunk) for chunk in state["chunks"]]
        else:
            state["smart_chunks"] = self.transform.transform(
                state["chunks"],
                context={
                    "trace_id": state["trace_id"],
                    "document_id": state["document_id"],
                    "source_path": state["source_path"],
                    "source_sha256": state["source_sha256"],
                },
            )

        payload = {"smart_chunk_count": len(state["smart_chunks"])}
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        self._emit_progress(state=state, completed_stages=index, current_stage=stage, status="running")

    def _run_embedding_stage(self, state: dict[str, Any], index: int) -> None:
        """Run the embedding stage or skip it when no embedding backend is configured."""

        stage = "embed"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()

        if self.embedding is None or not state["smart_chunks"]:
            state["vectors"] = []
        else:
            texts = [str(chunk.get("content", "")) for chunk in state["smart_chunks"]]
            state["vectors"] = self.embedding.embed_documents(texts)

        payload = {"vector_count": len(state["vectors"])}
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        self._emit_progress(state=state, completed_stages=index, current_stage=stage, status="running")

    def _run_store_stage(self, state: dict[str, Any], index: int) -> None:
        """Persist vectorized chunks or skip persistence when backend is absent."""

        stage = "store"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()

        if self.vector_store is None:
            state["stored_ids"] = []
        else:
            smart_chunks = list(state["smart_chunks"])
            vectors = list(state["vectors"])
            if smart_chunks and not vectors:
                raise ValueError("Embedding vectors are required before vector-store persistence")
            if len(smart_chunks) != len(vectors):
                raise ValueError("Chunk count and vector count must match before storage")
            if not smart_chunks:
                state["stored_ids"] = []
            else:
                ids = [
                    self._build_chunk_id(state["source_path"], chunk, chunk_index)
                    for chunk_index, chunk in enumerate(smart_chunks)
                ]
                documents = [str(chunk.get("content", "")) for chunk in smart_chunks]
                metadatas = [dict(chunk.get("metadata") or {}) for chunk in smart_chunks]
                state["stored_ids"] = self.vector_store.add(
                    ids,
                    vectors,
                    documents=documents,
                    metadatas=metadatas,
                )

        payload = {"stored_count": len(state["stored_ids"])}
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        self._emit_progress(state=state, completed_stages=index, current_stage=stage, status="running")

    def _run_lifecycle_stage(
        self,
        state: dict[str, Any],
        *,
        index: int,
        final_status: str,
        completed_before: int,
    ) -> dict[str, Any]:
        """Finalize the document lifecycle state for success, skip, or failure."""

        stage = "lifecycle"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()

        payload: dict[str, Any] = {
            "final_status": final_status,
            "registry_enabled": self.document_registry is not None,
            "document_id": state["document_id"],
        }
        if self.document_registry is not None and state["document_id"] is not None:
            record = self._update_registry_for_final_status(
                document_id=state["document_id"],
                final_status=final_status,
            )
            payload["registry_status"] = str(record["status"])
            payload["registry_stage"] = str(record["current_stage"])

        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        self._emit_progress(
            state=state,
            completed_stages=completed_before + 1,
            current_stage=stage,
            status="completed" if final_status == "indexed" else final_status,
        )
        return payload

    def _handle_failure(
        self,
        *,
        stage: str,
        index: int,
        error: Exception,
        state: dict[str, Any],
        completed_stages: int,
    ) -> dict[str, Any]:
        """Normalize stage failures, emit callbacks, and mark lifecycle failure."""

        state["status"] = "failed"
        state["current_stage"] = stage
        failure = self._build_failure_result(stage=stage, error=error, state=state)
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="failed",
            elapsed_ms=0.0,
            payload=failure["error"],
        )
        self._emit_progress(
            state=state,
            completed_stages=completed_stages,
            current_stage=stage,
            status="failed",
        )
        self._emit_error(
            state=state,
            stage=stage,
            error=failure["error"],
            completed_stages=completed_stages,
        )
        lifecycle_payload = self._run_lifecycle_stage(
            state,
            index=len(self.STAGE_ORDER),
            final_status="failed",
            completed_before=completed_stages,
        )
        failure["lifecycle"] = lifecycle_payload
        failure["current_stage"] = "lifecycle"
        return failure

    def _promote_registry_to_processing(self, document_id: str) -> None:
        """Move a registered document into processing before active ingestion begins."""

        if self.document_registry is None:
            return
        try:
            self.document_registry.update_status(
                document_id,
                status="processing",
                current_stage="load",
            )
        except DocumentRegistryError:
            # The lifecycle layer remains the source of truth; if the record is already in
            # a processing-compatible state, the ingestion run should continue.
            pass

    def _update_registry_for_final_status(self, *, document_id: str, final_status: str) -> dict[str, object]:
        """Update the registry to the final lifecycle status, tolerating repeated finalization."""

        assert self.document_registry is not None
        existing = self.document_registry.get(document_id)
        if existing is None:
            raise DocumentRegistryError(f"Unknown document: {document_id}")

        current_status = str(existing["status"])
        target_stage = "lifecycle"
        if current_status == final_status:
            return dict(existing)
        try:
            return self.document_registry.update_status(
                document_id,
                status=final_status,
                current_stage=target_stage,
            )
        except DocumentRegistryError:
            if final_status == "failed" and current_status == "pending":
                return self.document_registry.update_status(
                    document_id,
                    status="failed",
                    current_stage=target_stage,
                )
            raise

    def _emit_pipeline_start(self, state: dict[str, Any]) -> None:
        """Emit a pipeline start event with stable context fields."""

        event = PipelineEvent(
            trace_id=state["trace_id"],
            source_path=state["source_path"],
            document_id=state["document_id"],
            total_stages=len(self.STAGE_ORDER),
            status="started",
            metadata=dict(state["metadata"]),
        )
        for callback in self.callbacks:
            callback.on_pipeline_start(event)

    def _emit_stage_start(self, *, state: dict[str, Any], stage: str, index: int) -> None:
        """Emit a stage start event to all registered callbacks."""

        event = StageEvent(
            trace_id=state["trace_id"],
            source_path=state["source_path"],
            document_id=state["document_id"],
            stage=stage,
            index=index,
            total=len(self.STAGE_ORDER),
            status="started",
        )
        for callback in self.callbacks:
            callback.on_stage_start(event)

    def _emit_stage_end(
        self,
        *,
        state: dict[str, Any],
        stage: str,
        index: int,
        status: str,
        elapsed_ms: float,
        payload: dict[str, Any],
    ) -> None:
        """Emit a stage end event to all registered callbacks."""

        event = StageEvent(
            trace_id=state["trace_id"],
            source_path=state["source_path"],
            document_id=state["document_id"],
            stage=stage,
            index=index,
            total=len(self.STAGE_ORDER),
            status=status,
            elapsed_ms=elapsed_ms,
            payload=payload,
        )
        for callback in self.callbacks:
            callback.on_stage_end(event)

    def _emit_progress(
        self,
        *,
        state: dict[str, Any],
        completed_stages: int,
        current_stage: str,
        status: str,
    ) -> None:
        """Emit a progress update to all registered callbacks."""

        event = ProgressEvent(
            trace_id=state["trace_id"],
            source_path=state["source_path"],
            document_id=state["document_id"],
            completed_stages=completed_stages,
            total_stages=len(self.STAGE_ORDER),
            current_stage=current_stage,
            status=status,
        )
        for callback in self.callbacks:
            callback.on_progress(event)

    def _emit_error(
        self,
        *,
        state: dict[str, Any],
        stage: str,
        error: dict[str, Any],
        completed_stages: int,
    ) -> None:
        """Emit a normalized error event to all registered callbacks."""

        event = ErrorEvent(
            trace_id=state["trace_id"],
            source_path=state["source_path"],
            document_id=state["document_id"],
            stage=stage,
            completed_stages=completed_stages,
            total_stages=len(self.STAGE_ORDER),
            retry_count=0,
            error=error,
        )
        for callback in self.callbacks:
            callback.on_error(event)

    @staticmethod
    def _build_failure_result(
        *,
        stage: str,
        error: Exception,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize stage failures into a stable pipeline error payload."""

        return {
            "status": "failed",
            "trace_id": state["trace_id"],
            "document_id": state["document_id"],
            "stage": stage,
            "source_path": state["source_path"],
            "source_sha256": state["source_sha256"],
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

    @staticmethod
    def _resolve_document_id(*, source_path: str, source_sha256: str | None) -> str:
        """Build a stable default document id from the strongest available anchor."""

        if source_sha256:
            digest = hashlib.sha256(source_sha256.encode("utf-8")).hexdigest()
        else:
            digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
        return f"doc_{digest[:16]}"

    @staticmethod
    def _infer_document_id_from_documents(documents: list[dict[str, Any]]) -> str | None:
        """Infer the document id from loader output when one is available."""

        for document in documents:
            metadata = dict(document.get("metadata") or {})
            document_id = metadata.get("document_id") or document.get("document_id")
            if document_id:
                return str(document_id)
        return None

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        """Return the elapsed duration in milliseconds rounded for stable assertions."""

        return round((time.perf_counter() - started_at) * 1000, 3)

    @staticmethod
    def _hash_text(value: str) -> str:
        """Hash fallback text inputs when a byte-level source hash is unavailable."""

        return hashlib.sha256(value.encode("utf-8")).hexdigest()
