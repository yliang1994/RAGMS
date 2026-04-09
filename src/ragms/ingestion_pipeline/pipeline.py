"""Core ingestion pipeline orchestration with stage callbacks and unified failures."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from ragms.core.trace_collector import TraceManager
from ragms.core.trace_collector.trace_schema import BaseTrace
from ragms.storage.traces import TraceRepository
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


def attach_ingestion_trace(
    metadata: dict[str, Any] | None,
    *,
    trace_id: str,
    document_id: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Attach stable trace context to ingestion metadata."""

    attached = dict(metadata or {})
    attached["trace_id"] = trace_id
    if document_id is not None:
        attached["document_id"] = document_id
    if source_sha256 is not None:
        attached["source_sha256"] = source_sha256
    return attached


def record_ingestion_stage(
    trace_manager: TraceManager | None,
    trace: BaseTrace | None,
    *,
    stage_name: str,
    input_payload: Any = None,
    output_payload: Any = None,
    metadata: dict[str, Any] | None = None,
    status: str = "succeeded",
    error: BaseException | None = None,
) -> None:
    """Record one ingestion stage if tracing is enabled."""

    if trace_manager is None or trace is None:
        return
    if stage_name not in trace._active_stages:
        trace_manager.start_stage(
            trace,
            stage_name,
            input_payload=input_payload,
            metadata=metadata,
        )
    trace_manager.finish_stage(
        trace,
        stage_name,
        status=status,
        output_payload=output_payload,
        metadata=metadata,
        error=error,
    )


class IngestionPipeline:
    """Orchestrate the ingestion stages from integrity checks to lifecycle finalization."""

    STAGE_ORDER = (
        "file_integrity",
        "load",
        "chunking",
        "transform",
        "embedding",
        "storage",
        "lifecycle_finalize",
    )

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
        chunking_pipeline: Any | None = None,
        dense_encoder: Any | None = None,
        sparse_encoder: Any | None = None,
        storage_pipeline: Any | None = None,
        bm25_writer: Any | None = None,
        image_storage_writer: Any | None = None,
        trace_manager: TraceManager | None = None,
        trace_repository: TraceRepository | None = None,
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.transform = transform
        self.embedding = embedding
        self.vector_store = vector_store
        self.file_integrity = file_integrity
        self.document_registry = document_registry
        self.callbacks = list(callbacks or [])
        self.chunking_pipeline = chunking_pipeline
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.storage_pipeline = storage_pipeline
        self.bm25_writer = bm25_writer
        self.image_storage_writer = image_storage_writer
        self.trace_manager = trace_manager or TraceManager()
        self.trace_repository = trace_repository

    def run(
        self,
        source_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the ingestion stages and return a unified success, skipped, or failure payload."""

        normalized_source = str(Path(source_path))
        trace = self.trace_manager.start_trace(
            "ingestion",
            trace_id=str((metadata or {}).get("trace_id") or "").strip() or None,
            collection=(metadata or {}).get("collection"),
            metadata={
                "loader": getattr(self.loader, "implementation", self.loader.__class__.__name__),
                "splitter": getattr(self.splitter, "implementation", self.splitter.__class__.__name__),
            },
            source_path=normalized_source,
            document_id=None,
        )
        state: dict[str, Any] = {
            "status": "running",
            "trace_id": trace.trace_id,
            "trace": trace,
            "source_path": normalized_source,
            "document_id": None,
            "source_sha256": None,
            "current_stage": "pipeline_start",
            "metadata": dict(metadata or {}),
            "documents": [],
            "chunks": [],
            "smart_chunks": [],
            "vectors": [],
            "sparse_vectors": [],
            "chunk_records": [],
            "stored_ids": [],
            "bm25": {},
            "stored_images": {},
        }
        self._emit_pipeline_start(state)
        self._emit_progress(
            state=state,
            completed_stages=0,
            current_stage="pipeline_start",
            status="started",
            elapsed_ms=0.0,
        )

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
            state["current_stage"] = "lifecycle_finalize"
            state["lifecycle"] = lifecycle_payload
            self._finish_trace(
                state,
                status="skipped",
                skipped=file_integrity_payload.get("skip_reason") or True,
            )
            return state

        completed_stages = 1
        if self.document_registry is not None and state["document_id"] is not None:
            self._promote_registry_to_processing(state["document_id"])

        stage_handlers: list[tuple[str, Any]] = [
            ("load", self._run_load_stage),
            ("chunking", self._run_split_stage),
            ("transform", self._run_transform_stage),
            ("embedding", self._run_embedding_stage),
            ("storage", self._run_store_stage),
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
        state["current_stage"] = "lifecycle_finalize"
        state["lifecycle"] = lifecycle_payload
        self._finish_trace(state, status="succeeded", skipped=False)
        return state

    def _run_file_integrity_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the file integrity gate, register the document, and decide skip behavior."""

        stage = "file_integrity"
        index = 1
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()
        trace_metadata = {"method": "sha256", "retry_count": 0}
        self.trace_manager.start_stage(
            state["trace"],
            stage,
            input_payload={
                "source_path": state["source_path"],
                "force_rebuild": bool(state["metadata"].get("force_rebuild", False)),
            },
            metadata=trace_metadata,
        )

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
        skip_reason = "content_unchanged" if should_skip and not force_rebuild else None

        state["source_sha256"] = source_sha256
        state["document_id"] = self._resolve_document_id(
            source_path=state["source_path"],
            source_sha256=source_sha256,
        )
        state["metadata"] = attach_ingestion_trace(
            state["metadata"],
            trace_id=state["trace_id"],
            document_id=state["document_id"],
            source_sha256=source_sha256,
        )

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
            "skip_reason": skip_reason,
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
        record_ingestion_stage(
            self.trace_manager,
            state["trace"],
            stage_name=stage,
            output_payload=payload,
            metadata={
                **trace_metadata,
                "document_id": state["document_id"],
                "skipped": should_skip,
                "skip_reason": skip_reason,
            },
        )
        self._emit_progress(
            state=state,
            completed_stages=index,
            current_stage=stage,
            status="running",
            elapsed_ms=self._pipeline_elapsed_ms(state),
        )
        return payload

    def _run_load_stage(self, state: dict[str, Any], index: int) -> None:
        """Run the document loading stage."""

        stage = "load"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()
        trace_metadata = {
            "provider": getattr(self.loader, "implementation", self.loader.__class__.__name__),
            "retry_count": 0,
        }
        self.trace_manager.start_stage(
            state["trace"],
            stage,
            input_payload={"source_path": state["source_path"]},
            metadata=trace_metadata,
        )

        documents = self.loader.load(state["source_path"], metadata=state["metadata"])
        state["documents"] = list(documents)
        inferred_document_id = self._infer_document_id_from_documents(state["documents"])
        if inferred_document_id and state["document_id"] is None:
            state["document_id"] = inferred_document_id
            state["metadata"]["document_id"] = inferred_document_id

        payload = {
            "document_count": len(state["documents"]),
            "image_count": self._count_document_images(state["documents"]),
        }
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        record_ingestion_stage(
            self.trace_manager,
            state["trace"],
            stage_name=stage,
            output_payload=payload,
            metadata=trace_metadata,
        )
        self._emit_progress(
            state=state,
            completed_stages=index,
            current_stage=stage,
            status="running",
            elapsed_ms=self._pipeline_elapsed_ms(state),
        )

    def _run_split_stage(self, state: dict[str, Any], index: int) -> None:
        """Split canonical documents into chunk dictionaries."""

        stage = "chunking"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()
        trace_metadata = {
            "provider": getattr(self.splitter, "implementation", self.splitter.__class__.__name__),
            "retry_count": 0,
        }
        self.trace_manager.start_stage(
            state["trace"],
            stage,
            input_payload={"document_count": len(state["documents"])},
            metadata=trace_metadata,
        )

        chunks: list[dict[str, Any]] = []
        for document in state["documents"]:
            if self.chunking_pipeline is None:
                chunks.extend(self.splitter.split(document))
                continue
            normalized_chunks = self.chunking_pipeline.run(document)
            chunks.extend(
                chunk.to_dict() if hasattr(chunk, "to_dict") else dict(chunk)
                for chunk in normalized_chunks
            )
        state["chunks"] = chunks

        payload = {
            "chunk_count": len(chunks),
            "average_chunk_length": self._average_content_length(chunks),
            "image_occurrence_count": self._count_chunk_image_occurrences(chunks),
        }
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        record_ingestion_stage(
            self.trace_manager,
            state["trace"],
            stage_name=stage,
            output_payload=payload,
            metadata=trace_metadata,
        )
        self._emit_progress(
            state=state,
            completed_stages=index,
            current_stage=stage,
            status="running",
            elapsed_ms=self._pipeline_elapsed_ms(state),
        )

    def _run_transform_stage(self, state: dict[str, Any], index: int) -> None:
        """Run the transform stage or pass chunks through unchanged."""

        stage = "transform"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()
        trace_metadata = {
            "provider": self.transform.__class__.__name__ if self.transform is not None else "disabled",
            "retry_count": 0,
        }
        self.trace_manager.start_stage(
            state["trace"],
            stage,
            input_payload={"chunk_count": len(state["chunks"])},
            metadata=trace_metadata,
        )

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

        payload = {
            "smart_chunk_count": len(state["smart_chunks"]),
            "warning_count": self._count_transform_warnings(state["smart_chunks"]),
            "fallback_count": self._count_transform_fallbacks(state["smart_chunks"]),
        }
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        record_ingestion_stage(
            self.trace_manager,
            state["trace"],
            stage_name=stage,
            output_payload=payload,
            metadata=trace_metadata,
        )
        self._emit_progress(
            state=state,
            completed_stages=index,
            current_stage=stage,
            status="running",
            elapsed_ms=self._pipeline_elapsed_ms(state),
        )

    def _run_embedding_stage(self, state: dict[str, Any], index: int) -> None:
        """Run the embedding stage or skip it when no embedding backend is configured."""

        stage = "embedding"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()
        trace_metadata = {
            "provider": getattr(self.embedding, "implementation", self.embedding.__class__.__name__)
            if self.embedding is not None
            else "disabled",
            "retry_count": 0,
        }
        self.trace_manager.start_stage(
            state["trace"],
            stage,
            input_payload={"smart_chunk_count": len(state["smart_chunks"])},
            metadata=trace_metadata,
        )

        if self.embedding is None or not state["smart_chunks"]:
            state["vectors"] = []
            state["sparse_vectors"] = []
        else:
            if self.dense_encoder is not None:
                state["vectors"] = self.dense_encoder.encode_documents(state["smart_chunks"])
            else:
                texts = [str(chunk.get("content", "")) for chunk in state["smart_chunks"]]
                state["vectors"] = self.embedding.embed_documents(texts)

            if self.sparse_encoder is not None:
                state["sparse_vectors"] = self.sparse_encoder.encode(state["smart_chunks"])
            else:
                state["sparse_vectors"] = [
                    self._empty_sparse_vector(chunk) for chunk in state["smart_chunks"]
                ]

        payload = {
            "vector_count": len(state["vectors"]),
            "sparse_vector_count": len(state["sparse_vectors"]),
            "batch_count": 0 if not state["smart_chunks"] else 1,
            "vector_dimension": self._infer_vector_dimension(state["vectors"]),
        }
        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        record_ingestion_stage(
            self.trace_manager,
            state["trace"],
            stage_name=stage,
            output_payload=payload,
            metadata=trace_metadata,
        )
        self._emit_progress(
            state=state,
            completed_stages=index,
            current_stage=stage,
            status="running",
            elapsed_ms=self._pipeline_elapsed_ms(state),
        )

    def _run_store_stage(self, state: dict[str, Any], index: int) -> None:
        """Persist vectorized chunks or skip persistence when backend is absent."""

        stage = "storage"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()
        trace_metadata = {
            "provider": getattr(self.vector_store, "implementation", self.vector_store.__class__.__name__)
            if self.vector_store is not None
            else self.storage_pipeline.__class__.__name__ if self.storage_pipeline is not None else "disabled",
            "retry_count": 0,
        }
        self.trace_manager.start_stage(
            state["trace"],
            stage,
            input_payload={
                "smart_chunk_count": len(state["smart_chunks"]),
                "vector_count": len(state["vectors"]),
            },
            metadata=trace_metadata,
        )

        if self.storage_pipeline is None:
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
        else:
            smart_chunks = list(state["smart_chunks"])
            vectors = list(state["vectors"])
            sparse_vectors = list(state.get("sparse_vectors") or [])
            if smart_chunks and not vectors:
                raise ValueError("Embedding vectors are required before storage persistence")
            if not sparse_vectors and smart_chunks:
                sparse_vectors = [self._empty_sparse_vector(chunk) for chunk in smart_chunks]
            storage_result = self.storage_pipeline.run(
                smart_chunks,
                dense_vectors=vectors,
                sparse_vectors=sparse_vectors,
            )
            state["chunk_records"] = list(storage_result.get("chunk_records") or [])
            state["stored_ids"] = list(storage_result.get("written_ids") or [])
            bm25_result = {}
            image_result = {}
            if self.bm25_writer is not None:
                bm25_result = self.bm25_writer.index(state["chunk_records"])
                state["bm25"] = dict(bm25_result)
            if self.image_storage_writer is not None:
                image_result = self.image_storage_writer.save_all(state["chunk_records"])
                state["stored_images"] = dict(image_result)
            payload = {
                "stored_count": len(state["stored_ids"]),
                "record_count": int(storage_result.get("record_count", 0) or 0),
                "bm25_indexed_count": int(bm25_result.get("indexed_count", 0) or 0),
                "stored_image_count": int(image_result.get("stored_count", 0) or 0),
            }

        self._emit_stage_end(
            state=state,
            stage=stage,
            index=index,
            status="completed",
            elapsed_ms=self._elapsed_ms(started_at),
            payload=payload,
        )
        record_ingestion_stage(
            self.trace_manager,
            state["trace"],
            stage_name=stage,
            output_payload=payload,
            metadata=trace_metadata,
        )
        self._emit_progress(
            state=state,
            completed_stages=index,
            current_stage=stage,
            status="running",
            elapsed_ms=self._pipeline_elapsed_ms(state),
        )

    def _run_lifecycle_stage(
        self,
        state: dict[str, Any],
        *,
        index: int,
        final_status: str,
        completed_before: int,
    ) -> dict[str, Any]:
        """Finalize the document lifecycle state for success, skip, or failure."""

        stage = "lifecycle_finalize"
        self._emit_stage_start(state=state, stage=stage, index=index)
        started_at = time.perf_counter()
        trace_metadata = {
            "method": "document_registry_finalize",
            "retry_count": 0,
        }
        self.trace_manager.start_stage(
            state["trace"],
            stage,
            input_payload={
                "document_id": state["document_id"],
                "final_status": final_status,
            },
            metadata=trace_metadata,
        )

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
        if (
            final_status in {"indexed", "skipped"}
            and self.file_integrity is not None
            and hasattr(self.file_integrity, "mark_success")
        ):
            history_record = self.file_integrity.mark_success(
                state["source_path"],
                source_sha256=state["source_sha256"],
                document_id=state["document_id"],
                status=final_status,
            )
            payload["ingestion_history_status"] = str(history_record.get("status", final_status))

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
            elapsed_ms=self._pipeline_elapsed_ms(state),
        )
        record_ingestion_stage(
            self.trace_manager,
            state["trace"],
            stage_name=stage,
            output_payload=payload,
            metadata=trace_metadata,
            status="succeeded" if final_status == "indexed" else final_status,
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
        record_ingestion_stage(
            self.trace_manager,
            state["trace"],
            stage_name=stage,
            output_payload=failure["error"],
            metadata={"retry_count": 0},
            status="failed",
            error=error,
        )
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
            elapsed_ms=self._pipeline_elapsed_ms(state),
        )
        self._emit_error(
            state=state,
            stage=stage,
            error=failure["error"],
            completed_stages=completed_stages,
        )
        if (
            self.file_integrity is not None
            and hasattr(self.file_integrity, "mark_failed")
            and state["source_sha256"] is not None
        ):
            self.file_integrity.mark_failed(
                state["source_path"],
                error_message=failure["error"]["message"],
                source_sha256=state["source_sha256"],
                document_id=state["document_id"],
            )
        lifecycle_payload = self._run_lifecycle_stage(
            state,
            index=len(self.STAGE_ORDER),
            final_status="failed",
            completed_before=completed_stages,
        )
        failure["lifecycle"] = lifecycle_payload
        failure["current_stage"] = "lifecycle_finalize"
        self._finish_trace(state, status="failed", error=error, skipped=False)
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
        target_stage = "lifecycle_finalize"
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
        elapsed_ms: float,
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
            elapsed_ms=elapsed_ms,
            metadata={"collection": state["metadata"].get("collection")},
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
            "sparse_vectors": list(state.get("sparse_vectors", [])),
            "chunk_records": list(state.get("chunk_records", [])),
            "stored_ids": list(state.get("stored_ids", [])),
            "bm25": dict(state.get("bm25") or {}),
            "stored_images": dict(state.get("stored_images") or {}),
            "error": {
                "type": error.__class__.__name__,
                "message": str(error),
            },
        }

    def _finish_trace(
        self,
        state: dict[str, Any],
        *,
        status: str,
        error: BaseException | None = None,
        skipped: bool | str | None = None,
    ) -> None:
        """Finalize and append the ingestion trace."""

        trace = self.trace_manager.finish_trace(
            state["trace"],
            status=status,
            error=error,
            document_id=state["document_id"],
            total_chunks=len(state.get("smart_chunks") or state.get("chunks") or []),
            total_images=self._count_total_images(state),
            skipped=skipped,
            metadata={"collection": state["metadata"].get("collection")},
        )
        if self.trace_repository is not None:
            self.trace_repository.append(trace)

    @staticmethod
    def _average_content_length(items: list[dict[str, Any]]) -> float:
        if not items:
            return 0.0
        total = sum(len(str(item.get("content", ""))) for item in items)
        return round(total / len(items), 3)

    @staticmethod
    def _count_document_images(documents: list[dict[str, Any]]) -> int:
        return sum(len((document.get("metadata") or {}).get("images") or []) for document in documents)

    @staticmethod
    def _count_chunk_image_occurrences(chunks: list[dict[str, Any]]) -> int:
        return sum(len((chunk.get("metadata") or {}).get("image_occurrences") or []) for chunk in chunks)

    @staticmethod
    def _count_transform_warnings(chunks: list[dict[str, Any]]) -> int:
        warnings: set[str] = set()
        for chunk in chunks:
            warnings.update((chunk.get("metadata") or {}).get("transform_warnings") or [])
        return len(warnings)

    @staticmethod
    def _count_transform_fallbacks(chunks: list[dict[str, Any]]) -> int:
        return sum(len((chunk.get("metadata") or {}).get("transform_fallbacks") or []) for chunk in chunks)

    @staticmethod
    def _infer_vector_dimension(vectors: list[list[float]]) -> int:
        return len(vectors[0]) if vectors else 0

    @staticmethod
    def _count_total_images(state: dict[str, Any]) -> int:
        if state.get("stored_images"):
            return len(state["stored_images"])
        image_ids: set[str] = set()
        for chunk in state.get("smart_chunks") or []:
            image_ids.update(
                str(image_id)
                for image_id in (
                    chunk.get("image_refs")
                    or (chunk.get("metadata") or {}).get("image_refs")
                    or []
                )
            )
        if image_ids:
            return len(image_ids)
        return sum(
            len((document.get("metadata") or {}).get("images") or [])
            for document in state.get("documents", [])
        )

    @staticmethod
    def _pipeline_elapsed_ms(state: dict[str, Any]) -> float:
        trace = state.get("trace")
        started_at = getattr(trace, "_started_at_dt", None)
        if started_at is None:
            return 0.0
        return round((time.time() - started_at.timestamp()) * 1000, 3)

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

    @staticmethod
    def _empty_sparse_vector(chunk: dict[str, Any]) -> dict[str, Any]:
        """Return a minimal sparse payload when sparse encoding is unavailable."""

        content = str(chunk.get("content", ""))
        return {
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "tokens": [],
            "term_frequencies": {},
            "term_weights": {},
            "document_length": 0,
            "unique_terms": 0,
        }
