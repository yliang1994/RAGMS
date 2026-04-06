"""`ingest_documents` MCP tool adapter."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from mcp import types

from ragms.ingestion_pipeline.bootstrap import (
    build_ingestion_pipeline,
    discover_ingestion_sources,
    run_ingestion_batch,
)
from ragms.mcp_server.protocol_handler import ProtocolHandler
from ragms.runtime.container import ServiceContainer


def normalize_ingest_request(
    paths: list[str],
    collection: str | None = None,
    force_rebuild: bool = False,
    options: dict[str, Any] | None = None,
    *,
    protocol_handler: ProtocolHandler | None = None,
):
    """Validate and normalize ingest tool arguments."""

    handler = protocol_handler or ProtocolHandler()
    request = handler.validate_arguments(
        "ingest_documents",
        {
            "paths": paths,
            "collection": collection,
            "force_rebuild": force_rebuild,
            "options": options,
        },
    )

    deduped_paths: list[str] = []
    seen: set[str] = set()
    for raw_path in request.paths:
        normalized = str(raw_path).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_paths.append(normalized)

    return request.model_copy(update={"paths": deduped_paths, "options": dict(request.options or {})})


def serialize_ingestion_result(
    source_path: str | Path,
    result: dict[str, Any] | None,
    *,
    collection: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Convert a raw ingestion payload into a stable MCP document summary."""

    normalized_source = str(Path(source_path))
    payload = dict(result or {})
    lifecycle = dict(payload.get("lifecycle") or {})
    final_status = str(lifecycle.get("final_status") or payload.get("status") or ("failed" if error_message else "unknown"))
    error_payload = dict(payload.get("error") or {})
    if error_message and "message" not in error_payload:
        error_payload["message"] = error_message

    return {
        "source_path": normalized_source,
        "collection": collection,
        "document_id": payload.get("document_id"),
        "trace_id": payload.get("trace_id"),
        "status": final_status,
        "current_stage": payload.get("current_stage"),
        "source_sha256": payload.get("source_sha256"),
        "chunk_count": len(payload.get("smart_chunks") or payload.get("chunks") or []),
        "stored_count": len(payload.get("stored_ids") or []),
        "skipped": final_status == "skipped",
        "error": error_payload or None,
    }


def handle_ingest_documents(
    paths: list[str],
    collection: str | None = None,
    force_rebuild: bool = False,
    options: dict[str, Any] | None = None,
    *,
    runtime: ServiceContainer,
    pipeline_builder: Callable[..., Any] | None = None,
    source_discovery: Callable[[list[str]], tuple[list[Path], list[dict[str, str]]]] | None = None,
    batch_runner: Callable[..., list[dict[str, Any]]] | None = None,
    protocol_handler: ProtocolHandler | None = None,
) -> types.CallToolResult:
    """Execute the ingest tool and return a normalized MCP result."""

    handler = protocol_handler or ProtocolHandler()
    request = normalize_ingest_request(
        paths=paths,
        collection=collection,
        force_rebuild=force_rebuild,
        options=options,
        protocol_handler=handler,
    )
    resolved_collection = request.collection or runtime.settings.vector_store.collection
    build_pipeline = pipeline_builder or build_ingestion_pipeline
    discover_sources = source_discovery or discover_ingestion_sources
    run_batch = batch_runner or run_ingestion_batch

    try:
        pipeline = build_pipeline(runtime.settings, collection=request.collection)
        sources, source_errors = discover_sources(request.paths)
        batch_results = run_batch(
            pipeline,
            sources=sources,
            collection=resolved_collection,
            force_rebuild=request.force_rebuild,
        )
    except Exception as exc:
        error = handler.serialize_exception(exc)
        return handler.build_error_response(
            code=error.code,
            message=error.message,
            data=error.data,
        )

    documents = [
        serialize_ingestion_result(
            item["source_path"],
            item["result"],
            collection=resolved_collection,
        )
        for item in batch_results
    ]
    documents.extend(
        serialize_ingestion_result(
            error["path"],
            None,
            collection=resolved_collection,
            error_message=error["message"],
        )
        for error in source_errors
    )

    indexed_count = sum(1 for item in documents if item["status"] in {"indexed", "completed"})
    skipped_count = sum(1 for item in documents if item["status"] == "skipped")
    failed_count = sum(1 for item in documents if item["status"] == "failed")
    batch_trace_id = documents[0]["trace_id"] if len(documents) == 1 and documents[0]["trace_id"] else uuid.uuid4().hex
    summary = {
        "requested_path_count": len(request.paths),
        "resolved_source_count": len(batch_results),
        "accepted_count": len(batch_results),
        "document_count": len(documents),
        "indexed_count": indexed_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
    }
    structured_content = {
        "trace_id": batch_trace_id,
        "collection": resolved_collection,
        "requested_paths": request.paths,
        "options": request.options,
        "summary": summary,
        "documents": documents,
        "skipped_summary": {
            "count": skipped_count,
            "documents": [item for item in documents if item["status"] == "skipped"],
        },
        "failure_summary": {
            "count": failed_count,
            "documents": [item for item in documents if item["status"] == "failed"],
        },
    }
    text = (
        f"Ingestion accepted {summary['accepted_count']} source(s): "
        f"indexed {indexed_count}, skipped {skipped_count}, failed {failed_count}."
    )
    return handler.build_success_response(
        text=text,
        structured_content=structured_content,
    )


def bind_ingest_tool(runtime: ServiceContainer) -> Callable[..., types.CallToolResult]:
    """Bind the ingest tool to a concrete runtime container for MCP registration."""

    def ingest_documents(
        paths: list[str],
        collection: str | None = None,
        force_rebuild: bool = False,
        options: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        return handle_ingest_documents(
            paths=paths,
            collection=collection,
            force_rebuild=force_rebuild,
            options=options,
            runtime=runtime,
        )

    ingest_documents.__name__ = "ingest_documents"
    ingest_documents.__doc__ = "Trigger local document ingestion."
    return ingest_documents
