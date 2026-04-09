"""Data browser page renderer."""

from __future__ import annotations

from typing import Any

from ragms.observability.dashboard.components import render_empty_state, render_table


def render_data_browser(
    context: Any,
    *,
    collection: str | None = None,
    status: str | None = None,
    keyword: str | None = None,
    page: int | None = None,
    tag: str | None = None,
    document_id: str | None = None,
    chunk_id: str | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return the dashboard data browser payload."""

    collections_payload = context.data_service.list_collections()
    all_documents_payload = context.data_service.list_documents()
    filters = {
        key: value
        for key, value in {
            "collection": collection,
            "status": status,
            "keyword": keyword,
            "page": page,
            "tag": tag,
        }.items()
        if value not in (None, "")
    }
    documents_payload = context.data_service.list_documents(filters=filters)
    documents = documents_payload.get("documents") or []

    selected_document_id = document_id or (documents[0]["document_id"] if documents else None)
    selected_document = (
        context.data_service.get_document_detail(selected_document_id)
        if selected_document_id is not None
        else None
    )

    available_chunk_ids = [chunk["chunk_id"] for chunk in (selected_document or {}).get("chunks") or []]
    selected_chunk_id = chunk_id or (available_chunk_ids[0] if available_chunk_ids else None)
    selected_chunk = (
        context.data_service.get_chunk_detail(selected_chunk_id)
        if selected_chunk_id is not None
        else None
    )
    document_chunks = (selected_document or {}).get("chunks") or []
    related_traces = (selected_document or {}).get("related_traces") or []
    chunk_images = (selected_chunk or {}).get("images") or []

    payload = {
        "kind": "data_browser",
        "title": "数据浏览器",
        "filters": filters,
        "filter_model": _build_filter_model(
            collections_payload=collections_payload,
            all_documents_payload=all_documents_payload,
        ),
        "collections": render_table(
            [
                {**row, "target_page": "data_browser"}
                for row in collections_payload.get("collections") or []
            ],
            columns=["name", "document_count", "chunk_count", "image_count", "target_page"],
            empty_title="暂无集合",
            empty_description="尚未发现可浏览的 collection。",
        ),
        "documents": render_table(
            [
                {
                    **row,
                    "target_page": "data_browser",
                }
                for row in documents
            ],
            columns=["document_id", "primary_collection", "status", "chunk_count", "image_count", "page_count", "target_page"],
            empty_title="暂无文档",
            empty_description="当前过滤条件下没有匹配的文档。",
        ),
        "selected_document_id": selected_document_id,
        "selected_chunk_id": selected_chunk_id,
        "selected_document": selected_document,
        "selected_chunk": selected_chunk,
        "document_chunks": render_table(
            [
                {**row, "target_page": "data_browser"}
                for row in document_chunks
            ],
            columns=["chunk_id", "collection", "chunk_index", "page", "section_path", "tags", "summary", "target_page"],
            empty_title="暂无 Chunk",
            empty_description="当前文档还没有可浏览的 chunk。",
        ),
        "document_traces": render_table(
            related_traces,
            columns=["trace_id", "status", "duration_ms", "started_at", "target_page"],
            empty_title="暂无关联 Trace",
            empty_description="当前文档尚未发现可回跳的 ingestion trace。",
        ),
        "chunk_images": render_table(
            [
                {
                    **row,
                    "preview_label": f"第 {row.get('page') or '?'} 页图片",
                }
                for row in chunk_images
            ],
            columns=["image_id", "page", "position", "file_path", "preview_label"],
            empty_title="暂无图片预览",
            empty_description="当前 chunk 没有关联图片。",
        ),
        "document_empty_state": None if selected_document is not None else render_empty_state("暂无文档详情", "请选择文档后查看 chunk 与 trace 关联。"),
        "chunk_empty_state": None if selected_chunk is not None else render_empty_state("暂无 Chunk 详情", "当前文档下没有可展示的 chunk。"),
        "navigation": [
            {"label": "跳转到系统总览", "target_page": "system_overview"},
            {"label": "跳转到 Ingestion Trace", "target_page": "ingestion_trace", "document_id": selected_document_id},
        ],
    }
    if renderer is not None:
        _render_data_browser_streamlit(payload, renderer)
    return payload


def _render_data_browser_streamlit(payload: dict[str, Any], renderer: Any) -> None:
    renderer.subheader(payload["title"])
    renderer.caption(f"当前过滤条件: {payload['filters'] or '无'}")

    renderer.markdown("### Filters")
    renderer.code(str(payload["filter_model"]), language="python")

    renderer.markdown("### Collections")
    _render_table_or_empty(payload["collections"], renderer)

    renderer.markdown("### Documents")
    _render_table_or_empty(payload["documents"], renderer)

    renderer.markdown("### Document Detail")
    if payload["selected_document"] is None:
        render_empty_state(
            payload["document_empty_state"]["title"],
            payload["document_empty_state"]["description"],
            renderer=renderer,
        )
    else:
        renderer.code(str(payload["selected_document"]), language="python")
        renderer.markdown("#### Related Chunks")
        _render_table_or_empty(payload["document_chunks"], renderer)
        renderer.markdown("#### Related Ingestion Traces")
        _render_table_or_empty(payload["document_traces"], renderer)

    renderer.markdown("### Chunk Detail")
    if payload["selected_chunk"] is None:
        render_empty_state(
            payload["chunk_empty_state"]["title"],
            payload["chunk_empty_state"]["description"],
            renderer=renderer,
        )
    else:
        renderer.code(str(payload["selected_chunk"]), language="python")
        renderer.markdown("#### Chunk Images")
        _render_table_or_empty(payload["chunk_images"], renderer)


def _render_table_or_empty(table_payload: dict[str, Any], renderer: Any) -> None:
    if table_payload["kind"] == "empty":
        render_empty_state(
            table_payload["empty_state"]["title"],
            table_payload["empty_state"]["description"],
            renderer=renderer,
        )
        return
    render_table(
        table_payload["rows"],
        columns=table_payload["columns"],
        renderer=renderer,
    )


def _build_filter_model(
    *,
    collections_payload: dict[str, Any],
    all_documents_payload: dict[str, Any],
) -> dict[str, Any]:
    documents = all_documents_payload.get("documents") or []
    return {
        "collections": [row.get("name") for row in collections_payload.get("collections") or []],
        "statuses": sorted(
            {
                str(row.get("status") or "").strip()
                for row in documents
                if str(row.get("status") or "").strip()
            }
        ),
        "pages": sorted(
            {
                int(page)
                for row in documents
                for page in row.get("pages") or []
                if page is not None
            }
        ),
        "tags": sorted(
            {
                str(tag).strip()
                for row in documents
                for tag in row.get("tags") or []
                if str(tag).strip()
            }
        ),
    }
