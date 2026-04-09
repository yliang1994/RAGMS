"""Ingestion management page renderer."""

from __future__ import annotations

from typing import Any

from ragms.observability.dashboard.components import render_empty_state, render_table


def render_ingestion_management(
    context: Any,
    *,
    collection: str | None = None,
    status: str | None = None,
    action: str | None = None,
    document_id: str | None = None,
    uploads: list[dict[str, Any]] | None = None,
    force_rebuild: bool = False,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return the ingestion management page payload."""

    filters = {
        key: value
        for key, value in {
            "collection": collection,
            "status": status,
        }.items()
        if value not in (None, "")
    }
    documents_payload = context.data_service.list_documents(filters=filters)
    action_result = None
    if action == "ingest" and uploads:
        saved_paths = context.document_admin_service.save_uploads(uploads)
        action_result = context.document_admin_service.ingest_documents(
            saved_paths,
            collection=collection,
            force_rebuild=force_rebuild,
        )
        documents_payload = context.data_service.list_documents(filters=filters)
    elif action == "delete" and document_id:
        action_result = context.document_admin_service.delete_document(document_id)
        documents_payload = context.data_service.list_documents(filters=filters)
    elif action == "rebuild" and document_id:
        action_result = context.document_admin_service.rebuild_document(document_id)
        documents_payload = context.data_service.list_documents(filters=filters)

    payload = {
        "kind": "ingestion_management",
        "title": "Ingestion 管理",
        "filters": filters,
        "documents": render_table(
            documents_payload.get("documents") or [],
            columns=[
                "document_id",
                "primary_collection",
                "status",
                "current_stage",
                "failure_reason",
                "skip_reason",
                "version",
                "last_ingested_at",
                "updated_at",
                "page_count",
                "image_count",
            ],
            empty_title="暂无文档",
            empty_description="当前还没有可管理的文档记录。",
        ),
        "action_result": action_result,
        "progress": _build_progress_payload(action_result),
        "navigation": [
            {"label": "跳转到系统总览", "target_page": "system_overview"},
            {"label": "跳转到数据浏览", "target_page": "data_browser"},
            {"label": "跳转到 Ingestion Trace", "target_page": "ingestion_trace"},
        ],
    }
    if renderer is not None:
        _render_ingestion_management_streamlit(payload, renderer)
    return payload


def _build_progress_payload(action_result: dict[str, Any] | None) -> dict[str, Any]:
    if not action_result:
        return {
            "kind": "empty",
            "empty_state": render_empty_state("暂无执行进度", "上传、删除或重建后会在这里显示最近一次管理动作。"),
        }
    progress_events = list(action_result.get("progress_events") or [])
    if not progress_events:
        return {
            "kind": "summary",
            "latest": action_result,
            "events": [],
        }
    return {
        "kind": "progress",
        "latest": progress_events[-1],
        "events": progress_events,
    }


def _render_ingestion_management_streamlit(payload: dict[str, Any], renderer: Any) -> None:
    renderer.subheader(payload["title"])
    renderer.caption(f"当前过滤条件: {payload['filters'] or '无'}")
    renderer.markdown("### Documents")
    _render_table_or_empty(payload["documents"], renderer)
    renderer.markdown("### Latest Action")
    if payload["progress"]["kind"] == "empty":
        render_empty_state(
            payload["progress"]["empty_state"]["title"],
            payload["progress"]["empty_state"]["description"],
            renderer=renderer,
        )
    else:
        renderer.code(str(payload["progress"]["latest"]), language="python")


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
