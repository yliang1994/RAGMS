"""Shared table, badge, and empty-state helpers for dashboard pages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_STATUS_STYLES = {
    "succeeded": {"label": "Succeeded", "tone": "success"},
    "completed": {"label": "Completed", "tone": "success"},
    "indexed": {"label": "Indexed", "tone": "success"},
    "running": {"label": "Running", "tone": "info"},
    "started": {"label": "Started", "tone": "info"},
    "partial_success": {"label": "Partial Success", "tone": "warning"},
    "skipped": {"label": "Skipped", "tone": "warning"},
    "failed": {"label": "Failed", "tone": "error"},
    "placeholder": {"label": "Placeholder", "tone": "muted"},
}


def render_empty_state(
    title: str,
    description: str,
    *,
    renderer: Any | None = None,
) -> dict[str, str]:
    """Render or return a normalized empty-state payload."""

    payload = {
        "title": str(title).strip() or "No data",
        "description": str(description).strip() or "Nothing to display yet.",
    }
    if renderer is not None:
        renderer.info(f"{payload['title']}: {payload['description']}")
    return payload


def render_status_badge(status: str | None, *, renderer: Any | None = None) -> dict[str, str]:
    """Render or return a normalized status badge payload."""

    normalized = str(status or "unknown").strip().lower() or "unknown"
    style = _STATUS_STYLES.get(normalized, {"label": normalized.replace("_", " ").title(), "tone": "muted"})
    payload = {
        "status": normalized,
        "label": style["label"],
        "tone": style["tone"],
    }
    if renderer is not None:
        renderer.caption(f"[{payload['tone']}] {payload['label']}")
    return payload


def render_table(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    columns: list[str] | None = None,
    empty_title: str = "No records",
    empty_description: str = "No rows are available.",
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return a stable table payload with unified empty-state handling."""

    normalized_rows = [dict(row) for row in (rows or [])]
    if not normalized_rows:
        return {
            "kind": "empty",
            "empty_state": render_empty_state(empty_title, empty_description, renderer=renderer),
            "columns": list(columns or []),
            "rows": [],
            "row_count": 0,
        }

    resolved_columns = list(columns or _infer_columns(normalized_rows))
    payload = {
        "kind": "table",
        "columns": resolved_columns,
        "rows": [
            {column: row.get(column) for column in resolved_columns}
            for row in normalized_rows
        ],
        "row_count": len(normalized_rows),
    }
    if renderer is not None:
        renderer.dataframe(payload["rows"], use_container_width=True)
    return payload


def _infer_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            normalized = str(key)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered
