"""Minimal local query CLI bootstrap."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ragms.core.query_engine import build_query_engine
from ragms.runtime.config import load_settings
from ragms.runtime.container import build_container


def _apply_collection_override(settings, collection: str | None):
    """Return settings with an optional query-time collection override applied."""

    if not collection:
        return settings
    updated = settings.model_copy(deep=True)
    updated.vector_store.collection = collection
    return updated


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the local query CLI against the current query-engine pipeline."""

    parser = argparse.ArgumentParser(description="Run the local RagMS query CLI.")
    parser.add_argument("query", nargs="?", default="bootstrap smoke test")
    parser.add_argument(
        "--settings",
        default="settings.yaml",
        help="Path to the settings.yaml file.",
    )
    parser.add_argument("--collection", default=None, help="Override the target collection.")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum number of retrieved chunks.")
    parser.add_argument(
        "--print-top-chunks",
        type=int,
        default=0,
        help="Print the top N retrieved chunks with content previews.",
    )
    parser.add_argument(
        "--filters",
        default=None,
        help="Optional JSON metadata filters.",
    )
    parser.add_argument(
        "--return-debug",
        action="store_true",
        help="Include debug_info in the printed response.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_settings = load_settings(args.settings)
    settings = _apply_collection_override(base_settings, args.collection)
    container = build_container(settings)
    requested_collection = settings.vector_store.collection

    try:
        engine = build_query_engine(container, settings=settings)
        response = engine.run(
            query=args.query,
            collection=args.collection,
            top_k=args.top_k,
            filters=args.filters,
            return_debug=args.return_debug,
        )
        resolved_collection = (
            response["retrieved_chunks"][0]["metadata"].get("collection", requested_collection)
            if response["retrieved_chunks"]
            else requested_collection
        )
        print(
            "Query CLI ready: "
            f"strategy={settings.retrieval.strategy} "
            f"collection={resolved_collection} "
            f"query={args.query}"
        )
        print(f"Answer: {response['answer']}")
        if response["citations"]:
            print("Citations:")
            for citation in response["citations"]:
                location = citation.get("source_path") or citation.get("document_id")
                page = citation.get("page")
                suffix = "" if page is None else f" p.{page}"
                print(f"{citation['marker']} {location}{suffix}")
        else:
            print("Citations: none")
        print(f"Retrieved chunks: {len(response['retrieved_chunks'])}")
        if args.print_top_chunks > 0:
            _print_top_chunks(response["retrieved_chunks"], limit=args.print_top_chunks)
        if args.return_debug:
            print("Debug:")
            print(json.dumps(response.get("debug_info", {}), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(
            "Query CLI ready: "
            f"strategy={settings.retrieval.strategy} "
            f"collection={requested_collection} "
            f"query={args.query}"
        )
        print(f"Query execution unavailable: {exc}")
    return 0


def _print_top_chunks(retrieved_chunks: list[dict[str, object]], *, limit: int) -> None:
    """Print ranked chunk details for manual inspection."""

    print("Top Chunks:")
    if not retrieved_chunks:
        print("(none)")
        return

    for rank, chunk in enumerate(retrieved_chunks[:limit], start=1):
        metadata = dict(chunk.get("metadata") or {})
        source = metadata.get("source_path") or chunk.get("document_id") or "unknown"
        page = metadata.get("page")
        route = chunk.get("source_route") or "unknown"
        score = _select_display_score(chunk)
        suffix = "" if page is None else f" p.{page}"
        print(f"[{rank}] route={route} score={score:.6f} source={source}{suffix}")
        content = " ".join(str(chunk.get("content") or "").split()).strip()
        print(content if content else "(empty content)")


def _select_display_score(chunk: dict[str, object]) -> float:
    """Return the most relevant score already exposed in the response payload."""

    for key in ("rerank_score", "rrf_score", "score"):
        value = chunk.get(key)
        if value is None:
            continue
        return float(value)
    return 0.0


if __name__ == "__main__":
    raise SystemExit(run_cli())
