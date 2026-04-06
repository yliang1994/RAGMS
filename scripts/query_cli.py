"""Minimal local query CLI bootstrap."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ragms.core.query_engine import build_query_engine
from ragms.runtime.config import load_settings
from ragms.runtime.container import build_container


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

    settings = load_settings(args.settings)
    container = build_container(settings)
    requested_collection = args.collection or settings.vector_store.collection

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


if __name__ == "__main__":
    raise SystemExit(run_cli())
