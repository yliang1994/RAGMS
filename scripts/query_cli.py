"""Minimal local query CLI bootstrap."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ragms.runtime.config import load_settings
from ragms.runtime.container import build_container


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Load runtime settings and print a placeholder query execution summary."""

    parser = argparse.ArgumentParser(description="Run the local RagMS query CLI bootstrap.")
    parser.add_argument("query", nargs="?", default="bootstrap smoke test")
    parser.add_argument(
        "--settings",
        default="settings.yaml",
        help="Path to the settings.yaml file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.settings)
    container = build_container(settings)
    print(
        "Query CLI ready: "
        f"strategy={container.get('retrieval').implementation} "
        f"collection={settings.vector_store.collection} "
        f"query={args.query}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
