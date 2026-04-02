"""Minimal local ingestion bootstrap."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ragms.runtime.config import load_settings
from ragms.runtime.container import build_container


def ingest_documents_main(argv: Sequence[str] | None = None) -> int:
    """Load runtime settings and print a placeholder ingestion summary."""

    parser = argparse.ArgumentParser(description="Run the local RagMS ingestion bootstrap.")
    parser.add_argument(
        "--settings",
        default="settings.yaml",
        help="Path to the settings.yaml file.",
    )
    parser.add_argument(
        "--source-dir",
        default="data/raw/documents",
        help="Directory that would contain source documents.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.settings)
    container = build_container(settings)
    print(
        "Ingestion bootstrap ready: "
        f"source_dir={args.source_dir} "
        f"embedding={container.get('embedding').implementation} "
        f"data_dir={settings.paths.data_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(ingest_documents_main())
