from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ragms.runtime.config import load_settings


DEFAULT_SETTINGS = REPO_ROOT / "settings.yaml"
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "src_raw_data"
DEFAULT_COLLECTION = "real_c_ingestion_test"
RESULT_PATTERN = re.compile(
    r"^result source=(?P<source>.+?) status=(?P<status>\w+) chunks=(?P<chunks>\d+) stored=(?P<stored>\d+)$"
)


@dataclass(frozen=True)
class RunSummary:
    label: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    results: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real end-to-end ingestion validation against scripts/ingest_documents.py."
    )
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS),
        help="Path to settings.yaml.",
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory that contains the source PDFs to ingest.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Collection name used for this real ingestion test.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=5,
        help="Expected number of PDF files in the source directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings_path = Path(args.settings).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()

    ensure_preconditions(
        settings_path=settings_path,
        source_dir=source_dir,
        expected_count=args.expected_count,
    )

    pdf_paths = sorted(source_dir.glob("*.pdf"))
    env = build_pythonpath_env()

    run_summaries = [
        run_ingest(
            label="force-index",
            settings_path=settings_path,
            source_dir=source_dir,
            collection=args.collection,
            force=True,
            env=env,
        ),
        run_ingest(
            label="incremental-skip",
            settings_path=settings_path,
            source_dir=source_dir,
            collection=args.collection,
            force=False,
            env=env,
        ),
        run_ingest(
            label="force-rebuild",
            settings_path=settings_path,
            source_dir=source_dir,
            collection=args.collection,
            force=True,
            env=env,
        ),
    ]

    validate_run_results(run_summaries[0], expected_status="indexed", expected_count=len(pdf_paths))
    validate_run_results(run_summaries[1], expected_status="skipped", expected_count=len(pdf_paths))
    validate_run_results(run_summaries[2], expected_status="indexed", expected_count=len(pdf_paths))

    artifact_summary = inspect_artifacts(
        settings_path=settings_path,
        collection=args.collection,
        pdf_paths=pdf_paths,
    )

    print_summary(
        run_summaries=run_summaries,
        artifact_summary=artifact_summary,
        settings_path=settings_path,
        source_dir=source_dir,
        collection=args.collection,
    )
    return 0


def ensure_preconditions(*, settings_path: Path, source_dir: Path, expected_count: int) -> None:
    if not settings_path.is_file():
        raise SystemExit(f"settings file not found: {settings_path}")
    if not source_dir.is_dir():
        raise SystemExit(f"source directory not found: {source_dir}")
    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"no pdf files found under: {source_dir}")
    if len(pdf_paths) != expected_count:
        raise SystemExit(
            f"expected {expected_count} pdf files under {source_dir}, found {len(pdf_paths)}"
        )
    for pdf_path in pdf_paths:
        if not pdf_path.is_file():
            raise SystemExit(f"missing source file: {pdf_path}")


def build_pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    repo_root = str(REPO_ROOT)
    src_root = str(SRC_ROOT)
    parts = [repo_root, src_root]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_ingest(
    *,
    label: str,
    settings_path: Path,
    source_dir: Path,
    collection: str,
    force: bool,
    env: dict[str, str],
) -> RunSummary:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "ingest_documents.py"),
        "--settings",
        str(settings_path),
        "--path",
        str(source_dir),
        "--collection",
        collection,
    ]
    if force:
        command.append("--force")

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    summary = RunSummary(
        label=label,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        results=parse_result_lines(completed.stdout),
    )
    if summary.returncode != 0:
        raise SystemExit(build_failure_message(summary))
    return summary


def parse_result_lines(stdout: str) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for line in stdout.splitlines():
        matched = RESULT_PATTERN.match(line.strip())
        if matched is None:
            continue
        payload = matched.groupdict()
        parsed.append(
            {
                "source": payload["source"],
                "status": payload["status"],
                "chunks": int(payload["chunks"]),
                "stored": int(payload["stored"]),
            }
        )
    return parsed


def validate_run_results(summary: RunSummary, *, expected_status: str, expected_count: int) -> None:
    if "stage=store status=completed" not in summary.stdout and expected_status == "indexed":
        raise SystemExit(
            f"{summary.label}: missing store stage completion marker in CLI output.\n{summary.stdout}"
        )
    if len(summary.results) != expected_count:
        raise SystemExit(
            f"{summary.label}: expected {expected_count} result lines, got {len(summary.results)}.\n"
            f"{summary.stdout}"
        )
    actual_statuses = {str(item["status"]) for item in summary.results}
    if actual_statuses != {expected_status}:
        raise SystemExit(
            f"{summary.label}: expected status={expected_status}, got {sorted(actual_statuses)}.\n"
            f"{summary.stdout}"
        )
    if expected_status == "indexed":
        zero_chunk_sources = [str(item["source"]) for item in summary.results if int(item["chunks"]) <= 0]
        if zero_chunk_sources:
            raise SystemExit(
                f"{summary.label}: these sources produced zero chunks: {zero_chunk_sources}"
            )
        zero_store_sources = [str(item["source"]) for item in summary.results if int(item["stored"]) <= 0]
        if zero_store_sources:
            raise SystemExit(
                f"{summary.label}: these sources produced zero stored ids: {zero_store_sources}"
            )


def inspect_artifacts(
    *,
    settings_path: Path,
    collection: str,
    pdf_paths: list[Path],
) -> dict[str, object]:
    settings = load_settings(settings_path)
    metadata_db = settings.storage.sqlite.path
    bm25_index = settings.paths.data_dir / "indexes" / "sparse" / f"{collection}.json"
    vector_store_dir = settings.paths.data_dir / "vector_store" / "chroma"
    image_root = settings.paths.data_dir / "images" / collection

    if not metadata_db.is_file():
        raise SystemExit(f"metadata database not found: {metadata_db}")
    if not bm25_index.is_file():
        raise SystemExit(f"bm25 index not found: {bm25_index}")
    if not vector_store_dir.is_dir():
        raise SystemExit(f"vector store directory not found: {vector_store_dir}")

    source_paths = [str(path.resolve()) for path in pdf_paths]
    placeholders = ", ".join("?" for _ in source_paths)
    connection = sqlite3.connect(metadata_db)
    connection.row_factory = sqlite3.Row
    try:
        document_rows = connection.execute(
            f"""
            SELECT document_id, source_path, status, current_stage, version
            FROM documents
            WHERE source_path IN ({placeholders})
            ORDER BY source_path
            """,
            source_paths,
        ).fetchall()
        history_rows = connection.execute(
            f"""
            SELECT source_path, status, completed_at
            FROM ingestion_history
            WHERE source_path IN ({placeholders})
            ORDER BY source_path
            """,
            source_paths,
        ).fetchall()
        image_rows = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM images
            WHERE collection = ?
            """,
            (collection,),
        ).fetchone()
    finally:
        connection.close()

    if len(document_rows) != len(source_paths):
        raise SystemExit(
            f"documents table contains {len(document_rows)} matching rows, expected {len(source_paths)}"
        )
    if len(history_rows) != len(source_paths):
        raise SystemExit(
            f"ingestion_history contains {len(history_rows)} matching rows, expected {len(source_paths)}"
        )

    final_document_statuses = {str(row["status"]) for row in document_rows}
    final_history_statuses = {str(row["status"]) for row in history_rows}
    if final_document_statuses != {"indexed"}:
        raise SystemExit(f"unexpected document statuses: {sorted(final_document_statuses)}")
    if final_history_statuses != {"indexed"}:
        raise SystemExit(f"unexpected ingestion history statuses: {sorted(final_history_statuses)}")

    vector_files = sorted(path.name for path in vector_store_dir.rglob("*") if path.is_file())
    stored_image_files = sorted(path.name for path in image_root.rglob("*") if path.is_file()) if image_root.exists() else []
    image_count = int(image_rows["count"]) if image_rows is not None else 0

    return {
        "metadata_db": str(metadata_db),
        "bm25_index": str(bm25_index),
        "vector_store_dir": str(vector_store_dir),
        "vector_file_count": len(vector_files),
        "document_count": len(document_rows),
        "history_count": len(history_rows),
        "image_count": image_count,
        "image_dir": str(image_root),
        "stored_image_file_count": len(stored_image_files),
    }


def print_summary(
    *,
    run_summaries: list[RunSummary],
    artifact_summary: dict[str, object],
    settings_path: Path,
    source_dir: Path,
    collection: str,
) -> None:
    print("=== Real Ingestion Test Summary ===")
    print(f"python={sys.executable}")
    print(f"settings={settings_path}")
    print(f"source_dir={source_dir}")
    print(f"collection={collection}")
    for summary in run_summaries:
        print(f"[{summary.label}] returncode={summary.returncode}")
        for result in summary.results:
            print(
                "  "
                f"status={result['status']} "
                f"chunks={result['chunks']} "
                f"stored={result['stored']} "
                f"source={result['source']}"
            )
    print("artifacts:")
    print(f"  metadata_db={artifact_summary['metadata_db']}")
    print(f"  bm25_index={artifact_summary['bm25_index']}")
    print(f"  vector_store_dir={artifact_summary['vector_store_dir']}")
    print(f"  vector_file_count={artifact_summary['vector_file_count']}")
    print(f"  documents={artifact_summary['document_count']}")
    print(f"  ingestion_history={artifact_summary['history_count']}")
    print(f"  image_rows={artifact_summary['image_count']}")
    print(f"  image_dir={artifact_summary['image_dir']}")
    print(f"  stored_image_files={artifact_summary['stored_image_file_count']}")
    if int(artifact_summary["image_count"]) == 0:
        print("note: this dataset did not persist any image rows for the target collection.")


def build_failure_message(summary: RunSummary) -> str:
    lines = [
        f"{summary.label}: ingestion command failed with rc={summary.returncode}",
        f"command={' '.join(summary.command)}",
        "stdout:",
        summary.stdout or "<empty>",
        "stderr:",
        summary.stderr or "<empty>",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
