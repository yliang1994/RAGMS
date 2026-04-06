from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_SETTINGS = REPO_ROOT / "settings.yaml"
DEFAULT_QUERY_FILE = REPO_ROOT / "query_test.txt"
DEFAULT_COLLECTION = "real_c_ingestion_test"
RETRIEVED_PATTERN = re.compile(r"^Retrieved chunks:\s+(?P<count>\d+)$", re.MULTILINE)
TOP_CHUNK_PATTERN = re.compile(r"^\[(?P<rank>\d+)\]\s+route=", re.MULTILINE)


@dataclass(frozen=True)
class QueryRunSummary:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    retrieved_count: int
    printed_top_chunks: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real end-to-end query validation against scripts/query_cli.py."
    )
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS),
        help="Path to settings.yaml.",
    )
    parser.add_argument(
        "--query-file",
        default=str(DEFAULT_QUERY_FILE),
        help="Text file that contains the query to execute.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Collection name used for the query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of retrieved chunks requested from the query CLI.",
    )
    parser.add_argument(
        "--print-top-chunks",
        type=int,
        default=3,
        help="Number of ranked chunks to print for manual inspection.",
    )
    parser.add_argument(
        "--filters",
        default=None,
        help="Optional JSON metadata filters forwarded to the query CLI.",
    )
    parser.add_argument(
        "--return-debug",
        action="store_true",
        help="Forward --return-debug to the query CLI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings_path = Path(args.settings).expanduser().resolve()
    query_file = Path(args.query_file).expanduser().resolve()

    query_text = load_query_text(query_file)
    env = build_pythonpath_env()
    summary = run_query(
        settings_path=settings_path,
        query_text=query_text,
        collection=args.collection,
        top_k=args.top_k,
        print_top_chunks=args.print_top_chunks,
        filters=args.filters,
        return_debug=args.return_debug,
        env=env,
    )
    print_summary(
        summary=summary,
        settings_path=settings_path,
        query_file=query_file,
        collection=args.collection,
        query_text=query_text,
    )
    return 0


def load_query_text(query_file: Path) -> str:
    if not query_file.is_file():
        raise SystemExit(f"query file not found: {query_file}")
    query_text = " ".join(query_file.read_text(encoding="utf-8").split()).strip()
    if not query_text:
        raise SystemExit(f"query file is empty: {query_file}")
    return query_text


def build_pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    parts = [str(REPO_ROOT), str(SRC_ROOT)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_query(
    *,
    settings_path: Path,
    query_text: str,
    collection: str,
    top_k: int,
    print_top_chunks: int,
    filters: str | None,
    return_debug: bool,
    env: dict[str, str],
) -> QueryRunSummary:
    if top_k <= 0:
        raise SystemExit("--top-k must be greater than zero")
    if print_top_chunks < 0:
        raise SystemExit("--print-top-chunks must not be negative")

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "query_cli.py"),
        "--settings",
        str(settings_path),
        "--collection",
        collection,
        "--top-k",
        str(max(top_k, print_top_chunks or 0)),
        "--print-top-chunks",
        str(print_top_chunks),
    ]
    if filters:
        command.extend(["--filters", filters])
    if return_debug:
        command.append("--return-debug")
    command.append(query_text)

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout
    stderr = completed.stderr
    if completed.returncode != 0:
        raise SystemExit(build_failure_message(command, completed.returncode, stdout, stderr))
    if "Query execution unavailable:" in stdout:
        raise SystemExit(build_failure_message(command, completed.returncode, stdout, stderr))

    retrieved_count = parse_retrieved_count(stdout)
    printed_top_chunks = len(TOP_CHUNK_PATTERN.findall(stdout))
    expected_top_chunks = min(retrieved_count, print_top_chunks)
    if printed_top_chunks != expected_top_chunks:
        raise SystemExit(
            "query output did not print the expected number of top chunks.\n"
            f"expected={expected_top_chunks} actual={printed_top_chunks}\n"
            f"stdout:\n{stdout}\n"
        )

    return QueryRunSummary(
        command=command,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        retrieved_count=retrieved_count,
        printed_top_chunks=printed_top_chunks,
    )


def parse_retrieved_count(stdout: str) -> int:
    matched = RETRIEVED_PATTERN.search(stdout)
    if matched is None:
        raise SystemExit(f"query output is missing retrieved chunk count.\n{stdout}")
    return int(matched.group("count"))


def build_failure_message(
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> str:
    return (
        "real query test failed.\n"
        f"command: {' '.join(command)}\n"
        f"returncode: {returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}\n"
    )


def print_summary(
    *,
    summary: QueryRunSummary,
    settings_path: Path,
    query_file: Path,
    collection: str,
    query_text: str,
) -> None:
    print("=== Real Query Test Summary ===")
    print(f"settings: {settings_path}")
    print(f"query_file: {query_file}")
    print(f"collection: {collection}")
    print(f"query: {query_text}")
    print(f"retrieved_chunks: {summary.retrieved_count}")
    print(f"printed_top_chunks: {summary.printed_top_chunks}")
    print("--- CLI stdout ---")
    print(summary.stdout.rstrip())
    if summary.stderr.strip():
        print("--- CLI stderr ---")
        print(summary.stderr.rstrip())


if __name__ == "__main__":
    raise SystemExit(main())
