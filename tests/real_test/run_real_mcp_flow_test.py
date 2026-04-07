from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_SETTINGS = REPO_ROOT / "settings.yaml"
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "src_raw_data"
DEFAULT_QUERY_FILE = REPO_ROOT / "query_test.txt"
DEFAULT_COLLECTION = "real_c_ingestion_test"


@dataclass(frozen=True)
class ToolInvocation:
    step: str
    request: dict[str, Any]
    response: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real end-to-end MCP stdio flow against scripts/run_mcp_server.py."
    )
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS),
        help="Path to settings.yaml.",
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory that contains the source files to ingest.",
    )
    parser.add_argument(
        "--query-file",
        default=str(DEFAULT_QUERY_FILE),
        help="Text file that contains the query to execute.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Collection name used for ingestion and query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k passed to query_knowledge_hub.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Forward force_rebuild=true to ingest_documents.",
    )
    parser.add_argument(
        "--return-debug",
        action="store_true",
        help="Forward return_debug=true to query_knowledge_hub.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="Timeout applied to each MCP response.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level passed to the MCP server process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings_path = Path(args.settings).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    query_file = Path(args.query_file).expanduser().resolve()

    ensure_preconditions(
        settings_path=settings_path,
        source_dir=source_dir,
        query_file=query_file,
        top_k=args.top_k,
        timeout_seconds=args.timeout_seconds,
    )

    query_text = load_query_text(query_file)
    env = build_pythonpath_env()
    python_executable = resolve_python_executable(env)
    protocol_version = resolve_protocol_version(python_executable, env)

    invocations: list[ToolInvocation] = []
    process, stderr_path, server_command = start_server(
        python_executable=python_executable,
        settings_path=settings_path,
        log_level=args.log_level,
        env=env,
    )

    failure: BaseException | None = None

    try:
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "real-mcp-flow-test", "version": "1.0"},
            },
        }
        initialize_response = send_and_read(
            process,
            initialize_request,
            timeout_seconds=args.timeout_seconds,
            stderr_path=stderr_path,
        )
        invocations.append(
            ToolInvocation(
                step="initialize",
                request=initialize_request,
                response=initialize_response,
            )
        )
        assert_jsonrpc_success(
            step="initialize",
            response=initialize_response,
            stderr_path=stderr_path,
        )

        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        send_message(process, initialized_notification)
        invocations.append(
            ToolInvocation(
                step="notifications/initialized",
                request=initialized_notification,
                response={"notification": "sent"},
            )
        )

        tools_list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        tools_list_response = send_and_read(
            process,
            tools_list_request,
            timeout_seconds=args.timeout_seconds,
            stderr_path=stderr_path,
        )
        invocations.append(
            ToolInvocation(
                step="tools/list",
                request=tools_list_request,
                response=tools_list_response,
            )
        )
        assert_jsonrpc_success(
            step="tools/list",
            response=tools_list_response,
            stderr_path=stderr_path,
        )

        pre_collections = call_tool(
            process,
            request_id=3,
            name="list_collections",
            arguments={},
            timeout_seconds=args.timeout_seconds,
            stderr_path=stderr_path,
        )
        invocations.append(pre_collections)
        assert_tool_success(pre_collections, stderr_path=stderr_path)

        ingest = call_tool(
            process,
            request_id=4,
            name="ingest_documents",
            arguments={
                "paths": [str(source_dir)],
                "collection": args.collection,
                "force_rebuild": args.force_rebuild,
            },
            timeout_seconds=args.timeout_seconds,
            stderr_path=stderr_path,
        )
        invocations.append(ingest)
        assert_tool_success(ingest, stderr_path=stderr_path)

        post_collections = call_tool(
            process,
            request_id=5,
            name="list_collections",
            arguments={},
            timeout_seconds=args.timeout_seconds,
            stderr_path=stderr_path,
        )
        invocations.append(post_collections)
        assert_tool_success(post_collections, stderr_path=stderr_path)

        document_ids = extract_document_ids(ingest.response)
        for offset, document_id in enumerate(document_ids, start=6):
            summary_result = call_tool(
                process,
                request_id=offset,
                name="get_document_summary",
                arguments={"document_id": document_id},
                timeout_seconds=args.timeout_seconds,
                stderr_path=stderr_path,
            )
            invocations.append(summary_result)
            assert_tool_success(summary_result, stderr_path=stderr_path)

        query_result = call_tool(
            process,
            request_id=6 + len(document_ids),
            name="query_knowledge_hub",
            arguments={
                "query": query_text,
                "collection": args.collection,
                "top_k": args.top_k,
                "return_debug": args.return_debug,
            },
            timeout_seconds=args.timeout_seconds,
            stderr_path=stderr_path,
        )
        invocations.append(query_result)
        assert_tool_success(query_result, stderr_path=stderr_path)
    except BaseException as exc:
        failure = exc
    finally:
        stop_server(process)

    print_summary(
        settings_path=settings_path,
        source_dir=source_dir,
        query_file=query_file,
        query_text=query_text,
        collection=args.collection,
        python_executable=python_executable,
        protocol_version=protocol_version,
        server_command=server_command,
        invocations=invocations,
        stderr_path=stderr_path,
        failure=failure,
    )
    if failure is not None:
        raise failure
    return 0


def ensure_preconditions(
    *,
    settings_path: Path,
    source_dir: Path,
    query_file: Path,
    top_k: int,
    timeout_seconds: float,
) -> None:
    if not settings_path.is_file():
        raise SystemExit(f"settings file not found: {settings_path}")
    if not source_dir.is_dir():
        raise SystemExit(f"source directory not found: {source_dir}")
    if not any(source_dir.glob("*.pdf")):
        raise SystemExit(f"no pdf files found under: {source_dir}")
    if not query_file.is_file():
        raise SystemExit(f"query file not found: {query_file}")
    if top_k <= 0:
        raise SystemExit("--top-k must be greater than zero")
    if timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be greater than zero")


def load_query_text(query_file: Path) -> str:
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


def resolve_python_executable(env: dict[str, str]) -> Path:
    candidates = [Path(sys.executable), REPO_ROOT / ".venv" / "bin" / "python"]
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen or not candidate.is_file():
            continue
        seen.add(candidate_str)
        completed = subprocess.run(
            [candidate_str, "-c", "import mcp"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return candidate
    raise SystemExit(
        "unable to find a Python executable with the `mcp` package installed. "
        "Tried current interpreter and .venv/bin/python."
    )


def resolve_protocol_version(python_executable: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            "from mcp import types; print(types.LATEST_PROTOCOL_VERSION)",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "failed to resolve MCP protocol version.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
        )
    return completed.stdout.strip()


def start_server(
    *,
    python_executable: Path,
    settings_path: Path,
    log_level: str,
    env: dict[str, str],
) -> tuple[subprocess.Popen[bytes], Path, list[str]]:
    stderr_file = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="ragms-real-mcp-",
        suffix=".stderr.log",
        delete=False,
    )
    stderr_path = Path(stderr_file.name)
    command = [
        str(python_executable),
        "-u",
        "scripts/run_mcp_server.py",
        "--settings",
        str(settings_path),
        "--log-level",
        log_level,
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        env=env,
    )
    stderr_file.close()
    return process, stderr_path, command


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.kill()
    process.wait(timeout=10)


def send_message(process: subprocess.Popen[bytes], payload: dict[str, Any]) -> None:
    if process.stdin is None:
        raise SystemExit("stdio process stdin is not available")
    process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    process.stdin.flush()


def read_message(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    stderr_path: Path,
) -> dict[str, Any]:
    if process.stdout is None:
        raise SystemExit("stdio process stdout is not available")

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                "mcp server exited before sending a response.\n"
                f"returncode: {process.returncode}\n"
                f"stderr:\n{read_server_stderr(stderr_path)}\n"
            )
        ready, _, _ = select.select([process.stdout], [], [], 0.2)
        if process.stdout not in ready:
            continue
        raw_line = process.stdout.readline()
        if raw_line:
            return json.loads(raw_line.decode("utf-8"))

    raise SystemExit(
        f"timed out waiting for mcp response after {timeout_seconds} seconds.\n"
        f"stderr:\n{read_server_stderr(stderr_path)}\n"
    )


def send_and_read(
    process: subprocess.Popen[bytes],
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    stderr_path: Path,
) -> dict[str, Any]:
    send_message(process, payload)
    return read_message(process, timeout_seconds=timeout_seconds, stderr_path=stderr_path)


def call_tool(
    process: subprocess.Popen[bytes],
    *,
    request_id: int,
    name: str,
    arguments: dict[str, Any],
    timeout_seconds: float,
    stderr_path: Path,
) -> ToolInvocation:
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments,
        },
    }
    response = send_and_read(
        process,
        request,
        timeout_seconds=timeout_seconds,
        stderr_path=stderr_path,
    )
    return ToolInvocation(step=f"tools/call:{name}", request=request, response=response)


def assert_tool_success(invocation: ToolInvocation, *, stderr_path: Path) -> None:
    result = invocation.response.get("result")
    if not isinstance(result, dict):
        raise SystemExit(
            f"{invocation.step} returned an unexpected payload.\n"
            f"response:\n{pretty_json(invocation.response)}\n"
            f"stderr:\n{read_server_stderr(stderr_path)}\n"
        )
    if invocation.response.get("error") is not None:
        raise SystemExit(
            f"{invocation.step} returned a JSON-RPC error.\n"
            f"response:\n{pretty_json(invocation.response)}\n"
            f"stderr:\n{read_server_stderr(stderr_path)}\n"
        )
    if bool(result.get("isError")):
        raise SystemExit(
            f"{invocation.step} returned an MCP tool error.\n"
            f"response:\n{pretty_json(invocation.response)}\n"
            f"stderr:\n{read_server_stderr(stderr_path)}\n"
        )


def assert_jsonrpc_success(
    *,
    step: str,
    response: dict[str, Any],
    stderr_path: Path,
) -> None:
    if response.get("error") is None:
        return
    raise SystemExit(
        f"{step} returned a JSON-RPC error.\n"
        f"response:\n{pretty_json(response)}\n"
        f"stderr:\n{read_server_stderr(stderr_path)}\n"
    )


def extract_document_ids(response: dict[str, Any]) -> list[str]:
    result = dict(response.get("result") or {})
    structured_content = dict(result.get("structuredContent") or {})
    documents = list(structured_content.get("documents") or [])
    document_ids = []
    for item in documents:
        document_id = str(item.get("document_id") or "").strip()
        if document_id and document_id not in document_ids:
            document_ids.append(document_id)
    if not document_ids:
        raise SystemExit(
            "ingest_documents did not return any document_id values.\n"
            f"response:\n{pretty_json(response)}"
        )
    return document_ids


def read_server_stderr(stderr_path: Path) -> str:
    if not stderr_path.is_file():
        return ""
    return stderr_path.read_text(encoding="utf-8", errors="replace").strip()


def pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def print_summary(
    *,
    settings_path: Path,
    source_dir: Path,
    query_file: Path,
    query_text: str,
    collection: str,
    python_executable: Path,
    protocol_version: str,
    server_command: list[str],
    invocations: list[ToolInvocation],
    stderr_path: Path,
    failure: BaseException | None,
) -> None:
    print("=== Real MCP Flow Test Summary ===")
    print(f"python_executable: {python_executable}")
    print(f"protocol_version: {protocol_version}")
    print(f"settings: {settings_path}")
    print(f"source_dir: {source_dir}")
    print(f"query_file: {query_file}")
    print(f"collection: {collection}")
    print(f"query: {query_text}")
    print(f"server_command: {' '.join(server_command)}")
    print(f"status: {'failed' if failure is not None else 'passed'}")
    if failure is not None:
        print(f"failure: {failure!r}")

    for invocation in invocations:
        print(f"--- {invocation.step} request ---")
        print(pretty_json(invocation.request))
        print(f"--- {invocation.step} response ---")
        print(pretty_json(invocation.response))

    server_stderr = read_server_stderr(stderr_path)
    print("--- server stderr ---")
    print(server_stderr if server_stderr else "<empty>")


if __name__ == "__main__":
    raise SystemExit(main())
