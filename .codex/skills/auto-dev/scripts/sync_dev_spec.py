from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


CHAPTER_FILE_NAMES = {
    "1": "01-project-overview.md",
    "2": "02-core-features.md",
    "3": "03-technical-design.md",
    "4": "04-test-strategy.md",
    "5": "05-system-architecture.md",
    "6": "06-project-schedule.md",
    "7": "07-dev-rules.md",
}
SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
STATE_FILE_NAME = ".sync-state.json"
DEFAULT_ARTIFACTS_DIR = ".auto-dev"


@dataclass
class SyncResult:
    changed: bool
    dev_spec_path: Path
    sha256: str
    generated_files: list[str]


def resolve_dev_spec(start: Path, explicit_spec: str | None = None) -> Path:
    if explicit_spec:
        candidate = Path(explicit_spec).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"DEV_SPEC.md not found: {candidate}")

    current = start.resolve()
    search_roots = [current] + list(current.parents)
    for root in search_roots:
        candidate = root / "DEV_SPEC.md"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("DEV_SPEC.md not found in current directory or parents")


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_top_level_chapters(dev_spec_text: str) -> dict[str, str]:
    matches = list(SECTION_PATTERN.finditer(dev_spec_text))
    chapters: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(dev_spec_text)
        if title == "目录":
            continue
        number = title.split(".", 1)[0].strip()
        if number not in CHAPTER_FILE_NAMES:
            continue
        chapters[number] = dev_spec_text[start:end].strip() + "\n"
    return chapters


def write_references(references_dir: Path, chapters: dict[str, str]) -> list[str]:
    references_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []
    target_names = {CHAPTER_FILE_NAMES[number] for number in chapters}

    try:
        for number, content in chapters.items():
            filename = CHAPTER_FILE_NAMES[number]
            target_path = references_dir / filename
            target_path.write_text(content, encoding="utf-8")
            generated_files.append(filename)

        for existing in references_dir.glob("*.md"):
            if existing.name not in target_names:
                existing.unlink()
    except OSError as exc:
        raise OSError(
            f"Unable to write synced chapter references under {references_dir}. "
            "Ensure the target artifacts directory is writable."
        ) from exc

    return sorted(generated_files)


def default_references_dir(repo_root: Path) -> Path:
    return repo_root / DEFAULT_ARTIFACTS_DIR / "references"


def default_state_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_ARTIFACTS_DIR / STATE_FILE_NAME


def sync_dev_spec(
    repo_root: Path,
    spec_path: str | None = None,
    *,
    references_dir: Path | None = None,
    state_path: Path | None = None,
) -> SyncResult:
    dev_spec_path = resolve_dev_spec(repo_root, spec_path)
    references_dir = (references_dir or default_references_dir(repo_root)).expanduser().resolve()
    state_path = (state_path or default_state_path(repo_root)).expanduser().resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    current_sha = compute_sha256(dev_spec_path)
    state = load_state(state_path)

    if state.get("sha256") == current_sha:
        generated_files = list(state.get("generated_files", []))
        return SyncResult(False, dev_spec_path, current_sha, generated_files)

    chapters = parse_top_level_chapters(dev_spec_path.read_text(encoding="utf-8"))
    generated_files = write_references(references_dir, chapters)
    state_payload = {
        "version": 1,
        "dev_spec_path": str(dev_spec_path),
        "sha256": current_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_files": generated_files,
    }
    state_path.write_text(
        json.dumps(state_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return SyncResult(True, dev_spec_path, current_sha, generated_files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync DEV_SPEC.md into chapter references.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--dev-spec-path",
        "--spec-path",
        dest="spec_path",
        default=None,
        help="Optional explicit path to DEV_SPEC.md.",
    )
    parser.add_argument(
        "--references-dir",
        default=None,
        help="Directory where synced chapter references will be written.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to the sync state JSON file.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    references_dir = Path(args.references_dir).expanduser() if args.references_dir else None
    state_file = Path(args.state_file).expanduser() if args.state_file else None
    result = sync_dev_spec(
        repo_root,
        args.spec_path,
        references_dir=references_dir,
        state_path=state_file,
    )
    payload = {
        "changed": result.changed,
        "dev_spec_path": str(result.dev_spec_path),
        "sha256": result.sha256,
        "generated_files": result.generated_files,
        "references_dir": str((references_dir or default_references_dir(repo_root)).resolve()),
        "state_file": str((state_file or default_state_path(repo_root)).resolve()),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
