from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from sync_dev_spec import default_references_dir, default_state_path, sync_dev_spec


TASK_ROW_PATTERN = re.compile(
    r"^\|\s*(?P<task_id>[A-Z]\d+(?:\.\d+)?)\s*\|\s*(?P<title>.*?)\s*\|\s*(?P<status>\[[ ~x]\])\s*\|\s*(?P<done_date>.*?)\s*\|\s*(?P<note>.*?)\s*\|$",
    re.MULTILINE,
)
TASK_HEADER_TEMPLATE = r"^#####\s+{task_id}\s+(?P<title>.+)$"
STAGE_PROGRESS_PATTERN = re.compile(
    r"^\| (?P<label>阶段 [A-I]) \| (?P<total>\d+) \| (?P<done>\d+) \| (?P<progress>[\d.]+%) \|$",
    re.MULTILINE,
)
TOTAL_PROGRESS_PATTERN = re.compile(
    r"^\| \*\*总计\*\* \| \*\*(?P<total>\d+)\*\* \| \*\*(?P<done>\d+)\*\* \| \*\*(?P<progress>[\d.]+%)\*\* \|$",
    re.MULTILINE,
)


PACKAGE_HINTS = [
    ("settings.yaml", ["pydantic", "pyyaml", "python-dotenv"]),
    ("配置", ["pydantic", "pyyaml", "python-dotenv"]),
    ("markitdown", ["markitdown"]),
    ("splitter", ["langchain-text-splitters"]),
    ("chroma", ["chromadb"]),
    ("bm25", ["rank-bm25", "jieba"]),
    ("rerank", ["sentence-transformers"]),
    ("cross-encoder", ["sentence-transformers"]),
    ("retry", ["tenacity"]),
]
HEAVY_PACKAGES = {"sentence-transformers", "torch", "transformers"}


@dataclass
class TaskRow:
    task_id: str
    title: str
    status: str
    done_date: str
    note: str


@dataclass
class TaskCard:
    task_id: str
    title: str
    status: str
    goal: str
    files_to_modify: list[str]
    symbols: list[str]
    acceptance_criteria: list[str]
    dependency_preparation: list[str]
    test_command: str
    detail_block: str


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_task_rows(schedule_text: str) -> list[TaskRow]:
    return [
        TaskRow(
            task_id=match.group("task_id"),
            title=match.group("title").strip(),
            status=match.group("status"),
            done_date=match.group("done_date").strip(),
            note=match.group("note").strip(),
        )
        for match in TASK_ROW_PATTERN.finditer(schedule_text)
    ]


def select_task(schedule_text: str, explicit_task_id: str | None) -> TaskRow:
    rows = parse_task_rows(schedule_text)
    if explicit_task_id:
        normalized = explicit_task_id.strip()
        for row in rows:
            if row.task_id == normalized:
                return row
        raise ValueError(f"Unknown task id: {normalized}")

    for desired_status in ("[~]", "[ ]"):
        for row in rows:
            if row.status == desired_status:
                return row
    raise ValueError("No runnable task found in Chapter 6")


def extract_task_block(schedule_text: str, task_id: str) -> str:
    pattern = re.compile(TASK_HEADER_TEMPLATE.format(task_id=re.escape(task_id)), re.MULTILINE)
    match = pattern.search(schedule_text)
    if not match:
        raise ValueError(f"Task detail block missing for {task_id}")

    remaining = schedule_text[match.end() :]
    next_header = re.search(r"^#####\s+[A-Z]\d+(?:\.\d+)?\s+", remaining, re.MULTILINE)
    end = match.end() + next_header.start() if next_header else len(schedule_text)
    return schedule_text[match.start() : end].strip()


def extract_single_line(detail_block: str, label: str) -> str:
    match = re.search(rf"- {re.escape(label)}：(.+)", detail_block)
    if not match:
        return ""
    return match.group(1).strip()


def parse_backticked_csv(raw_value: str) -> list[str]:
    return re.findall(r"`([^`]+)`", raw_value)


def parse_acceptance(detail_block: str) -> list[str]:
    raw = extract_single_line(detail_block, "验收标准")
    if not raw:
        return []
    parts = [segment.strip("；。 ").strip() for segment in re.split(r"[；;]", raw) if segment.strip()]
    return parts


def parse_dependency_preparation(detail_block: str) -> list[str]:
    raw = extract_single_line(detail_block, "阶段依赖准备")
    if not raw:
        return []
    return [raw]


def extract_test_command(detail_block: str) -> str:
    match = re.search(r"- 测试方法：`([^`]+)`", detail_block)
    if not match:
        raise ValueError("Task detail block is missing the test command")
    return match.group(1)


def build_task_card(references_dir: Path, explicit_task_id: str | None) -> TaskCard:
    schedule_text = load_text(references_dir / "06-project-schedule.md")
    row = select_task(schedule_text, explicit_task_id)
    detail_block = extract_task_block(schedule_text, row.task_id)
    return TaskCard(
        task_id=row.task_id,
        title=row.title,
        status=row.status,
        goal=extract_single_line(detail_block, "目标"),
        files_to_modify=parse_backticked_csv(extract_single_line(detail_block, "修改文件")),
        symbols=parse_backticked_csv(extract_single_line(detail_block, "实现类/函数")),
        acceptance_criteria=parse_acceptance(detail_block),
        dependency_preparation=parse_dependency_preparation(detail_block),
        test_command=extract_test_command(detail_block),
        detail_block=detail_block,
    )


def suggest_packages(task_card: TaskCard) -> dict[str, object]:
    haystack = "\n".join(
        [
            task_card.title,
            task_card.goal,
            " ".join(task_card.files_to_modify),
            " ".join(task_card.symbols),
            task_card.detail_block,
        ]
    ).lower()
    packages: list[str] = []
    for keyword, hinted_packages in PACKAGE_HINTS:
        if keyword.lower() in haystack:
            for package in hinted_packages:
                if package not in packages:
                    packages.append(package)
    return {
        "important_packages": packages,
        "human_confirmation_needed": packages,
        "lazy_load_candidates": [package for package in packages if package in HEAVY_PACKAGES],
    }


def build_summary(references_dir: Path, task_card: TaskCard, max_fix_rounds: int) -> dict[str, object]:
    support_files = [
        "03-technical-design.md",
        "04-test-strategy.md",
        "05-system-architecture.md",
        "06-project-schedule.md",
        "07-dev-rules.md",
    ]
    return {
        "task_card": asdict(task_card),
        "must_read": [str(references_dir / name) for name in support_files],
        "package_report": suggest_packages(task_card),
        "max_fix_rounds": max_fix_rounds,
        "final_prompt": "After implementation, ask for one of: commit, skip, next.",
        "commit_message": f"{task_card.task_id.lower()}: {task_card.title}",
    }


def update_progress_table(dev_spec_text: str, completed_task_id: str, new_status: str) -> str:
    if new_status != "[x]":
        return dev_spec_text

    stage_label = f"阶段 {completed_task_id[0]}"

    def replace_stage(match: re.Match[str]) -> str:
        if match.group("label") != stage_label:
            return match.group(0)
        total = int(match.group("total"))
        done = int(match.group("done")) + 1
        progress = f"{(done / total) * 100:.0f}%"
        return f"| {match.group('label')} | {total} | {done} | {progress} |"

    updated = STAGE_PROGRESS_PATTERN.sub(replace_stage, dev_spec_text)

    def replace_total(match: re.Match[str]) -> str:
        total = int(match.group("total"))
        done = int(match.group("done")) + 1
        progress_value = f"{(done / total) * 100:.1f}%"
        return f"| **总计** | **{total}** | **{done}** | **{progress_value}** |"

    return TOTAL_PROGRESS_PATTERN.sub(replace_total, updated, count=1)


def update_task_status(
    dev_spec_path: Path,
    task_id: str,
    new_status: str,
    note: str,
    completed_on: str | None = None,
) -> None:
    dev_spec_text = load_text(dev_spec_path)
    completed_on = completed_on or date.today().isoformat()
    row_pattern = re.compile(
        rf"^\|\s*{re.escape(task_id)}\s*\|\s*(?P<title>.*?)\s*\|\s*(?P<status>\[[ ~x]\])\s*\|\s*(?P<done_date>.*?)\s*\|\s*(?P<note>.*?)\s*\|$",
        re.MULTILINE,
    )
    match = row_pattern.search(dev_spec_text)
    if not match:
        raise ValueError(f"Task row missing for {task_id}")
    current_status = match.group("status")
    current_done_date = match.group("done_date").strip()
    if current_status == new_status and match.group("note").strip() == note:
        return

    done_date = completed_on if new_status == "[x]" else current_done_date
    title = match.group("title").strip()
    replacement = f"| {task_id} | {title} | {new_status} | {done_date} | {note} |"
    updated = row_pattern.sub(replacement, dev_spec_text, count=1)
    if current_status != "[x]" and new_status == "[x]":
        updated = update_progress_table(updated, task_id, new_status)
    dev_spec_path.write_text(updated, encoding="utf-8")


def preview_next_task(references_dir: Path, current_task_id: str) -> str | None:
    rows = parse_task_rows(load_text(references_dir / "06-project-schedule.md"))
    seen_current = False
    for row in rows:
        if row.task_id == current_task_id:
            seen_current = True
            continue
        if seen_current and row.status in {"[~]", "[ ]"}:
            return row.task_id
    return None


def parse_changed_files(values: Iterable[str]) -> list[str]:
    changed: list[str] = []
    for value in values:
        for part in value.split(","):
            item = part.strip()
            if item and item not in changed:
                changed.append(item)
    return changed


def resolve_status_update(
    explicit_status: str | None,
    note: str,
    changed_files: list[str],
    tests_passed: int | None,
    tests_failed: int | None,
) -> str | None:
    if explicit_status:
        return explicit_status
    if not note:
        return None
    if tests_failed == 0 and tests_passed is not None:
        return "[x]"
    if changed_files:
        return "[~]"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or finalize an auto-dev task execution.")
    parser.add_argument("task_id", nargs="?", default=None, help="Optional explicit task id like B1 or B4.8.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--dev-spec-path", default=None, help="Optional explicit path to DEV_SPEC.md.")
    parser.add_argument("--references-dir", default=None, help="Directory containing synced chapter references.")
    parser.add_argument("--state-file", default=None, help="Path to the sync state JSON file.")
    parser.add_argument("--max-fix-rounds", type=int, default=3, help="Maximum test-fix rounds.")
    parser.add_argument("--mark-status", choices=["[~]", "[x]"], default=None, help="Update DEV_SPEC status for the selected task.")
    parser.add_argument("--note", default="", help="Status note written back to DEV_SPEC. When tests pass, status is auto-finalized.")
    parser.add_argument("--changed-files", action="append", default=[], help="Comma-separated changed files for the summary output.")
    parser.add_argument("--tests-passed", type=int, default=None, help="Optional passed test count for the summary output.")
    parser.add_argument("--tests-failed", type=int, default=None, help="Optional failed test count for the summary output.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    references_dir = (
        Path(args.references_dir).expanduser().resolve()
        if args.references_dir
        else default_references_dir(repo_root)
    )
    state_file = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else default_state_path(repo_root)
    )
    sync_result = sync_dev_spec(
        repo_root,
        args.dev_spec_path,
        references_dir=references_dir,
        state_path=state_file,
    )
    task_card = build_task_card(references_dir, args.task_id)
    changed_files = parse_changed_files(args.changed_files)
    resolved_status = resolve_status_update(
        explicit_status=args.mark_status,
        note=args.note,
        changed_files=changed_files,
        tests_passed=args.tests_passed,
        tests_failed=args.tests_failed,
    )

    if resolved_status:
        update_task_status(
            dev_spec_path=Path(sync_result.dev_spec_path),
            task_id=task_card.task_id,
            new_status=resolved_status,
            note=args.note,
        )
        sync_result = sync_dev_spec(
            repo_root,
            args.dev_spec_path,
            references_dir=references_dir,
            state_path=state_file,
        )
        task_card = build_task_card(references_dir, task_card.task_id)

    payload = build_summary(references_dir, task_card, args.max_fix_rounds)
    payload["sync_changed"] = sync_result.changed
    payload["dev_spec_path"] = str(sync_result.dev_spec_path)
    payload["references_dir"] = str(references_dir)
    payload["state_file"] = str(state_file)
    payload["changed_files"] = changed_files
    payload["test_stats"] = {
        "passed": args.tests_passed,
        "failed": args.tests_failed,
    }
    payload["status_writeback"] = resolved_status
    payload["next_task_preview"] = preview_next_task(references_dir, task_card.task_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
