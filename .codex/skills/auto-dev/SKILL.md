---
name: auto-dev
description: Use when the user asks for 自动开发, auto dev, or auto-dev in a repository that uses DEV_SPEC.md as the source of truth. This skill syncs DEV_SPEC.md into chapter references, selects the current task or a user-specified task like "auto dev B1", implements the task with spec-guided code and tests, retries test-and-fix up to three rounds, updates DEV_SPEC.md only when task state changes, and then offers commit, skip, or next.
---

# Auto Dev

## Overview

This skill automates spec-driven development for repositories that track implementation work in `DEV_SPEC.md`.
It first syncs the spec into chapter references with hash-based change detection, then chooses a task, implements it according to Chapters 3, 4, 5, 6, and 7, runs tests with up to three fix rounds, updates tracking only when task state changes, and finishes with a concise summary plus `commit` / `skip` / `next`.

## Triggering

Trigger this skill when the user says any of:

- `自动开发`
- `auto dev`
- `auto-dev`
- `auto dev <task-id>`
- `auto-dev <task-id>`

Task ids may be simple like `A1` or dotted like `B4.8`.

## Path Resolution

This skill separates static skill code from runtime-generated artifacts.

- `scripts/...` always means `.codex/skills/auto-dev/scripts/...`
- `.codex/skills/auto-dev/` is treated as read-only skill source
- synced references and sync state must be written under the repository-local `.auto-dev/` directory
- never resolve this skill's `scripts/...` against the repository root `scripts/` directory
- never write runtime files back into `.codex/skills/auto-dev/`

Runtime artifact layout:

- `.auto-dev/references/*.md`
- `.auto-dev/.sync-state.json`

When invoking helper scripts from the repository root:

- run sync with `python .codex/skills/auto-dev/scripts/sync_dev_spec.py --repo-root .`
- run task orchestration with `python .codex/skills/auto-dev/scripts/auto_dev.py --repo-root .`

## Workflow

### 1. Sync `DEV_SPEC.md`

Run `scripts/sync_dev_spec.py` first.

If calling it from the repository root, use:

- `python .codex/skills/auto-dev/scripts/sync_dev_spec.py --repo-root .`

Behavior:

- Resolve `DEV_SPEC.md` from the current working directory or walk upward until found.
- Compute SHA256 of the raw file.
- Compare with `.auto-dev/.sync-state.json`.
- If the hash is unchanged, do not rewrite chapter references.
- If changed:
  - split the top-level chapters `## 1` through `## 7`
  - ignore `## 目录`
  - write one markdown file per chapter into `.auto-dev/references/`
  - delete old generated chapter markdown files that are no longer valid
  - rewrite `.auto-dev/.sync-state.json`

Use the sync output as the source for later reads, but treat the root `DEV_SPEC.md` as the canonical file for hash checks and task status updates.

### 2. Select the task

If the user specified a task id, use it directly.

If no task id was specified:

- first choose the first `[~]` task in Chapter 6 order
- otherwise choose the first `[ ]` task in Chapter 6 order
- if no runnable task remains, stop and report it clearly

Build a task card from Chapter 6:

- task id and title
- goal
- files to modify
- classes or functions to implement
- acceptance criteria
- test command
- stage dependency preparation

### 3. Read the governing chapters

Always read these references after task selection:

- `.auto-dev/references/03-technical-design.md`
- `.auto-dev/references/04-test-strategy.md`
- `.auto-dev/references/05-system-architecture.md`
- `.auto-dev/references/06-project-schedule.md`
- `.auto-dev/references/07-dev-rules.md`

Interpret them this way:

- Chapter 3 decides the preferred technical choices and provider strategy.
- Chapter 4 decides whether the task needs unit tests, integration tests, or both.
- Chapter 5 decides where code belongs in the directory tree.
- Chapter 6 decides the current task, file targets, symbols, acceptance criteria, and test command.
- Chapter 7 decides coding style, exception handling, logging, and documentation rules.

### 4. Implement the task

During implementation:

- Follow the chapter-derived task card instead of improvising new scope.
- Put files in the Chapter 5 locations or the closest already-established equivalents.
- Add tests alongside code, keeping the scope as small as the task requires.
- Perform a code self-review before the test loop.
- Prefer standard Python packages instead of custom implementations for vector stores, splitters, rerankers, and similar infrastructure.
- Report the important third-party packages that likely need human confirmation before installation.
- For heavy libraries such as `sentence-transformers` or `torch` dependents, install only when the task needs them and import them lazily.

Default library preferences when the spec suggests them:

- `pydantic`, `pyyaml`, `python-dotenv`
- `markitdown`
- `langchain-text-splitters`
- `chromadb`
- `rank-bm25`, `jieba`
- `tenacity`
- `sentence-transformers`

### 5. Testing and auto-fix loop

Before every Python or pytest command:

- ensure `.venv` exists
- activate `.venv`

Then:

1. run the Chapter 6 test command for the selected task
2. if it fails, capture the error, fix the issue, and rerun
3. stop after at most 3 rounds
4. if still failing, report the remaining blocker and stop

Do not widen the test scope unless the failure proves the task broke an adjacent contract.

### 6. Persist and summarize

Only update `DEV_SPEC.md` when task status actually changed.

Status writeback is automatic and does not require user confirmation.

When a task completes:

- update the task row status and completion date
- update progress counts only when the task reaches `[x]`
- rerun `python .codex/skills/auto-dev/scripts/sync_dev_spec.py --repo-root .` if `DEV_SPEC.md` changed
- if implementation has started but tests are not yet fully green, write `[~]`

Summary output must include:

- task id
- task name
- changed files
- test command and result count
- recommended commit message

After automatic status writeback, ask the user for exactly one command:

- `commit`: run `git add` and `git commit`
- `skip`: stop without commit
- `next`: commit, then continue to the next task

## Failure handling

- If `DEV_SPEC.md` is missing, stop and report that the repository is not eligible for this skill.
- If the requested task id does not exist, stop and report the invalid task id.
- If Chapter 6 contains a task row but no matching detailed block, stop and report a spec inconsistency.
- If a required package is missing, identify the minimal package set before installing it.
- If a heavy dependency is only needed behind a runtime path, keep it lazily imported.
- If 3 test-fix rounds fail, stop and report the final error instead of looping indefinitely.
- If `.auto-dev/` cannot be written, stop and report the actual writable-path problem instead of falling back to `.codex/`.

## Anti-patterns

- Do not assume `scripts/...` refers to the repository root; for this skill it always refers to `.codex/skills/auto-dev/scripts/...`.
- Do not write synced references or state files into `.codex/skills/auto-dev/`; write them into `.auto-dev/`.
- Do not rewrite chapter references when the `DEV_SPEC.md` hash is unchanged.
- Do not update `DEV_SPEC.md` on every run; update it only when task state changed.
- Do not skip reading Chapters 3, 4, 5, and 7 after selecting a task.
- Do not hand-roll core infrastructure when a standard Python package is a better fit.
- Do not run Python or pytest commands outside the repository `.venv`.
- Do not silently continue after unresolved test failures beyond 3 rounds.

## Resources

- `.codex/skills/auto-dev/scripts/sync_dev_spec.py`: hash-based spec sync into chapter references
- `.codex/skills/auto-dev/scripts/auto_dev.py`: task selection, development orchestration, test/fix loop, and summary output
- `.auto-dev/references/*.md`: generated top-level chapter snapshots from `DEV_SPEC.md`

## Example invocations

- `自动开发`
- `auto dev`
- `auto dev B1`
- `auto-dev B4.8`
