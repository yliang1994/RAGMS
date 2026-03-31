# RAGMS

RAGMS is a local-first, modular RAG system designed around a pluggable architecture and MCP-native tooling.

## Project Status

The repository is being implemented incrementally from `DEV_SPEC.md`. The current stage establishes the project skeleton, package layout, and minimal runtime bootstrap entrypoint.

## Quick Start

```bash
python scripts/run_mcp_server.py --help
pytest --collect-only
```

## Layout

- `src/ragms/`: application package
- `scripts/`: local entrypoints
- `tests/`: unit, integration, and end-to-end tests
- `data/`: local working data
- `logs/`: runtime logs

