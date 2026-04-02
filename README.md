# RagServer

This repository follows `DEV_SPEC.md` as the primary source of truth.

## Current status

Stage A1 establishes the minimal Python project scaffold:

- `src/ragms` package
- `scripts/` entrypoint directory
- `tests/` baseline with isolated test environment defaults
- `data/` and `logs/` local runtime directories

## Local bootstrap

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pydantic pyyaml python-dotenv pytest pytest-mock pytest-check pytest-bdd
python -m pip install -e .
python scripts/run_mcp_server.py
pytest --collect-only
```

## Notes

- Activate `.venv` before running any Python command.
- `tests/conftest.py` sets isolated runtime paths for test runs.

