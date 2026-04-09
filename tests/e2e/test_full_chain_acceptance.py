from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_acceptance import render_acceptance_summary, run_full_acceptance


@pytest.mark.e2e
def test_full_chain_acceptance_returns_structured_summary(tmp_path: Path) -> None:
    summary = run_full_acceptance(tmp_path)
    rendered = render_acceptance_summary(summary)
    payload = json.loads(rendered)

    assert payload["status"] == "passed"
    assert payload["failed_steps"] == []
    assert payload["trace_ids"]
    assert payload["run_ids"]
    assert payload["artifact_paths"]
    assert payload["scenarios"]["scenario_1_data_preparation"]["status"] == "passed"
    assert payload["scenarios"]["scenario_2_recall_quality_evaluation"]["status"] == "passed"
    assert payload["scenarios"]["scenario_3_mcp_client_function"]["status"] == "passed"
