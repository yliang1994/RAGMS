from __future__ import annotations

from pathlib import Path

import pytest

from ragms import get_project_root


@pytest.fixture(scope="session")
def project_root() -> Path:
    return get_project_root()

