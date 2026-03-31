from __future__ import annotations

from ragms import get_project_root


def test_get_project_root_points_to_repository() -> None:
    assert get_project_root().joinpath("DEV_SPEC.md").exists()
