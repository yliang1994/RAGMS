from __future__ import annotations

from ragms import get_project_root


def test_get_project_root_points_to_repo_root() -> None:
    project_root = get_project_root()

    assert project_root.name == "RagServer"
    assert (project_root / "DEV_SPEC.md").exists()

