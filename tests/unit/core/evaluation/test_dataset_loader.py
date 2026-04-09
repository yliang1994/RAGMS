from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragms.core.evaluation import DatasetLoader, DatasetLoaderError


def test_dataset_loader_loads_three_dataset_types_and_filters_labels() -> None:
    loader = DatasetLoader(Path("data/evaluation/datasets"))

    golden = loader.load(dataset_name="golden", dataset_version="v1")
    filtered = loader.load(dataset_name="multimodal_or_filtering", labels=["dashboard"])

    assert golden["dataset_name"] == "golden"
    assert golden["samples"][0].dataset_version == "v1"
    assert golden["samples"][0].collection == "real_c_ingestion_test"
    assert filtered["samples"][0].labels == ["multimodal", "filtering", "dashboard"]


def test_dataset_loader_lists_versions_and_resolves_sample_source() -> None:
    loader = DatasetLoader(Path("data/evaluation/datasets"))
    versions = loader.list_dataset_versions("golden")
    manifest_path = Path("data/evaluation/datasets/golden/v1.json")
    source = loader.resolve_sample_source(manifest_path, {"sample_id": "sample-1"})

    assert versions == ["v1"]
    assert source.endswith("data/evaluation/datasets/golden/v1.json")


def test_dataset_loader_validates_duplicate_sample_ids_and_empty_samples(tmp_path: Path) -> None:
    loader = DatasetLoader(tmp_path)
    manifest = {
        "dataset_name": "golden",
        "dataset_version": "v1",
        "collection": "demo",
        "samples": [
            {"sample_id": "dup", "query": "q1", "evaluation_modes": ["retrieval"]},
            {"sample_id": "dup", "query": "q2", "evaluation_modes": ["retrieval"]},
        ],
    }

    with pytest.raises(DatasetLoaderError, match="duplicate sample_id"):
        loader.validate_manifest(manifest)

    with pytest.raises(DatasetLoaderError, match="samples must not be empty"):
        loader.validate_manifest(
            {
                "dataset_name": "golden",
                "dataset_version": "v1",
                "collection": "demo",
                "samples": [],
            }
        )


def test_dataset_loader_load_returns_json_safe_normalized_samples() -> None:
    loader = DatasetLoader(Path("data/evaluation/datasets"))
    payload = loader.load(dataset_name="badcase_regression")

    json.dumps(
        {
            **payload,
            "samples": [sample.to_dict() for sample in payload["samples"]],
        }
    )
    assert payload["samples"][0].sample_id == "badcase_001"
    assert payload["samples"][0].evaluation_modes == ["retrieval"]
