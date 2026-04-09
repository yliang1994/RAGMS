"""Dataset manifest loader for local evaluation assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragms.core.models import EvaluationSample, normalize_evaluation_sample
from ragms.runtime.exceptions import RagMSError


class DatasetLoaderError(RagMSError):
    """Raised when evaluation dataset manifests are invalid or missing."""


class DatasetLoader:
    """Load and validate local evaluation dataset manifests."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def load(
        self,
        *,
        dataset_name: str,
        dataset_version: str | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Load one dataset version and return normalized manifest plus samples."""

        manifest_path = self._resolve_manifest_path(dataset_name, dataset_version)
        manifest = self._load_manifest(manifest_path)
        self.validate_manifest(manifest, manifest_path=manifest_path)
        required_labels = {str(label).strip() for label in (labels or []) if str(label).strip()}
        normalized_samples: list[EvaluationSample] = []
        for sample_payload in manifest.get("samples") or []:
            sample = normalize_evaluation_sample(
                {
                    **sample_payload,
                    "dataset_name": manifest.get("dataset_name"),
                    "sample_source": self.resolve_sample_source(manifest_path, sample_payload),
                },
                defaults={
                    "collection": manifest.get("collection"),
                    "dataset_version": manifest.get("dataset_version"),
                    "config_snapshot": manifest.get("config_snapshot"),
                    "filters": manifest.get("default_filters"),
                },
            )
            if required_labels and not required_labels.issubset(set(sample.labels)):
                continue
            normalized_samples.append(sample)
        return {
            "dataset_name": manifest.get("dataset_name"),
            "dataset_version": manifest.get("dataset_version"),
            "collection": manifest.get("collection"),
            "config_snapshot": dict(manifest.get("config_snapshot") or {}),
            "default_filters": dict(manifest.get("default_filters") or {}),
            "samples": normalized_samples,
            "manifest_path": str(manifest_path),
        }

    def validate_manifest(
        self,
        manifest: dict[str, Any],
        *,
        manifest_path: str | Path | None = None,
    ) -> None:
        """Validate one manifest structure before normalization."""

        dataset_name = str(manifest.get("dataset_name") or "").strip()
        dataset_version = str(manifest.get("dataset_version") or "").strip()
        collection = str(manifest.get("collection") or "").strip()
        samples = list(manifest.get("samples") or [])
        if not dataset_name:
            raise DatasetLoaderError("dataset_name must not be empty")
        if not dataset_version:
            raise DatasetLoaderError("dataset_version must not be empty")
        if not collection:
            raise DatasetLoaderError("collection must not be empty")
        if not samples:
            raise DatasetLoaderError("samples must not be empty")

        seen_sample_ids: set[str] = set()
        for sample in samples:
            sample_id = str(sample.get("sample_id") or "").strip()
            if not sample_id:
                raise DatasetLoaderError("sample_id must not be empty")
            if sample_id in seen_sample_ids:
                location = f" in {manifest_path}" if manifest_path is not None else ""
                raise DatasetLoaderError(f"duplicate sample_id: {sample_id}{location}")
            seen_sample_ids.add(sample_id)
            normalize_evaluation_sample(
                sample,
                defaults={
                    "collection": collection,
                    "dataset_version": dataset_version,
                    "config_snapshot": manifest.get("config_snapshot"),
                    "filters": manifest.get("default_filters"),
                    "dataset_name": dataset_name,
                    "sample_source": self.resolve_sample_source(Path(manifest_path) if manifest_path else Path("."), sample),
                },
            )

    def list_dataset_versions(self, dataset_name: str) -> list[str]:
        """List all available versions for one dataset name."""

        dataset_dir = self.root_dir / dataset_name
        if not dataset_dir.is_dir():
            return []
        versions = []
        for path in sorted(dataset_dir.glob("*.json")):
            payload = self._load_manifest(path)
            version = str(payload.get("dataset_version") or path.stem).strip()
            if version:
                versions.append(version)
        return versions

    def resolve_sample_source(self, manifest_path: str | Path, sample_payload: dict[str, Any]) -> str:
        """Resolve the source file that defines one sample."""

        explicit = str(sample_payload.get("sample_source") or "").strip()
        if explicit:
            return str((Path(manifest_path).parent / explicit).resolve())
        return str(Path(manifest_path).resolve())

    def _resolve_manifest_path(self, dataset_name: str, dataset_version: str | None) -> Path:
        dataset_dir = self.root_dir / dataset_name
        if not dataset_dir.is_dir():
            raise DatasetLoaderError(f"Unknown dataset: {dataset_name}")
        if dataset_version is not None:
            manifest_path = dataset_dir / f"{dataset_version}.json"
            if not manifest_path.is_file():
                raise DatasetLoaderError(f"Unknown dataset version: {dataset_name}/{dataset_version}")
            return manifest_path
        candidates = sorted(dataset_dir.glob("*.json"))
        if not candidates:
            raise DatasetLoaderError(f"No manifest files found for dataset: {dataset_name}")
        return candidates[-1]

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise DatasetLoaderError(f"Failed to read manifest: {path}") from exc
        if not isinstance(payload, dict):
            raise DatasetLoaderError(f"Manifest must be a JSON object: {path}")
        return payload
