"""Configuration loading from settings.yaml plus environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .settings_models import AppSettings


ENV_PREFIX = "RAGMS_"


def _parse_env_value(raw_value: str) -> Any:
    value = raw_value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.isdigit():
        return int(value)
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def _assign_nested_value(target: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = target
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _load_env_overrides(prefix: str = ENV_PREFIX) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        normalized = key[len(prefix) :].lower()
        if "__" not in normalized:
            continue
        path = [part for part in normalized.split("__") if part]
        if not path:
            continue
        _assign_nested_value(overrides, path, _parse_env_value(value))
    return overrides


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(path_value: Path, base_dir: Path) -> Path:
    return path_value if path_value.is_absolute() else (base_dir / path_value).resolve()


def _normalize_paths(settings: AppSettings, settings_path: Path) -> AppSettings:
    base_dir = settings_path.parent
    updated = settings.model_copy(deep=True)
    updated.paths.project_root = _resolve_path(updated.paths.project_root, base_dir)
    updated.paths.data_dir = _resolve_path(updated.paths.data_dir, base_dir)
    updated.paths.logs_dir = _resolve_path(updated.paths.logs_dir, base_dir)
    updated.paths.settings_file = settings_path.resolve()
    updated.storage.sqlite.path = _resolve_path(updated.storage.sqlite.path, base_dir)
    updated.observability.log_file = _resolve_path(updated.observability.log_file, base_dir)
    updated.dashboard.traces_file = _resolve_path(updated.dashboard.traces_file, base_dir)
    updated.dashboard.title = updated.dashboard.title.strip() or f"{updated.app_name} Dashboard"
    return updated


def load_settings(settings_path: str | Path = "settings.yaml") -> AppSettings:
    """Load application settings from YAML, .env, and environment variables."""

    resolved_settings_path = Path(settings_path).expanduser().resolve()
    if not resolved_settings_path.is_file():
        raise FileNotFoundError(f"settings.yaml not found: {resolved_settings_path}")

    load_dotenv(resolved_settings_path.parent / ".env", override=False)
    raw_settings = yaml.safe_load(resolved_settings_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_settings, dict):
        raise ValueError("settings.yaml must contain a mapping at the top level")

    merged_settings = _deep_merge(raw_settings, _load_env_overrides())
    settings = AppSettings.model_validate(merged_settings)
    return _normalize_paths(settings, resolved_settings_path)
