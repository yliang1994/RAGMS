from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ragms import get_project_root
from ragms.runtime.settings_models import AppSettings


def load_settings(settings_path: str | Path | None = None) -> AppSettings:
    project_root = get_project_root()
    resolved_settings_path = Path(settings_path) if settings_path else project_root / "settings.yaml"
    resolved_settings_path = resolved_settings_path.resolve()

    raw_settings = _read_yaml(resolved_settings_path)
    runtime_config = raw_settings.get("runtime", {})
    env_file_name = runtime_config.get("env_file", ".env")
    env_values = _read_env_file(resolved_settings_path.parent / env_file_name)

    merged = _deep_copy(raw_settings)
    _apply_env_overrides(merged, env_values)
    _apply_env_overrides(merged, os.environ)
    merged["project_root"] = str(project_root)

    try:
        return AppSettings.model_validate(merged)
    except ValidationError:
        if runtime_config.get("fail_fast_on_invalid_config", True):
            raise
        return AppSettings.model_construct(**merged)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("settings.yaml must define a mapping at the top level")
    return data


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    env_values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_values[key.strip()] = value.strip().strip('"').strip("'")
    return env_values


def _apply_env_overrides(data: dict[str, Any], env: dict[str, str]) -> None:
    path_mappings = {
        "RAGMS_APP_NAME": ("app", "name"),
        "RAGMS_APP_ENV": ("app", "env"),
        "RAGMS_STORAGE_SQLITE_PATH": ("storage", "sqlite", "path"),
        "RAGMS_STORAGE_CHROMA_PATH": ("storage", "chroma", "path"),
        "RAGMS_STORAGE_BM25_INDEX_DIR": ("storage", "bm25", "index_dir"),
        "RAGMS_STORAGE_IMAGES_DIR": ("storage", "images", "dir"),
        "RAGMS_STORAGE_TRACES_FILE": ("storage", "traces", "file"),
        "RAGMS_STORAGE_APP_LOGS_DIR": ("storage", "app_logs", "dir"),
        "RAGMS_MODELS_LLM_PROVIDER": ("models", "llm", "provider"),
        "RAGMS_MODELS_DENSE_EMBEDDING_PROVIDER": ("models", "embedding", "dense", "provider"),
    }
    for env_name, path in path_mappings.items():
        if env_name in env:
            _set_nested_value(data, path, env[env_name])

    _resolve_secret(data, env, ("models", "llm"))
    _resolve_secret(data, env, ("models", "transform_llm"))
    _resolve_secret(data, env, ("models", "vision_llm"))
    _resolve_secret(data, env, ("models", "embedding", "dense"))


def _resolve_secret(data: dict[str, Any], env: dict[str, str], path: tuple[str, ...]) -> None:
    target = _get_nested_value(data, path)
    if not isinstance(target, dict):
        return
    env_name = target.get("api_key_env")
    if env_name and env_name in env:
        target["api_key"] = env[env_name]


def _get_nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set_nested_value(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = data
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value

