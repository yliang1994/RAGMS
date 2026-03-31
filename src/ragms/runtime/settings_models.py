from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RagmsBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AppConfig(RagmsBaseModel):
    name: str = "ragms"
    env: str = "local"
    log_level: str = "INFO"
    default_collection: str = "knowledge_hub"


class RuntimeConfig(RagmsBaseModel):
    settings_file: str = "settings.yaml"
    env_file: str = ".env"
    fail_fast_on_invalid_config: bool = True


class MCPConfig(RagmsBaseModel):
    transport: Literal["stdio"] = "stdio"
    server_name: str = "ragms-mcp-server"
    tools: list[str] = Field(default_factory=list)


class LLMSettings(RagmsBaseModel):
    provider: Literal["openai", "qwen", "deepseek"]
    model: str
    api_key_env: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class DenseEmbeddingSettings(RagmsBaseModel):
    provider: Literal["openai", "bge", "jina"]
    model: str
    api_key_env: str | None = None
    api_key: str | None = None
    batch_size: int = 64


class SparseEmbeddingSettings(RagmsBaseModel):
    provider: Literal["bm25"] = "bm25"
    tokenizer: str = "default"


class EmbeddingSettings(RagmsBaseModel):
    dense: DenseEmbeddingSettings
    sparse: SparseEmbeddingSettings


class RerankerSettings(RagmsBaseModel):
    mode: Literal["cross_encoder", "llm", "none"] = "none"
    model: str | None = None
    enabled: bool = False


class ModelsConfig(RagmsBaseModel):
    llm: LLMSettings
    transform_llm: LLMSettings
    vision_llm: LLMSettings
    embedding: EmbeddingSettings
    reranker: RerankerSettings


class SQLiteConfig(RagmsBaseModel):
    path: Path


class ChromaConfig(RagmsBaseModel):
    path: Path
    collection_prefix: str = "ragms_"


class BM25Config(RagmsBaseModel):
    index_dir: Path


class ImagesConfig(RagmsBaseModel):
    dir: Path


class TracesConfig(RagmsBaseModel):
    file: Path


class AppLogsConfig(RagmsBaseModel):
    dir: Path


class StorageConfig(RagmsBaseModel):
    sqlite: SQLiteConfig
    chroma: ChromaConfig
    bm25: BM25Config
    images: ImagesConfig
    traces: TracesConfig
    app_logs: AppLogsConfig


class DashboardConfig(RagmsBaseModel):
    enabled: bool = True
    title: str = "RAGMS Local Dashboard"
    auto_refresh: bool = True


class AppSettings(RagmsBaseModel):
    app: AppConfig
    runtime: RuntimeConfig
    mcp: MCPConfig
    models: ModelsConfig
    storage: StorageConfig
    dashboard: DashboardConfig
    project_root: Path | None = None

    @field_validator("project_root")
    @classmethod
    def _resolve_project_root(cls, value: Path | None) -> Path | None:
        if value is None:
            return value
        return value.resolve()

    @model_validator(mode="after")
    def _normalize_paths(self) -> "AppSettings":
        if self.project_root is None:
            return self

        base_dir = self.project_root
        self.storage.sqlite.path = _resolve_path(self.storage.sqlite.path, base_dir)
        self.storage.chroma.path = _resolve_path(self.storage.chroma.path, base_dir)
        self.storage.bm25.index_dir = _resolve_path(self.storage.bm25.index_dir, base_dir)
        self.storage.images.dir = _resolve_path(self.storage.images.dir, base_dir)
        self.storage.traces.file = _resolve_path(self.storage.traces.file, base_dir)
        self.storage.app_logs.dir = _resolve_path(self.storage.app_logs.dir, base_dir)
        return self


def _resolve_path(value: Path, base_dir: Path) -> Path:
    return value if value.is_absolute() else (base_dir / value).resolve()

