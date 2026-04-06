"""Strongly typed application settings models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unknown fields for fast config failure."""

    model_config = ConfigDict(extra="forbid")


class PathSettings(StrictModel):
    """Filesystem locations used by the local runtime."""

    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")
    settings_file: Path = Path("settings.yaml")


class LLMSettings(StrictModel):
    """Primary text-generation model configuration."""

    provider: Literal["openai", "qwen", "deepseek"] = "openai"
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    base_url: str | None = None


class EmbeddingSettings(StrictModel):
    """Dense embedding model configuration."""

    provider: Literal["openai", "qwen"] = "openai"
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = 10


class IngestionTransformSettings(StrictModel):
    """Feature flags for transform-stage optional LLM behaviors."""

    enable_llm_chunk_refine: bool = False
    enable_llm_metadata_enrich: bool = False


class IngestionSettings(StrictModel):
    """Ingestion pipeline runtime settings."""

    transform: IngestionTransformSettings = Field(default_factory=IngestionTransformSettings)


class VisionLLMSettings(StrictModel):
    """Vision-language model configuration with optional routing hints."""

    provider: Literal["auto", "gpt4o", "qwen_vl"] = "auto"
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    base_url: str | None = None
    language_providers: dict[str, Literal["gpt4o", "qwen_vl"]] = Field(
        default_factory=lambda: {"zh": "qwen_vl", "en": "gpt4o"}
    )
    environment_providers: dict[str, Literal["gpt4o", "qwen_vl"]] = Field(
        default_factory=lambda: {
            "development": "qwen_vl",
            "test": "qwen_vl",
            "production": "gpt4o",
        }
    )


class VectorStoreSettings(StrictModel):
    """Vector store backend selection."""

    backend: Literal["chroma"] = "chroma"
    collection: str = "default"


class RetrievalSettings(StrictModel):
    """Retrieval pipeline strategy configuration."""

    strategy: Literal["hybrid", "dense_only", "sparse_only"] = "hybrid"
    fusion_algorithm: Literal["rrf"] = "rrf"
    rerank_backend: Literal["disabled", "cross_encoder", "llm_reranker"] = "disabled"


class EvaluationSettings(StrictModel):
    """Evaluation backend configuration."""

    backends: list[str] = Field(default_factory=lambda: ["custom_metrics"])


class SQLiteStorageSettings(StrictModel):
    """SQLite metadata storage configuration."""

    path: Path = Path("data/metadata/ragms.db")


class StorageSettings(StrictModel):
    """Storage backend configuration used by ingestion and management flows."""

    sqlite: SQLiteStorageSettings = Field(default_factory=SQLiteStorageSettings)


class ObservabilitySettings(StrictModel):
    """Trace and application logging settings."""

    enabled: bool = True
    log_file: Path = Path("logs/traces.jsonl")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class DashboardSettings(StrictModel):
    """Local dashboard runtime settings."""

    enabled: bool = True
    port: int = 8501
    traces_file: Path = Path("logs/traces.jsonl")


class AppSettings(StrictModel):
    """Top-level application settings assembled from YAML and environment variables."""

    app_name: str = "ragms"
    environment: Literal["development", "test", "production"] = "development"
    paths: PathSettings = Field(default_factory=PathSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    vision_llm: VisionLLMSettings = Field(default_factory=VisionLLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
