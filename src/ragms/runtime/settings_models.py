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

    provider: Literal["openai"] = "openai"
    model: str = "text-embedding-3-small"
    api_key: str | None = None


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
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)

