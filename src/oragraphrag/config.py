"""Configuration loading: defaults < config.yaml < env vars."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OciGrokConfig(BaseModel):
    compartment_ocid: str = ""
    endpoint_id: str = ""
    model: str = "grok-4-3"
    region: str = "us-chicago-1"


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:35b-a3b"


class LlmConfig(BaseModel):
    provider: Literal["oci_grok", "ollama", "openai_compat"] = "oci_grok"
    oci_grok: OciGrokConfig = Field(default_factory=OciGrokConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    fallback_on_outage: bool = False
    request_timeout_s: float = 60.0
    max_retries: int = 5


class OracleEmbedConfig(BaseModel):
    model: str = "ALL_MINILM_L12_V2"


class EmbeddingsConfig(BaseModel):
    provider: Literal["oracle", "ollama", "sentence_transformers"] = "oracle"
    oracle: OracleEmbedConfig = Field(default_factory=OracleEmbedConfig)
    dim: int = 384


class OracleConfig(BaseModel):
    username: str = "ORAGRAPH"
    password: str = "Welcome12345*"
    dsn: str = "localhost:1521/FREEPDB1"
    pool_min: int = 1
    pool_max: int = 8


class AmplitudeConfig(BaseModel):
    alpha: float = 8.0
    beta: float = 0.0
    per_axis_overrides: dict[str, float] = Field(default_factory=dict)


class PageRankConfig(BaseModel):
    damping: float = 0.85
    top_m_entities: int = 20


class RetrievalConfig(BaseModel):
    seed_k_entities: int = 8
    seed_k_propositions: int = 16
    max_subgraph_nodes: int = 256
    max_subgraph_edges: int = 2048
    amplitude: AmplitudeConfig = Field(default_factory=AmplitudeConfig)
    pagerank: PageRankConfig = Field(default_factory=PageRankConfig)


class AnswerConfig(BaseModel):
    token_budget: int = 4000


class IngestConfig(BaseModel):
    span_max_tokens: int = 1200
    section_overlap_tokens: int = 100
    extract_concurrency: int = 8
    canonicalize_threshold: float = 0.92
    hnsw_refresh_after_inserts: int = 1000


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OGR__",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    llm: LlmConfig = Field(default_factory=LlmConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file. None-valued sections fall back to defaults."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        data = {k: v for k, v in data.items() if v is not None}
        return cls(**data)
