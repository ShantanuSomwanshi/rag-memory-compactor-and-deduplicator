"""Configuration for a compaction run."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class CompactorConfig(BaseModel):
    """All knobs for ingestion, candidate discovery, validation and summarization.

    The two thresholds are deliberately separate:

    ``similarity_threshold`` is the cheap, recall-oriented gate used during ANN
    candidate discovery - it decides which chunks are worth *looking at* as a
    possible duplicate group.

    ``validation_threshold`` is the strict, precision-oriented gate applied to
    the full pairwise similarity matrix of a candidate group before anything is
    summarized. It exists because ANN + graph grouping is transitive: A~B and
    B~C can drag A and C into one group even when A and C are not alike. The
    validation pass is what stops that becoming a bad merge.
    """

    model_config = {"extra": "forbid"}

    # --- storage -----------------------------------------------------------
    workdir: Path = Field(
        default=Path(".ragcompactor"),
        description="Directory holding the vector store and archive database.",
    )
    collection: str = Field(default="memory", description="Vector store collection name.")
    store_backend: str = Field(
        default="memory",
        description="Vector store backend: 'chroma' for persistent ChromaDB, 'memory' for in-process.",
    )

    # --- ingestion ---------------------------------------------------------
    chunk_size: int = Field(default=900, ge=50, description="Target chunk size in characters.")
    chunk_overlap: int = Field(default=120, ge=0, description="Character overlap between chunks.")

    # --- embedding ---------------------------------------------------------
    embedding_backend: str = Field(
        default="sentence-transformers",
        description="Embedder: 'sentence-transformers' or 'hashing' (dependency-free fallback).",
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dim: int = Field(
        default=384,
        description="Only used by the hashing embedder; the ST embedder reports its own dimension.",
    )

    # --- candidate discovery ----------------------------------------------
    top_k: int = Field(default=10, ge=1, description="ANN neighbours examined per chunk.")
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity above which two chunks are linked as duplicate candidates.",
    )
    max_group_size: int = Field(
        default=8,
        ge=2,
        description="Hard cap on chunks merged into one summary, so a dense cluster cannot collapse the store.",
    )

    # --- validation --------------------------------------------------------
    validation_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum pairwise cosine similarity required across a group before it may be merged.",
    )
    refine_rejected_groups: bool = Field(
        default=True,
        description="If a group fails validation, keep the largest subset that passes instead of discarding it.",
    )

    # --- summarization -----------------------------------------------------
    llm_backend: str = Field(
        default="litellm",
        description="Summarizer backend: 'litellm' (hosted or local via LiteLLM routing) or 'stub'.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LiteLLM model string, e.g. gpt-4o-mini, claude-3-5-haiku-20241022, ollama/llama3.2:3b.",
    )
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_summary_tokens: int = Field(default=512, ge=32)
    llm_timeout: float = Field(default=60.0, gt=0)

    # --- accounting --------------------------------------------------------
    token_model: str = Field(
        default="gpt-4o-mini",
        description="Model whose tokenizer is used for benchmark accounting.",
    )

    @model_validator(mode="after")
    def _check(self) -> "CompactorConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.validation_threshold < self.similarity_threshold:
            raise ValueError(
                "validation_threshold must be >= similarity_threshold; the validation pass is "
                "meant to be stricter than candidate discovery, not looser"
            )
        return self

    # --- paths -------------------------------------------------------------
    @property
    def store_path(self) -> Path:
        return self.workdir / "chroma"

    @property
    def archive_path(self) -> Path:
        return self.workdir / "archive.sqlite3"

    def ensure_workdir(self) -> Path:
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir

    # --- serialization -----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "CompactorConfig":
        """Load config from JSON, applying RAGCOMPACTOR_* environment overrides."""
        data: dict = {}
        if path is not None:
            p = Path(path)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
        for field in cls.model_fields:
            env_key = f"RAGCOMPACTOR_{field.upper()}"
            if env_key in os.environ:
                data[field] = os.environ[env_key]
        return cls.model_validate(data)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p
