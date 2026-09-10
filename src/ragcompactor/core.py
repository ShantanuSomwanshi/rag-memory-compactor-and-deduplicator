"""Configuration, data models and token accounting.

The plain-data layer of the package: the knobs (:class:`CompactorConfig`), the
structures every pipeline stage passes around, and token counting used by both
the summarizer and the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib
import json
import os

from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------------
# config
# Configuration for a compaction run.
# ------------------------------------------------------------------------

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
    llm_num_retries: int = Field(
        default=5,
        ge=0,
        description="Retries with exponential backoff on rate-limit / transient errors.",
    )
    max_group_failures: int = Field(
        default=3,
        ge=1,
        description="Abandon the run after this many groups fail at the LLM step. "
        "Merges already completed stay committed.",
    )
    llm_request_delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds to wait between summarization calls. Set this to stay "
        "under a provider's tokens-per-minute cap (Groq's free tier allows roughly "
        "one merge every 7s).",
    )

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

    @property
    def ledger_path(self) -> Path:
        """JSON record of which sources have already been ingested."""
        return self.workdir / "ingested.json"

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


# ------------------------------------------------------------------------
# models
# Core data structures passed between pipeline stages.
# ------------------------------------------------------------------------

def chunk_id_for(text: str, source: str, ordinal: int) -> str:
    """Stable content-addressed id, so re-ingesting the same file is idempotent."""
    digest = hashlib.sha256(f"{source}:{ordinal}:{text}".encode("utf-8")).hexdigest()
    return digest[:24]


@dataclass
class Chunk:
    """One unit of text in the vector store."""

    id: str
    text: str
    source: str = ""
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls, text: str, source: str = "", ordinal: int = 0, **metadata: Any
    ) -> "Chunk":
        return cls(
            id=chunk_id_for(text, source, ordinal),
            text=text,
            source=source,
            ordinal=ordinal,
            metadata=metadata,
        )

    @property
    def is_merged(self) -> bool:
        return bool(self.metadata.get("merged"))


@dataclass
class CandidateGroup:
    """A set of chunks that ANN + similarity mapping believes is redundant.

    This is a *proposal only*. Nothing here has passed the validation gate yet.
    """

    chunk_ids: list[str]
    mean_similarity: float
    min_similarity: float

    @property
    def size(self) -> int:
        return len(self.chunk_ids)


@dataclass
class ValidationOutcome:
    """Result of the strict pairwise check applied before summarization."""

    accepted: bool
    chunk_ids: list[str]
    min_similarity: float
    mean_similarity: float
    reason: str
    dropped_ids: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.chunk_ids)


@dataclass
class SummaryResult:
    """What a summarizer backend returns, including its own token cost."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class MergeRecord:
    """One merge, as persisted in the archive."""

    merge_id: str
    run_id: str
    merged_chunk_id: str
    merged_text: str
    original_ids: list[str]
    created_at: str
    undone: bool = False


@dataclass
class CompactionReport:
    """Summary of a single compaction run."""

    run_id: str
    chunks_before: int = 0
    chunks_after: int = 0
    candidate_groups: int = 0
    validated_groups: int = 0
    rejected_groups: int = 0
    refined_groups: int = 0
    chunks_merged: int = 0
    chunks_archived: int = 0
    summarization_tokens: int = 0
    failed_groups: int = 0
    last_error: str = ""
    dry_run: bool = False
    merges: list[MergeRecord] = field(default_factory=list)

    @property
    def chunks_removed(self) -> int:
        return self.chunks_before - self.chunks_after

    @property
    def reduction_pct(self) -> float:
        if self.chunks_before == 0:
            return 0.0
        return 100.0 * self.chunks_removed / self.chunks_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "chunks_before": self.chunks_before,
            "chunks_after": self.chunks_after,
            "chunks_removed": self.chunks_removed,
            "reduction_pct": round(self.reduction_pct, 2),
            "candidate_groups": self.candidate_groups,
            "validated_groups": self.validated_groups,
            "rejected_groups": self.rejected_groups,
            "refined_groups": self.refined_groups,
            "chunks_merged": self.chunks_merged,
            "chunks_archived": self.chunks_archived,
            "summarization_tokens": self.summarization_tokens,
            "failed_groups": self.failed_groups,
            "last_error": self.last_error,
            "dry_run": self.dry_run,
        }


@dataclass
class IngestReport:
    """Outcome of one ingestion pass, as decided against the ledger."""

    roots: list[str] = field(default_factory=list)
    added_files: int = 0
    skipped_files: int = 0
    updated_files: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    stale_merges: list[str] = field(default_factory=list)

    @property
    def files_seen(self) -> int:
        return self.added_files + self.skipped_files + self.updated_files

    def to_dict(self) -> dict[str, Any]:
        return {
            "roots": self.roots,
            "files_seen": self.files_seen,
            "added_files": self.added_files,
            "updated_files": self.updated_files,
            "skipped_files": self.skipped_files,
            "chunks_added": self.chunks_added,
            "chunks_removed": self.chunks_removed,
            "stale_merges": self.stale_merges,
        }


# ------------------------------------------------------------------------
# tokens
# Token accounting.
#
# Uses tiktoken when it is installed so the benchmark numbers are real, and falls
# back to a characters/4 estimate otherwise. The fallback is flagged on the
# result so a report never silently presents an estimate as a measurement.
# ------------------------------------------------------------------------

_FALLBACK_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=8)
def _encoder(model: str):
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - depends on optional extra
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - tiktoken present but unusable
            return None


def tokenizer_is_exact(model: str = "gpt-4o-mini") -> bool:
    """True when real tokenization is available (tiktoken installed)."""
    return _encoder(model) is not None


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in ``text``."""
    if not text:
        return 0
    enc = _encoder(model)
    if enc is None:
        return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)
    return len(enc.encode(text))


def count_many(texts: list[str], model: str = "gpt-4o-mini") -> int:
    return sum(count_tokens(t, model) for t in texts)
