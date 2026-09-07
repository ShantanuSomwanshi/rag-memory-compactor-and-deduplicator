"""Core data structures passed between pipeline stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


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
            "dry_run": self.dry_run,
        }
