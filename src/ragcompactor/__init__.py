"""ragcompactor - local RAG memory compactor and deduplicator.

Merges redundant chunks in a vector store into single summarized chunks so that
retrieval pulls less duplicated text into the context window. Originals are
archived rather than deleted, so every merge can be undone.
"""

from ragcompactor.config import CompactorConfig
from ragcompactor.models import (
    CandidateGroup,
    Chunk,
    CompactionReport,
    MergeRecord,
    ValidationOutcome,
)

__version__ = "0.1.0"

__all__ = [
    "CompactorConfig",
    "Chunk",
    "CandidateGroup",
    "ValidationOutcome",
    "MergeRecord",
    "CompactionReport",
    "__version__",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    # Compactor pulls in the store/embedder layer, so keep it lazy: importing the
    # package must stay cheap and must not require the optional heavy deps.
    if name == "Compactor":
        from ragcompactor.compactor import Compactor

        return Compactor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
