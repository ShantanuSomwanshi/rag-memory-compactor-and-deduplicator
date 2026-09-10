"""ragcompactor - local RAG memory compactor and deduplicator.

Merges redundant chunks in a vector store into single summarized chunks so that
retrieval pulls less duplicated text into the context window. Originals are
archived rather than deleted, so every merge can be undone.

Layout mirrors the pipeline:

    core       config, data models, token accounting
    ingest     load files and split them into chunks
    backends   embedding, vector storage and summarization (all swappable)
    pipeline   candidate discovery and the validation gate
    archive    SQLite record of every merge, and the undo path
    compactor  orchestrates the stages
    benchmark  before/after token accounting
    cli        command line entry point
"""

from ragcompactor.core import (
    CandidateGroup,
    Chunk,
    CompactionReport,
    CompactorConfig,
    MergeRecord,
    SummaryResult,
    ValidationOutcome,
)

__version__ = "0.1.0"

__all__ = [
    "CompactorConfig",
    "Chunk",
    "CandidateGroup",
    "ValidationOutcome",
    "SummaryResult",
    "MergeRecord",
    "CompactionReport",
    "__version__",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    # Compactor pulls in the backends layer, so keep it lazy: importing the
    # package must stay cheap and must not require the optional heavy deps.
    if name == "Compactor":
        from ragcompactor.compactor import Compactor

        return Compactor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
