"""The orchestrator: ingest -> discover -> validate -> summarize -> archive -> undo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ragcompactor.archive import Archive
from ragcompactor.candidates import find_candidate_groups
from ragcompactor.config import CompactorConfig
from ragcompactor.embeddings import Embedder, get_embedder
from ragcompactor.ingest import chunk_paths
from ragcompactor.models import Chunk, CompactionReport, MergeRecord
from ragcompactor.store import VectorStore, get_store
from ragcompactor.summarize import Summarizer, get_summarizer
from ragcompactor.validate import validate_group


class Compactor:
    """Wires the pipeline stages together over one store + archive pair."""

    def __init__(
        self,
        config: CompactorConfig | None = None,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        summarizer: Summarizer | None = None,
        archive: Archive | None = None,
    ) -> None:
        self.config = config or CompactorConfig()
        self.config.ensure_workdir()
        self.store = store if store is not None else get_store(self.config)
        self.embedder = embedder if embedder is not None else get_embedder(self.config)
        self.summarizer = (
            summarizer if summarizer is not None else get_summarizer(self.config)
        )
        self.archive = archive if archive is not None else Archive(self.config.archive_path)

    # --- ingestion ---------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def ingest(self, paths: list[str | Path]) -> int:
        """Chunk and embed every supported file under ``paths``."""
        chunks = chunk_paths(
            list(paths),
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        return self.add_chunks(chunks)

    # --- compaction --------------------------------------------------------
    def _merged_chunk(self, text: str, originals: list[Chunk], run_id: str) -> Chunk:
        sources = sorted({c.source for c in originals if c.source})
        return Chunk.create(
            text,
            source=sources[0] if len(sources) == 1 else "|".join(sources),
            ordinal=min(c.ordinal for c in originals),
            merged=True,
            merged_from=[c.id for c in originals],
            merged_sources=sources,
            run_id=run_id,
        )

    def compact(self, dry_run: bool = False) -> CompactionReport:
        """Run one full compaction pass.

        With ``dry_run`` the store is left untouched and no LLM call is made, so
        thresholds can be tuned against a real corpus without spending tokens.
        """
        cfg = self.config
        run_id = "dry-run" if dry_run else self.archive.start_run(cfg.model_dump_json())
        report = CompactionReport(
            run_id=run_id, chunks_before=self.store.count(), dry_run=dry_run
        )

        groups = find_candidate_groups(
            self.store,
            top_k=cfg.top_k,
            threshold=cfg.similarity_threshold,
            max_group_size=cfg.max_group_size,
        )
        report.candidate_groups = len(groups)

        for group in groups:
            outcome = validate_group(
                self.store,
                group,
                threshold=cfg.validation_threshold,
                refine=cfg.refine_rejected_groups,
            )
            if not outcome.accepted:
                report.rejected_groups += 1
                continue

            report.validated_groups += 1
            if outcome.dropped_ids:
                report.refined_groups += 1

            originals = [c for c in (self.store.get(i) for i in outcome.chunk_ids) if c]
            if len(originals) < 2:
                report.rejected_groups += 1
                report.validated_groups -= 1
                continue

            if dry_run:
                report.chunks_merged += len(originals)
                continue

            result = self.summarizer.summarize([c.text for c in originals])
            report.summarization_tokens += result.total_tokens
            if not result.text.strip():
                report.rejected_groups += 1
                report.validated_groups -= 1
                continue

            merged = self._merged_chunk(result.text, originals, run_id)
            vectors = {
                c.id: v
                for c in originals
                if (v := self.store.get_vector(c.id)) is not None
            }

            record = self.archive.record_merge(run_id, merged, originals, vectors)
            self.store.delete([c.id for c in originals])
            self.add_chunks([merged])

            report.merges.append(record)
            report.chunks_merged += len(originals)
            report.chunks_archived += len(originals)

        report.chunks_after = self.store.count()
        return report

    # --- undo --------------------------------------------------------------
    def undo_merge(self, merge_id: str) -> bool:
        """Restore one merge: delete the summary, put the originals back."""
        record = self.archive.get_merge(merge_id)
        if record is None or record.undone:
            return False

        originals = self.archive.originals_for(merge_id)
        if not originals:
            return False

        self.store.delete([record.merged_chunk_id])

        with_vectors = [(c, v) for c, v in originals if v is not None]
        without_vectors = [c for c, v in originals if v is None]

        if with_vectors:
            chunks = [c for c, _ in with_vectors]
            matrix = np.vstack([np.asarray(v, dtype=np.float32) for _, v in with_vectors])
            self.store.add(chunks, matrix)
        if without_vectors:
            # archived before vectors were kept, or a store that cannot return them
            self.add_chunks(without_vectors)

        self.archive.mark_undone(merge_id)
        return True

    def undo_run(self, run_id: str) -> int:
        """Undo every merge in a run. Returns how many were restored."""
        restored = 0
        for record in self.archive.list_merges(run_id=run_id, include_undone=False):
            if self.undo_merge(record.merge_id):
                restored += 1
        return restored

    def last_run_id(self) -> str | None:
        runs = self.archive.list_runs()
        return runs[0]["run_id"] if runs else None

    # --- misc --------------------------------------------------------------
    def stats(self) -> dict:
        chunks = self.store.all_chunks()
        merged = [c for c in chunks if c.is_merged]
        return {
            "chunks": len(chunks),
            "merged_chunks": len(merged),
            "archive": self.archive.stats(),
        }

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "Compactor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def merges_of(report: CompactionReport) -> list[MergeRecord]:
    return list(report.merges)
