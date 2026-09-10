"""The orchestrator: ingest -> discover -> validate -> summarize -> archive -> undo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ragcompactor.archive import Archive
from ragcompactor.backends import Embedder, Summarizer, VectorStore, get_embedder, get_store, get_summarizer
from ragcompactor.core import (
    Chunk,
    CompactionReport,
    CompactorConfig,
    IngestReport,
    MergeRecord,
)
from ragcompactor.ingest import (
    IngestLedger,
    chunk_file,
    file_fingerprint,
    iter_source_files,
)
from ragcompactor.pipeline import find_candidate_groups, validate_group


class Compactor:
    """Wires the pipeline stages together over one store + archive pair."""

    def __init__(
        self,
        config: CompactorConfig | None = None,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        summarizer: Summarizer | None = None,
        archive: Archive | None = None,
        ledger: IngestLedger | None = None,
    ) -> None:
        self.config = config or CompactorConfig()
        self.config.ensure_workdir()
        self.store = store if store is not None else get_store(self.config)
        self.embedder = embedder if embedder is not None else get_embedder(self.config)
        self.summarizer = (
            summarizer if summarizer is not None else get_summarizer(self.config)
        )
        self.archive = archive if archive is not None else Archive(self.config.archive_path)
        self.ledger = (
            ledger if ledger is not None else IngestLedger(self.config.ledger_path)
        )

    # --- ingestion ---------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def ingest(self, paths: list[str | Path], force: bool = False) -> IngestReport:
        """Ingest every supported file under ``paths``, skipping unchanged ones.

        The ledger decides what work is actually done. A file whose content hash
        matches its recorded one is skipped outright. A file that has changed has
        its previous chunks removed from the store before the new ones are added,
        so an edit does not leave orphaned chunks behind.

        ``force`` re-ingests everything regardless of the ledger, which is what
        you want after changing ``chunk_size`` or swapping the embedding model,
        since those invalidate stored chunks without changing any source file.
        """
        report = IngestReport()
        stale: set[str] = set()

        for root in paths:
            self.ledger.record_root(root)
            report.roots.append(str(root))

            for path in iter_source_files(root):
                fingerprint = file_fingerprint(path)
                status = self.ledger.status(path, fingerprint)

                if status == "unchanged" and not force:
                    report.skipped_files += 1
                    continue

                # An edited (or force-reingested) file's old chunks must go, or
                # they linger in the store with no source to justify them.
                old_ids = self.ledger.chunk_ids_for(path)
                if old_ids:
                    present = [i for i in old_ids if self.store.get(i) is not None]
                    stale.update(self.archive.merges_containing(old_ids))
                    self.store.delete(old_ids)
                    report.chunks_removed += len(present)

                chunks = chunk_file(
                    path,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap,
                )
                self.add_chunks(chunks)
                self.ledger.record_file(path, fingerprint, [c.id for c in chunks])
                report.chunks_added += len(chunks)

                if status == "new":
                    report.added_files += 1
                else:
                    report.updated_files += 1

        report.stale_merges = sorted(stale)
        self.ledger.save()
        return report

    def forget_source(self, path: str | Path) -> int:
        """Drop a source from the ledger and remove its chunks from the store."""
        chunk_ids = self.ledger.forget_file(path)
        present = [i for i in chunk_ids if self.store.get(i) is not None]
        if chunk_ids:
            self.store.delete(chunk_ids)
        self.ledger.save()
        return len(present)

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

            try:
                result = self.summarizer.summarize([c.text for c in originals])
            except Exception as exc:
                # A provider failure (rate limit, timeout, outage) must not throw
                # away the merges this run has already committed. Record it, and
                # give up only once failures look persistent rather than isolated.
                report.failed_groups += 1
                report.validated_groups -= 1
                report.last_error = f"{type(exc).__name__}: {exc}"[:300]
                if report.failed_groups >= cfg.max_group_failures:
                    break
                continue

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
            "sources": self.ledger.summary(),
        }

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "Compactor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def merges_of(report: CompactionReport) -> list[MergeRecord]:
    return list(report.merges)
