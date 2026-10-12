"""The ingest ledger, and compaction across separately-ingested batches."""

from ragcompactor.compactor import Compactor
from ragcompactor.ingest import IngestLedger, file_fingerprint
from tests.conftest import DISTINCT_TEXTS, DUPLICATE_TEXTS


def write(tmp_path, name: str, text: str):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# --- ledger mechanics ------------------------------------------------------


def test_fingerprint_tracks_content_not_timestamp(tmp_path):
    f = write(tmp_path, "a.md", "hello world")
    first = file_fingerprint(f)

    f.touch()  # mtime moves, content does not
    assert file_fingerprint(f)["content_hash"] == first["content_hash"]

    f.write_text("hello world!", encoding="utf-8")
    assert file_fingerprint(f)["content_hash"] != first["content_hash"]


def test_ledger_round_trips_through_json(tmp_path):
    path = tmp_path / "ingested.json"
    ledger = IngestLedger(path)
    f = write(tmp_path, "a.md", "some text")

    ledger.record_root(tmp_path)
    ledger.record_file(f, file_fingerprint(f), ["c1", "c2"])
    ledger.save()

    reloaded = IngestLedger(path)
    assert reloaded.status(f) == "unchanged"
    assert reloaded.chunk_ids_for(f) == ["c1", "c2"]
    assert reloaded.summary()["files"] == 1
    assert reloaded.summary()["chunks"] == 2
    assert str(tmp_path.resolve().as_posix()) in reloaded.roots


def test_ledger_reports_status_transitions(tmp_path):
    ledger = IngestLedger(tmp_path / "l.json")
    f = write(tmp_path, "a.md", "one")

    assert ledger.status(f) == "new"
    ledger.record_file(f, file_fingerprint(f), ["x"])
    assert ledger.status(f) == "unchanged"

    f.write_text("two", encoding="utf-8")
    assert ledger.status(f) == "changed"


def test_corrupt_ledger_starts_empty_rather_than_raising(tmp_path):
    path = tmp_path / "ingested.json"
    path.write_text("{not json", encoding="utf-8")
    assert IngestLedger(path).summary()["files"] == 0


def test_ledger_flags_sources_that_vanished(tmp_path):
    ledger = IngestLedger(tmp_path / "l.json")
    f = write(tmp_path, "gone.md", "text")
    ledger.record_file(f, file_fingerprint(f), ["x"])
    f.unlink()
    assert ledger.missing_files() == [ledger.key(f)]


# --- ingestion behaviour ---------------------------------------------------


def test_unchanged_files_are_skipped_on_reingest(tmp_path, offline_config):
    write(tmp_path / "corpus", "a.md", DISTINCT_TEXTS[0])
    write(tmp_path / "corpus", "b.md", DISTINCT_TEXTS[1])

    with Compactor(offline_config) as c:
        first = c.ingest([tmp_path / "corpus"])
        assert first.added_files == 2
        assert first.skipped_files == 0
        assert first.chunks_added == 2

        second = c.ingest([tmp_path / "corpus"])
        assert second.added_files == 0
        assert second.skipped_files == 2
        assert second.chunks_added == 0
        assert c.store.count() == 2


def test_changed_file_replaces_its_own_chunks(tmp_path, offline_config):
    f = write(tmp_path / "corpus", "a.md", DISTINCT_TEXTS[0])

    with Compactor(offline_config) as c:
        c.ingest([tmp_path / "corpus"])
        old_ids = c.ledger.chunk_ids_for(f)

        f.write_text(DISTINCT_TEXTS[1], encoding="utf-8")
        report = c.ingest([tmp_path / "corpus"])

        assert report.updated_files == 1
        assert report.chunks_removed == 1
        assert c.store.count() == 1
        # the superseded chunk is gone, not orphaned alongside the new one
        assert c.store.get(old_ids[0]) is None


def test_force_reingests_unchanged_files(tmp_path, offline_config):
    write(tmp_path / "corpus", "a.md", DISTINCT_TEXTS[0])

    with Compactor(offline_config) as c:
        c.ingest([tmp_path / "corpus"])
        forced = c.ingest([tmp_path / "corpus"], force=True)

        assert forced.skipped_files == 0
        assert forced.updated_files == 1
        assert c.store.count() == 1


def test_ledger_records_the_folders_ingested(tmp_path, offline_config):
    write(tmp_path / "corpus", "a.md", DISTINCT_TEXTS[0])

    with Compactor(offline_config) as c:
        c.ingest([tmp_path / "corpus"])
        assert c.config.ledger_path.exists()
        assert c.ledger.summary()["roots"] == [
            IngestLedger.key(tmp_path / "corpus")
        ]


def test_forget_source_removes_its_chunks(tmp_path, offline_config):
    f = write(tmp_path / "corpus", "a.md", DISTINCT_TEXTS[0])
    write(tmp_path / "corpus", "b.md", DISTINCT_TEXTS[1])

    with Compactor(offline_config) as c:
        c.ingest([tmp_path / "corpus"])
        removed = c.forget_source(f)

        assert removed == 1
        assert c.store.count() == 1
        assert c.ledger.status(f) == "new"


# --- interaction with compaction -------------------------------------------


def test_new_data_is_compacted_against_data_already_in_the_store(
    tmp_path, offline_config
):
    """The point of the feature: batch two must be compared with batch one."""
    write(tmp_path / "batch1", "a.md", DUPLICATE_TEXTS[0])
    write(tmp_path / "batch1", "far.md", DISTINCT_TEXTS[0])

    with Compactor(offline_config) as c:
        c.ingest([tmp_path / "batch1"])
        first = c.compact()
        assert first.validated_groups == 0  # nothing to merge within batch one
        assert c.store.count() == 2

        # a near-duplicate of batch one, ingested later as a separate folder
        write(tmp_path / "batch2", "b.md", DUPLICATE_TEXTS[1])
        c.ingest([tmp_path / "batch2"])
        assert c.store.count() == 3

        second = c.compact()

        assert second.validated_groups == 1
        assert second.chunks_after == 2
        merged = [ch for ch in c.store.all_chunks() if ch.is_merged]
        assert len(merged) == 1
        # the merge spans both folders
        assert len(set(merged[0].metadata["merged_sources"])) == 2


def test_reingest_does_not_resurrect_chunks_that_were_merged_away(
    tmp_path, offline_config
):
    """Without the ledger this silently undoes the previous compaction."""
    write(tmp_path / "corpus", "a.md", DUPLICATE_TEXTS[0])
    write(tmp_path / "corpus", "b.md", DUPLICATE_TEXTS[1])

    with Compactor(offline_config) as c:
        c.ingest([tmp_path / "corpus"])
        report = c.compact()
        assert report.validated_groups == 1
        after_compaction = c.store.count()

        again = c.ingest([tmp_path / "corpus"])

        assert again.skipped_files == 2
        assert again.chunks_added == 0
        assert c.store.count() == after_compaction


def test_editing_a_merged_source_warns_that_a_summary_is_stale(
    tmp_path, offline_config
):
    a = write(tmp_path / "corpus", "a.md", DUPLICATE_TEXTS[0])
    write(tmp_path / "corpus", "b.md", DUPLICATE_TEXTS[1])

    with Compactor(offline_config) as c:
        c.ingest([tmp_path / "corpus"])
        run = c.compact()
        merge_id = run.merges[0].merge_id

        a.write_text(DUPLICATE_TEXTS[0] + " Plus a new clause.", encoding="utf-8")
        report = c.ingest([tmp_path / "corpus"])

        assert report.updated_files == 1
        assert report.stale_merges == [merge_id]
