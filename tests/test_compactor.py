from ragcompactor.compactor import Compactor
from ragcompactor.core import Chunk
from ragcompactor.backends import StubSummarizer, build_prompt
from tests.conftest import DISTINCT_TEXTS, DUPLICATE_TEXTS


def seed(compactor: Compactor) -> list[Chunk]:
    texts = DUPLICATE_TEXTS + DISTINCT_TEXTS
    chunks = [Chunk.create(t, source="seed", ordinal=i) for i, t in enumerate(texts)]
    compactor.add_chunks(chunks)
    return chunks


def test_compaction_merges_duplicates_and_keeps_distinct_chunks(offline_config):
    with Compactor(offline_config) as c:
        chunks = seed(c)
        report = c.compact()

        assert report.validated_groups == 1
        assert report.chunks_before == 4
        assert report.chunks_after == 3
        assert report.chunks_archived == 2

        # the two distinct chunks are untouched
        remaining = {ch.id for ch in c.store.all_chunks()}
        assert chunks[2].id in remaining
        assert chunks[3].id in remaining
        # the duplicates are gone, replaced by one merged chunk
        assert chunks[0].id not in remaining
        assert chunks[1].id not in remaining
        merged = [ch for ch in c.store.all_chunks() if ch.is_merged]
        assert len(merged) == 1
        assert set(merged[0].metadata["merged_from"]) == {chunks[0].id, chunks[1].id}


def test_dry_run_changes_nothing(offline_config):
    with Compactor(offline_config) as c:
        seed(c)
        before = set(c.store.all_ids())

        report = c.compact(dry_run=True)

        assert report.dry_run is True
        assert report.run_id == "dry-run"
        assert report.candidate_groups == 1
        assert set(c.store.all_ids()) == before
        assert c.archive.list_runs() == []


def test_undo_restores_the_exact_originals(offline_config):
    with Compactor(offline_config) as c:
        chunks = seed(c)
        originals = {ch.id: ch.text for ch in chunks}

        report = c.compact()
        assert c.store.count() == 3

        restored = c.undo_run(report.run_id)

        assert restored == 1
        assert c.store.count() == 4
        assert set(c.store.all_ids()) == set(originals)
        for chunk_id, text in originals.items():
            assert c.store.get(chunk_id).text == text
        # the merged chunk is gone
        assert not [ch for ch in c.store.all_chunks() if ch.is_merged]


def test_undo_is_idempotent_and_reports_unknown_merges(offline_config):
    with Compactor(offline_config) as c:
        seed(c)
        report = c.compact()
        merge_id = report.merges[0].merge_id

        assert c.undo_merge(merge_id) is True
        assert c.undo_merge(merge_id) is False
        assert c.undo_merge("does-not-exist") is False


def test_archive_records_run_and_stats(offline_config):
    with Compactor(offline_config) as c:
        seed(c)
        report = c.compact()

        runs = c.archive.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == report.run_id
        assert runs[0]["merges"] == 1

        stats = c.stats()
        assert stats["chunks"] == 3
        assert stats["merged_chunks"] == 1
        assert stats["archive"]["archived_chunks"] == 2


def test_compaction_is_idempotent_on_a_clean_store(offline_config):
    with Compactor(offline_config) as c:
        seed(c)
        c.compact()
        after_first = c.store.count()

        second = c.compact()

        assert c.store.count() == after_first
        assert second.chunks_merged == 0


def test_stub_summarizer_preserves_distinct_sentences():
    result = StubSummarizer().summarize(
        [
            "Chroma stores the vectors. It supports cosine distance.",
            "Chroma stores the vectors. The archive lives in SQLite instead.",
        ]
    )
    assert "Chroma stores the vectors" in result.text
    assert "archive lives in SQLite" in result.text
    assert result.total_tokens == 0


def test_prompt_labels_every_chunk():
    prompt = build_prompt(["first", "second", "third"])
    assert "--- chunk 1 ---" in prompt
    assert "--- chunk 3 ---" in prompt
