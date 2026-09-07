import pytest

from ragcompactor.benchmark import (
    CorpusStats,
    RetrievalStats,
    build_report,
    corpus_stats,
    retrieval_stats,
)
from ragcompactor.compactor import Compactor
from ragcompactor.config import CompactorConfig
from ragcompactor.models import Chunk
from ragcompactor.tokens import count_tokens
from tests.conftest import DISTINCT_TEXTS, DUPLICATE_TEXTS


def test_config_rejects_looser_validation_than_discovery():
    with pytest.raises(ValueError):
        CompactorConfig(similarity_threshold=0.90, validation_threshold=0.80)


def test_config_rejects_overlap_larger_than_chunk():
    with pytest.raises(ValueError):
        CompactorConfig(chunk_size=100, chunk_overlap=100)


def test_config_round_trips_through_json(tmp_path):
    cfg = CompactorConfig(top_k=7, llm_model="claude-3-5-haiku-20241022")
    path = cfg.save(tmp_path / "cfg.json")
    loaded = CompactorConfig.load(path)
    assert loaded.top_k == 7
    assert loaded.llm_model == "claude-3-5-haiku-20241022"


def test_break_even_divides_cost_by_per_query_saving():
    report = build_report(
        before_corpus=CorpusStats(chunks=100, total_tokens=10_000),
        after_corpus=CorpusStats(chunks=80, total_tokens=8_000),
        before_retrieval=RetrievalStats(
            queries=10, top_k=5, total_context_tokens=5_000, unique_chunks_retrieved=40
        ),
        after_retrieval=RetrievalStats(
            queries=10, top_k=5, total_context_tokens=4_000, unique_chunks_retrieved=30
        ),
        compaction_tokens=2_000,
    )

    assert report.tokens_saved_per_query == 100.0  # 500 -> 400 per query
    assert report.break_even_queries == 20.0  # 2000 / 100
    assert report.corpus_reduction_pct == pytest.approx(20.0)
    assert report.context_reduction_pct == pytest.approx(20.0)


def test_break_even_is_none_when_compaction_saves_nothing():
    report = build_report(
        before_corpus=CorpusStats(chunks=10, total_tokens=1_000),
        after_corpus=CorpusStats(chunks=10, total_tokens=1_000),
        before_retrieval=RetrievalStats(4, 5, 400, 10),
        after_retrieval=RetrievalStats(4, 5, 400, 10),
        compaction_tokens=500,
    )
    assert report.break_even_queries is None
    assert "never" in report.render()


def test_zero_cost_run_is_flagged_as_not_meaningful():
    report = build_report(
        before_corpus=CorpusStats(4, 400),
        after_corpus=CorpusStats(3, 300),
        before_retrieval=RetrievalStats(2, 2, 200, 4),
        after_retrieval=RetrievalStats(2, 2, 150, 3),
        compaction_tokens=0,
    )
    assert any("stub summarizer" in n for n in report.notes)


def test_end_to_end_benchmark_shows_smaller_corpus(offline_config):
    with Compactor(offline_config) as c:
        texts = DUPLICATE_TEXTS + DISTINCT_TEXTS
        c.add_chunks([Chunk.create(t, source="s", ordinal=i) for i, t in enumerate(texts)])
        queries = [" ".join(t.split()[:8]) for t in texts]

        before_corpus = corpus_stats(c.store, c.config.token_model)
        before_retrieval = retrieval_stats(c.store, c.embedder, queries, 3, c.config.token_model)

        run = c.compact()

        after_corpus = corpus_stats(c.store, c.config.token_model)
        after_retrieval = retrieval_stats(c.store, c.embedder, queries, 3, c.config.token_model)

        report = build_report(
            before_corpus,
            after_corpus,
            before_retrieval,
            after_retrieval,
            compaction_tokens=run.summarization_tokens,
        )

        assert report.chunks_removed == 1
        assert report.after_corpus.chunks == 3
        assert "Compaction benchmark" in report.render()


def test_token_counting_is_positive_and_monotonic():
    short = count_tokens("hello world")
    longer = count_tokens("hello world " * 50)
    assert short > 0
    assert longer > short
    assert count_tokens("") == 0
