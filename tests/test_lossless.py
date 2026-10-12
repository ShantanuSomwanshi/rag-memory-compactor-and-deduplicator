"""The free merge path: collapsing a group without calling the LLM.

Duplicates that arrive from overlapping excerpts, or from the same file saved
twice, are *identical*. Paying a model to rewrite two identical paragraphs into
one is the most avoidable cost in the pipeline, so the compactor proves whether
a group can be collapsed by containment before it reaches for the model.
"""

import tempfile
from pathlib import Path

import pytest

from ragcompactor.backends import StubSummarizer
from ragcompactor.compactor import Compactor
from ragcompactor.core import Chunk, CompactorConfig
from ragcompactor.pipeline import lossless_merge

PASSAGE = (
    "Local governments must submit provisional accounts for the previous year "
    "to qualify for grants."
)


class CountingSummarizer(StubSummarizer):
    """A stub that records whether the model was consulted at all."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def summarize(self, texts):
        self.calls += 1
        return super().summarize(texts)


@pytest.fixture
def loose_config():
    """Thresholds low enough that the pairs below reach the merge step."""
    def _make(**overrides):
        return CompactorConfig(
            workdir=Path(tempfile.mkdtemp()) / "wd",
            store_backend="memory",
            embedding_backend="hashing",
            llm_backend="stub",
            similarity_threshold=0.50,
            validation_threshold=0.55,
            **overrides,
        )
    return _make


def compact_texts(config, texts):
    summarizer = CountingSummarizer()
    compactor = Compactor(config, summarizer=summarizer)
    compactor.add_chunks(
        [Chunk.create(t, source=f"f{i}.txt", ordinal=0) for i, t in enumerate(texts)]
    )
    report = compactor.compact()
    merged = [c for c in compactor.store.all_chunks() if c.is_merged]
    return report, summarizer, merged


# --- the containment proof itself ------------------------------------------


def test_identical_texts_collapse_to_one():
    assert lossless_merge([PASSAGE, PASSAGE]) == PASSAGE


def test_whitespace_differences_still_count_as_identical():
    assert lossless_merge([PASSAGE, PASSAGE.replace(" ", "  ")]) == PASSAGE


def test_containing_chunk_wins():
    longer = PASSAGE + " See Table 8.8 for the detailed criteria."
    assert lossless_merge([PASSAGE, longer]) == longer


def test_partial_overlap_is_refused():
    """Sharing a sentence is not containment - this must go to the model."""
    a = PASSAGE + " The State Finance Commission must be constituted on time."
    b = PASSAGE + " A challenge fund was earmarked for air quality."
    assert lossless_merge([a, b]) is None


def test_paraphrase_is_refused():
    other = (
        "Local bodies have to file provisional accounts from the prior year "
        "to become eligible."
    )
    assert lossless_merge([PASSAGE, other]) is None


def test_single_text_is_not_a_merge():
    assert lossless_merge([PASSAGE]) is None


# --- the pipeline honouring it ---------------------------------------------


def test_identical_chunks_merge_without_touching_the_llm(loose_config):
    report, summarizer, merged = compact_texts(loose_config(), [PASSAGE, PASSAGE])

    assert summarizer.calls == 0
    assert report.lossless_groups == 1
    assert report.llm_groups == 0
    assert report.summarization_tokens == 0
    assert len(merged) == 1
    assert merged[0].metadata["merge_method"] == "lossless"
    # and the surviving text is the original, word for word
    assert merged[0].text == PASSAGE


def test_contained_chunks_also_merge_for_free(loose_config):
    longer = PASSAGE + " See Table 8.8 for the detailed criteria."
    report, summarizer, merged = compact_texts(loose_config(), [PASSAGE, longer])

    assert summarizer.calls == 0
    assert report.lossless_groups == 1
    assert merged[0].text == longer


def test_genuine_paraphrase_still_goes_to_the_llm(loose_config):
    near = PASSAGE.replace("grants", "funds")
    report, summarizer, merged = compact_texts(loose_config(), [PASSAGE, near])

    assert summarizer.calls == 1
    assert report.lossless_groups == 0
    assert report.llm_groups == 1
    assert merged[0].metadata["merge_method"] == "llm"


def test_the_free_path_can_be_switched_off(loose_config):
    report, summarizer, merged = compact_texts(
        loose_config(lossless_merges=False), [PASSAGE, PASSAGE]
    )

    assert summarizer.calls == 1
    assert report.lossless_groups == 0
    assert report.llm_groups == 1
    assert merged[0].metadata["merge_method"] == "llm"


def test_a_free_merge_is_still_fully_reversible(loose_config):
    """Skipping the model must not skip the archive."""
    config = loose_config()
    summarizer = CountingSummarizer()
    compactor = Compactor(config, summarizer=summarizer)
    compactor.add_chunks(
        [Chunk.create(PASSAGE, source=f"f{i}.txt", ordinal=0) for i in range(2)]
    )
    report = compactor.compact()
    assert compactor.store.count() == 1

    restored = compactor.undo_run(report.run_id)

    assert restored == 1
    assert compactor.store.count() == 2
    assert all(c.text == PASSAGE for c in compactor.store.all_chunks())
