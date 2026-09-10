import numpy as np

from ragcompactor.pipeline import find_candidate_groups, group_stats, pairwise_matrix
from ragcompactor.backends import HashingEmbedder
from ragcompactor.core import CandidateGroup, Chunk
from ragcompactor.backends import InMemoryStore
from ragcompactor.pipeline import largest_valid_subset, validate_group
from tests.conftest import DISTINCT_TEXTS, DUPLICATE_TEXTS


def test_candidate_discovery_groups_duplicates_and_ignores_distinct_text():
    embedder = HashingEmbedder(384)
    texts = DUPLICATE_TEXTS + DISTINCT_TEXTS
    chunks = [Chunk.create(t, source="s", ordinal=i) for i, t in enumerate(texts)]
    store = InMemoryStore()
    store.add(chunks, embedder.embed(texts))

    groups = find_candidate_groups(store, top_k=5, threshold=0.80)

    assert len(groups) == 1
    grouped = set(groups[0].chunk_ids)
    assert grouped == {chunks[0].id, chunks[1].id}
    for distinct_chunk in chunks[2:]:
        assert distinct_chunk.id not in grouped


def test_candidate_discovery_chains_transitively(chain_store):
    # a~b and b~c are above 0.80; a~c (0.805) also is, so all three chain together
    groups = find_candidate_groups(chain_store, top_k=5, threshold=0.80)
    assert len(groups) == 1
    assert set(groups[0].chunk_ids) == {"a", "b", "c"}
    assert groups[0].min_similarity < 0.85


def test_max_group_size_caps_merge_width():
    embedder = HashingEmbedder(384)
    base = "The compactor merges redundant chunks in a vector store to reduce token usage"
    texts = [f"{base} {'really ' * i}now." for i in range(6)]
    chunks = [Chunk.create(t, source="s", ordinal=i) for i, t in enumerate(texts)]
    store = InMemoryStore()
    store.add(chunks, embedder.embed(texts))

    groups = find_candidate_groups(store, top_k=6, threshold=0.70, max_group_size=3)
    assert groups
    assert all(g.size <= 3 for g in groups)


def test_validation_rejects_or_refines_a_transitive_group(chain_store):
    group = CandidateGroup(chunk_ids=["a", "b", "c"], mean_similarity=0.9, min_similarity=0.805)

    strict = validate_group(chain_store, group, threshold=0.90, refine=False)
    assert strict.accepted is False
    assert "weakest pair" in strict.reason

    refined = validate_group(chain_store, group, threshold=0.90, refine=True)
    assert refined.accepted is True
    assert refined.size == 2
    assert refined.min_similarity >= 0.90
    assert len(refined.dropped_ids) == 1


def test_validation_accepts_a_genuinely_tight_group():
    store = InMemoryStore()
    chunks = [Chunk(id="x", text="one"), Chunk(id="y", text="two")]
    store.add(chunks, np.vstack([np.array([1.0, 0.0]), np.array([0.999, 0.0447])]))

    group = CandidateGroup(chunk_ids=["x", "y"], mean_similarity=0.99, min_similarity=0.99)
    outcome = validate_group(store, group, threshold=0.90)

    assert outcome.accepted is True
    assert outcome.dropped_ids == []
    assert outcome.size == 2


def test_validation_rejects_when_no_pair_clears_threshold():
    store = InMemoryStore()
    chunks = [Chunk(id="p", text="one"), Chunk(id="q", text="two")]
    store.add(chunks, np.vstack([np.array([1.0, 0.0]), np.array([0.0, 1.0])]))

    group = CandidateGroup(chunk_ids=["p", "q"], mean_similarity=0.0, min_similarity=0.0)
    outcome = validate_group(store, group, threshold=0.90, refine=True)

    assert outcome.accepted is False
    assert "no subset" in outcome.reason


def test_largest_valid_subset_finds_the_clique():
    # 0-1-2 mutually similar; 3 attached only to 0
    sims = np.array(
        [
            [1.00, 0.95, 0.94, 0.93],
            [0.95, 1.00, 0.96, 0.10],
            [0.94, 0.96, 1.00, 0.11],
            [0.93, 0.10, 0.11, 1.00],
        ]
    )
    assert largest_valid_subset(sims, 0.90) == [0, 1, 2]


def test_group_stats_matches_pairwise_matrix(chain_store):
    mean_sim, min_sim = group_stats(chain_store, ["a", "b", "c"])
    sims = pairwise_matrix(chain_store, ["a", "b", "c"])
    assert min_sim == min(sims[0, 1], sims[0, 2], sims[1, 2])
    assert 0.0 < mean_sim <= 1.0
