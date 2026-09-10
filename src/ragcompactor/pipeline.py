"""Candidate discovery and the validation gate.

Stages 2 and 3. Discovery is a generous recall filter - ANN top-k search plus
similarity mapping into connected components. Validation is the strict
precision filter that every group must clear before an LLM ever sees it.

The two live together because they are two halves of one decision: discovery
proposes, validation disposes.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np

from ragcompactor.backends import VectorStore
from ragcompactor.core import CandidateGroup, ValidationOutcome


# ------------------------------------------------------------------------
# candidates
# Candidate discovery: ANN top-k search followed by similarity mapping.
#
# Stage 2 of the pipeline. Every chunk queries the store for its ``top_k``
# nearest neighbours; pairs scoring above ``similarity_threshold`` become edges in
# a graph, and each connected component of that graph is a candidate group.
#
# Grouping by connected component is intentionally generous - it is a recall
# filter. Because similarity is not transitive, a component can chain together
# chunks that are not all alike (A~B, B~C, but A!~C), which is exactly why
# nothing here is merged until it clears the stricter gate in ``validate``.
# ------------------------------------------------------------------------

def build_similarity_graph(
    store: VectorStore, top_k: int = 10, threshold: float = 0.85
) -> nx.Graph:
    """ANN top-k per chunk, keeping edges above ``threshold``."""
    graph: nx.Graph = nx.Graph()
    ids = store.all_ids()
    graph.add_nodes_from(ids)

    for chunk_id in ids:
        vector = store.get_vector(chunk_id)
        if vector is None:
            continue
        for neighbour_id, score in store.query(vector, k=top_k, exclude={chunk_id}):
            if score >= threshold:
                # undirected: keep the best score seen for the pair
                existing = graph.get_edge_data(chunk_id, neighbour_id, default={}).get(
                    "weight", 0.0
                )
                graph.add_edge(chunk_id, neighbour_id, weight=max(existing, float(score)))
    return graph


def pairwise_matrix(store: VectorStore, chunk_ids: list[str]) -> np.ndarray:
    """Full cosine similarity matrix for the given chunks (vectors are unit norm)."""
    vectors = []
    for chunk_id in chunk_ids:
        vec = store.get_vector(chunk_id)
        if vec is None:
            raise KeyError(f"no vector stored for chunk {chunk_id}")
        vectors.append(np.asarray(vec, dtype=np.float32))
    matrix = np.vstack(vectors)
    return np.clip(matrix @ matrix.T, -1.0, 1.0)


def group_stats(store: VectorStore, chunk_ids: list[str]) -> tuple[float, float]:
    """(mean, min) similarity across all distinct pairs in the group."""
    if len(chunk_ids) < 2:
        return 1.0, 1.0
    sims = pairwise_matrix(store, chunk_ids)
    pairs = [sims[i, j] for i, j in combinations(range(len(chunk_ids)), 2)]
    return float(np.mean(pairs)), float(np.min(pairs))


def _split_oversized(
    graph: nx.Graph, component: list[str], max_size: int
) -> list[list[str]]:
    """Break a component larger than ``max_size`` into bounded sub-groups.

    Greedy and deterministic: repeatedly take the highest-degree remaining node
    and pull in its strongest neighbours until the cap is reached. Merging an
    unbounded cluster into one summary would destroy detail, so the cap matters.
    """
    remaining = set(component)
    groups: list[list[str]] = []

    while len(remaining) >= 2:
        sub = graph.subgraph(remaining)
        seed = max(sorted(remaining), key=lambda n: (sub.degree(n), n))
        neighbours = sorted(
            (n for n in sub.neighbors(seed)),
            key=lambda n: (-sub[seed][n].get("weight", 0.0), n),
        )
        group = [seed] + neighbours[: max_size - 1]
        if len(group) < 2:
            remaining.discard(seed)
            continue
        groups.append(group)
        remaining -= set(group)

    return groups


def find_candidate_groups(
    store: VectorStore,
    top_k: int = 10,
    threshold: float = 0.85,
    max_group_size: int = 8,
) -> list[CandidateGroup]:
    """Return proposed redundant groups, largest and tightest first."""
    graph = build_similarity_graph(store, top_k=top_k, threshold=threshold)

    raw_groups: list[list[str]] = []
    for component in nx.connected_components(graph):
        members = sorted(component)
        if len(members) < 2:
            continue
        if len(members) <= max_group_size:
            raw_groups.append(members)
        else:
            raw_groups.extend(_split_oversized(graph, members, max_group_size))

    groups: list[CandidateGroup] = []
    for members in raw_groups:
        mean_sim, min_sim = group_stats(store, members)
        groups.append(
            CandidateGroup(chunk_ids=members, mean_similarity=mean_sim, min_similarity=min_sim)
        )

    groups.sort(key=lambda g: (-g.size, -g.mean_similarity))
    return groups


# ------------------------------------------------------------------------
# validate
# Validation gate: confirm a candidate group is genuinely redundant.
#
# Stage 3 of the pipeline, and the step that separates this package from naive
# threshold deduplication. Candidate discovery links chunks transitively through
# ANN neighbours, so a proposed group can contain pairs that were never directly
# compared. Summarizing such a group merges unrelated material and silently loses
# information that retrieval can never get back (short of an undo).
#
# So before any LLM call, the full pairwise similarity matrix of the group is
# computed and every pair must clear ``validation_threshold``. A group that fails
# is not simply thrown away: the largest subset in which *all* pairs clear the
# threshold is a maximal clique in the strict-similarity graph, and with groups
# capped to single digits that clique can be found exactly rather than
# approximated.
# ------------------------------------------------------------------------

def _stats(sims: np.ndarray, indices: list[int]) -> tuple[float, float]:
    if len(indices) < 2:
        return 1.0, 1.0
    pairs = [sims[i, j] for i, j in combinations(indices, 2)]
    return float(np.mean(pairs)), float(np.min(pairs))


def largest_valid_subset(sims: np.ndarray, threshold: float) -> list[int]:
    """Indices of the largest subset whose every pair scores >= ``threshold``."""
    n = sims.shape[0]
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, j in combinations(range(n), 2):
        if sims[i, j] >= threshold:
            graph.add_edge(i, j)

    best: list[int] = []
    for clique in nx.find_cliques(graph):
        if len(clique) > len(best):
            best = sorted(clique)
    return best if len(best) >= 2 else []


def validate_group(
    store: VectorStore,
    group: CandidateGroup,
    threshold: float = 0.90,
    refine: bool = True,
) -> ValidationOutcome:
    """Decide whether ``group`` may be summarized, shrinking it if allowed."""
    chunk_ids = list(group.chunk_ids)
    if len(chunk_ids) < 2:
        return ValidationOutcome(
            accepted=False,
            chunk_ids=chunk_ids,
            min_similarity=1.0,
            mean_similarity=1.0,
            reason="group has fewer than 2 chunks",
        )

    sims = pairwise_matrix(store, chunk_ids)
    all_indices = list(range(len(chunk_ids)))
    mean_sim, min_sim = _stats(sims, all_indices)

    if min_sim >= threshold:
        return ValidationOutcome(
            accepted=True,
            chunk_ids=chunk_ids,
            min_similarity=min_sim,
            mean_similarity=mean_sim,
            reason=f"all pairs >= {threshold:.2f}",
        )

    if not refine:
        return ValidationOutcome(
            accepted=False,
            chunk_ids=chunk_ids,
            min_similarity=min_sim,
            mean_similarity=mean_sim,
            reason=f"weakest pair {min_sim:.3f} < {threshold:.2f}",
        )

    keep = largest_valid_subset(sims, threshold)
    if not keep:
        return ValidationOutcome(
            accepted=False,
            chunk_ids=chunk_ids,
            min_similarity=min_sim,
            mean_similarity=mean_sim,
            reason=f"no subset of 2+ chunks clears {threshold:.2f}",
        )

    kept_ids = [chunk_ids[i] for i in keep]
    dropped_ids = [c for c in chunk_ids if c not in set(kept_ids)]
    kept_mean, kept_min = _stats(sims, keep)
    return ValidationOutcome(
        accepted=True,
        chunk_ids=kept_ids,
        min_similarity=kept_min,
        mean_similarity=kept_mean,
        reason=(
            f"refined from {len(chunk_ids)} to {len(kept_ids)} chunks; "
            f"dropped {len(dropped_ids)} below {threshold:.2f}"
        ),
        dropped_ids=dropped_ids,
    )


def validate_groups(
    store: VectorStore,
    groups: list[CandidateGroup],
    threshold: float = 0.90,
    refine: bool = True,
) -> list[ValidationOutcome]:
    return [validate_group(store, g, threshold=threshold, refine=refine) for g in groups]
