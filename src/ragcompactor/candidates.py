"""Candidate discovery: ANN top-k search followed by similarity mapping.

Stage 2 of the pipeline. Every chunk queries the store for its ``top_k``
nearest neighbours; pairs scoring above ``similarity_threshold`` become edges in
a graph, and each connected component of that graph is a candidate group.

Grouping by connected component is intentionally generous - it is a recall
filter. Because similarity is not transitive, a component can chain together
chunks that are not all alike (A~B, B~C, but A!~C), which is exactly why
nothing here is merged until it clears the stricter gate in ``validate``.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np

from ragcompactor.models import CandidateGroup
from ragcompactor.store import VectorStore


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
