"""Validation gate: confirm a candidate group is genuinely redundant.

Stage 3 of the pipeline, and the step that separates this package from naive
threshold deduplication. Candidate discovery links chunks transitively through
ANN neighbours, so a proposed group can contain pairs that were never directly
compared. Summarizing such a group merges unrelated material and silently loses
information that retrieval can never get back (short of an undo).

So before any LLM call, the full pairwise similarity matrix of the group is
computed and every pair must clear ``validation_threshold``. A group that fails
is not simply thrown away: the largest subset in which *all* pairs clear the
threshold is a maximal clique in the strict-similarity graph, and with groups
capped to single digits that clique can be found exactly rather than
approximated.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np

from ragcompactor.candidates import pairwise_matrix
from ragcompactor.models import CandidateGroup, ValidationOutcome
from ragcompactor.store import VectorStore


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
