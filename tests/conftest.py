import math

import numpy as np
import pytest

from ragcompactor.config import CompactorConfig
from ragcompactor.models import Chunk
from ragcompactor.store import InMemoryStore


@pytest.fixture
def offline_config(tmp_path) -> CompactorConfig:
    """Config that needs no network, no API key and no model download."""
    return CompactorConfig(
        workdir=tmp_path / "wd",
        store_backend="memory",
        embedding_backend="hashing",
        llm_backend="stub",
        similarity_threshold=0.80,
        validation_threshold=0.85,
    )


def unit(angle_deg: float) -> np.ndarray:
    r = math.radians(angle_deg)
    return np.array([math.cos(r), math.sin(r)], dtype=np.float32)


@pytest.fixture
def chain_store() -> InMemoryStore:
    """Three chunks forming a similarity chain: A~B and B~C, but A!~C.

    cos(18.19 deg) ~= 0.95, cos(36.38 deg) ~= 0.805. This is the transitivity
    trap that candidate discovery falls into and validation must catch.
    """
    store = InMemoryStore()
    chunks = [
        Chunk(id="a", text="alpha text about vector stores", source="t"),
        Chunk(id="b", text="beta text about vector stores", source="t"),
        Chunk(id="c", text="gamma text about vector stores", source="t"),
    ]
    vectors = np.vstack([unit(0.0), unit(18.19), unit(36.38)])
    store.add(chunks, vectors)
    return store


DUPLICATE_TEXTS = [
    "The compactor merges redundant chunks in a vector store to reduce token usage.",
    "The compactor merges redundant chunks in a vector store so as to reduce token usage.",
]

DISTINCT_TEXTS = [
    "Cosine similarity between unit vectors reduces to a simple dot product.",
    "SQLite stores the archive because it needs no server and ships with Python.",
]
