"""Embedding backends.

Every embedder returns L2-normalized row vectors, which makes cosine similarity
a plain dot product everywhere downstream.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_WORD = re.compile(r"[a-z0-9']+")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into normalized vectors."""

    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dimension) float32 array of unit vectors."""
        ...


class HashingEmbedder:
    """Dependency-free embedder using the hashing trick over word bigrams.

    Not competitive with a trained model, but deterministic, instant and
    offline - which makes it the right default for tests, CI and demos where
    downloading a sentence-transformers checkpoint is not worth it.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    @staticmethod
    def _features(text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return l2_normalize(np.vstack([self._vector(t) for t in texts]))


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model (default all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The sentence-transformers embedder needs the 'embed' extra: "
                "pip install 'ragcompactor[embed]'"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return l2_normalize(vectors)


def get_embedder(config) -> Embedder:
    """Build the embedder named by ``config.embedding_backend``."""
    backend = (config.embedding_backend or "").lower()
    if backend in {"hashing", "hash", "fake"}:
        return HashingEmbedder(dimension=config.embedding_dim)
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(model_name=config.embedding_model)
    raise ValueError(f"unknown embedding_backend: {config.embedding_backend!r}")
