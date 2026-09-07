"""Vector store backends.

The pipeline only ever talks to the ``VectorStore`` protocol, so ChromaDB is a
swappable implementation detail rather than a hard dependency.
"""

from __future__ import annotations

import json
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from ragcompactor.embeddings import l2_normalize
from ragcompactor.models import Chunk


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def delete(self, ids: Iterable[str]) -> None: ...
    def get(self, chunk_id: str) -> Chunk | None: ...
    def get_vector(self, chunk_id: str) -> np.ndarray | None: ...
    def all_chunks(self) -> list[Chunk]: ...
    def all_ids(self) -> list[str]: ...
    def count(self) -> int: ...
    def query(
        self, vector: np.ndarray, k: int, exclude: Iterable[str] = ()
    ) -> list[tuple[str, float]]: ...


class InMemoryStore:
    """Exact-search store backed by a numpy matrix.

    Exhaustive rather than approximate, which is the right trade at the scale
    this package targets (tens of thousands of chunks) and makes test results
    deterministic.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors length mismatch")
        for chunk, vec in zip(chunks, vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vec.astype(np.float32)

    def delete(self, ids: Iterable[str]) -> None:
        for chunk_id in list(ids):
            self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_vector(self, chunk_id: str) -> np.ndarray | None:
        return self._vectors.get(chunk_id)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def all_ids(self) -> list[str]:
        return list(self._chunks.keys())

    def count(self) -> int:
        return len(self._chunks)

    def query(
        self, vector: np.ndarray, k: int, exclude: Iterable[str] = ()
    ) -> list[tuple[str, float]]:
        excluded = set(exclude)
        ids = [i for i in self._chunks if i not in excluded]
        if not ids:
            return []
        matrix = np.vstack([self._vectors[i] for i in ids])
        query_vec = l2_normalize(np.asarray(vector, dtype=np.float32))[0]
        scores = matrix @ query_vec
        order = np.argsort(-scores)[:k]
        return [(ids[int(i)], float(scores[int(i)])) for i in order]


class ChromaStore:
    """Persistent ChromaDB-backed store."""

    def __init__(self, path, collection: str = "memory") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The chroma store needs the 'chroma' extra: pip install 'ragcompactor[chroma]'"
            ) from exc
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    # Chroma stores metadata as flat scalars, so nested dicts are JSON-encoded.
    @staticmethod
    def _encode_meta(chunk: Chunk) -> dict:
        return {
            "source": chunk.source,
            "ordinal": chunk.ordinal,
            "extra": json.dumps(chunk.metadata or {}),
        }

    @staticmethod
    def _decode(chunk_id: str, text: str, meta: dict) -> Chunk:
        extra = {}
        raw = (meta or {}).get("extra")
        if raw:
            try:
                extra = json.loads(raw)
            except (TypeError, ValueError):
                extra = {}
        return Chunk(
            id=chunk_id,
            text=text,
            source=(meta or {}).get("source", ""),
            ordinal=int((meta or {}).get("ordinal", 0) or 0),
            metadata=extra,
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            metadatas=[self._encode_meta(c) for c in chunks],
        )

    def delete(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        if ids:
            self._collection.delete(ids=ids)

    def get(self, chunk_id: str) -> Chunk | None:
        res = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res or not res.get("ids"):
            return None
        return self._decode(res["ids"][0], res["documents"][0], res["metadatas"][0])

    def get_vector(self, chunk_id: str) -> np.ndarray | None:
        res = self._collection.get(ids=[chunk_id], include=["embeddings"])
        embeddings = (res or {}).get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return np.asarray(embeddings[0], dtype=np.float32)

    def all_chunks(self) -> list[Chunk]:
        res = self._collection.get(include=["documents", "metadatas"])
        return [
            self._decode(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def all_ids(self) -> list[str]:
        return list(self._collection.get()["ids"])

    def count(self) -> int:
        return int(self._collection.count())

    def query(
        self, vector: np.ndarray, k: int, exclude: Iterable[str] = ()
    ) -> list[tuple[str, float]]:
        excluded = set(exclude)
        query_vec = l2_normalize(np.asarray(vector, dtype=np.float32))[0]
        n = self.count()
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(k + len(excluded), n),
        )
        out: list[tuple[str, float]] = []
        for chunk_id, distance in zip(res["ids"][0], res["distances"][0]):
            if chunk_id in excluded:
                continue
            # chroma cosine distance -> similarity
            out.append((chunk_id, 1.0 - float(distance)))
            if len(out) >= k:
                break
        return out


def get_store(config) -> VectorStore:
    backend = (config.store_backend or "").lower()
    if backend in {"memory", "inmemory", "in-memory"}:
        return InMemoryStore()
    if backend == "chroma":
        return ChromaStore(config.store_path, collection=config.collection)
    raise ValueError(f"unknown store_backend: {config.store_backend!r}")
