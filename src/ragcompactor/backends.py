"""Swappable backends: embedding, vector storage and summarization.

Everything in this module sits behind a Protocol, and every heavy dependency
(sentence-transformers, chromadb, litellm) is imported lazily inside the class
that needs it. That is what keeps them optional extras and makes the pipeline
indifferent to which provider is configured.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable
import hashlib
import json
import os
import re
import time

import numpy as np

from ragcompactor.core import Chunk, SummaryResult, count_tokens


# ------------------------------------------------------------------------
# embeddings
# Embedding backends.
#
# Every embedder returns L2-normalized row vectors, which makes cosine similarity
# a plain dot product everywhere downstream.
# ------------------------------------------------------------------------

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


# ------------------------------------------------------------------------
# store
# Vector store backends.
#
# The pipeline only ever talks to the ``VectorStore`` protocol, so ChromaDB is a
# swappable implementation detail rather than a hard dependency.
# ------------------------------------------------------------------------

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


# ------------------------------------------------------------------------
# summarize
# Summarization backends.
#
# The compactor depends only on the ``Summarizer`` protocol, and LiteLLM does the
# provider routing underneath, so the same code path serves a hosted API
# (``gpt-4o-mini``, ``claude-3-5-haiku-20241022``) and a local model
# (``ollama/llama3.2:3b``) - the model string is the only thing that changes.
# ------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You compact a retrieval corpus. You are given several chunks that have been "
    "measured as near-duplicates of each other. Rewrite them as ONE self-contained "
    "chunk that preserves every distinct fact, number, name and qualifier found in "
    "any of them. Do not add information that is not present. Do not editorialize, "
    "and do not refer to the chunks or to the merging process. Return only the "
    "merged text."
)


# LiteLLM routes on the model string; each provider reads its own key. Checking
# up front turns "18 stack traces" into one sentence naming the variable to set.
_PROVIDER_KEYS: list[tuple[str, str | None]] = [
    ("ollama/", None),          # local, needs no key
    ("groq/", "GROQ_API_KEY"),
    ("anthropic/", "ANTHROPIC_API_KEY"),
    ("claude-", "ANTHROPIC_API_KEY"),
    ("together_ai/", "TOGETHER_API_KEY"),
    ("mistral/", "MISTRAL_API_KEY"),
    ("openai/", "OPENAI_API_KEY"),
    ("gpt-", "OPENAI_API_KEY"),
]


def missing_api_key(model: str) -> str | None:
    """Name of the env var this model needs, if it is not set. None if fine."""
    for prefix, env_var in _PROVIDER_KEYS:
        if model.startswith(prefix):
            if env_var is None or os.environ.get(env_var):
                return None
            return env_var
    return None  # unknown provider - let LiteLLM decide


def build_prompt(texts: list[str]) -> str:
    parts = [f"--- chunk {i + 1} ---\n{t.strip()}" for i, t in enumerate(texts)]
    return (
        "Merge the following near-duplicate chunks into a single chunk.\n\n"
        + "\n\n".join(parts)
    )


@runtime_checkable
class Summarizer(Protocol):
    model: str

    def summarize(self, texts: list[str]) -> SummaryResult: ...


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z0-9']+", a.lower()))
    wb = set(re.findall(r"[a-z0-9']+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class StubSummarizer:
    """Deterministic, offline, zero-cost extractive summarizer.

    Takes the longest chunk as the spine and adds a sentence from the other
    chunks only when it is not a near-repeat of something already kept
    (word-level Jaccard below ``sentence_threshold``). That makes it genuinely
    compressive rather than a concatenation, so offline demos and tests show the
    same *shape* of result a real model would - just cruder. It is not a
    substitute for an LLM: it can only select sentences, never rewrite them.
    """

    model = "stub"

    def __init__(self, sentence_threshold: float = 0.6) -> None:
        self.sentence_threshold = sentence_threshold

    def summarize(self, texts: list[str]) -> SummaryResult:
        if not texts:
            return SummaryResult(text="", model=self.model)

        ordered = sorted(texts, key=len, reverse=True)
        kept: list[str] = list(_sentences(ordered[0]))

        for other in ordered[1:]:
            for sentence in _sentences(other):
                if len(sentence) < 15:
                    continue
                if any(_jaccard(sentence, k) >= self.sentence_threshold for k in kept):
                    continue
                kept.append(sentence)

        return SummaryResult(text=" ".join(kept).strip(), model=self.model)


class LiteLLMSummarizer:
    """Routes the summarization call through LiteLLM to any supported provider."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 60.0,
        num_retries: int = 5,
        request_delay: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.num_retries = num_retries
        self.request_delay = request_delay
        self._last_call: float | None = None

    def _throttle(self) -> None:
        """Space calls out, so a per-minute token cap is respected rather than hit.

        Retries recover from a rate limit after the fact; throttling avoids
        provoking one. On a small free tier the second is much faster overall,
        because a refused request still costs a round trip and a backoff wait.
        """
        if self.request_delay <= 0 or self._last_call is None:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

    def summarize(self, texts: list[str]) -> SummaryResult:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The litellm summarizer needs the 'llm' extra: pip install 'ragcompactor[llm]'"
            ) from exc

        missing = missing_api_key(self.model)
        if missing:
            raise RuntimeError(
                f"{missing} is not set, so '{self.model}' cannot be reached. "
                f"Put it in a .env file next to ragcompactor.json as "
                f"{missing}=your_key, or set it in this shell."
            )

        prompt = build_prompt(texts)
        self._throttle()
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                num_retries=self.num_retries,
            )
        finally:
            self._last_call = time.monotonic()
        text = (response.choices[0].message.content or "").strip()

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if not prompt_tokens:  # provider did not report usage - estimate locally
            prompt_tokens = count_tokens(SYSTEM_PROMPT + prompt)
        if not completion_tokens:
            completion_tokens = count_tokens(text)

        return SummaryResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
        )


def get_summarizer(config) -> Summarizer:
    backend = (config.llm_backend or "").lower()
    if backend in {"stub", "none", "offline"}:
        return StubSummarizer()
    if backend in {"litellm", "api", "llm"}:
        return LiteLLMSummarizer(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.max_summary_tokens,
            timeout=config.llm_timeout,
            num_retries=config.llm_num_retries,
            request_delay=config.llm_request_delay,
        )
    raise ValueError(f"unknown llm_backend: {config.llm_backend!r}")
