"""Token accounting for the before/after comparison.

The honest comparison is *not* "raw files stuffed into a prompt" versus this
package - any RAG setup beats that, and it would credit compaction for savings
that plain retrieval already provides. What is measured here is the same
pipeline with and without compaction: identical corpus, identical queries,
identical top-k, differing only in whether redundant chunks were merged.

Two numbers matter, and the second is the one that survives scrutiny:

* context tokens per query - how much text top-k retrieval drags into the
  prompt, before and after;
* break-even - compaction spends tokens on summarization, so the win only
  starts after enough queries have amortized that one-off cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ragcompactor.embeddings import Embedder
from ragcompactor.store import VectorStore
from ragcompactor.tokens import count_tokens, tokenizer_is_exact


@dataclass
class CorpusStats:
    chunks: int
    total_tokens: int

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.chunks if self.chunks else 0.0


@dataclass
class RetrievalStats:
    queries: int
    top_k: int
    total_context_tokens: int
    unique_chunks_retrieved: int

    @property
    def avg_context_tokens(self) -> float:
        return self.total_context_tokens / self.queries if self.queries else 0.0


@dataclass
class BenchmarkReport:
    before_corpus: CorpusStats
    after_corpus: CorpusStats
    before_retrieval: RetrievalStats
    after_retrieval: RetrievalStats
    compaction_tokens: int
    exact_tokenizer: bool = True
    notes: list[str] = field(default_factory=list)

    # --- storage side ------------------------------------------------------
    @property
    def chunks_removed(self) -> int:
        return self.before_corpus.chunks - self.after_corpus.chunks

    @property
    def corpus_token_reduction(self) -> int:
        return self.before_corpus.total_tokens - self.after_corpus.total_tokens

    @property
    def corpus_reduction_pct(self) -> float:
        before = self.before_corpus.total_tokens
        return 100.0 * self.corpus_token_reduction / before if before else 0.0

    # --- query side --------------------------------------------------------
    @property
    def tokens_saved_per_query(self) -> float:
        return (
            self.before_retrieval.avg_context_tokens
            - self.after_retrieval.avg_context_tokens
        )

    @property
    def context_reduction_pct(self) -> float:
        before = self.before_retrieval.avg_context_tokens
        return 100.0 * self.tokens_saved_per_query / before if before else 0.0

    @property
    def break_even_queries(self) -> float | None:
        """Queries needed before summarization cost is repaid. None if never."""
        saved = self.tokens_saved_per_query
        if saved <= 0:
            return None
        return self.compaction_tokens / saved

    def to_dict(self) -> dict:
        return {
            "before": {
                "corpus": asdict(self.before_corpus),
                "avg_context_tokens_per_query": round(
                    self.before_retrieval.avg_context_tokens, 1
                ),
            },
            "after": {
                "corpus": asdict(self.after_corpus),
                "avg_context_tokens_per_query": round(
                    self.after_retrieval.avg_context_tokens, 1
                ),
            },
            "chunks_removed": self.chunks_removed,
            "corpus_token_reduction": self.corpus_token_reduction,
            "corpus_reduction_pct": round(self.corpus_reduction_pct, 2),
            "tokens_saved_per_query": round(self.tokens_saved_per_query, 1),
            "context_reduction_pct": round(self.context_reduction_pct, 2),
            "compaction_tokens": self.compaction_tokens,
            "break_even_queries": (
                round(self.break_even_queries, 1)
                if self.break_even_queries is not None
                else None
            ),
            "exact_tokenizer": self.exact_tokenizer,
            "notes": self.notes,
        }

    def render(self) -> str:
        d = self.to_dict()
        be = d["break_even_queries"]
        lines = [
            "Compaction benchmark",
            "====================",
            f"  corpus chunks      {self.before_corpus.chunks} -> {self.after_corpus.chunks} "
            f"({self.chunks_removed} removed)",
            f"  corpus tokens      {self.before_corpus.total_tokens} -> "
            f"{self.after_corpus.total_tokens} ({d['corpus_reduction_pct']}% smaller)",
            f"  context per query  {d['before']['avg_context_tokens_per_query']} -> "
            f"{d['after']['avg_context_tokens_per_query']} tokens "
            f"({d['context_reduction_pct']}% smaller)",
            f"  saved per query    {d['tokens_saved_per_query']} tokens",
            f"  compaction cost    {self.compaction_tokens} tokens (one-off)",
            f"  break-even         {be if be is not None else 'never - compaction did not reduce context'}"
            + (" queries" if be is not None else ""),
        ]
        if not self.exact_tokenizer:
            lines.append(
                "  NOTE: exact tokenizer unavailable (tiktoken missing, or its vocabulary "
                "could not be downloaded) - counts are ~chars/4 estimates."
            )
        lines.extend(f"  NOTE: {n}" for n in self.notes)
        return "\n".join(lines)


def corpus_stats(store: VectorStore, token_model: str = "gpt-4o-mini") -> CorpusStats:
    chunks = store.all_chunks()
    return CorpusStats(
        chunks=len(chunks),
        total_tokens=sum(count_tokens(c.text, token_model) for c in chunks),
    )


def retrieval_stats(
    store: VectorStore,
    embedder: Embedder,
    queries: list[str],
    top_k: int = 5,
    token_model: str = "gpt-4o-mini",
) -> RetrievalStats:
    """Measure the context a top-k retrieval would inject, per query."""
    if not queries:
        return RetrievalStats(0, top_k, 0, 0)

    vectors = embedder.embed(queries)
    total = 0
    unique: set[str] = set()
    for vector in vectors:
        hits = store.query(vector, k=top_k)
        for chunk_id, _score in hits:
            chunk = store.get(chunk_id)
            if chunk is None:
                continue
            unique.add(chunk_id)
            total += count_tokens(chunk.text, token_model)
    return RetrievalStats(
        queries=len(queries),
        top_k=top_k,
        total_context_tokens=total,
        unique_chunks_retrieved=len(unique),
    )


def build_report(
    before_corpus: CorpusStats,
    after_corpus: CorpusStats,
    before_retrieval: RetrievalStats,
    after_retrieval: RetrievalStats,
    compaction_tokens: int,
    token_model: str = "gpt-4o-mini",
) -> BenchmarkReport:
    report = BenchmarkReport(
        before_corpus=before_corpus,
        after_corpus=after_corpus,
        before_retrieval=before_retrieval,
        after_retrieval=after_retrieval,
        compaction_tokens=compaction_tokens,
        exact_tokenizer=tokenizer_is_exact(token_model),
    )
    if compaction_tokens == 0:
        report.notes.append(
            "compaction cost was 0 - the stub summarizer makes no API call, so "
            "break-even is not meaningful for this run"
        )
    return report
