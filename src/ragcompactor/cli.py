"""Command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ragcompactor import benchmark as bench
from ragcompactor.compactor import Compactor
from ragcompactor.config import CompactorConfig

app = typer.Typer(
    add_completion=False,
    help="Compact a RAG vector store by merging redundant chunks.",
)

DEFAULT_CONFIG = Path("ragcompactor.json")


def _load_config(config_path: Path | None) -> CompactorConfig:
    return CompactorConfig.load(config_path or DEFAULT_CONFIG)


def _compactor(config_path: Path | None) -> Compactor:
    return Compactor(_load_config(config_path))


ConfigOpt = typer.Option(
    None, "--config", "-c", help="Path to config JSON (default: ./ragcompactor.json)."
)


@app.command()
def init(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    store: str = typer.Option("chroma", help="Vector store backend: chroma | memory."),
    embedder: str = typer.Option(
        "sentence-transformers", help="Embedder: sentence-transformers | hashing."
    ),
    llm: str = typer.Option("litellm", help="Summarizer backend: litellm | stub."),
    model: str = typer.Option("gpt-4o-mini", help="LiteLLM model string."),
) -> None:
    """Write a starter config file."""
    cfg = CompactorConfig(
        store_backend=store, embedding_backend=embedder, llm_backend=llm, llm_model=model
    )
    cfg.save(config)
    typer.echo(f"wrote {config}")
    typer.echo(f"  store      {cfg.store_backend} -> {cfg.store_path}")
    typer.echo(f"  embedder   {cfg.embedding_backend} ({cfg.embedding_model})")
    typer.echo(f"  summarizer {cfg.llm_backend} ({cfg.llm_model})")


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="Files or directories to ingest."),
    config: Path = ConfigOpt,
) -> None:
    """Chunk, embed and store documents."""
    with _compactor(config) as c:
        added = c.ingest(list(paths))
        typer.echo(f"ingested {added} chunks; store now holds {c.store.count()}")


@app.command()
def compact(
    config: Path = ConfigOpt,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would merge without calling the LLM or writing."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Find, validate and merge redundant chunks."""
    with _compactor(config) as c:
        report = c.compact(dry_run=dry_run)
        if json_out:
            typer.echo(json.dumps(report.to_dict(), indent=2))
            return
        d = report.to_dict()
        typer.echo(f"run {report.run_id}{' (dry run)' if dry_run else ''}")
        typer.echo(f"  candidate groups   {d['candidate_groups']}")
        typer.echo(f"  validated          {d['validated_groups']} (refined {d['refined_groups']})")
        typer.echo(f"  rejected           {d['rejected_groups']}")
        typer.echo(f"  chunks             {d['chunks_before']} -> {d['chunks_after']} "
                   f"({d['reduction_pct']}% smaller)")
        typer.echo(f"  archived           {d['chunks_archived']}")
        typer.echo(f"  summarizer tokens  {d['summarization_tokens']}")
        if not dry_run and report.merges:
            typer.echo(f"  undo with: ragcompactor undo --run {report.run_id}")


@app.command()
def undo(
    config: Path = ConfigOpt,
    merge: str = typer.Option(None, "--merge", help="Undo a single merge by id."),
    run: str = typer.Option(None, "--run", help="Undo every merge in a run."),
    last: bool = typer.Option(False, "--last", help="Undo the most recent run."),
) -> None:
    """Restore archived originals and drop the merged chunk."""
    with _compactor(config) as c:
        if merge:
            ok = c.undo_merge(merge)
            typer.echo(f"merge {merge}: {'restored' if ok else 'not found or already undone'}")
            return
        run_id = run or (c.last_run_id() if last else None)
        if not run_id:
            typer.echo("specify --merge, --run or --last")
            raise typer.Exit(code=1)
        restored = c.undo_run(run_id)
        typer.echo(f"run {run_id}: restored {restored} merge(s); store now holds {c.store.count()}")


@app.command()
def runs(config: Path = ConfigOpt) -> None:
    """List compaction runs."""
    with _compactor(config) as c:
        rows = c.archive.list_runs()
        if not rows:
            typer.echo("no runs recorded")
            return
        for r in rows:
            typer.echo(
                f"{r['run_id']}  {r['created_at']}  merges={r['merges']}  undone={r['undone']}"
            )


@app.command()
def stats(config: Path = ConfigOpt) -> None:
    """Show store and archive counts."""
    with _compactor(config) as c:
        typer.echo(json.dumps(c.stats(), indent=2))


@app.command()
def benchmark(
    config: Path = ConfigOpt,
    ingest_paths: list[Path] = typer.Option(
        None,
        "--ingest",
        help="Ingest these files/dirs first. Required with the in-memory store, "
        "which does not persist between commands.",
    ),
    queries_file: Path = typer.Option(
        None, "--queries", help="Text file with one retrieval query per line."
    ),
    top_k: int = typer.Option(5, "--top-k", help="Chunks retrieved per query."),
    sample_queries: int = typer.Option(
        12, help="If no queries file is given, derive this many queries from the corpus."
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Measure tokens before and after compaction, and the break-even point.

    Compacts the store in place, so run it on a copy if you want the
    pre-compaction state preserved.
    """
    with _compactor(config) as c:
        cfg = c.config
        if ingest_paths:
            added = c.ingest(list(ingest_paths))
            typer.echo(f"ingested {added} chunks\n")

        queries: list[str]
        if queries_file:
            queries = [q.strip() for q in queries_file.read_text(encoding="utf-8").splitlines() if q.strip()]
        else:
            # Derive stand-in queries from the corpus itself so the benchmark
            # runs without hand-written questions. Real queries are better.
            texts = [ch.text for ch in c.store.all_chunks()][:sample_queries]
            queries = [" ".join(t.split()[:12]) for t in texts if t.strip()]

        if not queries:
            typer.echo("no queries available - ingest documents first")
            raise typer.Exit(code=1)

        before_corpus = bench.corpus_stats(c.store, cfg.token_model)
        before_retrieval = bench.retrieval_stats(
            c.store, c.embedder, queries, top_k, cfg.token_model
        )

        report = c.compact(dry_run=False)

        after_corpus = bench.corpus_stats(c.store, cfg.token_model)
        after_retrieval = bench.retrieval_stats(
            c.store, c.embedder, queries, top_k, cfg.token_model
        )

        result = bench.build_report(
            before_corpus,
            after_corpus,
            before_retrieval,
            after_retrieval,
            compaction_tokens=report.summarization_tokens,
            token_model=cfg.token_model,
        )
        if json_out:
            payload = result.to_dict()
            payload["compaction"] = report.to_dict()
            typer.echo(json.dumps(payload, indent=2))
        else:
            typer.echo(result.render())
            typer.echo(f"\n  (compaction run {report.run_id}; undo with "
                       f"ragcompactor undo --run {report.run_id})")


@app.command()
def demo(
    workdir: Path = typer.Option(Path(".ragcompactor-demo"), help="Where to build the demo."),
) -> None:
    """Build a small synthetic corpus with known duplicates and compact it.

    Runs fully offline - hashing embedder, in-memory store, stub summarizer - so
    it works with no API key and no model download.
    """
    from ragcompactor.models import Chunk

    base = [
        "The compactor merges redundant chunks in a vector store to reduce token usage.",
        "Retrieval augmented generation pulls the top k chunks into the prompt context.",
        "Archived originals make every merge reversible through the undo command.",
    ]
    paraphrases = [
        "The compactor merges redundant chunks in a vector store so as to reduce token usage.",
        "The compactor merges redundant chunks within a vector store to reduce token usage overall.",
        "Retrieval augmented generation pulls the top k chunks into the prompt context window.",
        "Archived originals make each merge reversible via the undo command.",
    ]
    distinct = [
        "Cosine similarity between unit vectors reduces to a simple dot product.",
        "SQLite stores the archive because it needs no server and ships with Python.",
    ]

    cfg = CompactorConfig(
        workdir=workdir,
        store_backend="memory",
        embedding_backend="hashing",
        llm_backend="stub",
        similarity_threshold=0.80,
        validation_threshold=0.85,
    )
    with Compactor(cfg) as c:
        texts = base + paraphrases + distinct
        c.add_chunks([Chunk.create(t, source="demo", ordinal=i) for i, t in enumerate(texts)])
        typer.echo(f"seeded {c.store.count()} chunks ({len(distinct)} of them genuinely distinct)")

        report = c.compact()
        d = report.to_dict()
        typer.echo(
            f"compacted: {d['chunks_before']} -> {d['chunks_after']} chunks, "
            f"{d['validated_groups']} group(s) merged, {d['rejected_groups']} rejected"
        )

        restored = c.undo_run(report.run_id)
        typer.echo(f"undo restored {restored} merge(s); store back to {c.store.count()} chunks")


if __name__ == "__main__":  # pragma: no cover
    app()
