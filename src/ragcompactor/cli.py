"""Command line interface."""

from __future__ import annotations

from pathlib import Path
import json

import typer

from ragcompactor.backends import get_summarizer
from ragcompactor.compactor import Compactor
from ragcompactor.core import CompactorConfig
from ragcompactor import benchmark as bench


app = typer.Typer(
    add_completion=False,
    help="Compact a RAG vector store by merging redundant chunks.",
)

DEFAULT_CONFIG = Path("ragcompactor.json")


@app.callback()
def _bootstrap() -> None:
    """Load .env before any command runs.

    Real environment variables win over the file (``override=False``), so an
    exported key still takes precedence over a stale one committed by mistake.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a core dep, but be safe
        return
    load_dotenv(override=False)


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
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-ingest everything, ignoring the ledger. Use after changing "
        "chunk_size or the embedding model.",
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Chunk, embed and store documents, skipping sources already ingested."""
    with _compactor(config) as c:
        report = c.ingest(list(paths), force=force)
        if json_out:
            typer.echo(json.dumps(report.to_dict(), indent=2))
            return
        d = report.to_dict()
        typer.echo(f"scanned {d['files_seen']} file(s)")
        typer.echo(f"  new        {d['added_files']}")
        typer.echo(f"  changed    {d['updated_files']}")
        typer.echo(f"  unchanged  {d['skipped_files']} (skipped)")
        typer.echo(f"  chunks     +{d['chunks_added']} / -{d['chunks_removed']}")
        typer.echo(f"  store now holds {c.store.count()} chunks")
        if d["stale_merges"]:
            typer.echo(
                f"\n  WARNING: {len(d['stale_merges'])} existing merge(s) contain text from "
                "files that have since changed."
            )
            typer.echo("  Those summaries still carry the superseded wording. Undo them with:")
            for merge_id in d["stale_merges"]:
                typer.echo(f"    ragcompactor undo --merge {merge_id}")


@app.command()
def sources(config: Path = ConfigOpt) -> None:
    """List the folders and files recorded as ingested."""
    with _compactor(config) as c:
        summary = c.ledger.summary()
        typer.echo(f"ledger: {summary['path']}")
        typer.echo(f"  {summary['files']} file(s), {summary['chunks']} chunk(s)")
        if summary["roots"]:
            typer.echo("\nfolders ingested:")
            for root in summary["roots"]:
                typer.echo(f"  {root}")
        if summary["files"]:
            typer.echo("\nfiles:")
            for path in c.ledger.known_files():
                record = c.ledger.files[path]
                flag = "" if Path(path).exists() else "   [missing on disk]"
                typer.echo(f"  {record['chunk_count']:>4} chunks  {path}{flag}")
        if summary["missing_files"]:
            typer.echo(
                f"\n{len(summary['missing_files'])} recorded source(s) no longer exist. "
                "Drop one with: ragcompactor forget <path>"
            )


@app.command()
def forget(
    path: Path = typer.Argument(..., help="Source file to drop from the store."),
    config: Path = ConfigOpt,
) -> None:
    """Remove a source's chunks from the store and forget it in the ledger."""
    with _compactor(config) as c:
        removed = c.forget_source(path)
        typer.echo(f"removed {removed} chunk(s); store now holds {c.store.count()}")


@app.command()
def compact(
    config: Path = ConfigOpt,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would merge without calling the LLM or writing."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    throttle: float = typer.Option(
        None,
        "--throttle",
        help="Seconds to wait between LLM calls, to stay under a provider's "
        "tokens-per-minute cap. Groq's free tier needs about 8.",
    ),
) -> None:
    """Find, validate and merge redundant chunks."""
    with _compactor(config) as c:
        if throttle is not None:
            c.config.llm_request_delay = throttle
            c.summarizer = get_summarizer(c.config)
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
        if d["failed_groups"]:
            typer.echo(f"  FAILED groups      {d['failed_groups']}")
            typer.echo(f"    last error: {d['last_error']}")
            typer.echo("    completed merges are committed - re-run to continue.")
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
    throttle: float = typer.Option(
        None,
        "--throttle",
        help="Seconds between LLM calls during the compaction step, to stay under "
        "a provider's tokens-per-minute cap. Groq's free tier needs about 8.",
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Measure tokens before and after compaction, and the break-even point.

    Compacts the store in place, so run it on a copy if you want the
    pre-compaction state preserved.
    """
    with _compactor(config) as c:
        cfg = c.config
        if throttle is not None:
            cfg.llm_request_delay = throttle
            c.summarizer = get_summarizer(cfg)
        if ingest_paths:
            ingested = c.ingest(list(ingest_paths))
            typer.echo(
                f"ingested {ingested.chunks_added} chunks "
                f"({ingested.skipped_files} unchanged file(s) skipped)\n"
            )

        queries: list[str]
        if queries_file:
            queries = [q.strip() for q in queries_file.read_text(encoding="utf-8").splitlines() if q.strip()]
        else:
            # Derive stand-in queries from the corpus itself so the benchmark
            # runs without hand-written questions. Real queries are better.
            #
            # Sample at an even stride across the whole store rather than taking
            # the first N chunks: store order groups chunks by document, so the
            # first N would all come from one file and would measure only that
            # corner of the corpus.
            all_chunks = [ch for ch in c.store.all_chunks() if ch.text.strip()]
            if 0 < sample_queries < len(all_chunks):
                step = len(all_chunks) / sample_queries
                picked = [all_chunks[int(i * step)] for i in range(sample_queries)]
            else:
                picked = all_chunks
            queries = [" ".join(ch.text.split()[:12]) for ch in picked]

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
    from ragcompactor.core import Chunk

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
