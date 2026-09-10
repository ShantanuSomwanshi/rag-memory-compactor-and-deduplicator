# ragcompactor

Local RAG memory compactor and deduplicator. It finds redundant chunks in a
vector store, merges each redundant group into one summarized chunk, and
archives the originals so any merge can be undone.

Vector stores bloat: the same material gets ingested twice, notes restate each
other, a PDF overlaps a summary of itself. Retrieval then spends its top-k slots
on near-copies, which costs tokens on every single query and crowds out
material that would actually have helped.

Self-hostable, LLM-agnostic (LiteLLM routes the summarization call to whichever
provider you configure), and reversible by design.

## Pipeline

```
ingest -> embed -> ANN top-k + similarity mapping -> VALIDATION GATE
       -> LLM summarize -> replace in store -> archive originals -> undo
```

1. **Ingest & embed** - files are chunked on paragraph boundaries and embedded
   (`all-MiniLM-L6-v2` by default), then stored in ChromaDB.
2. **Candidate discovery** - each chunk queries its `top_k` nearest neighbours;
   pairs above `similarity_threshold` become graph edges, and each connected
   component is a candidate group.
3. **Validation gate** - the full pairwise similarity matrix of the group is
   computed and *every* pair must clear the stricter `validation_threshold`.
4. **Summarize & replace** - only validated groups reach the LLM. The merged
   chunk replaces the originals.
5. **Archive** - originals (text, metadata and embedding vector) go to SQLite.
6. **Undo** - restores the exact originals and drops the merged chunk.

### Why the validation gate exists

Candidate discovery links chunks *transitively*. If A resembles B and B
resembles C, all three land in one component even when A and C were never
directly compared and are not alike. Merging that group summarizes unrelated
material together and quietly loses information.

So discovery and validation use two separate thresholds on purpose:
`similarity_threshold` (default 0.85) is a cheap recall filter deciding what is
worth *examining*; `validation_threshold` (default 0.90) is a strict precision
filter deciding what may actually be *merged*. Config enforces that validation
is never looser than discovery.

A group that fails validation is not discarded. The largest subset whose every
pair clears the threshold is a maximal clique in the strict-similarity graph,
and since groups are capped at `max_group_size` that clique is computed exactly.
The weak members are dropped and the tight core still merges.

## Install

```bash
pip install -r requirements.txt      # or: pip install -e ".[all]"
pip install -e .                     # puts the ragcompactor command on PATH
```

For development: `pip install -r requirements-dev.txt`. Extras are granular if
you want a smaller install: `chroma`, `embed`, `llm`, `tokens`, `pdf`.

## API keys

Copy `.env.example` to `.env` and fill in the key for whichever provider your
`llm_model` uses. Every command loads `.env` automatically, and `.env` is
gitignored so keys are never committed:

```bash
cp .env.example .env      # then edit it
```

A real environment variable still wins over the file. If the key a model needs
is missing, the run stops immediately with a message naming the variable,
rather than failing once per merge.

## Quick start

```bash
ragcompactor demo                          # offline end-to-end: seed, compact, undo

ragcompactor init                          # writes ragcompactor.json
ragcompactor ingest ./notes ./papers
ragcompactor sources                       # what has been ingested already
ragcompactor compact --dry-run             # what would merge, no LLM call, no writes
ragcompactor compact
ragcompactor runs
ragcompactor undo --last
```

`demo` runs fully offline - hashing embedder, in-memory store, stub summarizer -
so it needs no API key and no model download. It also deliberately shows the
validation gate refining an over-broad group.

## Incremental ingestion

Put your corpus in `data/` (or anywhere else — `ingest` takes any path and walks
directories recursively). These extensions are picked up; anything else in the
folder is skipped rather than causing an error:

| Extension | Notes |
| --- | --- |
| `.txt` `.md` `.markdown` `.rst` | Read directly |
| `.pdf` | Needs the `pdf` extra |
| `.csv` `.json` `.log` `.py` | Read as plain text |

Note that `.md` is on that list, so keep notes-about-the-corpus out of the
corpus folder — they would be ingested along with everything else.

`ingest` keeps a ledger at `<workdir>/ingested.json` recording every folder and
file it has taken in, with each file's content hash and the chunk ids it
produced. Re-running `ingest` on the same folder skips unchanged files entirely.

```bash
ragcompactor ingest ./data      # first run: 40 new files
ragcompactor ingest ./data      # second run: 40 unchanged, skipped
ragcompactor sources            # list ingested folders and files
ragcompactor ingest ./data --force   # ignore the ledger and redo everything
ragcompactor forget ./data/old.md    # drop one source and its chunks
```

Chunk ids are content-addressed, so re-adding an identical chunk is already
harmless. The ledger is there for the three things that does *not* cover:

- **Cost.** Skipping an unchanged file avoids re-reading it, re-parsing a PDF,
  re-chunking and — by far the most expensive part — re-embedding it.
- **Edited files.** When a file changes, its old chunks keep their old ids and
  would sit in the store forever with no source behind them. The ledger knows
  which ids came from which file, so they are removed before the new ones land.
- **Accidentally undoing a compaction.** After a merge the originals are
  archived and deleted from the store. Re-ingesting that source file would
  regenerate exactly those ids and silently put the duplicates back. Skipping
  already-ingested files prevents it.

Files are matched by content hash rather than timestamp, so a touched or
restored file is correctly seen as unchanged, and an edit made within the same
second is still caught.

Use `--force` after changing `chunk_size`, `chunk_overlap` or the embedding
model — those invalidate stored chunks without changing any source file.

### Editing a file whose chunks were already merged

If a changed file had chunks folded into a summary, that summary still carries
the superseded wording. `ingest` detects this and names the affected merges:

```
WARNING: 1 existing merge(s) contain text from files that have since changed.
Those summaries still carry the superseded wording. Undo them with:
  ragcompactor undo --merge 4f2a91c0b7de
```

Undoing restores the originals, after which the next `compact` re-merges them
with the updated text.

## New material is compacted against old

`compact` always runs over the whole store, so chunks ingested today are
compared against everything already in it — a duplicate that spans two separate
ingestion runs, months apart, is found and merged like any other. Compaction is
not scoped to the batch you just added.

## Configuration

`ragcompactor.json`, or `RAGCOMPACTOR_*` environment variables (which win).

| Key | Default | Meaning |
| --- | --- | --- |
| `store_backend` | `memory` | `chroma` (persistent) or `memory` (in-process) |
| `embedding_backend` | `sentence-transformers` | or `hashing` (offline, no download) |
| `chunk_size` / `chunk_overlap` | 900 / 120 | characters |
| `top_k` | 10 | neighbours examined per chunk |
| `similarity_threshold` | 0.85 | candidate discovery (recall) |
| `validation_threshold` | 0.90 | merge gate (precision) |
| `max_group_size` | 8 | cap on chunks merged into one summary |
| `refine_rejected_groups` | true | keep the tight subset of a failing group |
| `llm_backend` | `litellm` | or `stub` (offline) |
| `llm_model` | `gpt-4o-mini` | any LiteLLM model string |

**The in-memory store does not persist between CLI commands.** Use
`store_backend: "chroma"` for real use, or pass `--ingest` to `benchmark` for a
one-shot run.

### Choosing an LLM

The model string is the only thing that changes:

```jsonc
"llm_model": "gpt-4o-mini"                    // OPENAI_API_KEY
"llm_model": "claude-3-5-haiku-20241022"      // ANTHROPIC_API_KEY
"llm_model": "groq/llama-3.1-8b-instant"      // GROQ_API_KEY
"llm_model": "ollama/llama3.2:3b"             // fully local, no key
```

Copy `.env.example` to `.env` and set the key that matches.

## Benchmarking

```bash
ragcompactor benchmark --queries queries.txt --top-k 5
```

Compacts the store and reports corpus size, context tokens per query, and the
break-even point. **Run it on a copy** - it compacts in place.

Two things this deliberately does *not* do:

- It does not compare against stuffing raw files into a prompt. Any RAG setup
  beats that, so such a comparison would credit compaction for savings plain
  retrieval already provides. The comparison here is the same corpus and the
  same queries at the same top-k, differing only in whether compaction ran.
- It does not hide the cost. Summarization spends tokens, so the report states
  the one-off compaction cost and the number of queries needed to repay it.
  If compaction did not shrink retrieved context, break-even reads `never`.

With `llm_backend: "stub"` the compaction cost is zero because no API call is
made, so break-even is not meaningful and the report says so. The stub is
extractive - it can only select sentences, never rewrite them - so real numbers
need a real model.

## Library use

```python
from ragcompactor import CompactorConfig
from ragcompactor.compactor import Compactor

cfg = CompactorConfig(store_backend="chroma", llm_model="gpt-4o-mini")

with Compactor(cfg) as c:
    c.ingest(["./notes"])
    report = c.compact()
    print(report.to_dict())
    c.undo_run(report.run_id)      # changed your mind
```

Every stage is a protocol - `Embedder`, `VectorStore`, `Summarizer` - so any of
them can be swapped without touching the pipeline.

## Tests

```bash
pytest
```

The suite runs fully offline using the hashing embedder, in-memory store and
stub summarizer.

## Status

Early. The archive format and CLI flags may still change.
