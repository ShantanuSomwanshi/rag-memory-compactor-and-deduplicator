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
pip install -e ".[all]"        # chroma + sentence-transformers + litellm + tiktoken + pypdf
pip install -e ".[dev]"        # pytest
```

Extras are granular if you want a smaller install: `chroma`, `embed`, `llm`,
`tokens`, `pdf`.

## Quick start

```bash
ragcompactor demo                          # offline end-to-end: seed, compact, undo

ragcompactor init                          # writes ragcompactor.json
ragcompactor ingest ./notes ./papers
ragcompactor compact --dry-run             # what would merge, no LLM call, no writes
ragcompactor compact
ragcompactor runs
ragcompactor undo --last
```

`demo` runs fully offline - hashing embedder, in-memory store, stub summarizer -
so it needs no API key and no model download. It also deliberately shows the
validation gate refining an over-broad group.

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
