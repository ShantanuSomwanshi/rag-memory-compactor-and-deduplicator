# Running ragcompactor

This guide explains how to install, verify, and run the project from a fresh
Windows checkout. Run all commands from the repository root, the directory
that contains `pyproject.toml`.

## 1. Prerequisites

- Windows PowerShell or Command Prompt
- Python 3.10 or newer
- Internet access for installing packages and downloading the default embedding
  model
- An API key for the selected hosted LLM, unless using the offline demo, the
  stub summarizer, or a local Ollama model

Check Python before starting:

```powershell
python --version
```

The project was checked with Python 3.12.3.

## 2. Create and activate a virtual environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, either allow it for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

For Command Prompt, use:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

The prompt should show `(.venv)` after activation. The virtual environment
keeps project packages separate from globally installed Python packages.

## 3. Install the project

Upgrade packaging tools, install the complete runtime dependency set, and
install the package in editable mode:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

What these commands do:

- `pip install --upgrade pip` updates the package installer.
- `pip install -r requirements-dev.txt` installs the runtime packages,
  ChromaDB, sentence-transformers, LiteLLM, PDF support, token accounting, and
  test tools.
- `pip install -e .` installs this checkout and registers the `ragcompactor`
  command. `-e` means source changes are immediately used without reinstalling.

The equivalent package-extra install is:

```powershell
python -m pip install -e ".[all,dev]"
```

## 4. Verify the installation without an API key

First confirm that the CLI is available:

```powershell
python -m ragcompactor.cli --help
ragcompactor --help
```

Then run the complete offline demonstration:

```powershell
ragcompactor demo
```

The demo uses a hashing embedder, an in-memory store, and a stub summarizer.
It creates duplicate and distinct chunks, compacts the duplicates, and undoes
the merge. It does not need an API key, a model download, or ChromaDB.

To place its temporary work directory somewhere else:

```powershell
ragcompactor demo --workdir .ragcompactor-demo
```

## 5. Configure a real run

The checked-in example configuration is in `ragcompactor.json`. It uses a
persistent Chroma store, the `all-MiniLM-L6-v2` sentence-transformer, and a
LiteLLM model. Edit `llm_model` to match the provider you intend to use.

For a hosted provider, copy the environment template and fill in only the
matching key:

```powershell
Copy-Item .env.example .env
notepad .env
```

Examples:

```text
gpt-4o-mini                         -> OPENAI_API_KEY
claude-3-5-haiku-20241022           -> ANTHROPIC_API_KEY
groq/llama-3.1-8b-instant           -> GROQ_API_KEY
```

For a local Ollama model, set `llm_model` to something such as
`ollama/llama3.2:3b`; no hosted API key is required, but Ollama and the model
must be installed and running.

To generate a new starter configuration instead of editing the existing one:

```powershell
ragcompactor init --store chroma --embedder sentence-transformers --llm litellm --model gpt-4o-mini
```

`init` writes `ragcompactor.json` and can overwrite an existing configuration,
so inspect it before using the command on an established work directory.

## 6. Ingest and compact documents

The sample corpus is in `data/`. You can pass files or directories; directory
ingestion is recursive.

```powershell
ragcompactor ingest .\data
```

This chunks the supported files, creates embeddings, and stores them in the
configured Chroma collection. The ingestion ledger is written under the
configured work directory, normally `.ragcompactor\ingested.json`.

Useful inspection commands:

```powershell
ragcompactor sources
ragcompactor stats
```

Preview candidate merges without calling the LLM or changing the store:

```powershell
ragcompactor compact --dry-run
```

Run the actual compaction:

```powershell
ragcompactor compact
```

The command discovers similar chunks, applies the strict validation gate,
summarizes validated groups, replaces them in Chroma, and archives the
originals in SQLite.

For providers with a rate limit, add a delay between summarization calls:

```powershell
ragcompactor compact --throttle 8
```

View recorded runs:

```powershell
ragcompactor runs
```

## 7. Undo a compaction

Undo the most recent run:

```powershell
ragcompactor undo --last
```

Undo a specific run shown by `ragcompactor runs`:

```powershell
ragcompactor undo --run RUN_ID
```

Undo one merge instead:

```powershell
ragcompactor undo --merge MERGE_ID
```

The original text, metadata, and embedding vectors are restored from the
SQLite archive.

## 8. Re-ingestion and changed files

Repeated ingestion skips files whose content has not changed:

```powershell
ragcompactor ingest .\data
```

After changing chunking settings or the embedding model, force re-ingestion:

```powershell
ragcompactor ingest .\data --force
```

Remove one source and its chunks from the store:

```powershell
ragcompactor forget .\data\file.txt
```

If an edited source was part of an earlier merge, the ingest command reports
the affected merge IDs. Undo those merges before compacting the updated text.

## 9. Benchmark the result

Benchmarking compacts the store in place. Run it on a copy if the current state
must be preserved.

With a persistent store and already-ingested data:

```powershell
ragcompactor benchmark --top-k 5
```

With a one-line-per-query file:

```powershell
ragcompactor benchmark --queries .\queries.txt --top-k 5
```

For the in-memory backend, ingestion must happen in the same command:

```powershell
ragcompactor benchmark --ingest .\data --top-k 5
```

The report compares corpus tokens and retrieved-context tokens before and
after compaction, and estimates the query break-even point.

## 10. Run the tests

The tests are designed to run offline:

```powershell
python -m pytest -q
```

For coverage:

```powershell
python -m pytest --cov=ragcompactor --cov-report=term-missing
```

If pytest fails during startup with `AttributeError: module 'py' has no
attribute 'path'`, the active global Python installation has a conflicting
`py.py` module. Deactivate it and create the clean `.venv` described above,
then reinstall `requirements-dev.txt` inside that environment. Do not run the
tests from the conflicting global installation.

## Runtime files

Normal runs create local state under `.ragcompactor\`:

- Chroma vector-store data
- `archive.sqlite3`, which stores originals for undo
- `ingested.json`, which records the ingestion ledger

These runtime files and `.env` are ignored by Git. Do not commit API keys or
the generated vector-store data.
