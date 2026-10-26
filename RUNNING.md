# Run the Streamlit app

Follow these steps from a fresh Windows checkout to launch the app with an
ingested and compacted corpus. Run the PowerShell commands from the repository
root, the folder containing `app.py` and `pyproject.toml`.

## Prerequisites

- Windows and Python 3.10 or newer
- Internet access to install packages and download the embedding model
- An OpenAI API key (the supplied example config uses `gpt-4o-mini`)
- Documents to use as the corpus

## 1. Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell prevents activation, run this in the same terminal and activate
again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Create the local config and set the API key

```powershell
Copy-Item ragcompactor.example.json ragcompactor.json
$env:OPENAI_API_KEY = Read-Host "OpenAI API key"
```

Enter your OpenAI API key when prompted. Keep using this PowerShell window:
the key is set only for this session and is inherited by the commands below.
The example config uses `gpt-4o-mini`. If you use a different provider, update
`llm_model` in `ragcompactor.json` and set the matching provider key in the
PowerShell session instead.

## 3. Add and ingest your documents

Put your documents in the `data` folder. Supported formats include `.txt`,
`.md`, `.rst`, `.py`, `.json`, `.csv`, `.log`, and `.pdf` (PDF support is
included in `requirements.txt`).

```powershell
ragcompactor ingest .\data
```

The first run downloads the configured sentence-transformer model and creates
the persistent vector store under `.ragcompactor`.

## 4. Compact the corpus

```powershell
ragcompactor compact
```

Compaction records merged chunks and archives the originals. The Streamlit app
uses that archive to compare retrieval before and after compaction.

## 5. Launch the app

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal. Select a question and click
**Retrieve**. Keep the virtual environment active and the Streamlit process
running while using the app.
