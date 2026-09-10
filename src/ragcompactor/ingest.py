"""Loading source documents and splitting them into chunks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

from ragcompactor.core import Chunk


TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".py", ".json", ".csv", ".log"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "PDF ingestion needs the 'pdf' extra: pip install 'ragcompactor[pdf]'"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def read_file(path: Path) -> str:
    if path.suffix.lower() in PDF_SUFFIXES:
        return normalize(read_pdf(path))
    return normalize(path.read_text(encoding="utf-8", errors="replace"))


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split on paragraph boundaries, packing up to ``chunk_size`` characters.

    Paragraphs longer than ``chunk_size`` are hard-split with ``overlap`` carried
    between the pieces so a sentence straddling a boundary still retrieves.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = normalize(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            flush()
            start = 0
            while start < len(para):
                piece = para[start : start + chunk_size]
                chunks.append(piece.strip())
                if start + chunk_size >= len(para):
                    break
                start += chunk_size - overlap
            continue

        if not buffer:
            buffer = para
        elif len(buffer) + 2 + len(para) <= chunk_size:
            buffer = f"{buffer}\n\n{para}"
        else:
            flush()
            buffer = para

    flush()
    return [c for c in chunks if c]


def chunk_file(path: str | Path, chunk_size: int = 900, overlap: int = 120) -> list[Chunk]:
    p = Path(path)
    text = read_file(p)
    source = str(p)
    return [
        Chunk.create(piece, source=source, ordinal=i)
        for i, piece in enumerate(split_text(text, chunk_size, overlap))
    ]


def iter_source_files(root: str | Path) -> list[Path]:
    """Every supported file under ``root`` (or ``root`` itself if it is a file)."""
    p = Path(root)
    if p.is_file():
        return [p] if p.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        f
        for f in p.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    )


def chunk_paths(
    paths: list[str | Path], chunk_size: int = 900, overlap: int = 120
) -> list[Chunk]:
    """Chunk every supported file under each given path, de-duplicating exact repeats."""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for entry in paths:
        for f in iter_source_files(entry):
            for chunk in chunk_file(f, chunk_size, overlap):
                if chunk.id in seen:
                    continue
                seen.add(chunk.id)
                chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Ingest ledger
# ---------------------------------------------------------------------------


def file_fingerprint(path: str | Path) -> dict:
    """Identify a file by its content, not just its timestamp.

    Hashing the bytes means a file that was touched, moved back, or rewritten
    with identical content is correctly recognised as unchanged, and a file
    edited within the same second is still caught - which an mtime check alone
    would miss. Size and mtime are recorded alongside for human inspection.
    """
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(131072), b""):
            digest.update(block)
    stat = p.stat()
    return {
        "content_hash": digest.hexdigest(),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IngestLedger:
    """JSON record of what has already been ingested into the vector store.

    Chunk ids are content-addressed, so re-adding an identical chunk is already
    harmless at the store level. The ledger exists for the three things that
    property does *not* give you:

    * **Cost.** Skipping an unchanged file avoids re-reading it, re-parsing a
      PDF, re-chunking and - most expensively - re-embedding it.
    * **Edited files.** When a file changes, its old chunks keep their old ids
      and would linger in the store forever as orphans. The ledger remembers
      which ids came from which file, so they can be removed.
    * **Undoing compaction by accident.** After a merge, the original chunks are
      archived and deleted from the store. Re-ingesting that source file would
      regenerate exactly those ids and silently restore the duplicates the
      compactor had just removed. Skipping already-ingested files prevents it.
    """

    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.roots: list[str] = []
        self.files: dict[str, dict] = {}
        self.load()

    # --- persistence -------------------------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # A corrupt ledger must not block ingestion; the worst case of
            # starting empty is redundant work, not incorrect data.
            return
        self.roots = list(data.get("roots", []))
        self.files = dict(data.get("files", {}))

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "updated_at": _now(),
            "roots": self.roots,
            "files": self.files,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.path

    # --- keys --------------------------------------------------------------
    @staticmethod
    def key(path: str | Path) -> str:
        """Absolute, forward-slashed path, so the ledger survives a cwd change."""
        return Path(path).resolve().as_posix()

    # --- queries -----------------------------------------------------------
    def status(self, path: str | Path, fingerprint: dict | None = None) -> str:
        """``new``, ``unchanged`` or ``changed``."""
        record = self.files.get(self.key(path))
        if record is None:
            return "new"
        fingerprint = fingerprint or file_fingerprint(path)
        if record.get("content_hash") == fingerprint["content_hash"]:
            return "unchanged"
        return "changed"

    def chunk_ids_for(self, path: str | Path) -> list[str]:
        return list(self.files.get(self.key(path), {}).get("chunk_ids", []))

    def known_files(self) -> list[str]:
        return sorted(self.files)

    def missing_files(self) -> list[str]:
        """Sources recorded here that no longer exist on disk."""
        return [k for k in sorted(self.files) if not Path(k).exists()]

    def summary(self) -> dict:
        return {
            "roots": list(self.roots),
            "files": len(self.files),
            "chunks": sum(len(r.get("chunk_ids", [])) for r in self.files.values()),
            "missing_files": self.missing_files(),
            "path": str(self.path),
        }

    # --- writes ------------------------------------------------------------
    def record_root(self, path: str | Path) -> None:
        key = self.key(path)
        if key not in self.roots:
            self.roots.append(key)

    def record_file(
        self, path: str | Path, fingerprint: dict, chunk_ids: list[str]
    ) -> None:
        self.files[self.key(path)] = {
            **fingerprint,
            "chunk_ids": list(chunk_ids),
            "chunk_count": len(chunk_ids),
            "ingested_at": _now(),
        }

    def forget_file(self, path: str | Path) -> list[str]:
        """Drop a source from the ledger, returning the chunk ids it owned."""
        record = self.files.pop(self.key(path), None)
        return list(record.get("chunk_ids", [])) if record else []
