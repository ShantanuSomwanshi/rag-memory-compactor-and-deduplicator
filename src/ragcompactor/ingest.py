"""Loading source documents and splitting them into chunks."""

from __future__ import annotations

import re
from pathlib import Path

from ragcompactor.models import Chunk

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
