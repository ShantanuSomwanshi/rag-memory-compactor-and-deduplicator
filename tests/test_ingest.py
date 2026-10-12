from ragcompactor.ingest import chunk_file, chunk_paths, normalize, split_text
from ragcompactor.core import Chunk


def test_normalize_collapses_whitespace_and_blank_lines():
    assert normalize("a   b\r\n\r\n\r\n\r\nc") == "a b\n\nc"


def test_split_text_packs_paragraphs_under_chunk_size():
    text = "\n\n".join(["para one." * 5, "para two." * 5, "para three." * 5])
    chunks = split_text(text, chunk_size=200, overlap=20)
    assert chunks
    assert all(len(c) <= 200 for c in chunks)
    # nothing is lost
    assert "para three." in " ".join(chunks)


def test_split_text_hard_splits_oversized_paragraph_with_overlap():
    para = "x" * 500
    chunks = split_text(para, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_split_text_rejects_overlap_bigger_than_chunk():
    try:
        split_text("abc", chunk_size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_chunk_ids_are_content_addressed_and_stable():
    first = Chunk.create("same text", source="s", ordinal=0)
    second = Chunk.create("same text", source="s", ordinal=0)
    different = Chunk.create("same text", source="s", ordinal=1)
    assert first.id == second.id
    assert first.id != different.id


def test_chunk_file_and_paths(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("First paragraph here.\n\nSecond paragraph here.", encoding="utf-8")

    chunks = chunk_file(doc, chunk_size=100, overlap=10)
    assert chunks and all(c.source == str(doc) for c in chunks)

    # a directory walk finds the same content, and repeats are dropped
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")
    walked = chunk_paths([tmp_path, doc], chunk_size=100, overlap=10)
    assert len({c.id for c in walked}) == len(walked)
