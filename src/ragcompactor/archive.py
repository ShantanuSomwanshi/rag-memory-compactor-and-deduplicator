"""SQLite archive of everything a merge replaced, and the undo path.

Compaction is lossy by construction: several chunks become one. That is only
safe to run unattended if the originals survive, so every merge writes its
source chunks - text, metadata and embedding vector - here before they leave the
live store. Keeping the vector means an undo restores the exact original
embedding instead of re-embedding (which could drift with a model change).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ragcompactor.models import Chunk, MergeRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    config_json TEXT
);
CREATE TABLE IF NOT EXISTS merges (
    merge_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    merged_chunk_id TEXT NOT NULL,
    merged_text     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    undone          INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS originals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    merge_id      TEXT NOT NULL,
    chunk_id      TEXT NOT NULL,
    text          TEXT NOT NULL,
    source        TEXT,
    ordinal       INTEGER,
    metadata_json TEXT,
    vector        BLOB,
    FOREIGN KEY (merge_id) REFERENCES merges(merge_id)
);
CREATE INDEX IF NOT EXISTS idx_merges_run ON merges(run_id);
CREATE INDEX IF NOT EXISTS idx_originals_merge ON originals(merge_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Archive:
    """Durable record of merges, and the source of truth for undo."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Archive":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- writing -----------------------------------------------------------
    def start_run(self, config_json: str = "") -> str:
        run_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO runs (run_id, created_at, config_json) VALUES (?, ?, ?)",
            (run_id, _now(), config_json),
        )
        self._conn.commit()
        return run_id

    def record_merge(
        self,
        run_id: str,
        merged_chunk: Chunk,
        originals: list[Chunk],
        vectors: dict[str, np.ndarray] | None = None,
    ) -> MergeRecord:
        merge_id = uuid.uuid4().hex[:12]
        created = _now()
        self._conn.execute(
            "INSERT INTO merges (merge_id, run_id, merged_chunk_id, merged_text, created_at, undone)"
            " VALUES (?, ?, ?, ?, ?, 0)",
            (merge_id, run_id, merged_chunk.id, merged_chunk.text, created),
        )
        for chunk in originals:
            vec = (vectors or {}).get(chunk.id)
            blob = (
                np.asarray(vec, dtype=np.float32).tobytes() if vec is not None else None
            )
            self._conn.execute(
                "INSERT INTO originals (merge_id, chunk_id, text, source, ordinal, metadata_json, vector)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    merge_id,
                    chunk.id,
                    chunk.text,
                    chunk.source,
                    chunk.ordinal,
                    json.dumps(chunk.metadata or {}),
                    blob,
                ),
            )
        self._conn.commit()
        return MergeRecord(
            merge_id=merge_id,
            run_id=run_id,
            merged_chunk_id=merged_chunk.id,
            merged_text=merged_chunk.text,
            original_ids=[c.id for c in originals],
            created_at=created,
        )

    # --- reading -----------------------------------------------------------
    def list_runs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT r.run_id, r.created_at,"
            " (SELECT COUNT(*) FROM merges m WHERE m.run_id = r.run_id) AS merges,"
            " (SELECT COUNT(*) FROM merges m WHERE m.run_id = r.run_id AND m.undone = 1) AS undone"
            " FROM runs r ORDER BY r.created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_merge(self, merge_id: str) -> MergeRecord | None:
        row = self._conn.execute(
            "SELECT * FROM merges WHERE merge_id = ?", (merge_id,)
        ).fetchone()
        if row is None:
            return None
        ids = [
            r["chunk_id"]
            for r in self._conn.execute(
                "SELECT chunk_id FROM originals WHERE merge_id = ? ORDER BY id", (merge_id,)
            )
        ]
        return MergeRecord(
            merge_id=row["merge_id"],
            run_id=row["run_id"],
            merged_chunk_id=row["merged_chunk_id"],
            merged_text=row["merged_text"],
            original_ids=ids,
            created_at=row["created_at"],
            undone=bool(row["undone"]),
        )

    def list_merges(self, run_id: str | None = None, include_undone: bool = True) -> list[MergeRecord]:
        sql = "SELECT merge_id FROM merges"
        params: tuple = ()
        clauses = []
        if run_id:
            clauses.append("run_id = ?")
            params += (run_id,)
        if not include_undone:
            clauses.append("undone = 0")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC"
        merges = [self.get_merge(r["merge_id"]) for r in self._conn.execute(sql, params)]
        return [m for m in merges if m is not None]

    def originals_for(self, merge_id: str) -> list[tuple[Chunk, np.ndarray | None]]:
        rows = self._conn.execute(
            "SELECT * FROM originals WHERE merge_id = ? ORDER BY id", (merge_id,)
        ).fetchall()
        out: list[tuple[Chunk, np.ndarray | None]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError):
                metadata = {}
            chunk = Chunk(
                id=row["chunk_id"],
                text=row["text"],
                source=row["source"] or "",
                ordinal=int(row["ordinal"] or 0),
                metadata=metadata,
            )
            vector = (
                np.frombuffer(row["vector"], dtype=np.float32) if row["vector"] else None
            )
            out.append((chunk, vector))
        return out

    def mark_undone(self, merge_id: str) -> None:
        self._conn.execute("UPDATE merges SET undone = 1 WHERE merge_id = ?", (merge_id,))
        self._conn.commit()

    # --- stats -------------------------------------------------------------
    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT (SELECT COUNT(*) FROM runs) AS runs,"
            " (SELECT COUNT(*) FROM merges) AS merges,"
            " (SELECT COUNT(*) FROM merges WHERE undone = 1) AS undone,"
            " (SELECT COUNT(*) FROM originals) AS archived_chunks"
        ).fetchone()
        return dict(row)
