import json
import os
import sqlite3
from typing import Any


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash TEXT,
    source_type TEXT NOT NULL,
    extension TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    preview_path TEXT NOT NULL,
    preview_width INTEGER NOT NULL,
    preview_height INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extractions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL,
    filename TEXT NOT NULL,
    line1 TEXT NOT NULL,
    line2 TEXT NOT NULL,
    parsed_json TEXT NOT NULL,
    report_path TEXT,
    duration_ms REAL,
    crop_json TEXT,
    rotation INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_truth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    line1 TEXT NOT NULL,
    line2 TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(document_id),
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
"""


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def get_connection(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        _ensure_document_file_hash(conn)
        conn.commit()


def _ensure_document_file_hash(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(documents)").fetchall()
    }
    if "file_hash" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash)"
    )


def insert_document(db_path: str, record: dict[str, Any]) -> dict[str, Any]:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO documents (
                id, filename, file_hash, source_type, extension, stored_path, preview_path,
                preview_width, preview_height, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["filename"],
                record.get("file_hash"),
                record["source_type"],
                record["extension"],
                record["stored_path"],
                record["preview_path"],
                int(record["preview_width"]),
                int(record["preview_height"]),
                record["created_at"],
            ),
        )
        conn.commit()
    return record


def get_document(db_path: str, document_id: str) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return _row_to_dict(row)


def get_document_by_hash(db_path: str, file_hash: str) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM documents
            WHERE file_hash = ?
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (file_hash,),
        ).fetchone()
    return _row_to_dict(row)


def insert_extraction(db_path: str, record: dict[str, Any]) -> dict[str, Any]:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO extractions (
                id, document_id, status, filename, line1, line2, parsed_json,
                report_path, duration_ms, crop_json, rotation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["document_id"],
                record["status"],
                record["filename"],
                record["line1"],
                record["line2"],
                json.dumps(record.get("parsed") or {}, ensure_ascii=True),
                record.get("report_path"),
                record.get("duration_ms"),
                json.dumps(record.get("crop") or {}, ensure_ascii=True) if record.get("crop") else None,
                int(record.get("rotation", 0)),
                record["created_at"],
            ),
        )
        conn.commit()
    return record


def get_extraction(db_path: str, extraction_id: str) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM extractions WHERE id = ?", (extraction_id,)).fetchone()
    payload = _row_to_dict(row)
    if payload is None:
        return None
    payload["parsed"] = json.loads(payload.pop("parsed_json") or "{}")
    payload["crop"] = json.loads(payload.pop("crop_json") or "{}") if payload.get("crop_json") else None
    return payload


def upsert_reference(db_path: str, record: dict[str, Any]) -> dict[str, Any]:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO reference_truth (
                document_id, filename, line1, line2, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                filename=excluded.filename,
                line1=excluded.line1,
                line2=excluded.line2,
                notes=excluded.notes,
                updated_at=excluded.updated_at
            """,
            (
                record["document_id"],
                record["filename"],
                record["line1"],
                record["line2"],
                record.get("notes", ""),
                record["created_at"],
                record["updated_at"],
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM reference_truth WHERE document_id = ?", (record["document_id"],)).fetchone()
    return dict(row)


def list_references(db_path: str) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM reference_truth ORDER BY updated_at DESC, id DESC").fetchall()
    return [dict(row) for row in rows]
