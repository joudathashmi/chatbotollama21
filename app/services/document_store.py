"""Persistent document library store (Postgres FTS or in-memory for tests).

Visibility:
  private — owner_username only
  org     — any authenticated caller

Files live under MISA_DOCUMENTS_ROOT/{document_id}/{safe_filename}.
Metadata + chunks are in Postgres (or a process-local memory backend).
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import psycopg2
import psycopg2.extras

from app import config
from app.logger import logger

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\- ]+")
_VISIBILITIES = frozenset({"private", "org"})
_STATUSES = frozenset({"pending", "ready", "failed"})

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    owner_username TEXT NOT NULL,
    visibility TEXT NOT NULL CHECK (visibility IN ('private', 'org')),
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
    sha256 TEXT NOT NULL,
    byte_size BIGINT NOT NULL DEFAULT 0,
    storage_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'ready', 'failed')),
    source TEXT NOT NULL CHECK (source IN ('upload', 'ingest')),
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS documents_sha_private_uidx
    ON documents (sha256, owner_username) WHERE visibility = 'private';
CREATE UNIQUE INDEX IF NOT EXISTS documents_sha_org_uidx
    ON documents (sha256) WHERE visibility = 'org';
CREATE INDEX IF NOT EXISTS documents_owner_idx ON documents (owner_username);
CREATE INDEX IF NOT EXISTS documents_visibility_idx ON documents (visibility);

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx
    ON document_chunks USING GIN (tsv);
"""


@dataclass
class DocumentRecord:
    id: str
    owner_username: str
    visibility: str
    filename: str
    content_type: str
    sha256: str
    byte_size: int
    storage_path: str
    status: str
    source: str
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner_username": self.owner_username,
            "visibility": self.visibility,
            "filename": self.filename,
            "content_type": self.content_type,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "status": self.status,
            "source": self.source,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ChunkHit:
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


def safe_filename(name: str) -> str:
    base = os.path.basename(name or "upload.bin").strip() or "upload.bin"
    cleaned = _SAFE_NAME_RE.sub("_", base).strip("._") or "upload.bin"
    return cleaned[:180]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_dirs() -> None:
    Path(config.DOCUMENTS_ROOT).mkdir(parents=True, exist_ok=True)
    inbox = Path(config.DOCUMENTS_INGEST_DIR)
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "processed").mkdir(exist_ok=True)
    (inbox / "failed").mkdir(exist_ok=True)


def storage_path_for(doc_id: str, filename: str) -> Path:
    root = Path(config.DOCUMENTS_ROOT).resolve()
    dest_dir = (root / doc_id).resolve()
    if not str(dest_dir).startswith(str(root)):
        raise ValueError("Invalid document storage path.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = (dest_dir / safe_filename(filename)).resolve()
    if not str(dest).startswith(str(root)):
        raise ValueError("Invalid document filename path.")
    return dest


def _visible_clause(username: str, alias: str = "d") -> tuple[str, list]:
    return (
        f"({alias}.visibility = 'org' OR {alias}.owner_username = %s)",
        [username],
    )


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------

_pg_tls = threading.local()
_schema_lock = Lock()
_schema_ready = False


def _pg_connect():
    """Writable connection dedicated to the document library (not read-only)."""
    conn = psycopg2.connect(**config.DB_CONFIG)
    conn.autocommit = True
    return conn


def _pg_conn():
    conn = getattr(_pg_tls, "conn", None)
    if conn is None or getattr(conn, "closed", 1):
        conn = _pg_connect()
        _pg_tls.conn = conn
    return conn


def ensure_postgres_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        conn = _pg_conn()
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        _schema_ready = True
        logger.info("Document library schema ready (postgres).")


class PostgresDocumentStore:
    def ensure(self) -> None:
        ensure_dirs()
        ensure_postgres_schema()

    def find_duplicate(
        self, *, sha256: str, visibility: str, owner_username: str
    ) -> DocumentRecord | None:
        self.ensure()
        if visibility == "org":
            sql = "SELECT * FROM documents WHERE sha256 = %s AND visibility = 'org' LIMIT 1"
            params: tuple = (sha256,)
        else:
            sql = """
                SELECT * FROM documents
                WHERE sha256 = %s AND visibility = 'private' AND owner_username = %s
                LIMIT 1
            """
            params = (sha256, owner_username)
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return _row_to_doc(row) if row else None

    def create_pending(
        self,
        *,
        owner_username: str,
        visibility: str,
        filename: str,
        content_type: str,
        sha256: str,
        byte_size: int,
        storage_path: str,
        source: str,
        doc_id: str | None = None,
    ) -> DocumentRecord:
        self.ensure()
        if visibility not in _VISIBILITIES:
            raise ValueError("visibility must be private or org")
        doc_id = doc_id or str(uuid.uuid4())
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO documents (
                    id, owner_username, visibility, filename, content_type,
                    sha256, byte_size, storage_path, status, source, created_at, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'pending',%s,NOW(),NOW())
                RETURNING *
                """,
                (
                    doc_id, owner_username, visibility, filename, content_type,
                    sha256, byte_size, storage_path, source,
                ),
            )
            row = cur.fetchone()
        return _row_to_doc(row)

    def set_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        self.ensure()
        with _pg_conn().cursor() as cur:
            cur.execute(
                "UPDATE documents SET status=%s, error=%s, updated_at=NOW() WHERE id=%s",
                (status, error, doc_id),
            )

    def replace_chunks(self, doc_id: str, chunks: list[str]) -> None:
        self.ensure()
        with _pg_conn().cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE document_id=%s", (doc_id,))
            for i, text in enumerate(chunks):
                cur.execute(
                    "INSERT INTO document_chunks (document_id, chunk_index, text) VALUES (%s,%s,%s)",
                    (doc_id, i, text),
                )

    def get(self, doc_id: str, username: str) -> DocumentRecord | None:
        self.ensure()
        vis_sql, vis_params = _visible_clause(username)
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM documents d WHERE d.id=%s AND {vis_sql}",
                [doc_id, *vis_params],
            )
            row = cur.fetchone()
        return _row_to_doc(row) if row else None

    def get_raw(self, doc_id: str) -> DocumentRecord | None:
        self.ensure()
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM documents WHERE id=%s", (doc_id,))
            row = cur.fetchone()
        return _row_to_doc(row) if row else None

    def list_visible(self, username: str, limit: int = 100) -> list[DocumentRecord]:
        self.ensure()
        vis_sql, vis_params = _visible_clause(username)
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT * FROM documents d
                WHERE {vis_sql}
                ORDER BY d.created_at DESC
                LIMIT %s
                """,
                [*vis_params, max(1, min(limit, 500))],
            )
            rows = cur.fetchall()
        return [_row_to_doc(r) for r in rows]

    def delete(self, doc_id: str, username: str, *, is_admin: bool) -> bool:
        self.ensure()
        doc = self.get_raw(doc_id)
        if doc is None:
            return False
        if doc.owner_username != username and not is_admin:
            return False
        if doc.visibility == "org" and doc.owner_username != username and not is_admin:
            return False
        # Org docs: owner or admin may delete.
        with _pg_conn().cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id=%s", (doc_id,))
        try:
            path = Path(doc.storage_path)
            if path.is_file():
                path.unlink(missing_ok=True)
            parent = path.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass
        return True

    def retrieve(
        self, query: str, username: str, *, top_k: int | None = None
    ) -> list[ChunkHit]:
        self.ensure()
        q = (query or "").strip()
        if not q:
            return []
        k = top_k or config.DOCUMENTS_RETRIEVAL_TOP_K
        vis_sql, vis_params = _visible_clause(username)
        sql = f"""
            SELECT c.document_id, d.filename, c.chunk_index, c.text,
                   ts_rank_cd(c.tsv, plainto_tsquery('english', %s)) AS score
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'ready'
              AND {vis_sql}
              AND c.tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
        """
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, [q, *vis_params, q, k])
            rows = cur.fetchall()
        return [
            ChunkHit(
                document_id=str(r["document_id"]),
                filename=r["filename"],
                chunk_index=int(r["chunk_index"]),
                text=r["text"],
                score=float(r["score"] or 0),
            )
            for r in rows
        ]

    def reset_for_tests(self) -> None:
        self.ensure()
        with _pg_conn().cursor() as cur:
            cur.execute("DELETE FROM document_chunks")
            cur.execute("DELETE FROM documents")


def _row_to_doc(row: dict) -> DocumentRecord:
    return DocumentRecord(
        id=str(row["id"]),
        owner_username=row["owner_username"],
        visibility=row["visibility"],
        filename=row["filename"],
        content_type=row["content_type"],
        sha256=row["sha256"],
        byte_size=int(row["byte_size"] or 0),
        storage_path=row["storage_path"],
        status=row["status"],
        source=row["source"],
        error=row.get("error"),
        created_at=str(row["created_at"]) if row.get("created_at") is not None else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") is not None else None,
    )


# ---------------------------------------------------------------------------
# Memory backend (tests / offline)
# ---------------------------------------------------------------------------

@dataclass
class _MemState:
    docs: dict[str, DocumentRecord] = field(default_factory=dict)
    chunks: dict[str, list[str]] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)


class MemoryDocumentStore:
    def __init__(self) -> None:
        self._state = _MemState()

    def ensure(self) -> None:
        ensure_dirs()

    def find_duplicate(
        self, *, sha256: str, visibility: str, owner_username: str
    ) -> DocumentRecord | None:
        with self._state.lock:
            for d in self._state.docs.values():
                if d.sha256 != sha256 or d.visibility != visibility:
                    continue
                if visibility == "org" or d.owner_username == owner_username:
                    return d
        return None

    def create_pending(
        self,
        *,
        owner_username: str,
        visibility: str,
        filename: str,
        content_type: str,
        sha256: str,
        byte_size: int,
        storage_path: str,
        source: str,
        doc_id: str | None = None,
    ) -> DocumentRecord:
        now = datetime.now(timezone.utc).isoformat()
        doc = DocumentRecord(
            id=doc_id or str(uuid.uuid4()),
            owner_username=owner_username,
            visibility=visibility,
            filename=filename,
            content_type=content_type,
            sha256=sha256,
            byte_size=byte_size,
            storage_path=storage_path,
            status="pending",
            source=source,
            created_at=now,
            updated_at=now,
        )
        with self._state.lock:
            self._state.docs[doc.id] = doc
        return doc

    def set_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        with self._state.lock:
            d = self._state.docs.get(doc_id)
            if d:
                d.status = status
                d.error = error
                d.updated_at = datetime.now(timezone.utc).isoformat()

    def replace_chunks(self, doc_id: str, chunks: list[str]) -> None:
        with self._state.lock:
            self._state.chunks[doc_id] = list(chunks)

    def get(self, doc_id: str, username: str) -> DocumentRecord | None:
        with self._state.lock:
            d = self._state.docs.get(doc_id)
            if d is None:
                return None
            if d.visibility == "org" or d.owner_username == username:
                return d
        return None

    def get_raw(self, doc_id: str) -> DocumentRecord | None:
        with self._state.lock:
            return self._state.docs.get(doc_id)

    def list_visible(self, username: str, limit: int = 100) -> list[DocumentRecord]:
        with self._state.lock:
            rows = [
                d for d in self._state.docs.values()
                if d.visibility == "org" or d.owner_username == username
            ]
        rows.sort(key=lambda d: d.created_at or "", reverse=True)
        return rows[: max(1, min(limit, 500))]

    def delete(self, doc_id: str, username: str, *, is_admin: bool) -> bool:
        with self._state.lock:
            d = self._state.docs.get(doc_id)
            if d is None:
                return False
            if d.owner_username != username and not is_admin:
                return False
            self._state.docs.pop(doc_id, None)
            self._state.chunks.pop(doc_id, None)
        try:
            path = Path(d.storage_path)
            if path.is_file():
                path.unlink(missing_ok=True)
        except Exception:
            pass
        return True

    def retrieve(
        self, query: str, username: str, *, top_k: int | None = None
    ) -> list[ChunkHit]:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]{2,}", query or "")]
        if not tokens:
            return []
        k = top_k or config.DOCUMENTS_RETRIEVAL_TOP_K
        hits: list[ChunkHit] = []
        with self._state.lock:
            for doc_id, chunks in self._state.chunks.items():
                d = self._state.docs.get(doc_id)
                if d is None or d.status != "ready":
                    continue
                if not (d.visibility == "org" or d.owner_username == username):
                    continue
                for i, text in enumerate(chunks):
                    low = text.lower()
                    score = sum(1.0 for t in tokens if t in low) / max(len(tokens), 1)
                    if score > 0:
                        hits.append(ChunkHit(
                            document_id=doc_id,
                            filename=d.filename,
                            chunk_index=i,
                            text=text,
                            score=score,
                        ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def reset_for_tests(self) -> None:
        with self._state.lock:
            self._state.docs.clear()
            self._state.chunks.clear()


_store: PostgresDocumentStore | MemoryDocumentStore | None = None
_store_lock = Lock()


def get_document_store() -> PostgresDocumentStore | MemoryDocumentStore:
    global _store
    with _store_lock:
        if _store is None:
            backend = (config.DOCUMENTS_BACKEND or "postgres").strip().lower()
            if backend == "memory":
                _store = MemoryDocumentStore()
                _store.ensure()
            else:
                try:
                    s = PostgresDocumentStore()
                    s.ensure()
                    _store = s
                except Exception as e:
                    logger.warning(
                        f"Document postgres backend unavailable ({e}); using memory store."
                    )
                    _store = MemoryDocumentStore()
                    _store.ensure()
        return _store


def reset_document_store_for_tests() -> None:
    global _store
    with _store_lock:
        if _store is not None:
            try:
                _store.reset_for_tests()
            except Exception:
                pass
        _store = MemoryDocumentStore()
