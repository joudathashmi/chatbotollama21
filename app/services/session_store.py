"""Persistent chat sessions + messages (Postgres or in-memory for tests).

Each authenticated user owns their sessions. Messages are append-only within
a session; deleting a session cascades to its messages.

World-class extras: pin, soft-archive (TTL), searchable titles, state card
+ rolling summary for smart prompt context (see chat_context.py).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

import psycopg2
import psycopg2.extras

from app import config
from app.logger import logger

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY,
    owner_username TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT 'New chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pinned BOOLEAN NOT NULL DEFAULT FALSE,
    archived_at TIMESTAMPTZ,
    state_json JSONB,
    summary TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS chat_sessions_owner_updated_idx
    ON chat_sessions (owner_username, updated_at DESC);
CREATE INDEX IF NOT EXISTS chat_sessions_owner_pinned_idx
    ON chat_sessions (owner_username, pinned DESC, updated_at DESC);
CREATE INDEX IF NOT EXISTS chat_sessions_owner_archived_idx
    ON chat_sessions (owner_username, archived_at);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    answer_source TEXT,
    web_sources JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chat_messages_session_idx
    ON chat_messages (session_id, id);
"""

# Idempotent upgrades for DBs created before 003.
_SCHEMA_UPGRADE_SQL = """
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS state_json JSONB;
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT '';
"""

_ROLES = frozenset({"user", "assistant"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def title_from_question(question: str, limit: int = 72) -> str:
    text = " ".join((question or "").strip().split())
    if not text:
        return "New chat"
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass
class SessionRecord:
    id: str
    owner_username: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None
    message_count: int = 0
    pinned: bool = False
    archived_at: str | None = None
    state: dict | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner_username": self.owner_username,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "pinned": self.pinned,
            "archived_at": self.archived_at,
            "state": self.state,
            "summary": self.summary,
            "active_entity": (self.state or {}).get("active_entity") if self.state else None,
        }


@dataclass
class MessageRecord:
    id: int | str
    session_id: str
    role: str
    content: str
    answer_source: str | None = None
    web_sources: list | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "answer_source": self.answer_source,
            "web_sources": self.web_sources,
            "created_at": self.created_at,
        }


def _parse_state(raw) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def _row_to_session(r: dict, message_count: int | None = None) -> SessionRecord:
    return SessionRecord(
        id=str(r["id"]),
        owner_username=r["owner_username"],
        title=r["title"],
        created_at=str(r["created_at"]) if r.get("created_at") else None,
        updated_at=str(r["updated_at"]) if r.get("updated_at") else None,
        message_count=int(
            message_count if message_count is not None else (r.get("message_count") or 0)
        ),
        pinned=bool(r.get("pinned")),
        archived_at=str(r["archived_at"]) if r.get("archived_at") else None,
        state=_parse_state(r.get("state_json") if "state_json" in r else r.get("state")),
        summary=str(r.get("summary") or ""),
    )


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

_pg_tls = threading.local()
_schema_lock = Lock()
_schema_ready = False


def _pg_connect():
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
            cur.execute(_SCHEMA_UPGRADE_SQL)
        _schema_ready = True
        logger.info("Chat sessions schema ready (postgres).")


class PostgresSessionStore:
    def ensure(self) -> None:
        ensure_postgres_schema()

    def maintain_ttl(self, owner_username: str | None = None) -> dict[str, int]:
        """Soft-archive idle sessions; hard-delete old archives. Lazy on list."""
        self.ensure()
        archived = deleted = 0
        idle_days = int(getattr(config, "SESSIONS_IDLE_ARCHIVE_DAYS", 90) or 0)
        hard_days = int(getattr(config, "SESSIONS_HARD_DELETE_DAYS", 365) or 0)
        with _pg_conn().cursor() as cur:
            if idle_days > 0:
                if owner_username:
                    cur.execute(
                        """
                        UPDATE chat_sessions
                        SET archived_at = NOW()
                        WHERE owner_username = %s
                          AND archived_at IS NULL
                          AND pinned = FALSE
                          AND updated_at < NOW() - (%s || ' days')::interval
                        """,
                        (owner_username, str(idle_days)),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE chat_sessions
                        SET archived_at = NOW()
                        WHERE archived_at IS NULL
                          AND pinned = FALSE
                          AND updated_at < NOW() - (%s || ' days')::interval
                        """,
                        (str(idle_days),),
                    )
                archived = cur.rowcount or 0
            if hard_days > 0:
                if owner_username:
                    cur.execute(
                        """
                        DELETE FROM chat_sessions
                        WHERE owner_username = %s
                          AND archived_at IS NOT NULL
                          AND archived_at < NOW() - (%s || ' days')::interval
                        """,
                        (owner_username, str(hard_days)),
                    )
                else:
                    cur.execute(
                        """
                        DELETE FROM chat_sessions
                        WHERE archived_at IS NOT NULL
                          AND archived_at < NOW() - (%s || ' days')::interval
                        """,
                        (str(hard_days),),
                    )
                deleted = cur.rowcount or 0
        return {"archived": archived, "deleted": deleted}

    def create(self, owner_username: str, title: str | None = None) -> SessionRecord:
        self.ensure()
        sid = str(uuid.uuid4())
        title = (title or "New chat").strip() or "New chat"
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (id, owner_username, title)
                VALUES (%s::uuid, %s, %s)
                RETURNING id::text AS id, owner_username, title, created_at, updated_at,
                          pinned, archived_at, state_json, summary
                """,
                (sid, owner_username, title),
            )
            r = cur.fetchone()
        return _row_to_session(r, 0)

    def list_for_user(
        self,
        owner_username: str,
        limit: int = 50,
        *,
        include_archived: bool = False,
        q: str | None = None,
    ) -> list[SessionRecord]:
        self.ensure()
        self.maintain_ttl(owner_username)
        limit = max(1, min(int(limit), 200))
        params: list[Any] = [owner_username]
        where = ["s.owner_username = %s"]
        if not include_archived:
            where.append("s.archived_at IS NULL")
        if q and q.strip():
            where.append("s.title ILIKE %s")
            params.append(f"%{q.strip()}%")
        params.append(limit)
        sql = f"""
            SELECT s.id::text AS id, s.owner_username, s.title,
                   s.created_at, s.updated_at, s.pinned, s.archived_at,
                   s.state_json, s.summary,
                   COALESCE(m.cnt, 0)::int AS message_count
            FROM chat_sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS cnt
                FROM chat_messages GROUP BY session_id
            ) m ON m.session_id = s.id
            WHERE {' AND '.join(where)}
            ORDER BY s.pinned DESC, s.updated_at DESC
            LIMIT %s
        """
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [_row_to_session(r) for r in rows]

    def get(self, session_id: str, owner_username: str) -> SessionRecord | None:
        self.ensure()
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT s.id::text AS id, s.owner_username, s.title,
                       s.created_at, s.updated_at, s.pinned, s.archived_at,
                       s.state_json, s.summary,
                       (SELECT COUNT(*) FROM chat_messages m
                        WHERE m.session_id = s.id)::int AS message_count
                FROM chat_sessions s
                WHERE s.id = %s::uuid AND s.owner_username = %s
                """,
                (session_id, owner_username),
            )
            r = cur.fetchone()
        return _row_to_session(r) if r else None

    def rename(self, session_id: str, owner_username: str, title: str) -> SessionRecord | None:
        self.ensure()
        title = (title or "").strip() or "New chat"
        with _pg_conn().cursor() as cur:
            cur.execute(
                """
                UPDATE chat_sessions
                SET title = %s, updated_at = NOW()
                WHERE id = %s::uuid AND owner_username = %s
                """,
                (title[:200], session_id, owner_username),
            )
            if cur.rowcount == 0:
                return None
        return self.get(session_id, owner_username)

    def set_pinned(
        self, session_id: str, owner_username: str, pinned: bool
    ) -> SessionRecord | None:
        self.ensure()
        with _pg_conn().cursor() as cur:
            cur.execute(
                """
                UPDATE chat_sessions
                SET pinned = %s, updated_at = NOW(),
                    archived_at = CASE WHEN %s THEN NULL ELSE archived_at END
                WHERE id = %s::uuid AND owner_username = %s
                """,
                (bool(pinned), bool(pinned), session_id, owner_username),
            )
            if cur.rowcount == 0:
                return None
        return self.get(session_id, owner_username)

    def set_archived(
        self, session_id: str, owner_username: str, archived: bool
    ) -> SessionRecord | None:
        self.ensure()
        with _pg_conn().cursor() as cur:
            if archived:
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET archived_at = NOW(), pinned = FALSE, updated_at = NOW()
                    WHERE id = %s::uuid AND owner_username = %s
                    """,
                    (session_id, owner_username),
                )
            else:
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET archived_at = NULL, updated_at = NOW()
                    WHERE id = %s::uuid AND owner_username = %s
                    """,
                    (session_id, owner_username),
                )
            if cur.rowcount == 0:
                return None
        return self.get(session_id, owner_username)

    def save_state(
        self,
        session_id: str,
        owner_username: str,
        state: dict | None,
        summary: str | None = None,
    ) -> SessionRecord | None:
        self.ensure()
        with _pg_conn().cursor() as cur:
            cur.execute(
                """
                UPDATE chat_sessions
                SET state_json = %s,
                    summary = COALESCE(%s, summary),
                    updated_at = NOW(),
                    archived_at = NULL
                WHERE id = %s::uuid AND owner_username = %s
                """,
                (
                    json.dumps(state) if state is not None else None,
                    summary,
                    session_id,
                    owner_username,
                ),
            )
            if cur.rowcount == 0:
                return None
        return self.get(session_id, owner_username)

    def delete(self, session_id: str, owner_username: str) -> bool:
        self.ensure()
        with _pg_conn().cursor() as cur:
            cur.execute(
                """
                DELETE FROM chat_sessions
                WHERE id = %s::uuid AND owner_username = %s
                """,
                (session_id, owner_username),
            )
            return cur.rowcount > 0

    def list_messages(
        self, session_id: str, owner_username: str, limit: int = 500
    ) -> list[MessageRecord] | None:
        if self.get(session_id, owner_username) is None:
            return None
        limit = max(1, min(int(limit), 2000))
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, session_id::text AS session_id, role, content,
                       answer_source, web_sources, created_at
                FROM chat_messages
                WHERE session_id = %s::uuid
                ORDER BY id ASC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
        out: list[MessageRecord] = []
        for r in rows:
            ws = r.get("web_sources")
            if isinstance(ws, str):
                try:
                    ws = json.loads(ws)
                except Exception:
                    ws = None
            out.append(MessageRecord(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                answer_source=r.get("answer_source"),
                web_sources=ws if isinstance(ws, list) else None,
                created_at=str(r["created_at"]) if r["created_at"] else None,
            ))
        return out

    def append_message(
        self,
        session_id: str,
        owner_username: str,
        role: str,
        content: str,
        *,
        answer_source: str | None = None,
        web_sources: list | None = None,
        auto_title: bool = False,
    ) -> MessageRecord | None:
        if role not in _ROLES:
            raise ValueError(f"Invalid role: {role}")
        sess = self.get(session_id, owner_username)
        if sess is None:
            return None
        content = content or ""
        with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO chat_messages
                    (session_id, role, content, answer_source, web_sources)
                VALUES (%s::uuid, %s, %s, %s, %s)
                RETURNING id, session_id::text AS session_id, role, content,
                          answer_source, web_sources, created_at
                """,
                (
                    session_id,
                    role,
                    content,
                    answer_source,
                    json.dumps(web_sources) if web_sources is not None else None,
                ),
            )
            r = cur.fetchone()
            cur.execute(
                """
                UPDATE chat_sessions
                SET updated_at = NOW(), archived_at = NULL
                WHERE id = %s::uuid
                """,
                (session_id,),
            )
            if auto_title and role == "user" and (
                not sess.title or sess.title == "New chat"
            ):
                cur.execute(
                    """
                    UPDATE chat_sessions SET title = %s
                    WHERE id = %s::uuid AND owner_username = %s
                    """,
                    (title_from_question(content), session_id, owner_username),
                )
        ws = r.get("web_sources") if r else None
        if isinstance(ws, str):
            try:
                ws = json.loads(ws)
            except Exception:
                ws = None
        return MessageRecord(
            id=r["id"],
            session_id=r["session_id"],
            role=r["role"],
            content=r["content"],
            answer_source=r.get("answer_source"),
            web_sources=ws if isinstance(ws, list) else None,
            created_at=str(r["created_at"]) if r and r["created_at"] else None,
        )

    def reset_for_tests(self) -> None:
        self.ensure()
        with _pg_conn().cursor() as cur:
            cur.execute("DELETE FROM chat_messages")
            cur.execute("DELETE FROM chat_sessions")


# ---------------------------------------------------------------------------
# Memory (tests / fallback)
# ---------------------------------------------------------------------------

@dataclass
class _MemState:
    sessions: dict[str, SessionRecord] = field(default_factory=dict)
    messages: dict[str, list[MessageRecord]] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)
    _msg_seq: int = 0


class MemorySessionStore:
    def __init__(self) -> None:
        self._state = _MemState()

    def ensure(self) -> None:
        return

    def maintain_ttl(self, owner_username: str | None = None) -> dict[str, int]:
        idle_days = int(getattr(config, "SESSIONS_IDLE_ARCHIVE_DAYS", 90) or 0)
        hard_days = int(getattr(config, "SESSIONS_HARD_DELETE_DAYS", 365) or 0)
        archived = deleted = 0
        now = _now()
        with self._state.lock:
            to_del: list[str] = []
            for sid, s in self._state.sessions.items():
                if owner_username and s.owner_username != owner_username:
                    continue
                try:
                    updated = datetime.fromisoformat(s.updated_at) if s.updated_at else now
                except Exception:
                    updated = now
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                if (
                    idle_days > 0
                    and not s.archived_at
                    and not s.pinned
                    and updated < now - timedelta(days=idle_days)
                ):
                    s.archived_at = _now_iso()
                    archived += 1
                if hard_days > 0 and s.archived_at:
                    try:
                        arch = datetime.fromisoformat(s.archived_at)
                    except Exception:
                        continue
                    if arch.tzinfo is None:
                        arch = arch.replace(tzinfo=timezone.utc)
                    if arch < now - timedelta(days=hard_days):
                        to_del.append(sid)
                        deleted += 1
            for sid in to_del:
                self._state.sessions.pop(sid, None)
                self._state.messages.pop(sid, None)
        return {"archived": archived, "deleted": deleted}

    def create(self, owner_username: str, title: str | None = None) -> SessionRecord:
        sid = str(uuid.uuid4())
        now = _now_iso()
        rec = SessionRecord(
            id=sid,
            owner_username=owner_username,
            title=(title or "New chat").strip() or "New chat",
            created_at=now,
            updated_at=now,
            message_count=0,
        )
        with self._state.lock:
            self._state.sessions[sid] = rec
            self._state.messages[sid] = []
        return rec

    def list_for_user(
        self,
        owner_username: str,
        limit: int = 50,
        *,
        include_archived: bool = False,
        q: str | None = None,
    ) -> list[SessionRecord]:
        self.maintain_ttl(owner_username)
        limit = max(1, min(int(limit), 200))
        qn = (q or "").strip().lower()
        with self._state.lock:
            rows = [
                s for s in self._state.sessions.values()
                if s.owner_username == owner_username
                and (include_archived or not s.archived_at)
                and (not qn or qn in (s.title or "").lower())
            ]
            rows.sort(
                key=lambda s: (not s.pinned, s.updated_at or ""),
                reverse=False,
            )
            # pinned first: sort pinned DESC then updated DESC
            rows.sort(key=lambda s: (s.pinned, s.updated_at or ""), reverse=True)
            out = []
            for s in rows[:limit]:
                out.append(SessionRecord(
                    id=s.id,
                    owner_username=s.owner_username,
                    title=s.title,
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                    message_count=len(self._state.messages.get(s.id, [])),
                    pinned=s.pinned,
                    archived_at=s.archived_at,
                    state=dict(s.state) if s.state else None,
                    summary=s.summary or "",
                ))
            return out

    def get(self, session_id: str, owner_username: str) -> SessionRecord | None:
        with self._state.lock:
            s = self._state.sessions.get(session_id)
            if s is None or s.owner_username != owner_username:
                return None
            return SessionRecord(
                id=s.id,
                owner_username=s.owner_username,
                title=s.title,
                created_at=s.created_at,
                updated_at=s.updated_at,
                message_count=len(self._state.messages.get(s.id, [])),
                pinned=s.pinned,
                archived_at=s.archived_at,
                state=dict(s.state) if s.state else None,
                summary=s.summary or "",
            )

    def rename(self, session_id: str, owner_username: str, title: str) -> SessionRecord | None:
        title = (title or "").strip() or "New chat"
        with self._state.lock:
            s = self._state.sessions.get(session_id)
            if s is None or s.owner_username != owner_username:
                return None
            s.title = title[:200]
            s.updated_at = _now_iso()
        return self.get(session_id, owner_username)

    def set_pinned(
        self, session_id: str, owner_username: str, pinned: bool
    ) -> SessionRecord | None:
        with self._state.lock:
            s = self._state.sessions.get(session_id)
            if s is None or s.owner_username != owner_username:
                return None
            s.pinned = bool(pinned)
            if pinned:
                s.archived_at = None
            s.updated_at = _now_iso()
        return self.get(session_id, owner_username)

    def set_archived(
        self, session_id: str, owner_username: str, archived: bool
    ) -> SessionRecord | None:
        with self._state.lock:
            s = self._state.sessions.get(session_id)
            if s is None or s.owner_username != owner_username:
                return None
            if archived:
                s.archived_at = _now_iso()
                s.pinned = False
            else:
                s.archived_at = None
            s.updated_at = _now_iso()
        return self.get(session_id, owner_username)

    def save_state(
        self,
        session_id: str,
        owner_username: str,
        state: dict | None,
        summary: str | None = None,
    ) -> SessionRecord | None:
        with self._state.lock:
            s = self._state.sessions.get(session_id)
            if s is None or s.owner_username != owner_username:
                return None
            s.state = dict(state) if state else None
            if summary is not None:
                s.summary = summary
            s.archived_at = None
            s.updated_at = _now_iso()
        return self.get(session_id, owner_username)

    def delete(self, session_id: str, owner_username: str) -> bool:
        with self._state.lock:
            s = self._state.sessions.get(session_id)
            if s is None or s.owner_username != owner_username:
                return False
            del self._state.sessions[session_id]
            self._state.messages.pop(session_id, None)
            return True

    def list_messages(
        self, session_id: str, owner_username: str, limit: int = 500
    ) -> list[MessageRecord] | None:
        if self.get(session_id, owner_username) is None:
            return None
        limit = max(1, min(int(limit), 2000))
        with self._state.lock:
            return list(self._state.messages.get(session_id, [])[:limit])

    def append_message(
        self,
        session_id: str,
        owner_username: str,
        role: str,
        content: str,
        *,
        answer_source: str | None = None,
        web_sources: list | None = None,
        auto_title: bool = False,
    ) -> MessageRecord | None:
        if role not in _ROLES:
            raise ValueError(f"Invalid role: {role}")
        with self._state.lock:
            s = self._state.sessions.get(session_id)
            if s is None or s.owner_username != owner_username:
                return None
            self._state._msg_seq += 1
            msg = MessageRecord(
                id=self._state._msg_seq,
                session_id=session_id,
                role=role,
                content=content or "",
                answer_source=answer_source,
                web_sources=list(web_sources) if web_sources else None,
                created_at=_now_iso(),
            )
            self._state.messages.setdefault(session_id, []).append(msg)
            s.updated_at = _now_iso()
            s.archived_at = None
            s.message_count = len(self._state.messages[session_id])
            if auto_title and role == "user" and (
                not s.title or s.title == "New chat"
            ):
                s.title = title_from_question(content)
            return msg

    def reset_for_tests(self) -> None:
        with self._state.lock:
            self._state.sessions.clear()
            self._state.messages.clear()
            self._state._msg_seq = 0


_store: PostgresSessionStore | MemorySessionStore | None = None
_store_lock = Lock()


def get_session_store() -> PostgresSessionStore | MemorySessionStore:
    global _store
    with _store_lock:
        if _store is None:
            if not getattr(config, "SESSIONS_ENABLED", True):
                _store = MemorySessionStore()
                _store.ensure()
                return _store
            backend = (config.SESSIONS_BACKEND or "postgres").strip().lower()
            if backend == "memory":
                _store = MemorySessionStore()
                _store.ensure()
            else:
                try:
                    s = PostgresSessionStore()
                    s.ensure()
                    _store = s
                except Exception as e:
                    logger.warning(
                        f"Chat sessions postgres backend unavailable ({e}); "
                        "using memory store."
                    )
                    _store = MemorySessionStore()
                    _store.ensure()
        return _store


def reset_session_store_for_tests() -> None:
    global _store
    with _store_lock:
        if _store is not None:
            try:
                _store.reset_for_tests()
            except Exception:
                pass
        _store = MemorySessionStore()
