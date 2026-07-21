"""DB retry, cache invalidation, and structured turn logging."""

import json

import psycopg2
import pytest

from app import database as mc
from app.services import chat_engine as ce


def test_connect_pg_retry_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _StubConn:
        # Stand-in for a psycopg2 connection: we just need a writeable
        # `autocommit` attribute since _connect_pg_with_retry sets it on
        # every fresh connection (autocommit-on prevents the "current
        # transaction is aborted" cascade across multi-tool-call turns).
        autocommit = False

    def fake_connect(**kw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise psycopg2.OperationalError("connection reset")
        return _StubConn()

    monkeypatch.setattr(mc.psycopg2, "connect", fake_connect)
    monkeypatch.setenv("PG_CONNECT_RETRIES", "3")
    monkeypatch.setenv("PG_RETRY_DELAY_SEC", "0")
    conn = mc._connect_pg_with_retry()
    assert conn is not None
    assert conn.autocommit is True
    assert calls["n"] == 2


def test_connect_pg_exhausts_retries(monkeypatch):
    def boom(**kw):
        raise psycopg2.OperationalError("down")

    monkeypatch.setattr(mc.psycopg2, "connect", boom)
    monkeypatch.setenv("PG_CONNECT_RETRIES", "2")
    monkeypatch.setenv("PG_RETRY_DELAY_SEC", "0")
    with pytest.raises(psycopg2.OperationalError):
        mc._connect_pg_with_retry()


def test_run_with_db_transient_retry_invalidates(monkeypatch):
    class _FakeCursor:
        def __init__(self, _conn):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, sql, params=None):
            pass

    class _FakeConn:
        def __init__(self, tag):
            self.tag = tag

        def cursor(self):
            return _FakeCursor(self)

    c1, c2 = _FakeConn(1), _FakeConn(2)
    seq = iter([c1, c2])

    def fake_get_db():
        return next(seq)

    def work(conn):
        if conn.tag == 1:
            raise psycopg2.OperationalError("stale")
        return "ok"

    monkeypatch.setattr(mc, "get_db", fake_get_db)
    assert mc._run_with_db_transient_retry(work) == "ok"


def test_run_with_db_transient_retry_ping_invalidates(monkeypatch):
    class _FakeCursor:
        def __init__(self, conn):
            self._conn = conn

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def execute(self, sql, params=None):
            if "SELECT 1" in (sql or "") and self._conn.fail_ping:
                raise psycopg2.OperationalError("gone")

    class _FakeConn:
        def __init__(self, tag, fail_ping=False):
            self.tag = tag
            self.fail_ping = fail_ping

        def cursor(self):
            return _FakeCursor(self)

    c1 = _FakeConn(1, fail_ping=True)
    c2 = _FakeConn(2, fail_ping=False)
    seq = iter([c1, c2])

    def fake_get_db():
        return next(seq)

    def work(conn):
        assert conn.tag == 2
        return "pong"

    monkeypatch.setattr(mc, "get_db", fake_get_db)
    assert mc._run_with_db_transient_retry(work) == "pong"


def test_chat_completions_retry_then_ok(monkeypatch):
    from unittest.mock import MagicMock

    from openai import RateLimitError

    monkeypatch.setenv("OPENAI_MAX_RETRIES", "4")
    monkeypatch.setenv("OPENAI_RETRY_DELAY_SEC", "0")
    client = MagicMock()
    ok_resp = object()
    client.chat.completions.create.side_effect = [
        RateLimitError("nope", response=MagicMock(), body=None),
        ok_resp,
    ]
    out = ce._chat_completions_create_with_retry(client, model="x", messages=[])
    assert out is ok_resp
    assert client.chat.completions.create.call_count == 2


def test_openai_max_completion_tokens_kw_default(monkeypatch):
    monkeypatch.delenv("OPENAI_MAX_COMPLETION_TOKENS", raising=False)
    assert ce._openai_max_completion_tokens_kw() == {"max_completion_tokens": 3072}


def test_openai_max_completion_tokens_kw_disabled(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_COMPLETION_TOKENS", "0")
    assert ce._openai_max_completion_tokens_kw() == {}


def test_max_history_user_turns(monkeypatch):
    monkeypatch.setenv("MISA_MAX_HISTORY_USER_TURNS", "3")
    assert ce._max_history_user_turns() == 3


def test_chat_completions_non_retryable_raises(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("OPENAI_MAX_RETRIES", "4")
    monkeypatch.setenv("OPENAI_RETRY_DELAY_SEC", "0")
    client = MagicMock()
    client.chat.completions.create.side_effect = ValueError("bad")
    with pytest.raises(ValueError):
        ce._chat_completions_create_with_retry(client, model="x", messages=[])


def test_structured_turn_log_writes_file(tmp_path, monkeypatch, capsys):
    logf = tmp_path / "turns.jsonl"
    monkeypatch.setenv("MISA_LOG_TURNS", "true")
    monkeypatch.setenv("MISA_LOG_FILE", str(logf))
    ce._structured_turn_log({"event": "test", "x": 1})
    lines = logf.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "test"
    err = capsys.readouterr().err
    assert "test" in err


def test_structured_turn_log_skips_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("MISA_LOG_TURNS", raising=False)
    logf = tmp_path / "turns.jsonl"
    monkeypatch.setenv("MISA_LOG_FILE", str(logf))
    ce._structured_turn_log({"event": "x"})
    assert not logf.exists()
