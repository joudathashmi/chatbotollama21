"""Scalability tests for the changes on the concurrency track:

  1. Thread-local Postgres connections (app/database.py) — each worker
     thread gets its own connection, reused within the thread, and
     close_all_db_connections() tears them all down. Verified with a fake
     connection factory so no real database is required.

  2. The Redis-backed rate limiter (app/services/rate_limiter.py) — shared
     sliding-window semantics, exercised against fakeredis, plus the
     build_rate_limiter() factory selection and graceful fallback to the
     in-memory limiter when Redis is unavailable.
"""

import threading

import pytest

import app.database as database
from app.services.rate_limiter import (
    RateLimiter,
    RedisRateLimiter,
    build_rate_limiter,
)


# ─── Thread-local DB connections ────────────────────────────────────────

class _FakeConn:
    """Minimal stand-in for a psycopg2 connection."""
    _counter = 0

    def __init__(self):
        type(self)._counter += 1
        self.id = type(self)._counter
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def _fake_pg(monkeypatch):
    """Patch the connection factory so get_db() hands out FakeConns and
    reset the module's per-thread + registry state around each test."""
    monkeypatch.setattr(database, "_connect_pg_with_retry", lambda: _FakeConn())
    database.close_all_db_connections()
    database._tls = threading.local()
    yield
    database.close_all_db_connections()
    database._tls = threading.local()


def test_get_db_reuses_same_connection_within_thread(_fake_pg):
    c1 = database.get_db()
    c2 = database.get_db()
    assert c1 is c2  # one connection per thread, reused


def test_get_db_gives_distinct_connections_per_thread(_fake_pg):
    main_conn = database.get_db()
    other = {}

    def worker():
        other["conn"] = database.get_db()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert other["conn"] is not None
    assert other["conn"] is not main_conn  # separate thread -> separate conn


def test_close_all_db_connections_closes_and_clears(_fake_pg):
    conn = database.get_db()
    assert conn.closed is False
    database.close_all_db_connections()
    assert conn.closed is True
    # After teardown a fresh connection is issued (not the closed one).
    new_conn = database.get_db()
    assert new_conn is not conn
    assert new_conn.closed is False


def test_invalidate_db_cache_only_affects_current_thread(_fake_pg):
    conn = database.get_db()
    database._invalidate_db_cache()
    assert conn.closed is True
    replacement = database.get_db()
    assert replacement is not conn


# ─── Redis-backed rate limiter (via fakeredis) ──────────────────────────

@pytest.fixture
def fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeStrictRedis()


def test_redis_limiter_allows_up_to_max_then_blocks(fake_redis):
    limiter = RedisRateLimiter(max_requests=3, window_seconds=60, client=fake_redis)
    for _ in range(3):
        allowed, _ = limiter.check("user:ip")
        assert allowed is True
    allowed, retry_after = limiter.check("user:ip")
    assert allowed is False
    assert retry_after > 0


def test_redis_limiter_is_per_key(fake_redis):
    limiter = RedisRateLimiter(max_requests=1, window_seconds=60, client=fake_redis)
    assert limiter.check("a")[0] is True
    assert limiter.check("a")[0] is False
    assert limiter.check("b")[0] is True  # different key, own window


def test_redis_limiter_shared_across_instances(fake_redis):
    """Two limiter instances (simulating two worker processes) sharing one
    Redis must enforce ONE combined window — the whole point of the backend."""
    w1 = RedisRateLimiter(max_requests=2, window_seconds=60, client=fake_redis)
    w2 = RedisRateLimiter(max_requests=2, window_seconds=60, client=fake_redis)
    assert w1.check("k")[0] is True
    assert w2.check("k")[0] is True
    # Third hit across the two "workers" exceeds the shared limit of 2.
    assert w1.check("k")[0] is False
    assert w2.check("k")[0] is False


def test_redis_limiter_reset(fake_redis):
    limiter = RedisRateLimiter(max_requests=1, window_seconds=60, client=fake_redis)
    assert limiter.check("k")[0] is True
    assert limiter.check("k")[0] is False
    limiter.reset()
    assert limiter.check("k")[0] is True


def test_redis_limiter_falls_back_when_client_errors():
    """If Redis raises at check time, the limiter degrades to its embedded
    in-memory window rather than erroring the request."""
    class _BrokenClient:
        def register_script(self, _lua):
            def _run(*a, **k):
                raise ConnectionError("redis down")
            return _run

    limiter = RedisRateLimiter(max_requests=1, window_seconds=60, client=_BrokenClient())
    # Falls back to in-memory RateLimiter: first allowed, second blocked.
    assert limiter.check("k")[0] is True
    assert limiter.check("k")[0] is False


# ─── Factory selection + fallback ───────────────────────────────────────

def test_factory_defaults_to_memory():
    limiter = build_rate_limiter(5, 60, backend="memory")
    assert isinstance(limiter, RateLimiter)


def test_factory_redis_with_injected_client(fake_redis):
    limiter = build_rate_limiter(5, 60, backend="redis", client=fake_redis)
    assert isinstance(limiter, RedisRateLimiter)


def test_factory_redis_unreachable_falls_back_to_memory():
    # No client injected + an unroutable URL → connection fails → memory.
    limiter = build_rate_limiter(
        5, 60, backend="redis", redis_url="redis://127.0.0.1:1/0"
    )
    assert isinstance(limiter, RateLimiter)
