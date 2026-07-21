"""Rate-limiting tests:
  - the RateLimiter sliding-window primitive
  - the app.rate_limit FastAPI dependency factory (429, keying, toggle)
  - one live end-to-end 429 through the cheap /feedback endpoint

The autouse conftest fixture disables rate limiting for the suite; the
endpoint test here re-enables it locally so it can observe throttling.
"""

import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import rate_limit
from app.main import app
from app.services.rate_limiter import RateLimiter

client = TestClient(app)


# ─── RateLimiter primitive ──────────────────────────────────────────────

def test_rate_limiter_allows_up_to_max():
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        allowed, _ = limiter.check("k")
        assert allowed is True
    allowed, retry_after = limiter.check("k")
    assert allowed is False
    assert retry_after > 0


def test_rate_limiter_window_slides():
    limiter = RateLimiter(max_requests=1, window_seconds=0.2)
    assert limiter.check("k")[0] is True
    assert limiter.check("k")[0] is False
    time.sleep(0.35)
    assert limiter.check("k")[0] is True


def test_rate_limiter_is_per_key():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.check("a")[0] is True
    assert limiter.check("b")[0] is True   # different key unaffected
    assert limiter.check("a")[0] is False


def test_rate_limiter_reset():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.check("k")
    assert limiter.check("k")[0] is False
    limiter.reset()
    assert limiter.check("k")[0] is True


def test_rate_limiter_rejects_bad_config():
    with pytest.raises(ValueError):
        RateLimiter(max_requests=0, window_seconds=60)
    with pytest.raises(ValueError):
        RateLimiter(max_requests=5, window_seconds=0)


def test_rate_limiter_sweeps_stale_keys():
    limiter = RateLimiter(max_requests=10, window_seconds=0.05)
    limiter._sweep_every = 1  # force a sweep on the next call
    limiter.check("idle-key")
    time.sleep(0.15)  # past 2x window
    limiter.check("trigger-sweep")
    assert "idle-key" not in limiter._hits


# ─── Dependency factory ─────────────────────────────────────────────────

class _FakeClient:
    host = "1.2.3.4"


class _FakeRequest:
    client = _FakeClient()


def _call(dep, user="u"):
    """Invoke a factory-built dependency the way FastAPI would, with a
    fake request + resolved user."""
    return dep(request=_FakeRequest(), user=user)


def test_factory_dependency_blocks_after_threshold():
    rate_limit.set_enabled(True)
    dep = rate_limit.rate_limit("unit-test-a", max_requests=2, window_seconds=60)
    _call(dep)   # 1
    _call(dep)   # 2
    with pytest.raises(HTTPException) as exc:
        _call(dep)  # 3 → blocked
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_factory_dependency_keys_by_user_and_ip():
    rate_limit.set_enabled(True)
    dep = rate_limit.rate_limit("unit-test-b", max_requests=1, window_seconds=60)
    _call(dep, user="alice")           # ok
    _call(dep, user="bob")             # different user → ok
    with pytest.raises(HTTPException):
        _call(dep, user="alice")       # alice again → blocked


def test_factory_dependency_respects_global_toggle():
    dep = rate_limit.rate_limit("unit-test-c", max_requests=1, window_seconds=60)
    rate_limit.set_enabled(False)
    for _ in range(10):
        assert _call(dep) is None  # never raises while disabled


# ─── Live endpoint (cheap /feedback path, no LLM) ───────────────────────

def test_feedback_endpoint_returns_429_when_throttled(tmp_path, monkeypatch):
    """End-to-end: exhaust the real feedback budget (default 20/60s) and
    confirm the endpoint starts returning 429 with a Retry-After header.
    Cheap to exercise — feedback is a disk append, no LLM."""
    monkeypatch.setenv("MISA_FEEDBACK_LOG", str(tmp_path / "fb.jsonl"))
    rate_limit.set_enabled(True)
    rate_limit.reset_all()

    from app.config import FEEDBACK_RATE_LIMIT
    budget = int(FEEDBACK_RATE_LIMIT[0])
    payload = {"verdict": "up", "question": "q", "answer": "a"}

    responses = [
        client.post("/api/v1/feedback", json=payload)
        for _ in range(budget + 3)
    ]
    statuses = [r.status_code for r in responses]
    assert statuses[:budget] == [200] * budget          # all within budget pass
    assert statuses[budget] == 429                        # first over-budget blocked

    # The 429 body follows the app's standard error shape, not FastAPI's
    # bare {"detail": "..."} — same contract as every other endpoint error.
    blocked_body = responses[budget].json()
    assert blocked_body["success"] is False
    assert blocked_body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert isinstance(blocked_body["error"]["message"], str)
    assert blocked_body["status"] == 429
    assert blocked_body["path"] == "/api/v1/feedback"
    assert isinstance(blocked_body["retry_after_seconds"], int)
    assert "Retry-After" in responses[budget].headers

    blocked = client.post("/api/v1/feedback", json=payload)
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers
