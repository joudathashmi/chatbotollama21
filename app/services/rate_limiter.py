"""
In-memory sliding-window rate limiter.

Guards abuse-prone, low-value-per-call endpoints (e.g. feedback
submission) against unbounded scripted requests from a single client.
Not a general request-throttling framework — deliberately small and
dependency-free so it can be reasoned about at a glance.

Design:
  - Sliding window per key (not fixed-bucket), so a client can't burst
    right at a window boundary to get ~2x the intended rate.
  - `deque` per key holding recent-hit timestamps; O(1) amortized
    check (only trims events older than the window off the front).
  - A single `Lock` guards the whole store. Feedback-class endpoints
    are not hot paths (unlike /chat), so this is not a contention risk
    in practice; if it ever becomes one, shard by key hash first
    rather than reaching for a heavier dependency.
  - Stale keys (no hits within 2x the window) are swept on every Nth
    call so idle clients don't leak memory forever. This is a process-
    local store: restarting the app clears all counters, and it does
    NOT share state across multiple worker processes/replicas. If this
    is ever horizontally scaled behind a load balancer, replace the
    store with Redis (INCR + EXPIRE, or a sorted-set sliding window) —
    the `RateLimiter.check()` call signature below is designed to stay
    the same across that swap.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from threading import Lock

from app.logger import logger


class RateLimiter:
    """Sliding-window limiter: at most `max_requests` per `window_seconds`
    per key."""

    def __init__(self, max_requests: int, window_seconds: float):
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()
        self._calls_since_sweep = 0
        # Sweep stale keys every N calls rather than on a timer thread —
        # keeps this module free of background-thread lifecycle concerns.
        self._sweep_every = 500

    def check(self, key: str) -> tuple[bool, float]:
        """Record a hit attempt for `key`. Returns (allowed, retry_after_sec).

        If allowed is False, retry_after_sec is how long the caller
        should wait before the oldest hit in the window expires.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            dq = self._hits.setdefault(key, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()

            if len(dq) >= self.max_requests:
                retry_after = dq[0] + self.window_seconds - now
                return False, max(retry_after, 0.0)

            dq.append(now)

            self._calls_since_sweep += 1
            if self._calls_since_sweep >= self._sweep_every:
                self._calls_since_sweep = 0
                self._sweep_stale(now)

            return True, 0.0

    def reset(self) -> None:
        """Drop all recorded hits. Intended for test isolation — not
        used on the hot path."""
        with self._lock:
            self._hits.clear()
            self._calls_since_sweep = 0

    def _sweep_stale(self, now: float) -> None:
        """Drop keys with no activity in the last 2x the window, so a
        long-running process doesn't accumulate one deque per ever-seen
        client forever. Caller already holds `self._lock`."""
        stale_cutoff = now - (2 * self.window_seconds)
        stale_keys = [
            k for k, dq in self._hits.items()
            if not dq or dq[-1] < stale_cutoff
        ]
        for k in stale_keys:
            del self._hits[k]


# ---------------------------------------------------------------------------
# Redis-backed sliding window (shared across workers / replicas)
# ---------------------------------------------------------------------------

# Atomic sliding-window check in a single round-trip. Uses a per-key sorted
# set of hit timestamps (score = ms epoch); trims expired entries, counts
# what's left, and either rejects (returning ms until the oldest hit ages
# out) or records the new hit and sets a TTL so idle keys self-expire.
_REDIS_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local maxr = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count >= maxr then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local retry = 0
  if oldest[2] then
    retry = (tonumber(oldest[2]) + window) - now
  end
  return {0, retry}
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, window)
return {1, 0}
"""


class RedisRateLimiter:
    """Sliding-window limiter backed by Redis, so every worker/replica
    shares one counter per key. Drop-in for `RateLimiter`: same
    `check(key) -> (allowed, retry_after_sec)` and `reset()` surface.

    Resilience: if a Redis call fails (store down, network blip), the
    limiter transparently falls back to a process-local in-memory window
    so SOME throttling survives instead of the endpoint erroring out.
    """

    def __init__(self, max_requests: int, window_seconds: float, client, key_prefix: str = "misa:rl"):
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._client = client
        self._prefix = key_prefix
        self._script = client.register_script(_REDIS_SLIDING_WINDOW_LUA)
        # Local fallback used only when Redis is unreachable at check time.
        self._fallback = RateLimiter(max_requests, window_seconds)
        self._warned = False

    def _rkey(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def check(self, key: str) -> tuple[bool, float]:
        now_ms = int(time.time() * 1000)
        window_ms = int(self.window_seconds * 1000)
        member = f"{now_ms}:{uuid.uuid4().hex}"
        try:
            allowed, retry_ms = self._script(
                keys=[self._rkey(key)],
                args=[now_ms, window_ms, self.max_requests, member],
            )
            allowed = bool(allowed)
            retry_after = max(float(retry_ms) / 1000.0, 0.0)
            # Guarantee a positive Retry-After when throttled: the exact
            # value depends on the store returning the oldest hit's score,
            # so fall back to the full window (a safe over-estimate) if it
            # comes back as 0.
            if not allowed and retry_after <= 0:
                retry_after = self.window_seconds
            return allowed, retry_after
        except Exception as e:
            if not self._warned:
                logger.warning(
                    f"Redis rate limiter unavailable ({e}); "
                    f"falling back to in-memory throttling for this process."
                )
                self._warned = True
            return self._fallback.check(key)

    def reset(self) -> None:
        """Clear this limiter's keys. Best-effort; test isolation only."""
        try:
            for k in self._client.scan_iter(match=f"{self._prefix}:*"):
                self._client.delete(k)
        except Exception:
            pass
        self._fallback.reset()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rate_limiter(
    max_requests: int,
    window_seconds: float,
    *,
    backend: str | None = None,
    redis_url: str | None = None,
    key_prefix: str | None = None,
    client=None,
):
    """Construct the configured limiter. Defaults come from app.config but
    can be overridden (used by tests to inject a fake Redis client).

    Returns a `RateLimiter` (in-memory) or `RedisRateLimiter`. Any failure
    to set up Redis logs a warning and degrades to in-memory so the app
    always starts."""
    from app.config import RATE_LIMIT_BACKEND, REDIS_URL, RATE_LIMIT_REDIS_PREFIX

    backend = (backend or RATE_LIMIT_BACKEND or "memory").strip().lower()
    if backend != "redis" and client is None:
        return RateLimiter(max_requests, window_seconds)

    prefix = key_prefix or RATE_LIMIT_REDIS_PREFIX
    try:
        if client is None:
            import redis  # lazy: only needed when the redis backend is on
            client = redis.Redis.from_url(
                redis_url or REDIS_URL,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
        return RedisRateLimiter(max_requests, window_seconds, client, key_prefix=prefix)
    except Exception as e:
        logger.warning(
            f"Rate-limit backend 'redis' unavailable ({e}); "
            f"using in-memory limiter instead."
        )
        return RateLimiter(max_requests, window_seconds)
