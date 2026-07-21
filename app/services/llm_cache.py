"""
LRU + TTL cache for repeat-expensive LLM calls.

Three LLM calls fire on every chat turn that aren't worth re-running
for the same question text:

  - classify_intent(question, history)   ~ 2.5s
  - _extract_exec_target(question)       ~ 1.3s
  - _extract_country_from_question(q)    ~ 0.8s

Same question text on the next turn (or by another user in the same
session) should hit cache and skip the round-trip. Saves ~3-5s per
repeat turn — common during a working session when an executive
iterates on a question.

DESIGN
======
Per-process in-memory cache, thread-safe. Keys are hashes of the
input text + relevant context (history snippet for the classifier
since history can flip the intent classification). LRU eviction at
500 entries per cache. 10-minute TTL — long enough to cover a
working session, short enough that stale data can't outlive an
actual fact change.

The cache is purely a wrapper — the underlying functions are
untouched, callers opt in by using `cached_call(...)`. No magic.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable


_LOCKS: dict[str, Lock] = {}
_CACHES: dict[str, OrderedDict[str, tuple[float, Any]]] = {}
_TTL_SEC = 600.0   # 10 minutes
_MAX_ENTRIES = 500


def _cache_for(name: str) -> tuple[OrderedDict[str, tuple[float, Any]], Lock]:
    """Get-or-create the cache + lock for a named cache namespace."""
    if name not in _CACHES:
        _CACHES[name] = OrderedDict()
        _LOCKS[name] = Lock()
    return _CACHES[name], _LOCKS[name]


def _make_key(*parts: Any) -> str:
    """Stable hash of (possibly multi-part) cache key. Uses sha256
    truncated to 24 chars — collision-safe at our volume."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def cached_call(
    name: str,
    key_parts: tuple,
    fn: Callable,
    *args, **kwargs,
):
    """Look up (name, key_parts) in the named LRU cache. On hit,
    return the cached value (extending LRU position). On miss, call
    `fn(*args, **kwargs)`, store the result, and return it.

    Failures are NOT cached — if `fn` raises or returns None, the
    next call will retry.
    """
    cache, lock = _cache_for(name)
    key = _make_key(*key_parts)
    now = time.time()

    # Fast path: cache hit and not expired
    with lock:
        entry = cache.get(key)
        if entry is not None:
            expires_at, value = entry
            if now < expires_at:
                # LRU bump
                cache.move_to_end(key)
                return value
            else:
                cache.pop(key, None)

    # Cache miss — call through (outside lock to avoid blocking)
    result = fn(*args, **kwargs)
    if result is None:
        return result  # don't cache None / failure

    with lock:
        cache[key] = (now + _TTL_SEC, result)
        # LRU evict oldest
        while len(cache) > _MAX_ENTRIES:
            cache.popitem(last=False)
    return result


def clear_all() -> None:
    """For tests / admin: drop every cache."""
    for name, lock in _LOCKS.items():
        with lock:
            _CACHES[name].clear()


def stats() -> dict:
    """Return per-cache size for /debug visibility."""
    out: dict = {}
    for name, cache in _CACHES.items():
        lock = _LOCKS.get(name)
        if lock:
            with lock:
                out[name] = len(cache)
        else:
            out[name] = len(cache)
    return out
