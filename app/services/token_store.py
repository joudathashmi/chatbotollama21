"""Refresh-token store with rotation + reuse detection.

Refresh JWTs carry a `jti` (token id) and an opaque `fid` (family id).
On each successful refresh the old jti is invalidated and a new one is
registered under the same family. Presenting an already-rotated (or
unknown) jti is treated as theft: the entire family is revoked so both
the attacker and the legitimate client must re-authenticate.

Backends:
  - memory (default) — process-local; fine for single-worker / tests
  - redis — shared across workers (same URL as rate limiting)
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

from app.logger import logger


class MemoryRefreshTokenStore:
    """In-process jti → {sub, fid, exp} map with family revocation."""

    def __init__(self) -> None:
        self._lock = Lock()
        # jti → {"sub": str, "fid": str, "exp": float}
        self._live: dict[str, dict[str, Any]] = {}
        # fid → set of jtis (including revoked) so reuse can find the family
        self._families: dict[str, set[str]] = {}
        # jtis known to have been rotated/revoked (reuse detection)
        self._used: set[str] = set()

    def register(self, jti: str, *, sub: str, family_id: str, exp: float) -> None:
        with self._lock:
            self._live[jti] = {"sub": sub, "fid": family_id, "exp": exp}
            self._families.setdefault(family_id, set()).add(jti)

    def rotate(
        self,
        old_jti: str,
        new_jti: str,
        *,
        sub: str,
        family_id: str,
        exp: float,
    ) -> bool:
        """Invalidate old_jti and register new_jti. Returns False on reuse
        or unknown token (caller must revoke the family + 401)."""
        with self._lock:
            now = time.time()
            if old_jti in self._used:
                return False
            meta = self._live.get(old_jti)
            if meta is None:
                return False
            if meta["sub"] != sub or meta["fid"] != family_id:
                return False
            if float(meta["exp"]) < now:
                self._live.pop(old_jti, None)
                return False
            self._live.pop(old_jti, None)
            self._used.add(old_jti)
            self._live[new_jti] = {"sub": sub, "fid": family_id, "exp": exp}
            self._families.setdefault(family_id, set()).add(new_jti)
            return True

    def revoke_family(self, family_id: str) -> None:
        with self._lock:
            jtis = self._families.pop(family_id, set())
            for jti in jtis:
                self._live.pop(jti, None)
                self._used.add(jti)

    def reset(self) -> None:
        """Test helper — clear all state."""
        with self._lock:
            self._live.clear()
            self._families.clear()
            self._used.clear()


class RedisRefreshTokenStore:
    """Redis-backed refresh store. Keys:
      {prefix}:jti:{jti}  → JSON {sub,fid,exp} with TTL
      {prefix}:used:{jti} → "1" with TTL (reuse marker)
      {prefix}:fam:{fid}  → set of jtis
    """

    def __init__(self, client, key_prefix: str = "misa:rt") -> None:
        self._client = client
        self._prefix = key_prefix
        self._fallback = MemoryRefreshTokenStore()
        self._warned = False

    def _jkey(self, jti: str) -> str:
        return f"{self._prefix}:jti:{jti}"

    def _ukey(self, jti: str) -> str:
        return f"{self._prefix}:used:{jti}"

    def _fkey(self, fid: str) -> str:
        return f"{self._prefix}:fam:{fid}"

    def _ttl(self, exp: float) -> int:
        return max(1, int(exp - time.time()) + 60)

    def register(self, jti: str, *, sub: str, family_id: str, exp: float) -> None:
        import json
        try:
            pipe = self._client.pipeline()
            pipe.set(self._jkey(jti), json.dumps({"sub": sub, "fid": family_id, "exp": exp}), ex=self._ttl(exp))
            pipe.sadd(self._fkey(family_id), jti)
            pipe.expire(self._fkey(family_id), self._ttl(exp))
            pipe.execute()
        except Exception as e:
            self._warn(e)
            self._fallback.register(jti, sub=sub, family_id=family_id, exp=exp)

    def rotate(
        self,
        old_jti: str,
        new_jti: str,
        *,
        sub: str,
        family_id: str,
        exp: float,
    ) -> bool:
        import json
        try:
            if self._client.exists(self._ukey(old_jti)):
                return False
            raw = self._client.get(self._jkey(old_jti))
            if not raw:
                return False
            meta = json.loads(raw)
            if meta.get("sub") != sub or meta.get("fid") != family_id:
                return False
            if float(meta.get("exp") or 0) < time.time():
                self._client.delete(self._jkey(old_jti))
                return False
            pipe = self._client.pipeline()
            pipe.delete(self._jkey(old_jti))
            pipe.set(self._ukey(old_jti), "1", ex=self._ttl(float(meta["exp"])))
            pipe.set(
                self._jkey(new_jti),
                json.dumps({"sub": sub, "fid": family_id, "exp": exp}),
                ex=self._ttl(exp),
            )
            pipe.sadd(self._fkey(family_id), new_jti)
            pipe.expire(self._fkey(family_id), self._ttl(exp))
            pipe.execute()
            return True
        except Exception as e:
            self._warn(e)
            return self._fallback.rotate(
                old_jti, new_jti, sub=sub, family_id=family_id, exp=exp
            )

    def revoke_family(self, family_id: str) -> None:
        try:
            members = self._client.smembers(self._fkey(family_id)) or set()
            pipe = self._client.pipeline()
            for jti in members:
                jti_s = jti.decode() if isinstance(jti, bytes) else str(jti)
                pipe.delete(self._jkey(jti_s))
                pipe.set(self._ukey(jti_s), "1", ex=max(1, 7 * 86400))
            pipe.delete(self._fkey(family_id))
            pipe.execute()
        except Exception as e:
            self._warn(e)
            self._fallback.revoke_family(family_id)

    def reset(self) -> None:
        try:
            for k in self._client.scan_iter(match=f"{self._prefix}:*"):
                self._client.delete(k)
        except Exception:
            pass
        self._fallback.reset()

    def _warn(self, e: Exception) -> None:
        if not self._warned:
            logger.warning(
                f"Redis refresh-token store unavailable ({e}); "
                f"falling back to in-memory store for this process."
            )
            self._warned = True


_store: MemoryRefreshTokenStore | RedisRefreshTokenStore | None = None


def build_refresh_token_store(*, client=None, backend: str | None = None):
    """Construct the configured refresh-token store (memory or redis)."""
    from app.config import RATE_LIMIT_BACKEND, REDIS_URL, RATE_LIMIT_REDIS_PREFIX

    backend = (backend or RATE_LIMIT_BACKEND or "memory").strip().lower()
    if backend != "redis" and client is None:
        return MemoryRefreshTokenStore()
    prefix = f"{RATE_LIMIT_REDIS_PREFIX}:rt"
    try:
        if client is None:
            import redis
            client = redis.Redis.from_url(
                REDIS_URL,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
        return RedisRefreshTokenStore(client, key_prefix=prefix)
    except Exception as e:
        logger.warning(
            f"Refresh-token backend 'redis' unavailable ({e}); using in-memory store."
        )
        return MemoryRefreshTokenStore()


def get_refresh_token_store():
    global _store
    if _store is None:
        _store = build_refresh_token_store()
    return _store


def reset_refresh_token_store_for_tests() -> None:
    """Reset singleton + clear contents (tests only)."""
    global _store
    if _store is not None:
        _store.reset()
    _store = MemoryRefreshTokenStore()
