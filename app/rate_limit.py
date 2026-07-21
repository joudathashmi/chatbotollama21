"""
FastAPI rate-limit dependencies — the web-layer glue over the pure
`RateLimiter` primitive (app/services/rate_limiter.py).

One place to build per-endpoint limiters so every route throttles the
same way: keyed by authenticated-user + client IP, honoring a global
enable switch, returning HTTP 429 with a `Retry-After` header.

Why user+IP and not just the user: all clients currently share one
Basic Auth credential, so keying on username alone would let a single
abusive source burn the whole quota for every legitimate user. Adding
the IP contains each source's abuse to itself. (When SSO lands and each
caller has a distinct identity, the IP component can be dropped.)

Usage (build ONCE at import, not per request):

    from app.rate_limit import rate_limit
    from app.config import CHAT_RATE_LIMIT
    _chat_rl = rate_limit("chat", *CHAT_RATE_LIMIT)

    @router.post("/chat", dependencies=[Depends(_chat_rl)])
    async def chat_endpoint(...): ...
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from app.auth import verify_credentials
from app.config import RATE_LIMIT_ENABLED
from app.services.rate_limiter import build_rate_limiter


class RateLimitExceeded(HTTPException):
    """429 with the extra fields `app/exception_handlers.py`'s global
    HTTPException handler needs to render a structured, machine-
    readable body (error.code + top-level retry_after_seconds) instead
    of FastAPI's bare `{"detail": "..."}`. Still a plain HTTPException
    under the hood, so any code path that bypasses the global handler
    degrades to a normal 429 rather than breaking."""

    def __init__(self, message: str, retry_after_seconds: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message,
            headers={"Retry-After": str(retry_after_seconds)},
        )
        self.error_code = "RATE_LIMIT_EXCEEDED"
        self.retry_after_seconds = retry_after_seconds

# Runtime-mutable enable flag (starts from config; tests flip it off).
_enabled: bool = RATE_LIMIT_ENABLED

# name → limiter, so tests and ops can reset/inspect a known set.
# Value may be an in-memory RateLimiter or a RedisRateLimiter depending on
# MISA_RATE_LIMIT_BACKEND; both share the same check()/reset() interface.
_registry: dict[str, object] = {}


def set_enabled(flag: bool) -> None:
    """Toggle all rate limiting at runtime. Tests disable it so the
    suite isn't throttled; ops can disable it behind a gateway that
    already throttles."""
    global _enabled
    _enabled = bool(flag)


def is_enabled() -> bool:
    return _enabled


def reset_all() -> None:
    """Clear every limiter's recorded hits. Test isolation only."""
    for limiter in _registry.values():
        limiter.reset()


def _client_key(request: Request, user: str) -> str:
    client_ip = request.client.host if request.client else "unknown"
    return f"{user}:{client_ip}"


def rate_limit(name: str, max_requests: int, window_seconds: float):
    """Build a FastAPI dependency enforcing `max_requests` per
    `window_seconds` per (user + IP) for one endpoint. Call once at
    import time and reuse the returned callable."""
    limiter = build_rate_limiter(max_requests, window_seconds)
    _registry[name] = limiter

    def _dependency(
        request: Request,
        user: str = Depends(verify_credentials),
    ) -> None:
        if not _enabled:
            return
        allowed, retry_after = limiter.check(_client_key(request, user))
        if not allowed:
            raise RateLimitExceeded(
                message=(
                    f"Rate limit exceeded — max {max_requests} requests per "
                    f"{int(window_seconds)}s for this endpoint. Try again shortly."
                ),
                retry_after_seconds=int(retry_after) + 1,
            )

    _dependency.__name__ = f"rate_limit_{name}"
    return _dependency


def rate_limit_ip(name: str, max_requests: int, window_seconds: float):
    """Like `rate_limit`, but keyed by client IP ALONE — for pre-auth
    endpoints (login/refresh) where no authenticated user exists yet, so
    `verify_credentials` can't be a sub-dependency. Blunts credential
    brute-forcing from a single source."""
    limiter = build_rate_limiter(max_requests, window_seconds)
    _registry[name] = limiter

    def _dependency(request: Request) -> None:
        if not _enabled:
            return
        client_ip = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.check(client_ip)
        if not allowed:
            raise RateLimitExceeded(
                message=(
                    f"Rate limit exceeded — max {max_requests} requests per "
                    f"{int(window_seconds)}s for this endpoint. Try again shortly."
                ),
                retry_after_seconds=int(retry_after) + 1,
            )

    _dependency.__name__ = f"rate_limit_ip_{name}"
    return _dependency
