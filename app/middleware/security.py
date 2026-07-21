"""HTTP-layer security middleware.

Two independent concerns, kept small and dependency-free:

  1. SecurityHeadersMiddleware — stamps standard hardening headers
     (nosniff, frame-deny, CSP, referrer/permissions policy, optional HSTS)
     on every response, and `Cache-Control: no-store` on API responses so
     sensitive JSON is never cached by browsers/proxies. Existing per-route
     headers (SSE / PDF set their own Cache-Control) are preserved.

  2. MaxBodySizeMiddleware — rejects requests whose body exceeds the
     configured ceiling with a standard 413. When Content-Length is present
     it rejects before buffering; when absent (chunked), it counts streamed
     bytes so the cap cannot be bypassed by omitting the header.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app import config
from app.utils.error_handler import create_error_response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        h = response.headers

        # setdefault so a route that deliberately set a header wins.
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "no-referrer")
        h.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=(), payment=(), usb=()",
        )
        h.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        h.setdefault("X-Permitted-Cross-Domain-Policies", "none")
        if config.CONTENT_SECURITY_POLICY:
            h.setdefault("Content-Security-Policy", config.CONTENT_SECURITY_POLICY)

        # HSTS only over TLS (ignored by browsers on plain HTTP anyway); gated
        # so a dev/proxy-terminated HTTP box doesn't advertise it spuriously.
        if config.HSTS_ENABLED:
            h.setdefault(
                "Strict-Transport-Security",
                f"max-age={config.HSTS_MAX_AGE}; includeSubDomains",
            )

        # Sensitive API JSON must never be cached.
        if request.url.path.startswith("/api/") and "cache-control" not in h:
            h["Cache-Control"] = "no-store"
            h.setdefault("Pragma", "no-cache")

        return response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = int(max_bytes)

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self.max_bytes:
                    return self._too_large(request)
            except ValueError:
                pass
            # Declared size is within the cap — let the framework stream
            # normally. (Re-reading the body here deadlocks TestClient /
            # BaseHTTPMiddleware; CL covers the common path.)
            return await call_next(request)

        # No Content-Length (chunked / streaming). Count bytes as they arrive
        # so omitting CL cannot bypass the ceiling.
        if request.method in ("GET", "HEAD", "OPTIONS", "TRACE"):
            return await call_next(request)

        received = 0
        chunks: list[bytes] = []

        while True:
            message = await request.receive()
            if message["type"] != "http.request":
                break
            chunk = message.get("body", b"") or b""
            received += len(chunk)
            if received > self.max_bytes:
                return self._too_large(request)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)
        return await call_next(request)

    def _too_large(self, request: Request) -> JSONResponse:
        body = create_error_response(
            code="PAYLOAD_TOO_LARGE",
            message="Request body exceeds the maximum allowed size.",
            status=413,
            path=str(request.url.path),
        ).model_dump()
        return JSONResponse(status_code=413, content=body)
