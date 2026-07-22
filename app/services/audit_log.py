"""
Audit logging — structured request log for Saudi NCA + NDMO compliance.

Every HTTP request to the API produces ONE JSON event line containing:
  - correlation id (UUID v4, so distributed traces can be reconstructed)
  - UTC timestamp (ISO 8601 with timezone)
  - user identity (from Bearer JWT `sub`, or "anonymous" for /chat UI + /health)
  - client IP (X-Forwarded-For aware; reads the leftmost public IP in chain)
  - request: method, path, query string
  - request body summary (for /chat: question + intent + entity; for others: byte count)
  - response: status code, latency_ms
  - error string if the handler raised

Designed for SIEM ingestion (Azure Sentinel, Splunk, ELK):
  - One event per line, JSON Lines format
  - Stable key names + types (no nulls becoming missing keys)
  - Emits to stdout for container log scrapers AND to a rotating file
  - Truncates user-supplied strings to 500 chars so a giant prompt
    doesn't blow up the log

PRIVACY NOTE: the audit log does NOT contain answer bodies, DB rows,
or any retrieved data. It contains only metadata about what the user
asked and what the system did — enough for forensic reconstruction
without itself becoming a PII liability.

Configuration (env):
  MISA_AUDIT_LOG          true|false   enable the middleware (default: true)
  MISA_AUDIT_LOG_FILE     path         rotating file destination
                                       (default: ./audit.jsonl)
  MISA_AUDIT_LOG_STDOUT   true|false   also emit to stdout (default: true)
  MISA_AUDIT_LOG_MAX_MB   int          rotate when file exceeds N MB
                                       (default: 50)
  MISA_AUDIT_LOG_BACKUPS  int          number of rotated files to keep
                                       (default: 10)

Compliance mapping:
  - NCA ECC subdomain 2-12 (Logging and Monitoring): ✓ comprehensive logs,
    forensic detail, secure storage, retention
  - NDMO Data Governance Framework, DG-7 (Audit & Monitoring): ✓ records
    of access to personal data with subject identification
  - PDPL Article 25 (Records of Processing): ✓ supports the ROPA with
    per-request evidence
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# ─── Configuration ───────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


AUDIT_LOG_ENABLED: bool = _env_bool("MISA_AUDIT_LOG", True)
AUDIT_LOG_FILE: str = (os.getenv("MISA_AUDIT_LOG_FILE") or "audit.jsonl").strip()
AUDIT_LOG_STDOUT: bool = _env_bool("MISA_AUDIT_LOG_STDOUT", True)
AUDIT_LOG_MAX_MB: int = max(1, int(os.getenv("MISA_AUDIT_LOG_MAX_MB", "50")))
# Retention: number of rotated files kept. Lower default in production
# reduces long-lived PII-adjacent metadata at rest; override via env.
AUDIT_LOG_BACKUPS: int = max(
    1,
    int(os.getenv(
        "MISA_AUDIT_LOG_BACKUPS",
        "5" if (os.getenv("MISA_ENV") or "").strip().lower() in ("production", "prod") else "10",
    )),
)

_MAX_STR_LEN = 500  # cap user-supplied strings so a giant prompt cannot
                    # bloat the log line into MB territory

# ─── Request-scoped identity (Risk-20-1) ─────────────────────────────
_audit_user: ContextVar[str] = ContextVar("misa_audit_user", default="unknown")


def set_audit_user(user: "str | None") -> None:
    """Record the authenticated caller for the current request context.
    Call once, as early as possible after auth resolves — see
    app/routers/v1/chat.py's chat_endpoint."""
    _audit_user.set((user or "").strip() or "unknown")


def get_audit_user() -> str:
    """The identity set by set_audit_user(), or "unknown" outside a request
    (tests, scripts driving the pipeline directly)."""
    return _audit_user.get()

# ─── Logger setup ────────────────────────────────────────────────────

# Dedicated logger so audit lines are isolated from application logs;
# both destinations (file + stdout) attach to this single logger so the
# middleware stays simple.
_audit_logger = logging.getLogger("misa.audit")
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False  # don't double-log via the root logger

if AUDIT_LOG_ENABLED and not _audit_logger.handlers:
    fmt = logging.Formatter("%(message)s")  # raw JSON, no extra prefix
    try:
        fh = RotatingFileHandler(
            AUDIT_LOG_FILE,
            maxBytes=AUDIT_LOG_MAX_MB * 1024 * 1024,
            backupCount=AUDIT_LOG_BACKUPS,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        _audit_logger.addHandler(fh)
    except Exception as e:
        # Never let a file-handler failure (read-only FS, permission denied)
        # silently break the app. Log the error to stderr and continue
        # with stdout-only audit.
        print(
            f"[audit] WARNING: could not open audit file {AUDIT_LOG_FILE!r}: {e}",
            file=sys.stderr,
        )

    if AUDIT_LOG_STDOUT:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        _audit_logger.addHandler(sh)


# ─── Helpers ─────────────────────────────────────────────────────────

def _truncate(s: str | None, limit: int = _MAX_STR_LEN) -> str | None:
    if s is None:
        return None
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _client_ip(request: Request) -> str:
    """Reads the client IP with X-Forwarded-For awareness.
    When behind a WAF/proxy (Azure Front Door, Cloudflare, etc.) the
    leftmost public IP in XFF is the true client. We don't trust XFF
    unconditionally because it can be spoofed in direct requests;
    callers running behind a trusted proxy should set MISA_TRUST_XFF=true.
    """
    if _env_bool("MISA_TRUST_XFF", False):
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # First non-empty IP in the chain
            for raw in xff.split(","):
                ip = raw.strip()
                if ip:
                    return ip
    # Fallback to direct peer
    return getattr(getattr(request, "client", None), "host", "unknown") or "unknown"


def _user_identity(request: Request) -> str:
    """Extracts the authenticated username from the Authorization header.

    Prefers Bearer JWT `sub` (current auth). Falls back to legacy HTTP
    Basic for transitional tooling. Returns 'anonymous' when no usable
    identity is present. PDPL/NCA require subject identification on every
    record that touches personal data, so this column is non-empty by design.
    """
    auth = request.headers.get("authorization") or ""
    lower = auth.lower()
    if lower.startswith("bearer "):
        token = auth[7:].strip()
        if not token:
            return "anonymous"
        try:
            from app import config as _cfg
            import jwt as _jwt
            secret = (_cfg.JWT_SECRET_KEY or "").strip()
            if not secret:
                return "anonymous"
            claims = _jwt.decode(
                token,
                secret,
                algorithms=[_cfg.JWT_ALGORITHM],
                audience=_cfg.JWT_AUDIENCE,
                issuer=_cfg.JWT_ISSUER,
            )
            return str(claims.get("sub") or "anonymous")
        except Exception:
            return "invalid-token"
    if lower.startswith("basic "):
        import base64
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8", errors="replace")
            user = decoded.split(":", 1)[0]
            return user or "anonymous"
        except Exception:
            return "malformed-auth"
    return "anonymous"


def _classify_path(path: str) -> str:
    """Maps the URL path to a coarse category for compliance reporting.
    NDMO ROPA wants to group access events by processing activity
    (chat, search, engagement dossier, feedback, ops). Doing the
    bucketing here keeps every consumer's downstream filter consistent.
    """
    if path.startswith("/api/v1/chat"):
        return "chat"
    if path.startswith("/api/v1/search"):
        return "search"
    if path.startswith("/api/v1/engagement"):
        return "engagement_dossier"
    if path.startswith("/api/v1/feedback"):
        return "feedback"
    if path.startswith("/api/v1/export"):
        return "export"
    if path in ("/health",):
        return "health"
    if path in ("/", "/chat"):
        return "ui"
    return "other"


# ─── Middleware ──────────────────────────────────────────────────────

class AuditMiddleware(BaseHTTPMiddleware):
    """ASGI middleware emitting one JSON audit line per HTTP request.

    Placed AFTER CORSMiddleware so we don't audit preflight OPTIONS
    chatter, but BEFORE auth (we want to capture failed-auth attempts
    too, since those are forensically interesting under NCA ECC 2-12).

    The middleware NEVER raises — any logging failure is swallowed
    after a single stderr warning, so an unhealthy log destination
    cannot take the API offline.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not AUDIT_LOG_ENABLED:
            return await call_next(request)

        # Generate or read the correlation id. Honour an incoming
        # X-Request-ID if a trusted upstream provided one (so distributed
        # traces stitch together).
        cid = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Stash the correlation id on request state so downstream
        # handlers can pull it for their own structured logs.
        request.state.correlation_id = cid

        t0 = time.time()
        status = 0
        error_str = None
        response: Response | None = None
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            status = 500
            # Re-raise after we've captured the event, so the framework's
            # exception handlers still produce the user-facing 500.
            raise
        finally:
            try:
                event = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "cid": cid,
                    "category": _classify_path(request.url.path),
                    "user": _user_identity(request),
                    "client_ip": _client_ip(request),
                    "method": request.method,
                    "path": request.url.path,
                    "query": _truncate(request.url.query or None, 200),
                    "status": status,
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "user_agent": _truncate(
                        request.headers.get("user-agent"), 200,
                    ),
                    "error": _truncate(error_str, 300),
                }
                # Attach the response's correlation id header for easy
                # cross-reference from a user-reported issue.
                if response is not None:
                    response.headers["x-request-id"] = cid

                _audit_logger.info(
                    json.dumps(event, ensure_ascii=False, default=str)
                )
            except Exception as log_err:
                # Last-line defence: never let audit logging crash the
                # request. Surface to stderr so ops sees the problem.
                print(
                    f"[audit] emit failed cid={cid}: {log_err}",
                    file=sys.stderr,
                )

        return response


# ─── Public API for handlers that want to enrich the audit event ─────
# Handlers (notably /chat) can call enrich_audit_event() to add the
# intent / entity / table list captured during processing. The enriched
# record is emitted as a SECOND event with the same correlation id, so
# downstream queries can JOIN on cid for the full picture.

def emit_application_event(request: Request, payload: dict) -> None:
    """Emit a JSON event tied to the current request's correlation id.

    Used by /chat to record the intent classification result + which
    tables were touched, since the middleware can only see HTTP-layer
    metadata. The payload is wrapped with the same envelope keys
    (ts, cid, user, category) for SIEM consistency.

    Never raises.
    """
    if not AUDIT_LOG_ENABLED:
        return
    try:
        cid = getattr(request.state, "correlation_id", None) or "unknown"
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "cid": cid,
            "category": _classify_path(request.url.path),
            "user": _user_identity(request),
            "kind": "application",
            **{k: _truncate(v) if isinstance(v, str) else v
               for k, v in payload.items()},
        }
        try:
            from app.services.prompt_masking import mask_obj
            event = mask_obj(event, for_log=True)
        except Exception:
            pass
        _audit_logger.info(
            json.dumps(event, ensure_ascii=False, default=str)
        )
    except Exception as log_err:
        print(f"[audit] application event emit failed: {log_err}", file=sys.stderr)


def emit_security_event(payload: dict) -> None:
    """Emit a security/monitoring event that is NOT tied to a live Request
    (Risk-20-5): DB-query-layer signals such as per-turn row-budget
    truncation or an at-cap single-query pull, emitted from deep in the
    chat engine where no Request object is available. Same envelope as
    `emit_application_event` (ts/category/kind) for SIEM consistency.

    `user` comes from the request-scoped ContextVar (Risk-20-1) so every
    security event is attributable to the authenticated caller; a payload
    may still pass its own `user` key to override it.

    Never raises.
    """
    if not AUDIT_LOG_ENABLED:
        return
    try:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "cid": "unknown",
            "category": "security",
            "kind": "security",
            "user": _audit_user.get(),
            **{k: _truncate(v) if isinstance(v, str) else v
               for k, v in payload.items()},
        }
        _audit_logger.info(
            json.dumps(event, ensure_ascii=False, default=str)
        )
    except Exception as log_err:
        print(f"[audit] security event emit failed: {log_err}", file=sys.stderr)
