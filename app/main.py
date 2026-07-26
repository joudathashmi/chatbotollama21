"""
MISA Intelligence API — FastAPI application factory.

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
Or use the convenience wrapper:
    python run.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logger import logger, configure_logging
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.security import MaxBodySizeMiddleware, SecurityHeadersMiddleware
from app.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)

# Configure logging as early as possible (important for startup/shutdown).
configure_logging()

from app.auth import optional_user, require_role, verify_credentials
from app import config
from app.config import (
    CORS_ALLOWED_ORIGINS,
    DB_MAX_CONCURRENCY,
    OPENAI_MODEL,
)
from app.database import close_all_db_connections, get_db
from app.routers.v1 import auth as auth_router
from app.routers.v1 import business_card as business_card_router
from app.routers.v1 import chat as chat_router
from app.routers.v1 import documents as documents_router
from app.routers.v1 import sessions as sessions_router
from app.routers.v1 import engagement as engagement_router
from app.routers.v1 import search as search_router
from app.services.audit_log import AuditMiddleware


# Lifespan events (startup/shutdown) with required logs.
async def _lifespan(app: FastAPI):
    logger.info("Server started")

    # Bound the thread pool that runs all blocking DB work. Every
    # asyncio.to_thread / loop.run_in_executor(None, ...) call in the app
    # (chat pipeline, search, engagement data, PDF render, malware scan)
    # dispatches here. Because app/database.py keeps ONE Postgres
    # connection per worker thread, this ceiling is also the per-process
    # open-connection ceiling — keeping (workers x DB_MAX_CONCURRENCY)
    # safely under Postgres `max_connections`.
    loop = asyncio.get_running_loop()
    db_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=DB_MAX_CONCURRENCY,
        thread_name_prefix="misa-db",
    )
    loop.set_default_executor(db_executor)
    app.state.db_executor = db_executor
    logger.info(f"DB worker pool: max_workers={DB_MAX_CONCURRENCY}")

    if CORS_ALLOWED_ORIGINS == ["*"]:
        logger.warning(
            "CORS: MISA_CORS_ALLOWED_ORIGINS is unset — allowing ALL origins (\"*\"). "
            "Set it to a comma-separated allowlist of your frontend/backend URLs "
            "before deploying (see env.sample)."
        )
    elif not CORS_ALLOWED_ORIGINS:
        logger.error(
            "CORS: allowlist is empty — browser clients will be blocked. "
            "Set MISA_CORS_ALLOWED_ORIGINS."
        )
    else:
        logger.info(f"CORS: allowed origins = {CORS_ALLOWED_ORIGINS}")

    # Production refuses to boot on fatal misconfig; local only warns.
    sec_errors = config.validate_security_config()
    for err in sec_errors:
        if config.IS_PRODUCTION:
            logger.error(f"SECURITY: {err}")
        else:
            logger.warning(f"SECURITY: {err}")
    if sec_errors and config.IS_PRODUCTION:
        raise RuntimeError(
            "Refusing to start in production due to security misconfiguration: "
            + "; ".join(sec_errors)
        )

    if not config.JWT_SECRET_KEY:
        logger.warning(
            "AUTH: JWT_SECRET_KEY is unset — login/token endpoints will return "
            "503 until it is configured. Generate one with: "
            "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )
    elif not config.jwt_secret_is_strong():
        logger.warning(
            f"AUTH: JWT_SECRET_KEY is shorter than {config.JWT_SECRET_MIN_LEN} "
            "chars — use a longer, random secret for HS256."
        )
    if config.ALLOW_PLAINTEXT_BOOTSTRAP and config.API_PASSWORD:
        logger.warning(
            "AUTH: plaintext bootstrap password is enabled "
            "(MISA_ALLOW_PLAINTEXT_BOOTSTRAP). Prefer bcrypt accounts via "
            "MISA_AUTH_USERS; disable bootstrap in production."
        )
    if config.AUTH_DISABLED:
        logger.warning(
            "AUTH: MISA_AUTH_DISABLED=true — /chat and /api/v1/* are open "
            f"as user {config.AUTH_DISABLED_USERNAME!r}. Do not use in production."
        )
    if not config.ENABLE_DOCS:
        logger.info("DOCS: interactive API docs disabled (MISA_ENABLE_DOCS=false).")
    if not config.openai_configured():
        logger.warning("OPENAI: no OPENAI_API_KEY configured — model features are disabled.")
    if config.RATE_LIMIT_BACKEND != "redis":
        logger.info(
            f"RATE LIMIT: backend={config.RATE_LIMIT_BACKEND!r} "
            "(use redis when running multiple workers — see serve.py)."
        )

    try:
        from app.services.malware_scanner import log_status_on_startup
        log_status_on_startup()
    except Exception as e:
        # Never let a scanner-status probe block startup.
        logger.warning(f"Malware-scan status check failed at startup: {e}")
    yield
    logger.info("Server shutting down")
    try:
        close_all_db_connections()
    except Exception as e:
        logger.warning(f"Error closing DB connections at shutdown: {e}")
    try:
        db_executor.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        logger.warning(f"Error shutting down DB worker pool: {e}")


app = FastAPI(

    title="MISA Intelligence API",
    description=(
        "Streaming endpoints over `company_profiles.misa_details`.\n\n"
        "**POST /api/v1/chat** — NL question → OpenAI SQL routing → local DB → SSE. "
        "Row data is never sent to OpenAI.\n\n"
        "**POST /api/v1/search** — Structured filter lookup → no OpenAI → NDJSON.\n\n"
        "**POST /api/v1/engagement/generate** — OpenAI Responses API + web_search dossier."
    ),
    version="1.0.0",
    lifespan=_lifespan,
    # Interactive docs expose the full API surface — gate them behind config
    # so they can be switched off for internet-facing deployments.
    docs_url="/docs" if config.ENABLE_DOCS else None,
    redoc_url="/redoc" if config.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if config.ENABLE_DOCS else None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    # Only the headers the API actually needs — Authorization for the bearer
    # token and Content-Type for JSON/multipart bodies.
    allow_headers=["Authorization", "Content-Type"],
)

# Audit logging — NCA ECC 2-12 (Logging & Monitoring) + NDMO DG-7
# (Audit & Monitoring). Emits one JSON event line per HTTP request
# to ./audit.jsonl (rotating) and stdout (SIEM-friendly). Disable
# with MISA_AUDIT_LOG=false. See app/services/audit_log.py for the
# full event schema and configuration knobs.
app.add_middleware(AuditMiddleware)

_auth = [Depends(verify_credentials)]
# RBAC: analyst+ for PII uploads / expensive dossier generation;
# viewer (default auth) for chat/search; admin for data probes.
_auth_analyst = [Depends(require_role(config.ROLE_ANALYST))]
_auth_admin = [Depends(require_role(config.ROLE_ADMIN))]


app.add_middleware(LoggingMiddleware)

# Added last → outermost. Body-size guard rejects oversized payloads before
# any inner processing; security headers wrap every response (incl. errors).
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=config.MAX_REQUEST_BYTES)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return request_validation_exception_handler(request, exc)


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return unhandled_exception_handler(request, exc)



# Auth endpoints (/api/v1/auth/login, /refresh) are intentionally NOT behind
# `_auth` — a caller can't present a bearer token before obtaining one.
app.include_router(auth_router.router, prefix="/api/v1")
app.include_router(business_card_router.router, prefix="/api/v1", dependencies=_auth_analyst)
app.include_router(documents_router.router, prefix="/api/v1", dependencies=_auth)
app.include_router(sessions_router.router, prefix="/api/v1", dependencies=_auth)
app.include_router(chat_router.router, prefix="/api/v1", dependencies=_auth)
app.include_router(search_router.router, prefix="/api/v1", dependencies=_auth)
app.include_router(engagement_router.router, prefix="/api/v1", dependencies=_auth_analyst)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect the bare URL to the chat UI."""
    return RedirectResponse(url="/chat")


_CHAT_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>MISA Chat</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/marked@13.0.0/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<style>
 * { box-sizing: border-box; }
 body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0; background: #f5f5f7; color: #1d1d1f; }
 header { padding: 12px 18px; background: #fff; border-bottom: 1px solid #e5e5e7;
          display: flex; align-items: center; gap: 12px; }
 header h1 { font-size: 16px; margin: 0; }
 header .meta { color: #86868b; font-size: 13px; }
 .workspace { max-width: 1280px; margin: 0 auto; padding: 14px 16px 130px;
              display: grid; grid-template-columns: 220px 1fr; gap: 14px; align-items: start; }
 .workspace.docs-open { grid-template-columns: 220px 300px 1fr; }
 @media (max-width: 960px) {
   .workspace, .workspace.docs-open { grid-template-columns: 1fr; }
   .sess-panel { position: static; max-height: none; }
 }
 .sess-panel { background: #fff; border: 1px solid #e5e5e7; border-radius: 14px;
               padding: 12px 12px; position: sticky; top: 12px;
               max-height: calc(100vh - 160px); overflow: auto; }
 .sess-panel .sess-head { display: flex; align-items: center; justify-content: space-between;
                          gap: 8px; margin-bottom: 10px; }
 .sess-panel h2 { margin: 0; font-size: 14px; font-weight: 600; }
 .sess-panel .new-chat { padding: 6px 10px; font-size: 12px; border-radius: 8px; }
 .sess-search { width: 100%; padding: 7px 9px; border: 1px solid #d2d2d7; border-radius: 8px;
                font: inherit; font-size: 12px; margin-bottom: 8px; }
 .sess-arch-toggle { display: flex; align-items: center; gap: 6px; font-size: 11px;
                     color: #86868b; margin-bottom: 8px; cursor: pointer; }
 .continue-chip { display: none; margin-bottom: 8px; padding: 8px 10px; border-radius: 8px;
                  background: #e8f0fe; color: #0055cc; font-size: 12px; cursor: pointer;
                  border: 1px solid #b3d0ff; }
 .continue-chip:not([hidden]) { display: block; }
 .continue-chip:hover { background: #dce9fd; }
 .sess-list { list-style: none; margin: 0; padding: 0; }
 .sess-list li { display: flex; gap: 2px; align-items: stretch; margin-bottom: 4px; }
 .sess-list button.sess-item {
   flex: 1; text-align: left; background: transparent; color: #1d1d1f;
   border: 1px solid transparent; border-radius: 8px; padding: 8px 10px;
   font: inherit; font-size: 12px; cursor: pointer; line-height: 1.35;
 }
 .sess-list button.sess-item:hover { background: #f5f5f7; }
 .sess-list button.sess-item.active { background: #e8f0fe; border-color: #b3d0ff; color: #0055cc; }
 .sess-list button.sess-item.archived { opacity: 0.65; }
 .sess-list .sess-title { display: block; word-break: break-word; }
 .sess-list .sess-meta { display: block; color: #86868b; font-size: 10px; margin-top: 2px; }
 .sess-list .sess-actions { display: flex; flex-direction: column; }
 .sess-list button.sess-ico {
   background: transparent; color: #86868b; border: 0; padding: 2px 6px;
   font-size: 12px; cursor: pointer; border-radius: 6px; line-height: 1.2;
 }
 .sess-list button.sess-ico:hover { color: #1d1d1f; background: #f5f5f7; }
 .sess-list button.sess-ico.on { color: #0055cc; }
 .sess-empty { color: #86868b; font-size: 12px; padding: 8px 4px; }
 .docs-toggle { background: #f5f5f7; color: #1d1d1f; border: 1px solid #d2d2d7;
                padding: 4px 12px; font-size: 13px; border-radius: 8px; cursor: pointer;
                display: inline-flex; align-items: center; gap: 6px; }
 .docs-toggle:hover { background: #e8e8ed; }
 .docs-toggle[aria-expanded="true"] { background: #e8f0fe; border-color: #b3d0ff; color: #0055cc; }
 .docs-toggle .doc-count { font-size: 11px; color: #86868b; background: #fff;
                           border: 1px solid #e5e5e7; border-radius: 999px; padding: 0 6px;
                           min-width: 18px; text-align: center; }
 .docs-toggle[aria-expanded="true"] .doc-count { border-color: #b3d0ff; color: #0055cc; }
 .doc-panel { display: none; background: #fff; border: 1px solid #e5e5e7; border-radius: 14px;
              padding: 12px 14px; position: sticky; top: 12px; max-height: calc(100vh - 160px);
              overflow: auto; }
 .workspace.docs-open .doc-panel { display: block; }
 .doc-panel .doc-head { display: flex; align-items: center; justify-content: space-between;
                        gap: 8px; margin-bottom: 4px; }
 .doc-panel h2 { margin: 0; font-size: 14px; font-weight: 600; }
 .doc-panel .doc-close { background: transparent; color: #86868b; border: 0; padding: 2px 6px;
                         font-size: 18px; line-height: 1; cursor: pointer; border-radius: 6px; }
 .doc-panel .doc-close:hover { background: #f5f5f7; color: #1d1d1f; }
 .doc-panel .hint { margin: 0 0 10px; color: #86868b; font-size: 12px; line-height: 1.4; }
 .doc-panel label { display: block; font-size: 12px; color: #4a4a4f; margin: 8px 0 4px; }
 .doc-panel input[type=file], .doc-panel select { width: 100%; font: inherit; font-size: 13px; }
 .doc-panel .doc-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
 .doc-panel .doc-actions button { padding: 6px 10px; font-size: 12px; border-radius: 8px; }
 .doc-panel button.secondary { background: #f5f5f7; color: #1d1d1f; border: 1px solid #d2d2d7; }
 .doc-msg { font-size: 12px; margin-top: 8px; min-height: 16px; }
 .doc-msg.ok { color: #0a7; }
 .doc-msg.err { color: #d32f2f; }
 .doc-list { list-style: none; margin: 12px 0 0; padding: 0; }
 .doc-list li { display: flex; gap: 6px; align-items: flex-start; padding: 8px 0;
                border-top: 1px solid #f0f0f2; font-size: 12px; }
 .doc-list .name { flex: 1; word-break: break-word; }
 .doc-list .badge { display: inline-block; padding: 1px 6px; border-radius: 999px;
                    background: #f0f0f2; font-size: 10px; margin-right: 4px; }
 .doc-list .badge.org { background: #e8f1ff; color: #0055cc; }
 .doc-list .badge.ready { background: #e6f7ed; color: #0a7; }
 .doc-list .badge.failed { background: #fdecea; color: #c00; }
 .doc-list button.del { background: transparent; color: #86868b; border: 0;
                        padding: 0 4px; font-size: 11px; cursor: pointer; }
 .doc-list button.del:hover { color: #d32f2f; }
 #chat { min-width: 0; }
 .form-inner { max-width: 1280px; }
 .msg { padding: 14px 18px; border-radius: 14px; margin: 12px 0;
        word-wrap: break-word; overflow-wrap: anywhere; }
 .user { background: #007aff; color: #fff; margin-left: 60px;
         white-space: pre-wrap; }
 .assistant { background: #fff; border: 1px solid #e5e5e7; }
 /* Markdown styling inside the assistant bubble */
 .assistant h1, .assistant h2, .assistant h3, .assistant h4 {
   margin: 18px 0 8px; line-height: 1.25; font-weight: 600;
   color: #111; border-bottom: 1px solid #f0f0f2; padding-bottom: 4px; }
 .assistant h1 { font-size: 19px; }
 .assistant h2 { font-size: 17px; }
 .assistant h3 { font-size: 15px; }
 .assistant h4 { font-size: 14px; color: #4a4a4f; border: 0; }
 .assistant h1:first-child, .assistant h2:first-child, .assistant h3:first-child { margin-top: 0; }
 .assistant p  { margin: 8px 0; }
 .assistant ul, .assistant ol { margin: 8px 0 8px 20px; padding-left: 6px; }
 .assistant li { margin: 4px 0; }
 .assistant strong { font-weight: 600; color: #0a0a0c; }
 .assistant em { color: #4a4a4f; }
 .assistant code { background: #f5f5f7; padding: 1px 5px; border-radius: 4px;
                   font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
 .assistant pre { background: #f5f5f7; padding: 10px; border-radius: 8px;
                  overflow-x: auto; }
 .assistant blockquote { border-left: 3px solid #d2d2d7; padding: 4px 12px;
                         margin: 8px 0; color: #4a4a4f; }
 .assistant a { color: #007aff; text-decoration: none; }
 .assistant a:hover { text-decoration: underline; }
 .assistant table { border-collapse: collapse; margin: 10px 0; }
 .assistant th, .assistant td { border: 1px solid #e5e5e7; padding: 6px 10px; }
 .assistant th { background: #f5f5f7; text-align: left; }
 .meta-row { font-size: 12px; color: #86868b; margin: 4px 4px 14px; }
 .feedback-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
                 margin: 0 4px 18px; font-size: 13px; }
 .feedback-btn { background: #f5f5f7; color: #1d1d1f; border: 1px solid #d2d2d7;
                 padding: 4px 10px; font-size: 14px; border-radius: 8px;
                 cursor: pointer; line-height: 1.2; }
 .feedback-btn:hover { background: #e8e8ed; }
 .feedback-row.feedback-locked .feedback-btn { opacity: 0.4; pointer-events: none; }
 .feedback-status { color: #86868b; font-size: 12px; margin-left: 6px; }
 .feedback-comment { width: 100%; min-height: 56px; padding: 6px 8px;
                     border: 1px solid #d2d2d7; border-radius: 8px;
                     font: inherit; font-size: 13px; resize: vertical;
                     margin-top: 6px; }
 .feedback-submit { background: #007aff; color: #fff; padding: 4px 12px;
                    font-size: 13px; }
 /* Inline citation chips — [web:N] markers in the LLM output that the
    UI rewrites to <sup class="cite"> superscripted links. Visible but
    not loud; hover reveals the link. */
 sup.cite { font-size: 11px; font-weight: 600; color: #0a73f0;
            background: #e8f0fe; padding: 1px 5px; border-radius: 8px;
            margin: 0 1px; vertical-align: super; }
 sup.cite:hover { background: #d0e2fd; }
 .assistant a:has(sup.cite) { text-decoration: none; }
 /* Sources footer — numbered list of URLs cited by the answer. */
 .sources-row { margin: 0 4px 18px; font-size: 13px;
                background: #f9f9fb; border: 1px solid #e5e5e7;
                border-radius: 10px; padding: 10px 14px; }
 .sources-head { font-weight: 600; color: #1d1d1f; margin-bottom: 6px;
                 font-size: 13px; }
 .sources-list { margin: 0; padding-left: 22px; }
 .sources-list li { margin: 4px 0; line-height: 1.4; }
 .sources-list a { color: #0a73f0; text-decoration: none; }
 .sources-list a:hover { text-decoration: underline; }
 .sources-snippet { color: #6e6e73; font-size: 12px; margin-top: 2px; }
 .source-type { display: inline-block; font-size: 10px; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.03em;
                color: #6e6e73; background: #e8e8ed; padding: 1px 6px;
                border-radius: 4px; margin-right: 6px; vertical-align: middle; }
 .source-type.doc { background: #e8f5e9; color: #2e7d32; }
 .source-type.web { background: #e3f2fd; color: #1565c0; }
 .source-type.db { background: #fff3e0; color: #e65100; }
 .locale-toggle { display: inline-flex; gap: 2px; margin-left: 10px;
                  border: 1px solid #d2d2d7; border-radius: 8px; padding: 2px; }
 .locale-toggle button { background: transparent; color: #6e6e73;
                         border: 0; padding: 3px 8px; font-size: 12px;
                         border-radius: 6px; cursor: pointer; }
 .locale-toggle button.active { background: #1d1d1f; color: #fff; }
 .recovery-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 4px 18px; }
 .recovery-row button { background: #f5f5f7; color: #1d1d1f; border: 1px solid #d2d2d7;
                        padding: 6px 12px; font-size: 13px; border-radius: 8px;
                        cursor: pointer; }
 .recovery-row button:hover { background: #e8e8ed; }
 /* Scoped to #form (the chat composer) on purpose: a bare `form` selector
    also caught the login card, pinning it to the bottom of the viewport and
    laying its fields out in a row. */
 #form { position: fixed; bottom: 0; left: 0; right: 0; padding: 12px 16px;
        background: #fff; border-top: 1px solid #e5e5e7; display: flex; gap: 8px; }
 .form-inner { max-width: 1280px; margin: 0 auto; display: flex; gap: 8px; width: 100%; }
 textarea { flex: 1; resize: none; padding: 10px 12px; border: 1px solid #d2d2d7;
            border-radius: 10px; font: inherit; min-height: 44px; max-height: 140px; }
 button { padding: 10px 18px; border: 0; border-radius: 10px;
          background: #007aff; color: #fff; font: inherit; cursor: pointer; }
 button:disabled { background: #b0b0b0; cursor: not-allowed; }
 .empty { color: #86868b; text-align: center; margin: 60px 16px; font-size: 14px; }
 /* Logout control in the header (visible once signed in). */
 .logout-btn { margin-left: auto; background: #f5f5f7; color: #1d1d1f;
               border: 1px solid #d2d2d7; padding: 4px 12px; font-size: 13px;
               border-radius: 8px; cursor: pointer; }
 .logout-btn:hover { background: #e8e8ed; }
 /* Login overlay — full-screen gate shown until a JWT is obtained. */
 .login-overlay { position: fixed; inset: 0; background: rgba(245,245,247,0.96);
                  display: flex; align-items: center; justify-content: center;
                  z-index: 1000; }
 .login-overlay[hidden] { display: none; }
 .login-card { background: #fff; border: 1px solid #e5e5e7; border-radius: 14px;
               padding: 28px 26px; width: 320px; max-width: calc(100vw - 32px);
               box-shadow: 0 10px 40px rgba(0,0,0,0.08); }
 .login-card h2 { margin: 0 0 4px; font-size: 18px; }
 .login-card p.sub { margin: 0 0 18px; color: #86868b; font-size: 13px; }
 .login-card label { display: block; font-size: 13px; color: #4a4a4f; margin: 10px 0 4px; }
 .login-card input { width: 100%; padding: 9px 11px; border: 1px solid #d2d2d7;
                     border-radius: 9px; font: inherit; font-size: 14px; }
 .login-card button { width: 100%; margin-top: 18px; padding: 10px;
                      background: #007aff; color: #fff; border: 0; border-radius: 9px;
                      font: inherit; font-size: 15px; cursor: pointer; }
 .login-card button:disabled { background: #b0b0b0; cursor: not-allowed; }
 .login-err { color: #d32f2f; font-size: 13px; margin-top: 12px; min-height: 16px; }
 details { margin: 8px 0; }
 details summary { font-size: 12px; color: #86868b; cursor: pointer; }
 details pre { font-size: 12px; background: #f5f5f7; padding: 8px;
               border-radius: 6px; max-height: 220px; overflow: auto; }
 .err { color: #d32f2f; }
</style></head>
<body>
<div class="login-overlay" id="loginOverlay" hidden>
  <form class="login-card" id="loginForm">
    <h2>MISA Intelligence</h2>
    <p class="sub">Sign in to continue.</p>
    <label for="loginUser">Username</label>
    <input id="loginUser" name="username" autocomplete="username" autofocus>
    <label for="loginPass">Password</label>
    <input id="loginPass" name="password" type="password" autocomplete="current-password">
    <button id="loginBtn" type="submit">Sign in</button>
    <div class="login-err" id="loginErr"></div>
  </form>
</div>
<header>
  <h1>MISA Chat</h1>
  <span class="meta" id="health">checking…</span>
  <div class="locale-toggle" id="localeToggle" title="Answer language">
    <button type="button" data-locale="en" class="active">EN</button>
    <button type="button" data-locale="ar">AR</button>
  </div>
  <button class="docs-toggle" id="docToggle" type="button" aria-expanded="false" aria-controls="docPanel">
    Documents <span class="doc-count" id="docCount" hidden>0</span>
  </button>
  <button class="logout-btn" id="logoutBtn" type="button" hidden>Sign out</button>
</header>
<div class="workspace" id="workspace">
  <aside class="sess-panel" id="sessPanel">
    <div class="sess-head">
      <h2>Chats</h2>
      <button class="new-chat" id="newChatBtn" type="button">New</button>
    </div>
    <input id="sessSearch" class="sess-search" type="search" placeholder="Search chats…" autocomplete="off">
    <label class="sess-arch-toggle"><input type="checkbox" id="sessShowArchived"> Show archived</label>
    <div class="continue-chip" id="continueChip" hidden></div>
    <ul class="sess-list" id="sessList"><li class="sess-empty">Loading…</li></ul>
  </aside>
  <aside class="doc-panel" id="docPanel" hidden>
    <div class="doc-head">
      <h2>Documents</h2>
      <button class="doc-close" id="docClose" type="button" aria-label="Close documents">×</button>
    </div>
    <p class="hint">Upload here (or drop files in the server inbox). Answers lead with your documents and can add live web context when useful.</p>
    <label for="docFile">File (PDF, DOCX, TXT, MD)</label>
    <input id="docFile" type="file" multiple accept=".pdf,.docx,.txt,.md,.markdown">
    <label for="docVis">Visibility</label>
    <select id="docVis"><option value="private">Private</option><option value="org">Org shared</option></select>
    <div class="hint" style="margin-top:8px;">
      Classification is checked automatically. Documents marked Restricted,
      Secret, or Top Secret are rejected and never stored. Only Public
      documents are processed.
    </div>
    <div id="docConsent" style="margin-top:10px; padding:10px; border:1px solid #e5e5e7; border-radius:10px; background:#fafafa;">
      <strong style="font-size:12px;" id="docConsentTitle">Upload consent declaration</strong>
      <p class="hint" id="docConsentPre" style="margin:6px 0;"></p>
      <ul id="docConsentTerms" style="margin:0 0 8px 16px; padding:0; font-size:12px; color:#4a4a4f; line-height:1.45;"></ul>
      <label style="display:flex; gap:6px; align-items:flex-start; font-size:12px; margin:0;">
        <input type="checkbox" id="docConsentChk" style="margin-top:2px;">
        <span>I confirm all of the above and consent to processing.</span>
      </label>
    </div>
    <div class="doc-actions">
      <button id="docUploadBtn" type="button">Upload</button>
      <button id="docIngestBtn" class="secondary" type="button">Ingest inbox</button>
      <button id="docRefreshBtn" class="secondary" type="button">Refresh</button>
    </div>
    <div class="doc-msg" id="docMsg"></div>
    <ul class="doc-list" id="docList"><li class="hint">Sign in to see your library.</li></ul>
  </aside>
  <main id="chat">
    <div class="empty">Ask about a company, country, deal, executive, or any allowed table.<br>
    Examples: <em>Tell me about Alphabet</em> · <em>Tell me about Pakistan as a country</em> · <em>Show me the latest deals</em></div>
  </main>
</div>
<form id="form">
  <div class="form-inner">
    <textarea id="q" placeholder='Ask anything. Tip: type "/profile Apple" for an executive deep-profile.' rows="1"></textarea>
    <button id="send" type="submit">Send</button>
  </div>
</form>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const q = document.getElementById('q');
const sendBtn = document.getElementById('send');
const health = document.getElementById('health');
const loginOverlay = document.getElementById('loginOverlay');
const loginForm = document.getElementById('loginForm');
const loginUser = document.getElementById('loginUser');
const loginPass = document.getElementById('loginPass');
const loginBtn = document.getElementById('loginBtn');
const loginErr = document.getElementById('loginErr');
const logoutBtn = document.getElementById('logoutBtn');

// ── JWT bearer-token auth (Risk-20-3) ──────────────────────────────────────
// Tokens live in sessionStorage (cleared when the tab closes). All /api/v1/*
// fetches go through authFetch, which attaches the access token and, on a
// 401, transparently refreshes once before retrying. /health is open and is
// fetched without a token.
// When AUTH_DISABLED is true (MISA_AUTH_DISABLED), the login overlay is
// skipped and API calls need no Bearer token.
const AUTH_DISABLED = __AUTH_DISABLED__;
const SESSIONS_ENABLED = __SESSIONS_ENABLED__;
const ACCESS_KEY = 'misa_access', REFRESH_KEY = 'misa_refresh';
const SESSION_KEY = 'misa_session_id';
const LOCALE_KEY = 'misa_ui_locale';
const getAccess  = () => sessionStorage.getItem(ACCESS_KEY);
const getRefresh = () => sessionStorage.getItem(REFRESH_KEY);
const isAuthed   = () => AUTH_DISABLED || !!getAccess();
let currentSessionId = null;
let uiLocale = 'en';
try { currentSessionId = sessionStorage.getItem(SESSION_KEY); } catch (e) {}
try {
  const saved = localStorage.getItem(LOCALE_KEY);
  if (saved === 'ar' || saved === 'en') uiLocale = saved;
} catch (e) {}
function setUiLocale(loc) {
  uiLocale = (loc === 'ar') ? 'ar' : 'en';
  try { localStorage.setItem(LOCALE_KEY, uiLocale); } catch (e) {}
  document.querySelectorAll('#localeToggle button').forEach(b => {
    b.classList.toggle('active', b.dataset.locale === uiLocale);
  });
  document.documentElement.lang = uiLocale;
  document.documentElement.dir = uiLocale === 'ar' ? 'rtl' : 'ltr';
}
function setCurrentSession(id) {
  currentSessionId = id || null;
  try {
    if (id) sessionStorage.setItem(SESSION_KEY, id);
    else sessionStorage.removeItem(SESSION_KEY);
  } catch (e) {}
}
function chatPayload(question, extra) {
  // Never send citation HTML back in history — the API rejects markup on
  // questions and used to 422 on history too (UI showed "Error: [object Object]").
  const cleanHistory = (history || []).map(m => ({
    role: m.role,
    content: stripHtmlForHistory(m.content || ''),
  }));
  const body = Object.assign({
    question, history: cleanHistory, locale: uiLocale,
  }, extra || {});
  if (SESSIONS_ENABLED && currentSessionId) body.session_id = currentSessionId;
  return body;
}
function stripHtmlForHistory(text) {
  if (!text) return text;
  return String(text)
    .replace(/<sup\b[^>]*class=["']?cite["']?[^>]*>(\d+)<\/sup>/gi, '[web:$1]')
    .replace(/<sup\b[^>]*class=["']?cite["']?[^>]*>[^<]*<\/sup>/gi, '[doc]')
    .replace(/<\/?[a-zA-Z][^>]*>/g, '');
}
function formatApiError(d) {
  if (d == null) return 'unknown';
  if (typeof d === 'string') return d;
  if (typeof d.error === 'string') return d.error;
  if (d.error && typeof d.error === 'object') {
    const msg = d.error.message || d.error.code || '';
    const details = d.error.details ? ' — ' + String(d.error.details).slice(0, 240) : '';
    return (msg || 'Request failed') + details;
  }
  if (typeof d.message === 'string') return d.message;
  if (d.detail) {
    if (typeof d.detail === 'string') return d.detail;
    if (Array.isArray(d.detail)) {
      return d.detail.map(x => x.msg || JSON.stringify(x)).join('; ');
    }
  }
  try { return JSON.stringify(d).slice(0, 300); } catch (e) { return 'Request failed'; }
}
function setTokens(a, r) {
  if (a) sessionStorage.setItem(ACCESS_KEY, a);
  if (r) sessionStorage.setItem(REFRESH_KEY, r);
}
function clearTokens() {
  sessionStorage.removeItem(ACCESS_KEY);
  sessionStorage.removeItem(REFRESH_KEY);
}

function showLogin() {
  if (AUTH_DISABLED) return;
  loginOverlay.hidden = false;
  logoutBtn.hidden = true;
  loginPass.value = '';
  loginErr.textContent = '';
  loginUser.focus();
  const list = document.getElementById('docList');
  if (list) list.innerHTML = '<li class="hint">Sign in to see your library.</li>';
}
function hideLogin() {
  loginOverlay.hidden = true;
  logoutBtn.hidden = AUTH_DISABLED;
  loadDocLibrary();
  if (SESSIONS_ENABLED) initSessions();
}

async function tryRefresh() {
  if (AUTH_DISABLED) return false;
  const rt = getRefresh();
  if (!rt) return false;
  try {
    const r = await authFetch('/api/v1/auth/refresh', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({refresh_token: rt}),
    });
    if (!r.ok) return false;
    const d = await r.json();
    setTokens(d.access_token, d.refresh_token);
    return true;
  } catch (e) { return false; }
}

// fetch wrapper that attaches the bearer token and retries once after a
// silent refresh on 401. Returns the (possibly retried) Response so callers
// — including the SSE streaming path — can read the body as usual.
async function authFetch(url, opts = {}) {
  const build = () => {
    const h = new Headers(opts.headers || {});
    const t = getAccess();
    if (t) h.set('Authorization', 'Bearer ' + t);
    return Object.assign({}, opts, {headers: h});
  };
  let res = await fetch(url, build());
  if (!AUTH_DISABLED && res.status === 401) {
    if (await tryRefresh()) {
      res = await fetch(url, build());
      if (res.status === 401) { showLogin(); }
    } else {
      showLogin();
    }
  }
  return res;
}

async function doLogin(username, password) {
  const r = await authFetch('/api/v1/auth/login', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({username, password}),
  });
  if (!r.ok) {
    let msg = 'Sign-in failed.';
    try { const d = await r.json(); msg = (d.error && d.error.message) || d.detail || msg; } catch (e) {}
    if (r.status === 401) msg = 'Invalid username or password.';
    throw new Error(msg);
  }
  const d = await r.json();
  setTokens(d.access_token, d.refresh_token);
}

loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  loginErr.textContent = '';
  loginBtn.disabled = true;
  try {
    await doLogin(loginUser.value.trim(), loginPass.value);
    hideLogin();
    fetchHealth();
    q.focus();
  } catch (err) {
    loginErr.textContent = err.message || 'Sign-in failed.';
  } finally {
    loginBtn.disabled = false;
  }
});

logoutBtn.addEventListener('click', () => {
  clearTokens();
  showLogin();
});

let firstMsg = true;
// Conversation history — sent on every turn so the server can inherit
// the prior turn's entity for follow-up questions like "engage with them".
// Capped at the most recent 12 user turns (matches MISA_MAX_HISTORY_USER_TURNS).
let history = [];
const MAX_HISTORY = 24; // up to 12 user + 12 assistant pairs

// Configure marked: GitHub-flavoured, no auto-IDs on headings.
if (window.marked) {
  marked.setOptions({ gfm: true, breaks: true, headerIds: false, mangle: false });
}

// Tiny markdown fallback used only if the CDN-loaded `marked` failed (offline,
// CSP-blocked, etc). Handles the subset our prompts actually produce:
// ## h2, ### h3, **bold**, *em*, `code`, - and 1. lists, paragraphs.
function tinyMd(src) {
  const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  const lines = (src || '').replace(/\r\n/g, '\n').split('\n');
  let html = '', inUL = false, inOL = false, para = [];
  const flushPara = () => { if (para.length) { html += '<p>' + inline(para.join(' ')) + '</p>'; para = []; } };
  const closeLists = () => { if (inUL) { html += '</ul>'; inUL = false; } if (inOL) { html += '</ol>'; inOL = false; } };
  const inline = s => esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(?<![*\w])\*([^*\n]+)\*(?!\w)/g, '<em>$1</em>');
  for (const raw of lines) {
    const line = raw;
    let m;
    if ((m = line.match(/^\s*(#{1,4})\s+(.+)$/))) {
      flushPara(); closeLists();
      html += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`;
    } else if ((m = line.match(/^\s*[-*]\s+(.+)$/))) {
      flushPara(); if (inOL) { html += '</ol>'; inOL = false; }
      if (!inUL) { html += '<ul>'; inUL = true; }
      html += '<li>' + inline(m[1]) + '</li>';
    } else if ((m = line.match(/^\s*\d+\.\s+(.+)$/))) {
      flushPara(); if (inUL) { html += '</ul>'; inUL = false; }
      if (!inOL) { html += '<ol>'; inOL = true; }
      html += '<li>' + inline(m[1]) + '</li>';
    } else if (line.trim() === '') {
      flushPara(); closeLists();
    } else {
      para.push(line.trim());
    }
  }
  flushPara(); closeLists();
  return html;
}

function safeMarkdownHtml(text) {
  // Model output can contain HTML/JS — never inject unsanitized markup.
  const raw = window.marked ? marked.parse(text || '') : tinyMd(text || '');
  if (window.DOMPurify) {
    return DOMPurify.sanitize(raw, {USE_PROFILES: {html: true}});
  }
  // CDN failed: escape to text so XSS cannot land via assistant bubbles.
  const d = document.createElement('div');
  d.textContent = text || '';
  return d.innerHTML;
}

function addMsg(role, text) {
  if (firstMsg) { chat.innerHTML = ''; firstMsg = false; }
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  if (role === 'assistant') {
    d.innerHTML = safeMarkdownHtml(text);
  } else {
    d.textContent = text;
  }
  chat.appendChild(d);
  d.scrollIntoView({behavior:'smooth', block:'end'});
  return d;
}

// Render a "Sources" footer: documents, web URLs, and DB tables.
function addSourcesFooter(sources) {
  if (!sources || !sources.length) return;
  const wrap = document.createElement('div');
  wrap.className = 'sources-row';
  const head = document.createElement('div');
  head.className = 'sources-head';
  head.textContent = uiLocale === 'ar' ? 'المصادر' : 'Sources';
  wrap.appendChild(head);
  const ol = document.createElement('ol');
  ol.className = 'sources-list';
  sources.forEach((s) => {
    const li = document.createElement('li');
    const typ = (s.type || (
      (s.url || '').startsWith('doc://') ? 'document' :
      (s.url || '').startsWith('db://') ? 'db' : 'web'
    ));
    const badge = document.createElement('span');
    badge.className = 'source-type ' + (typ === 'document' ? 'doc' : typ);
    badge.textContent = typ === 'document' ? 'doc' : typ;
    li.appendChild(badge);
    const url = s.url || '';
    const label = s.title || url || '(source)';
    if (url && (url.startsWith('http://') || url.startsWith('https://'))) {
      const a = document.createElement('a');
      a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer';
      a.textContent = label;
      li.appendChild(a);
    } else {
      const span = document.createElement('span');
      span.textContent = label;
      li.appendChild(span);
    }
    if (s.snippet) {
      const sn = document.createElement('div');
      sn.className = 'sources-snippet';
      sn.textContent = s.snippet.length > 220 ? s.snippet.slice(0, 220) + '…' : s.snippet;
      li.appendChild(sn);
    }
    ol.appendChild(li);
  });
  wrap.appendChild(ol);
  chat.appendChild(wrap);
}

function linkCitations(ans, sources) {
  if (!ans || !sources || !sources.length) return ans;
  const web = sources.filter(s =>
    (s.type === 'web' || (!s.type && (s.url || '').startsWith('http')))
  );
  const docs = sources.filter(s =>
    s.type === 'document' || (s.url || '').startsWith('doc://')
  );
  let out = ans;
  out = out.replace(/\[web:(\d+)\]/g, (m, n) => {
    const idx = parseInt(n, 10) - 1;
    const src = web[idx];
    if (!src || !src.url) return m;
    const title = (src.title || src.url).replace(/[\[\]]/g, '');
    return ` [<sup class="cite">${n}</sup>](${src.url} "${title.replace(/"/g, '&quot;')}")`;
  });
  out = out.replace(/\[doc:([^\]]+)\]/g, (m, key) => {
    const base = String(key).split('#')[0];
    const src = docs.find(d =>
      (d.title && (d.title === base || String(key).startsWith(d.title))) ||
      (d.url && d.url.includes(base))
    );
    if (!src) return m;
    const href = (src.url && src.url.startsWith('http')) ? src.url : '#';
    const title = (src.title || key).replace(/[\[\]"]/g, '');
    const label = title.length > 28 ? title.slice(0, 26) + '…' : title;
    if (href === '#') {
      return ` <sup class="cite" title="${title}">doc</sup>`;
    }
    return ` [<sup class="cite" title="${title}">doc</sup>](${href} "${label}")`;
  });
  return out;
}

function addRecoveryRow(actions) {
  const wrap = document.createElement('div');
  wrap.className = 'recovery-row';
  const specs = {
    retry: { label: uiLocale === 'ar' ? 'أعد المحاولة' : 'Try again',
             run: () => form.requestSubmit() },
    rephrase: { label: uiLocale === 'ar' ? 'أعد الصياغة' : 'Rephrase',
                run: () => { q.focus(); q.select && q.select(); } },
    documents: { label: uiLocale === 'ar' ? 'المستندات' : 'Open documents',
                 run: () => { if (typeof setDocsOpen === 'function') setDocsOpen(true); } },
  };
  (actions || ['retry', 'rephrase']).forEach(key => {
    const spec = specs[key];
    if (!spec) return;
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = spec.label;
    b.onclick = spec.run;
    wrap.appendChild(b);
  });
  chat.appendChild(wrap);
}

function addMeta(text, trace) {
  const m = document.createElement('div');
  m.className = 'meta-row';
  m.textContent = text;
  if (trace && trace.length) {
    const det = document.createElement('details');
    const sum = document.createElement('summary'); sum.textContent = 'trace';
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(trace, null, 2);
    det.appendChild(sum); det.appendChild(pre);
    m.appendChild(document.createElement('br'));
    m.appendChild(det);
  }
  chat.appendChild(m);
}

// Renders thumbs up/down + optional comment under each assistant
// message. Posts to /api/v1/feedback so the quality-loop log picks
// up the verdict + full debug payload for later review. Once a
// verdict is recorded, the buttons lock to prevent double-voting.
function addFeedbackRow({ question, answer, trace, debug, webSources }) {
  const wrap = document.createElement('div');
  wrap.className = 'feedback-row';
  const status = document.createElement('span');
  status.className = 'feedback-status';

  function send(verdict, comment) {
    const tables = (trace || []).map(t => t.table).filter(Boolean);
    const payload = {
      verdict,
      question,
      answer,
      comment: comment || null,
      intent: debug ? debug.intent : null,
      intent_confidence: debug ? debug.intent_confidence : null,
      entity_resolved: debug ? debug.entity_resolved : null,
      tables_searched: tables,
      row_count_total: debug ? debug.evidence_row_count_total : null,
      ui_locale: uiLocale,
    };
    status.textContent = 'sending…';
    authFetch('/api/v1/feedback', {
      method: 'POST', credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    }).then(r => r.json()).then(d => {
      status.textContent = d.persisted
        ? (verdict === 'up' ? '✓ thanks' : '✓ logged — will review')
        : ('feedback failed: ' + (d.error || 'unknown'));
      wrap.classList.add('feedback-locked');
    }).catch(err => {
      status.textContent = 'feedback failed: ' + err;
    });
  }

  const up = document.createElement('button');
  up.type = 'button'; up.className = 'feedback-btn'; up.textContent = '👍';
  up.title = 'Helpful answer';
  up.onclick = () => send('up', null);

  const down = document.createElement('button');
  down.type = 'button'; down.className = 'feedback-btn'; down.textContent = '👎';
  down.title = 'Wrong / unhelpful — opens a comment box';
  down.onclick = () => {
    // For thumbs-down, ask for an optional comment so the reviewer
    // knows WHY. Empty comment is fine (just thumbs-down logged).
    if (wrap.querySelector('.feedback-comment')) return;
    const ta = document.createElement('textarea');
    ta.className = 'feedback-comment';
    ta.placeholder = 'What went wrong? (optional, helps quality loop)';
    ta.rows = 2;
    const sb = document.createElement('button');
    sb.type = 'button'; sb.className = 'feedback-submit'; sb.textContent = 'Submit';
    sb.onclick = () => send('down', ta.value.trim());
    wrap.appendChild(ta);
    wrap.appendChild(sb);
    ta.focus();
  };

  // Download PDF button. Triggers POST /api/v1/export/pdf and saves
  // the response as a file. Lives in the same row as thumbs so the
  // user has all post-answer actions in one place.
  const pdfBtn = document.createElement('button');
  pdfBtn.type = 'button';
  pdfBtn.className = 'feedback-btn pdf-btn';
  pdfBtn.textContent = '📄 PDF';
  pdfBtn.title = 'Download this briefing as a PDF';
  pdfBtn.onclick = async () => {
    pdfBtn.disabled = true;
    pdfBtn.textContent = '⏳ rendering…';
    try {
      const r = await authFetch('/api/v1/export/pdf', {
        method: 'POST', credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          question, answer, web_sources: webSources || null,
        }),
      });
      if (!r.ok) {
        const msg = await r.text();
        status.textContent = 'PDF failed: ' + msg.slice(0, 120);
        pdfBtn.disabled = false; pdfBtn.textContent = '📄 PDF';
        return;
      }
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      // Pull filename from Content-Disposition if present
      const cd = r.headers.get('Content-Disposition') || '';
      const m = cd.match(/filename="?([^";]+)"?/);
      const a = document.createElement('a');
      a.href = url;
      a.download = m ? m[1] : 'MISA-briefing.pdf';
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
      pdfBtn.textContent = '✓ downloaded';
      setTimeout(() => { pdfBtn.textContent = '📄 PDF'; pdfBtn.disabled = false; }, 2000);
    } catch (err) {
      status.textContent = 'PDF failed: ' + err;
      pdfBtn.disabled = false; pdfBtn.textContent = '📄 PDF';
    }
  };

  wrap.appendChild(up);
  wrap.appendChild(down);
  wrap.appendChild(pdfBtn);
  wrap.appendChild(status);
  chat.appendChild(wrap);
}

// Try the fast streaming path. Returns true when streamed successfully
// (caller should NOT fall through to the JSON path). Returns false when
// the backend either returned no chunks (other-intent fallback signal)
// or hit an error worth retrying via JSON. The assistant bubble is
// populated progressively as SSE chunks arrive.
async function tryStreaming(question, t0) {
  let res;
  try {
    res = await authFetch('/api/v1/chat', {
      method: 'POST', credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(chatPayload(question, {stream: true, debug: false})),
    });
  } catch (err) {
    return false;
  }
  if (!res.ok || !res.body) return false;
  const contentType = res.headers.get('content-type') || '';
  if (!contentType.includes('text/event-stream')) {
    // Backend returned JSON despite stream=true (fallback path). The
    // outer code will retry with stream=false to get the rich payload.
    return false;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let assistantEl = null;
  let accumulated = '';
  let plainAnswer = '';
  let sawChunk = false;
  let statusEl = null;
  let lastSources = null;

  const processEvent = (evt) => {
    if (!evt || !evt.type) return;
    if (evt.type === 'status') {
      if (assistantEl) {
        // Answer already streaming — don't leave a stuck status bubble.
        return;
      }
      if (!statusEl) {
        statusEl = document.createElement('div');
        statusEl.className = 'msg assistant';
        statusEl.style.opacity = '0.7';
        statusEl.style.fontStyle = 'italic';
        if (firstMsg) { chat.innerHTML = ''; firstMsg = false; }
        chat.appendChild(statusEl);
      }
      statusEl.textContent = '⏳ ' + (evt.message || 'Working…');
      statusEl.scrollIntoView({behavior:'smooth', block:'end'});
    } else if (evt.type === 'chunk') {
      if (!assistantEl) {
        if (statusEl) { statusEl.remove(); statusEl = null; }
        assistantEl = addMsg('assistant', '');
      }
      sawChunk = true;
      accumulated += evt.text || '';
      plainAnswer = accumulated;
      // Re-render markdown each tick. For long answers this is fine;
      // marked.parse is fast enough; DOMPurify strips script/on* XSS.
      assistantEl.innerHTML = safeMarkdownHtml(accumulated);
      assistantEl.scrollIntoView({behavior:'smooth', block:'end'});
    } else if (evt.type === 'final') {
      // Polished post-stream rewrite (citations preserved when present).
      if (statusEl) { statusEl.remove(); statusEl = null; }
      if (evt.text) {
        plainAnswer = evt.text;
        accumulated = evt.text;
        if (!assistantEl) {
          assistantEl = addMsg('assistant', '');
        }
        sawChunk = true;
        assistantEl.innerHTML = safeMarkdownHtml(accumulated);
      }
    } else if (evt.type === 'rows') {
      // we don't render the structured rows panel in the test UI
    } else if (evt.type === 'error') {
      if (statusEl) { statusEl.remove(); statusEl = null; }
      const e = assistantEl || addMsg('assistant', '');
      e.textContent = 'Error: ' + formatApiError(evt);
      e.classList.add('err');
      addRecoveryRow(evt.recovery || ['retry', 'rephrase', 'documents']);
    } else if (evt.type === 'done') {
      if (statusEl) { statusEl.remove(); statusEl = null; }
      if (evt.session_id) {
        setCurrentSession(evt.session_id);
        if (SESSIONS_ENABLED) loadSessionList();
      }
      lastSources = evt.sources || evt.web_sources || null;
      if (assistantEl && (plainAnswer || accumulated)) {
        const base = plainAnswer || accumulated;
        const linked = linkCitations(base, lastSources || []);
        assistantEl.innerHTML = safeMarkdownHtml(linked);
        accumulated = linked;
      }
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      if (lastSources && lastSources.length) addSourcesFooter(lastSources);
      addMeta(dt + 's · streaming', evt.trace || []);
      // Update history + feedback row
      if (sawChunk) {
        addFeedbackRow({
          question, answer: plainAnswer || accumulated,
          trace: evt.trace || [], debug: null,
          webSources: lastSources || null,
        });
        history.push({role: 'user',      content: question});
        history.push({role: 'assistant', content: plainAnswer || stripHtmlForHistory(accumulated)});
        if (history.length > MAX_HISTORY) history = history.slice(-MAX_HISTORY);
      } else {
        addRecoveryRow(['retry', 'rephrase', 'documents']);
      }
    }
  };

  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream: true});
    // SSE messages are separated by blank lines (\n\n)
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      // each event has lines starting with "data: "
      for (const line of raw.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (!payload) continue;
        try {
          processEvent(JSON.parse(payload));
        } catch (e) { /* ignore parse errors on individual events */ }
      }
    }
  }
  // If we received no chunks, the backend's fast-path returned nothing —
  // tell caller to fall back to the JSON path.
  return sawChunk;
}

async function fetchHealth() {
  try {
    // /health only returns the diagnostic detail (pg / model) to an
    // authenticated caller — authFetch attaches the bearer token when we
    // have one. Before sign-in the response is a bare {"status":"ok"}, so
    // there is simply nothing to show in the header.
    const r = await authFetch('/health');
    if (!r.ok) { health.textContent = 'health: ' + r.status; return; }
    const d = await r.json();
    if (!d.postgres) { health.textContent = ''; return; }
    health.textContent = `pg: ${d.postgres} · openai: ${d.openai_configured ? d.openai_model : 'NOT configured'}`;
  } catch (e) { health.textContent = 'health: error'; }
}
fetchHealth();
setUiLocale(uiLocale);
(function bindLocaleToggle() {
  const el = document.getElementById('localeToggle');
  if (!el) return;
  el.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-locale]');
    if (!btn) return;
    setUiLocale(btn.dataset.locale);
  });
})();

q.addEventListener('input', () => {
  q.style.height = 'auto';
  q.style.height = Math.min(140, q.scrollHeight) + 'px';
});
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = q.value.trim();
  if (!question) return;
  addMsg('user', question);
  q.value = ''; q.style.height = 'auto';
  sendBtn.disabled = true;

  const t0 = performance.now();
  // STREAMING UX (added 2026-06): when the backend's fast streaming
  // path fires, we read SSE deltas and render text into the assistant
  // bubble as tokens arrive. Time-to-first-token drops from ~18s to
  // ~1-2s for company_profile queries. Falls back to single-JSON
  // when the fast path returns null (other intents). The progressive
  // render uses streaming=true; we accumulate the full text on the
  // client to drive the post-answer feedback/PDF buttons.
  let useStreaming = true;
  try {
    if (useStreaming) {
      const streamed = await tryStreaming(question, t0);
      if (streamed) { sendBtn.disabled = false; q.focus(); return; }
    }
    const r = await authFetch('/api/v1/chat', {
      method: 'POST', credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      // Send the running history so server-side follow-up inheritance
      // ("how do I engage with them" after "tell me about Apple") can
      // anchor on the prior entity. Without this, every turn looked
      // brand-new to the backend and follow-ups returned garbage.
      body: JSON.stringify(chatPayload(question, {stream: false, debug: true})),
    });
    if (r.status === 401) {
      // authFetch already surfaced the login overlay after a failed refresh.
      sendBtn.disabled = false; return;
    }
    const d = await r.json();
    if (d.session_id) {
      setCurrentSession(d.session_id);
      if (SESSIONS_ENABLED) loadSessionList();
    }
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    if (!r.ok || d.error || d.success === false) {
      const e = addMsg('assistant', 'Error: ' + formatApiError(d)); e.classList.add('err');
      addRecoveryRow(['retry', 'rephrase', 'documents']);
    } else {
      let ans = d.answer || '(empty)';
      const srcList = d.sources || d.web_sources || [];
      const plainForHistory = ans; // markdown without citation HTML
      ans = linkCitations(ans, srcList);
      const ansEl = addMsg('assistant', ans);
      if (srcList.length) addSourcesFooter(srcList);
      if (!d.answer || !String(d.answer).trim()) {
        addRecoveryRow(['retry', 'rephrase', 'documents']);
      }
      const tables = (d.trace || []).map(t => `${t.table}(${t.row_count})`).join(', ') || 'no tool calls';
      addMeta(`${dt}s · ${tables}`, d.trace);
      // Quality-loop foundation: thumbs up/down + optional comment
      // under every assistant message. Posts to /api/v1/feedback which
      // appends to feedback.jsonl for later review and golden-test
      // promotion. The buttons stay visible until clicked, then lock.
      addFeedbackRow({
        question,
        answer: plainForHistory,
        trace: d.trace || [],
        debug: d.debug || null,
        webSources: srcList || null,
      });
      // Append the user question and the assistant's answer to history
      // so the NEXT turn carries the context.
      history.push({role: 'user',      content: question});
      history.push({role: 'assistant', content: plainForHistory});
      if (history.length > MAX_HISTORY) {
        history = history.slice(-MAX_HISTORY);
      }
    }
  } catch (err) {
    const e = addMsg('assistant', 'Network error: ' + err.message); e.classList.add('err');
    addRecoveryRow(['retry', 'rephrase']);
  }
  sendBtn.disabled = false;
  q.focus();
});
q.focus();

// ── Document library (same screen as chat; collapsed until toggled) ─────────
const workspace = document.getElementById('workspace');
const docPanel = document.getElementById('docPanel');
const docToggle = document.getElementById('docToggle');
const docClose = document.getElementById('docClose');
const docCount = document.getElementById('docCount');
const docFile = document.getElementById('docFile');
const docVis = document.getElementById('docVis');
const docMsg = document.getElementById('docMsg');
const docList = document.getElementById('docList');
const DOCS_OPEN_KEY = 'misa_docs_open';

function setDocsOpen(open) {
  workspace.classList.toggle('docs-open', open);
  docPanel.hidden = !open;
  docToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  try { sessionStorage.setItem(DOCS_OPEN_KEY, open ? '1' : '0'); } catch (e) {}
  if (open) loadDocLibrary();
}
function setDocMsg(text, ok) {
  docMsg.className = 'doc-msg ' + (ok ? 'ok' : 'err');
  docMsg.textContent = text || '';
}
function setDocCount(n) {
  if (n > 0) { docCount.hidden = false; docCount.textContent = String(n); }
  else { docCount.hidden = true; docCount.textContent = '0'; }
}
function escapeHtml(s) {
  return String(s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function fmtSize(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
  return (n/1024/1024).toFixed(1) + ' MB';
}
async function loadDocLibrary() {
  if (!isAuthed()) {
    docList.innerHTML = '<li class="hint">Sign in to see your library.</li>';
    setDocCount(0);
    return;
  }
  loadConsentPolicy();
  const r = await authFetch('/api/v1/documents');
  if (!r.ok) {
    docList.innerHTML = '<li class="hint">Could not load documents.</li>';
    return;
  }
  const docs = (await r.json()).documents || [];
  setDocCount(docs.length);
  if (!docs.length) {
    docList.innerHTML = '<li class="hint">No documents yet — upload a file to prioritize it in answers.</li>';
    return;
  }
  docList.innerHTML = docs.map(d => `
    <li>
      <div class="name">
        <span class="badge ${d.visibility}">${d.visibility}</span>
        <span class="badge ${d.status}">${d.status}</span>
        ${escapeHtml(d.filename)}
        <div style="color:#86868b;margin-top:2px">${fmtSize(d.byte_size)} · ${escapeHtml(d.owner_username)}</div>
      </div>
      <button class="del" type="button" data-del="${d.id}">Delete</button>
    </li>`).join('');
  docList.querySelectorAll('[data-del]').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (!confirm('Delete this document?')) return;
      const res = await authFetch('/api/v1/documents/' + btn.dataset.del, {method: 'DELETE'});
      if (res.ok) { setDocMsg('Deleted.', true); loadDocLibrary(); }
      else setDocMsg('Delete failed.', false);
    });
  });
}
docToggle.addEventListener('click', () => {
  setDocsOpen(!workspace.classList.contains('docs-open'));
});
docClose.addEventListener('click', () => setDocsOpen(false));
const docConsentChk = document.getElementById('docConsentChk');
const docUploadBtn = document.getElementById('docUploadBtn');

function refreshDocGate() {
  docUploadBtn.disabled = !docConsentChk.checked;
}
docConsentChk.addEventListener('change', refreshDocGate);
refreshDocGate();

let consentPolicyLoaded = false;

async function loadConsentPolicy() {
  if (consentPolicyLoaded) return;
  try {
    const r = await authFetch('/api/v1/documents/consent-policy');
    if (!r.ok) return;
    consentPolicyLoaded = true;
    const p = await r.json();
    document.getElementById('docConsentTitle').textContent = p.title || 'Upload consent declaration';
    document.getElementById('docConsentPre').textContent = p.preamble || '';
    document.getElementById('docConsentTerms').innerHTML =
      (p.terms || []).map(t => '<li>' + escapeHtml(t.text) + '</li>').join('');
  } catch {}
}

docUploadBtn.addEventListener('click', async () => {
  if (!docConsentChk.checked) {
    setDocMsg('Please accept the consent declaration first.', false); return;
  }
  const files = docFile.files;
  if (!files.length) { setDocMsg('Choose a file first.', false); return; }
  let ok = 0; const errors = [];
  for (const f of files) {
    const fd = new FormData();
    fd.append('file', f);
    fd.append('visibility', docVis.value);
    fd.append('consent', 'true');
    const r = await authFetch('/api/v1/documents/upload', {method: 'POST', body: fd});
    if (r.ok) { ok++; continue; }
    let msg = 'upload failed';
    try { msg = (await r.json()).error?.message || msg; } catch {}
    errors.push(f.name + ': ' + msg);
  }
  setDocMsg(
    'Uploaded ' + ok + (errors.length ? ' · ' + errors.join(' · ') : ''),
    errors.length === 0
  );
  docFile.value = '';
  docConsentChk.checked = false;
  refreshDocGate();
  loadDocLibrary();
});
document.getElementById('docIngestBtn').addEventListener('click', async () => {
  const r = await authFetch('/api/v1/documents/ingest', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({visibility: 'org'}),
  });
  if (!r.ok) { setDocMsg('Ingest failed (analyst role required).', false); return; }
  const d = await r.json();
  setDocMsg('Ingested ' + (d.ingested||[]).length + ', dup ' + (d.duplicates||[]).length, true);
  loadDocLibrary();
});
document.getElementById('docRefreshBtn').addEventListener('click', loadDocLibrary);

// ── Chat sessions (persistent history) ──────────────────────────────────────
const sessList = document.getElementById('sessList');
const sessPanel = document.getElementById('sessPanel');
const newChatBtn = document.getElementById('newChatBtn');
const sessSearch = document.getElementById('sessSearch');
const sessShowArchived = document.getElementById('sessShowArchived');
const continueChip = document.getElementById('continueChip');
let sessionCache = [];

function clearChatView() {
  history = [];
  firstMsg = true;
  chat.innerHTML = '<div class="empty">Ask about a company, country, deal, executive, or any allowed table.<br>'
    + 'Examples: <em>Tell me about Alphabet</em> · <em>Tell me about Pakistan as a country</em> · <em>Show me the latest deals</em></div>';
  updateContinueChip(null);
}

function updateContinueChip(session) {
  const ent = session && (session.active_entity || (session.state && session.state.active_entity));
  if (!ent) { continueChip.hidden = true; continueChip.textContent = ''; return; }
  continueChip.hidden = false;
  continueChip.textContent = 'Continue about ' + ent;
  continueChip.onclick = () => {
    q.value = 'Tell me more about ' + ent;
    q.focus();
  };
}

function renderSessionList(sessions) {
  sessionCache = sessions || [];
  if (!sessions.length) {
    sessList.innerHTML = '<li class="sess-empty">No chats yet — send a message or click New.</li>';
    return;
  }
  sessList.innerHTML = sessions.map(s => `
    <li>
      <button type="button" class="sess-item${s.id === currentSessionId ? ' active' : ''}${s.archived_at ? ' archived' : ''}" data-sid="${s.id}">
        <span class="sess-title">${s.pinned ? '📌 ' : ''}${escapeHtml(s.title || 'New chat')}</span>
        <span class="sess-meta">${s.message_count || 0} msgs${s.active_entity ? ' · ' + escapeHtml(s.active_entity) : ''}${s.archived_at ? ' · archived' : ''}</span>
      </button>
      <div class="sess-actions">
        <button type="button" class="sess-ico${s.pinned ? ' on' : ''}" data-pin="${s.id}" title="Pin">📌</button>
        <button type="button" class="sess-ico" data-arch="${s.id}" title="${s.archived_at ? 'Unarchive' : 'Archive'}">${s.archived_at ? '↩' : '⬇'}</button>
        <button type="button" class="sess-ico" data-del="${s.id}" title="Delete">×</button>
      </div>
    </li>`).join('');
  sessList.querySelectorAll('[data-sid]').forEach(btn => {
    btn.addEventListener('click', () => selectSession(btn.dataset.sid));
  });
  sessList.querySelectorAll('[data-pin]').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const s = sessionCache.find(x => x.id === btn.dataset.pin);
      await authFetch('/api/v1/sessions/' + btn.dataset.pin, {
        method: 'PATCH', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pinned: !(s && s.pinned)}),
      });
      await loadSessionList();
    });
  });
  sessList.querySelectorAll('[data-arch]').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const s = sessionCache.find(x => x.id === btn.dataset.arch);
      await authFetch('/api/v1/sessions/' + btn.dataset.arch, {
        method: 'PATCH', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({archived: !(s && s.archived_at)}),
      });
      if (currentSessionId === btn.dataset.arch && !(s && s.archived_at)) {
        setCurrentSession(null);
        clearChatView();
      }
      await loadSessionList();
    });
  });
  sessList.querySelectorAll('[data-del]').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('Delete this chat?')) return;
      const id = btn.dataset.del;
      const r = await authFetch('/api/v1/sessions/' + id, {method: 'DELETE'});
      if (r.ok) {
        if (currentSessionId === id) {
          setCurrentSession(null);
          clearChatView();
        }
        await loadSessionList();
        if (!currentSessionId) await ensureActiveSession();
      }
    });
  });
}

async function loadSessionList() {
  if (!SESSIONS_ENABLED || !isAuthed()) {
    sessList.innerHTML = '<li class="sess-empty">Sign in to see chats.</li>';
    return;
  }
  const params = new URLSearchParams();
  if (sessSearch && sessSearch.value.trim()) params.set('q', sessSearch.value.trim());
  if (sessShowArchived && sessShowArchived.checked) params.set('include_archived', 'true');
  const url = '/api/v1/sessions' + (params.toString() ? ('?' + params.toString()) : '');
  const r = await authFetch(url);
  if (!r.ok) {
    sessList.innerHTML = '<li class="sess-empty">Could not load chats.</li>';
    return;
  }
  const sessions = (await r.json()).sessions || [];
  renderSessionList(sessions);
  const cur = sessions.find(s => s.id === currentSessionId);
  if (cur) updateContinueChip(cur);
  return sessions;
}

async function ensureActiveSession() {
  const sessions = await loadSessionList() || [];
  if (currentSessionId && sessions.some(s => s.id === currentSessionId)) {
    return currentSessionId;
  }
  if (sessions.length) {
    await selectSession(sessions[0].id);
    return currentSessionId;
  }
  const r = await authFetch('/api/v1/sessions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({}),
  });
  if (r.ok) {
    const s = await r.json();
    setCurrentSession(s.id);
    clearChatView();
    await loadSessionList();
  }
  return currentSessionId;
}

async function selectSession(id) {
  if (!id) return;
  const r = await authFetch('/api/v1/sessions/' + id);
  if (!r.ok) return;
  const d = await r.json();
  setCurrentSession(d.session.id);
  updateContinueChip(d.session);
  const msgs = d.messages || [];
  history = [];
  firstMsg = true;
  chat.innerHTML = '';
  if (!msgs.length) {
    clearChatView();
    updateContinueChip(d.session);
  } else {
    firstMsg = false;
    for (const m of msgs) {
      if (m.role === 'user') addMsg('user', m.content);
      else {
        addMsg('assistant', m.content || '');
        if (m.web_sources && m.web_sources.length) {
          const linked = linkCitations(m.content || '', m.web_sources);
          // Re-render last assistant bubble with citation chips.
          const last = chat.querySelector('.msg.assistant:last-of-type');
          if (last) last.innerHTML = safeMarkdownHtml(linked);
          addSourcesFooter(m.web_sources);
        }
      }
      history.push({role: m.role, content: m.content || ''});
    }
    if (history.length > MAX_HISTORY) history = history.slice(-MAX_HISTORY);
  }
  await loadSessionList();
}

async function createNewSession() {
  const r = await authFetch('/api/v1/sessions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({}),
  });
  if (!r.ok) return;
  const s = await r.json();
  setCurrentSession(s.id);
  clearChatView();
  await loadSessionList();
  q.focus();
}

async function initSessions() {
  if (!SESSIONS_ENABLED) {
    if (sessPanel) sessPanel.hidden = true;
    return;
  }
  if (!initSessions._bound) {
    newChatBtn.addEventListener('click', createNewSession);
    let searchTimer = null;
    sessSearch.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => loadSessionList(), 200);
    });
    sessShowArchived.addEventListener('change', () => loadSessionList());
    initSessions._bound = true;
  }
  await ensureActiveSession();
}

// Gate the UI on a token at load (after the document panel is wired).
if (AUTH_DISABLED || getAccess()) { hideLogin(); } else { showLogin(); }
try {
  if (sessionStorage.getItem(DOCS_OPEN_KEY) === '1') setDocsOpen(true);
} catch (e) {}
// Still refresh the count badge while the panel is closed.
if (isAuthed()) loadDocLibrary();
</script>
</body></html>
"""


@app.get("/chat", include_in_schema=False, response_class=HTMLResponse)
async def chat_ui():
    """Minimal browser chat UI for local testing.

    When ``MISA_AUTH_DISABLED`` is set, the login overlay is omitted and
    API calls need no Bearer token. Otherwise JWT login is required.
    """
    html = (
        _CHAT_HTML
        .replace("__AUTH_DISABLED__", "true" if config.AUTH_DISABLED else "false")
        .replace(
            "__SESSIONS_ENABLED__",
            "true" if config.SESSIONS_ENABLED else "false",
        )
    )
    return HTMLResponse(
        html,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )



@app.get("/documents", include_in_schema=False)
async def documents_ui():
    """Documents live on the chat screen; keep /documents as a redirect."""
    return RedirectResponse(url="/chat")


@app.get("/health", summary="Liveness check", tags=["ops"])
async def health(user: "str | None" = Depends(optional_user)):
    """Liveness probe with admin-gated diagnostics (Risk-20-3).

    Unauthenticated (load balancers, uptime probes) get a bare
    `{"status": "ok"}`. Authenticated non-admin callers get the same
    minimal response (no recon). Admin role additionally gets diagnostic
    detail (Postgres, OpenAI, malware scanner, version).
    """
    if user is None or not config.role_at_least(user, config.ROLE_ADMIN):
        return {"status": "ok"}

    pg_ok = False
    try:
        await asyncio.to_thread(get_db)
        pg_ok = True
    except Exception:
        pass

    from app.services.malware_scanner import check_status as check_malware_scan_status
    try:
        scan_status = await asyncio.to_thread(check_malware_scan_status)
    except Exception as e:
        scan_status = {"backend": "unknown", "enabled": True, "available": False, "detail": str(e)[:300]}

    return {
        "status": "ok",
        "postgres": "ok" if pg_ok else "error",
        "openai_configured": config.openai_configured(),
        "openai_model": OPENAI_MODEL,
        "malware_scan": {
            "backend": scan_status["backend"],
            "enabled": scan_status["enabled"],
            "available": scan_status["available"],
            "detail": scan_status["detail"],
        },
        "version": "0.0.10",
    }


@app.get("/health/data", summary="Data-sanity probe", tags=["ops"])
async def health_data(country: str = "India",
                      user: str = Depends(require_role(config.ROLE_ADMIN))):
    """Admin-only data probe. Does NOT return DB host/port/dbname
    (recon for lateral movement) — only row counts for the requested
    country filter.
    """
    def _probe() -> dict:
        conn = get_db()
        out: dict = {}
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM company_profiles")
            out["company_profiles_rows"] = cur.fetchone()[0]
            from app.services.engagement_data import _licensing_predicates
            preds = _licensing_predicates(cur)
            cur.execute(
                f"SELECT COUNT(*) FROM company_profiles WHERE {preds['licensed']}")
            out["licensed_total"] = cur.fetchone()[0]
            cur.execute(
                f"SELECT COUNT(*) FROM company_profiles WHERE {preds['rhq']}")
            out["rhq_total"] = cur.fetchone()[0]
            out["predicate_source"] = preds.get("source")
            like = f"%{country.strip()}%"
            cur.execute(
                "SELECT COUNT(*) FROM company_profiles "
                "WHERE headquarters ILIKE %s", (like,))
            out[f"hq_{country.lower()}_rows"] = cur.fetchone()[0]
            cur.execute(
                f"SELECT COUNT(*) FROM company_profiles "
                f"WHERE headquarters ILIKE %s AND {preds['licensed']}", (like,))
            out[f"hq_{country.lower()}_licensed"] = cur.fetchone()[0]
            cur.execute(
                f"SELECT COUNT(*) FROM company_profiles "
                f"WHERE headquarters ILIKE %s AND {preds['rhq']}", (like,))
            out[f"hq_{country.lower()}_rhq"] = cur.fetchone()[0]
        return out

    result: dict = {"status": "ok"}
    try:
        result["counts"] = await asyncio.to_thread(_probe)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:500]
    return result
