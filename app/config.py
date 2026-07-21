"""Central configuration — reads .env once at import time."""

from __future__ import annotations

import json
import os

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Chat / SQL routing (shared OpenAI key + model)
# ---------------------------------------------------------------------------
# Fail closed: an unset key stays empty (never a usable-looking placeholder),
# so client factories that check truthiness simply don't build a client.
OPENAI_API_KEY: str = (os.getenv("OPENAI_API_KEY") or "").strip()


def openai_configured() -> bool:
    """True only when a real (non-placeholder) OpenAI key is present."""
    return bool(OPENAI_API_KEY) and not OPENAI_API_KEY.startswith("sk-REPLACE")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_RETRIES: int = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "4")))
OPENAI_RETRY_DELAY_SEC: float = float(os.getenv("OPENAI_RETRY_DELAY_SEC", "0.8"))

# ---------------------------------------------------------------------------
# Azure OpenAI (data-residency-preserving alternative to public OpenAI API)
# ---------------------------------------------------------------------------
# When MISA_USE_AZURE_OPENAI=true, the chat/curation/composition pipeline
# routes through an Azure OpenAI deployment in a customer-controlled
# region (UAE North, Sweden Central, etc.) instead of the public
# api.openai.com endpoint. Data stays inside the Azure tenant under
# Microsoft's enterprise DPA — no training, no retention.
#
# The four values below are mandatory ONLY when the flag is true. With
# the flag false (default), nothing changes from the existing OpenAI
# public-API behaviour and these vars can stay unset.
USE_AZURE_OPENAI: bool = (os.getenv("MISA_USE_AZURE_OPENAI") or "").strip().lower() in (
    "1", "true", "yes", "on",
)
AZURE_OPENAI_ENDPOINT: str = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
AZURE_OPENAI_API_KEY: str = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
# API version — pinned to a known-good preview. Bump per Microsoft's
# release notes when needed; the OpenAI Python SDK supports forward
# compatibility within the same major series.
AZURE_OPENAI_API_VERSION: str = (
    os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"
).strip()
# Deployment name on the Azure side. Convention: deploy under the same
# name as the OpenAI model family (e.g. `gpt-4o-mini`) so OPENAI_MODEL
# values across the codebase map cleanly to Azure deployments without
# per-call rewriting.
AZURE_OPENAI_DEPLOYMENT: str = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT") or OPENAI_MODEL
).strip()
# Optional second Azure deployment for advisory / deep curation (gpt-4o tier).
# When unset, falls back to AZURE_OPENAI_DEPLOYMENT (legacy single-deployment).
AZURE_OPENAI_ADVISORY_DEPLOYMENT: str = (
    os.getenv("AZURE_OPENAI_ADVISORY_DEPLOYMENT") or AZURE_OPENAI_DEPLOYMENT
).strip()


def openai_max_completion_tokens_kw() -> dict:
    raw = (os.getenv("OPENAI_MAX_COMPLETION_TOKENS") or "3072").strip()
    if raw in ("", "0"):
        return {}
    try:
        n = int(raw)
    except ValueError:
        return {}
    return {"max_completion_tokens": n} if n > 0 else {}

# Model for the strategic-advisory report path. Advisory documents are
# long consultant-grade deliverables; the mini-tier chat model produces
# template-grade filler ("growing demand", "lucrative opportunities"),
# so this defaults to the full gpt-4o tier rather than OPENAI_MODEL.
ADVISORY_MODEL: str = (
    os.getenv("MISA_ADVISORY_OPENAI_MODEL") or "gpt-4o"
).strip()

# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------
# The same question should yield substantially the SAME answer each time —
# a government decision-support tool must not reword its intelligence on
# every run. Content-generating calls therefore use a low temperature and a
# fixed seed. Note: OpenAI is only *best-effort* deterministic even at
# temperature 0 (backend/MoE non-determinism), so minor wording drift can
# still occur; this removes the large, deliberate randomness, not all of it.
CHAT_TEMPERATURE: float = float(os.getenv("MISA_CHAT_TEMPERATURE", "0") or "0")
_seed_raw = (os.getenv("MISA_CHAT_SEED") or "7").strip()

def openai_determinism_kw() -> dict:
    """temperature + seed for content-generating calls, so the same
    question reproduces the same answer as closely as the API allows.
    Set MISA_CHAT_SEED='' to disable seeding."""
    kw: dict = {"temperature": CHAT_TEMPERATURE}
    if _seed_raw not in ("", "off", "none"):
        try:
            kw["seed"] = int(_seed_raw)
        except ValueError:
            pass
    return kw

def curation_model_for_depth(depth: str | None, default: str) -> str:
    """Model tiering for row-backed curation: quick facts stay on the
    cheap chat model; anything needing analysis (operational_detail,
    executive_briefing, strategic_recommendation) uses the advisory
    tier. The mini tier reliably produces 2-sentence answers and skips
    the mandated Strategic Read section no matter how directive the
    prompt is — model quality is the binding constraint, not prompt
    wording. Set MISA_DEEP_CURATION=false to keep everything on the
    default chat model (cost-saver switch)."""
    if _env_bool("MISA_DEEP_CURATION", True) and depth and depth != "simple_fact":
        return ADVISORY_MODEL or default
    return default

def openai_advisory_max_tokens_kw() -> dict:
    """Token budget for the strategic-advisory report path. Advisory
    answers are full consultant-grade documents (sector tables, tiered
    deep-dives, MISA targeting recommendations), so they need a far
    larger completion budget than the standard 3072-token chat cap."""
    raw = (os.getenv("MISA_ADVISORY_MAX_COMPLETION_TOKENS") or "8000").strip()
    if raw in ("", "0"):
        return {}
    try:
        n = int(raw)
    except ValueError:
        return {}
    return {"max_completion_tokens": n} if n > 0 else {}

def max_history_user_turns() -> int:
    raw = (os.getenv("MISA_MAX_HISTORY_USER_TURNS") or "12").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 12

# ---------------------------------------------------------------------------
# Chat curation / general-knowledge fallback
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


# ---------------------------------------------------------------------------
# Deployment environment (gates fail-closed production defaults)
# ---------------------------------------------------------------------------
# MISA_ENV=production|prod → IS_PRODUCTION. Local/dev leave unset or use
# "development". Production refuses weak secrets, open CORS, plaintext
# bootstrap passwords, docs exposure, and memory-only rate limits with
# multiple workers (see validate_security_config / serve.py).
_ENV_RAW: str = (os.getenv("MISA_ENV") or os.getenv("APP_ENV") or "development").strip().lower()
IS_PRODUCTION: bool = _ENV_RAW in ("production", "prod")

# Role hierarchy for RBAC (viewer < analyst < admin).
ROLE_VIEWER = "viewer"
ROLE_ANALYST = "analyst"
ROLE_ADMIN = "admin"
ROLE_RANK: "dict[str, int]" = {
    ROLE_VIEWER: 0,
    ROLE_ANALYST: 1,
    ROLE_ADMIN: 2,
}

# Send (privacy-filtered) DB rows to OpenAI to compose insight-rich answers.
CHAT_CURATION_ENABLED: bool = _env_bool("MISA_CHAT_CURATION", True)
# When the DB returns no rows, let OpenAI answer from general knowledge.
CHAT_FALLBACK_ENABLED: bool = _env_bool("MISA_CHAT_FALLBACK", True)
# Whether OpenAI may retain these requests. Default False for data privacy.
CHAT_OPENAI_STORE: bool = _env_bool("MISA_CHAT_OPENAI_STORE", False)
# Max rows sent to OpenAI for curation (caps token use and data exposure).
CHAT_CURATION_MAX_ROWS: int = max(1, int(os.getenv("MISA_CHAT_CURATION_MAX_ROWS", "15")))

# ---------------------------------------------------------------------------
# Engagement dossier (separate key + model; falls back to shared)
# ---------------------------------------------------------------------------
ENGAGEMENT_OPENAI_KEY: str = (
    os.getenv("MISA_ENGAGEMENT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
).strip()
ENGAGEMENT_MODEL: str = (
    os.getenv("MISA_ENGAGEMENT_OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
).strip()

# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------
DB_CONFIG: dict = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     os.getenv("PG_PORT", "5432"),
    "dbname":   os.getenv("PG_DB",   "postgres"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}
PG_CONNECT_RETRIES: int = max(1, int(os.getenv("PG_CONNECT_RETRIES", "4")))
PG_RETRY_DELAY_SEC: float = float(os.getenv("PG_RETRY_DELAY_SEC", "0.35"))

# Max concurrent synchronous DB operations per worker process. The app runs
# blocking DB work on a bounded thread pool and keeps ONE Postgres connection
# per pool thread (see app/database.py), so this value is also the ceiling on
# open connections per worker. Keep (workers x MISA_DB_MAX_CONCURRENCY) safely
# under the server's `max_connections`. Default 10 → e.g. 4 workers = 40 conns.
DB_MAX_CONCURRENCY: int = max(1, int(os.getenv("MISA_DB_MAX_CONCURRENCY", "10")))

# ---------------------------------------------------------------------------
# DB query hardening (Risk-20-5) — response-transparent by design.
# ---------------------------------------------------------------------------
DB_READONLY: bool = _env_bool("MISA_DB_READONLY", True)
QUERY_MAX_LIMIT: int = max(1, int(os.getenv("MISA_QUERY_MAX_LIMIT", "200")))
CHAT_MAX_ROWS_PER_TURN: int = max(1, int(os.getenv("MISA_CHAT_MAX_ROWS_PER_TURN", "200")))

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
LOG_TURNS: bool = os.getenv("MISA_LOG_TURNS", "").strip().lower() in ("1", "true", "yes")
LOG_FILE: str = (os.getenv("MISA_LOG_FILE") or "").strip()

MAX_USER_MESSAGE_CHARS: int = 12_000

# ---------------------------------------------------------------------------
# CORS (Risk-20-3 — restricts the wildcard "*" origin)
# ---------------------------------------------------------------------------
# Comma-separated list of exact origins (scheme + host + port, NO trailing
# slash, NO path) allowed to call this API from a browser — e.g. every
# frontend and backend URL that legitimately talks to it:
#   MISA_CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com,http://localhost:3000
# Unset/empty falls back to "*" (open) ONLY for local dev convenience —
# every real deployment must set this explicitly. A misconfigured entry
# (typo, wrong port) fails closed: that origin's browser requests are
# blocked, not silently widened.
def _parse_cors_origins() -> list[str]:
    raw = (os.getenv("MISA_CORS_ALLOWED_ORIGINS") or "").strip()
    if not raw:
        # Production must never default to "*"; local/dev keeps the open
        # fallback for convenience (warned loudly at startup).
        return [] if IS_PRODUCTION else ["*"]
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


CORS_ALLOWED_ORIGINS: list[str] = _parse_cors_origins()

# ---------------------------------------------------------------------------
# Authentication — bootstrap account (Risk-20-3)
# ---------------------------------------------------------------------------
# HTTP Basic Auth has been replaced by JWT bearer tokens (see below and
# app/auth.py). API_USERNAME/API_PASSWORD are retained as an always-present
# "bootstrap" account so existing .env files keep a working login and there
# is always at least one account that can obtain a token. Additional named
# accounts are provisioned via MISA_AUTH_USERS (hashed — see AUTH_USERS).
#
# In production the plaintext bootstrap is DISABLED by default
# (MISA_ALLOW_PLAINTEXT_BOOTSTRAP=false): provision bcrypt accounts via
# MISA_AUTH_USERS instead. Local/dev keeps the bootstrap for convenience.
API_USERNAME: str = (os.getenv("API_USERNAME") or "").strip()
API_PASSWORD: str = (os.getenv("API_PASSWORD") or "").strip()
ALLOW_PLAINTEXT_BOOTSTRAP: bool = _env_bool(
    "MISA_ALLOW_PLAINTEXT_BOOTSTRAP", default=not IS_PRODUCTION
)
# Role assigned to the bootstrap account when it is enabled.
BOOTSTRAP_ROLE: str = (
    (os.getenv("MISA_BOOTSTRAP_ROLE") or ROLE_ADMIN).strip().lower()
)
if BOOTSTRAP_ROLE not in ROLE_RANK:
    BOOTSTRAP_ROLE = ROLE_ADMIN

# Open /chat and /api/v1/* without login (local demos). Refused in production
# by validate_required_secrets(). When enabled, verify_credentials returns
# AUTH_DISABLED_USERNAME with admin privileges.
AUTH_DISABLED: bool = _env_bool("MISA_AUTH_DISABLED", False)
AUTH_DISABLED_USERNAME: str = (
    os.getenv("MISA_AUTH_DISABLED_USERNAME")
    or API_USERNAME
    or "local"
).strip() or "local"

# ---------------------------------------------------------------------------
# JWT authentication (Risk-20-3 — replaces HTTP Basic Auth)
# ---------------------------------------------------------------------------
# Self-issued bearer tokens: POST /api/v1/auth/login exchanges username +
# password for a short-lived access token and a longer-lived refresh token,
# both signed with JWT_SECRET_KEY (HS256). All /api/v1/* endpoints then
# require `Authorization: Bearer <access-token>`.
#
# JWT_SECRET_KEY is mandatory in any deployment that serves traffic — if it
# is unset, the auth dependency returns 503 (mirroring the old missing-Basic-
# Auth-config behaviour) rather than silently accepting unsigned tokens.
# Generate one with e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
JWT_SECRET_KEY: str = (os.getenv("JWT_SECRET_KEY") or "").strip()
# Algorithm is constrained to a symmetric HMAC allowlist. This makes the
# classic `alg=none` (and asymmetric-key-confusion) downgrade impossible even
# if the env var is tampered with — an unknown/empty value falls back to HS256.
_ALLOWED_JWT_ALGS: "frozenset[str]" = frozenset({"HS256", "HS384", "HS512"})
_jwt_alg_raw: str = (os.getenv("JWT_ALGORITHM") or "HS256").strip()
JWT_ALGORITHM: str = _jwt_alg_raw if _jwt_alg_raw in _ALLOWED_JWT_ALGS else "HS256"
# Bind tokens to this issuer/audience so a token minted for another service
# signed with the same secret can't be replayed here (verified on decode).
JWT_ISSUER: str = (os.getenv("MISA_JWT_ISSUER") or "misa-intelligence-api").strip()
JWT_AUDIENCE: str = (os.getenv("MISA_JWT_AUDIENCE") or "misa-intelligence-api").strip()
JWT_ACCESS_TTL_MIN: int = max(1, int(os.getenv("JWT_ACCESS_TTL_MIN", "30")))
JWT_REFRESH_TTL_DAYS: int = max(1, int(os.getenv("JWT_REFRESH_TTL_DAYS", "7")))
# Minimum acceptable signing-secret length (bytes). Shorter keys are brute-
# forceable for HS256; production refuses to boot (see validate_security_config).
JWT_SECRET_MIN_LEN: int = max(16, int(os.getenv("MISA_JWT_SECRET_MIN_LEN", "32")))


def jwt_secret_is_strong() -> bool:
    return len((JWT_SECRET_KEY or "").strip()) >= JWT_SECRET_MIN_LEN


# ---------------------------------------------------------------------------
# HTTP hardening (security headers, docs exposure, body-size cap)
# ---------------------------------------------------------------------------
# Interactive API docs (/docs, /redoc, /openapi.json) expose the full API
# surface. Default OFF in production; ON for local/dev convenience.
ENABLE_DOCS: bool = _env_bool("MISA_ENABLE_DOCS", default=not IS_PRODUCTION)
# HSTS is only meaningful over HTTPS. Default ON in production (assume TLS
# termination); OFF locally so plain-HTTP boxes aren't pinned incorrectly.
HSTS_ENABLED: bool = _env_bool("MISA_HSTS_ENABLED", default=IS_PRODUCTION)
HSTS_MAX_AGE: int = max(0, int(os.getenv("MISA_HSTS_MAX_AGE", "31536000")))
# Global request-body ceiling (bytes). Rejects oversized payloads with 413
# before they're buffered into memory. Default 10 MiB comfortably covers the
# 5 MiB business-card upload plus multipart overhead.
MAX_REQUEST_BYTES: int = max(
    64 * 1024, int(os.getenv("MISA_MAX_REQUEST_BYTES", str(10 * 1024 * 1024)))
)
# Content-Security-Policy for the served HTML UI. Allows marked + DOMPurify
# from jsdelivr (chat UI sanitizes model markdown before innerHTML).
CONTENT_SECURITY_POLICY: str = (
    os.getenv("MISA_CONTENT_SECURITY_POLICY")
    or (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "font-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )
).strip()


def _load_auth_users() -> "tuple[dict[str, str], dict[str, str]]":
    """Parse MISA_AUTH_USERS JSON list of
    {username, password_hash, role?} into (hashes, roles) maps.
    Malformed entries are skipped rather than crashing startup."""
    raw = (os.getenv("MISA_AUTH_USERS") or "").strip()
    if not raw:
        return {}, {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}, {}
    users: dict[str, str] = {}
    roles: dict[str, str] = {}
    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            username = str(entry.get("username") or "").strip()
            pw_hash = str(entry.get("password_hash") or "").strip()
            role = str(entry.get("role") or ROLE_ANALYST).strip().lower()
            if role not in ROLE_RANK:
                role = ROLE_ANALYST
            if username and pw_hash:
                users[username] = pw_hash
                roles[username] = role
    return users, roles


# username → bcrypt hash. Additional accounts beyond the bootstrap one.
AUTH_USERS: "dict[str, str]"
AUTH_USER_ROLES: "dict[str, str]"
AUTH_USERS, AUTH_USER_ROLES = _load_auth_users()


def user_role(username: str) -> str:
    """Return the RBAC role for an authenticated username."""
    if not username:
        return ROLE_VIEWER
    if AUTH_DISABLED and username == AUTH_DISABLED_USERNAME:
        return ROLE_ADMIN
    if username in AUTH_USER_ROLES:
        return AUTH_USER_ROLES[username]
    if username == API_USERNAME:
        return BOOTSTRAP_ROLE
    return ROLE_VIEWER


def role_at_least(username: str, minimum: str) -> bool:
    return ROLE_RANK.get(user_role(username), 0) >= ROLE_RANK.get(minimum, 0)

# ---------------------------------------------------------------------------
# Rate limiting — per-client sliding-window throttle for abuse-prone,
# low-value-per-call endpoints (feedback submission, etc). In-memory,
# per-process: fine for the current single-connection/few-worker
# deployment (app/database.py already has a similar per-process
# constraint); if this is ever horizontally scaled behind a load
# balancer, swap the backing store for Redis (see rate_limiter.py).
# ---------------------------------------------------------------------------
#
# Master switch: MISA_RATE_LIMIT_ENABLED=false disables app-wide (e.g.
# behind an API gateway that already throttles). Per-endpoint limits are
# the "conservative" tier — comfortable for humans, tight on automation.
RATE_LIMIT_ENABLED: bool = (
    os.getenv("MISA_RATE_LIMIT_ENABLED", "true").strip().lower()
    in ("1", "true", "yes", "on")
)

# Rate-limit backing store.
#   "memory" (default) — per-process, in-memory sliding window. Correct
#            for a single worker; each worker counts independently, so N
#            workers allow up to N x the configured limit in aggregate.
#   "redis"  — shared sliding window across all workers/replicas. Required
#            for accurate global limits behind a load balancer or when
#            serve.py runs multiple workers. Falls back to in-memory
#            automatically if Redis can't be reached (fail-open on the
#            store, not on throttling).
RATE_LIMIT_BACKEND: str = (os.getenv("MISA_RATE_LIMIT_BACKEND") or "memory").strip().lower()
REDIS_URL: str = (
    os.getenv("MISA_REDIS_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
).strip()
# Namespaced so rate-limit keys don't collide with other Redis users.
RATE_LIMIT_REDIS_PREFIX: str = (os.getenv("MISA_RATE_LIMIT_REDIS_PREFIX") or "misa:rl").strip()


def _rate_limit(env_prefix: str, default_max: int, default_window: int = 60) -> "tuple[int, float]":
    """Read `<PREFIX>_RATE_LIMIT_MAX` / `<PREFIX>_RATE_LIMIT_WINDOW_SEC`
    with sane fallbacks. Returns (max_requests, window_seconds)."""
    mx = max(1, int(os.getenv(f"{env_prefix}_RATE_LIMIT_MAX", str(default_max))))
    win = float(os.getenv(f"{env_prefix}_RATE_LIMIT_WINDOW_SEC", str(default_window)))
    return mx, win


# (max_requests, window_seconds) per endpoint family.
FEEDBACK_RATE_LIMIT: "tuple[int, float]" = _rate_limit("FEEDBACK", 20)
# Login/refresh — tight, IP-keyed (pre-auth) to blunt credential brute force.
AUTH_RATE_LIMIT: "tuple[int, float]" = _rate_limit("AUTH", 10)
CHAT_RATE_LIMIT: "tuple[int, float]" = _rate_limit("CHAT", 20)
SEARCH_RATE_LIMIT: "tuple[int, float]" = _rate_limit("SEARCH", 60)
ENGAGEMENT_RATE_LIMIT: "tuple[int, float]" = _rate_limit("ENGAGEMENT", 10)
BUSINESS_CARD_RATE_LIMIT: "tuple[int, float]" = _rate_limit("BUSINESS_CARD", 15)
# PDF export renders a document server-side (CPU-bound). Tighter than
# chat because each call is comparatively expensive and low-frequency.
PDF_EXPORT_RATE_LIMIT: "tuple[int, float]" = _rate_limit("PDF_EXPORT", 10)
# Cheap static metadata GETs (/questions, /engagement/modes). Generous
# ceiling — this only exists to blunt trivial hammering, not real use.
META_RATE_LIMIT: "tuple[int, float]" = _rate_limit("META", 60)

# Risk-20-6 (inference attacks): count-only `query_table` calls are the
# COUNT_ONLY_RATE_LIMIT_MAX / COUNT_ONLY_RATE_LIMIT_WINDOW_SEC.
_COUNT_ONLY_PER_TURN_HEADROOM = 10
COUNT_ONLY_RATE_LIMIT: "tuple[int, float]" = _rate_limit(
    "COUNT_ONLY",
    CHAT_RATE_LIMIT[0] * _COUNT_ONLY_PER_TURN_HEADROOM,
    int(CHAT_RATE_LIMIT[1]),
)

# Back-compat aliases (feedback endpoint referenced these names first).
FEEDBACK_RATE_LIMIT_MAX: int = int(FEEDBACK_RATE_LIMIT[0])
FEEDBACK_RATE_LIMIT_WINDOW_SEC: float = FEEDBACK_RATE_LIMIT[1]

# ---------------------------------------------------------------------------
# Malware scanning (business-card upload)
# ---------------------------------------------------------------------------
# No daemon, no separate container — a heuristic pre-check followed by
_MALWARE_SCAN_YAML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "malware_scan.yaml",
)


def _load_malware_scan_config() -> dict:
    try:
        with open(_MALWARE_SCAN_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("malware_scan") or {}
    except (OSError, yaml.YAMLError):
        # Missing/unreadable/malformed file — scan disabled rather than
        # crashing the whole app on a config typo.
        return {}


_malware_scan_cfg = _load_malware_scan_config()

# Env overrides win over YAML so deployers can flip backend without editing
# the checked-in file (e.g. MISA_MALWARE_SCAN_BACKEND=clamscan in prod).
MALWARE_SCAN_ENABLED: bool = _env_bool(
    "MISA_MALWARE_SCAN_ENABLED",
    default=bool(_malware_scan_cfg.get("enabled", False)),
)
_backend_env = (os.getenv("MISA_MALWARE_SCAN_BACKEND") or "").strip().lower()
MALWARE_SCAN_BACKEND: str = (
    _backend_env or str(_malware_scan_cfg.get("backend") or "none").strip().lower()
)
CLAMSCAN_PATH: str = (
    (os.getenv("MISA_CLAMSCAN_PATH") or "").strip()
    or str(_malware_scan_cfg.get("clamscan_path") or "").strip()
)
WINDOWS_DEFENDER_PATH: str = (
    (os.getenv("MISA_WINDOWS_DEFENDER_PATH") or "").strip()
    or str(_malware_scan_cfg.get("windows_defender_path") or "").strip()
)
CLAMSCAN_TIMEOUT_SEC: float = float(
    os.getenv("MISA_CLAMSCAN_TIMEOUT_SEC")
    or _malware_scan_cfg.get("timeout_sec", 30)
)
# When a backend IS configured but the scan itself fails (binary
# missing, process error, timeout) — default is fail-CLOSED (reject
# the upload) since an operator who turned scanning on is asserting
# "uploads must be scanned"; a broken scanner shouldn't silently
# degrade to "unscanned but accepted". Set true to fail-open instead
# (accept on scan failure, prioritizing availability).
MALWARE_SCAN_FAIL_OPEN: bool = _env_bool(
    "MISA_MALWARE_SCAN_FAIL_OPEN",
    default=bool(_malware_scan_cfg.get("fail_open", False)),
)


def validate_security_config() -> list[str]:
    """Return fatal misconfiguration messages for the current environment.

    In production these are refuse-to-boot errors. Locally they are warnings
    only (see app/main.py lifespan)."""
    errors: list[str] = []
    if not JWT_SECRET_KEY:
        errors.append("JWT_SECRET_KEY is unset.")
    elif not jwt_secret_is_strong():
        errors.append(
            f"JWT_SECRET_KEY is shorter than {JWT_SECRET_MIN_LEN} characters."
        )
    if IS_PRODUCTION:
        if AUTH_DISABLED:
            errors.append(
                "MISA_AUTH_DISABLED=true is not allowed in production — "
                "set MISA_AUTH_DISABLED=false and require JWT login."
            )
        if not CORS_ALLOWED_ORIGINS or CORS_ALLOWED_ORIGINS == ["*"]:
            errors.append(
                "MISA_CORS_ALLOWED_ORIGINS must be an explicit allowlist in production "
                "(wildcard '*' is refused)."
            )
        if ALLOW_PLAINTEXT_BOOTSTRAP and API_PASSWORD:
            errors.append(
                "Plaintext bootstrap password is enabled in production — set "
                "MISA_ALLOW_PLAINTEXT_BOOTSTRAP=false and provision MISA_AUTH_USERS."
            )
        if not AUTH_USERS and not (ALLOW_PLAINTEXT_BOOTSTRAP and API_USERNAME and API_PASSWORD):
            errors.append(
                "No auth accounts configured — provision MISA_AUTH_USERS "
                "(bcrypt) or temporarily allow the bootstrap account."
            )
        if not MALWARE_SCAN_ENABLED or MALWARE_SCAN_BACKEND in ("", "none"):
            errors.append(
                "Malware scanning backend is disabled — set "
                "MISA_MALWARE_SCAN_BACKEND=clamscan (or defender) for production uploads."
            )
        if ENABLE_DOCS:
            errors.append(
                "Interactive API docs are enabled — set MISA_ENABLE_DOCS=false in production."
            )
    return errors


def require_redis_rate_limit_for_workers(workers: int) -> str | None:
    """Return an error string if multi-worker prod lacks a shared rate-limit store."""
    if workers > 1 and RATE_LIMIT_BACKEND != "redis":
        msg = (
            f"MISA_WORKERS={workers} with MISA_RATE_LIMIT_BACKEND={RATE_LIMIT_BACKEND!r}: "
            "each worker counts independently (effective limit × workers). "
            "Set MISA_RATE_LIMIT_BACKEND=redis for accurate global throttling."
        )
        if IS_PRODUCTION:
            return msg
    return None

# ---------------------------------------------------------------------------
# Document library (upload + folder ingest + document-first chat)
# ---------------------------------------------------------------------------
DOCUMENTS_ENABLED: bool = _env_bool("MISA_DOCUMENTS_ENABLED", True)
# memory = process-local (tests / no PG write grants); postgres = durable FTS.
DOCUMENTS_BACKEND: str = (
    os.getenv("MISA_DOCUMENTS_BACKEND") or "postgres"
).strip().lower()
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_ROOT: str = (
    os.getenv("MISA_DOCUMENTS_ROOT")
    or os.path.join(_REPO_ROOT, "data", "documents")
).strip()
DOCUMENTS_INGEST_DIR: str = (
    os.getenv("MISA_DOCUMENTS_INGEST_DIR")
    or os.path.join(_REPO_ROOT, "data", "documents_inbox")
).strip()
DOCUMENTS_MAX_BYTES: int = max(
    64 * 1024, int(os.getenv("MISA_DOCUMENTS_MAX_BYTES", str(20 * 1024 * 1024)))
)
# Rate limit for document upload / ingest (analyst+).
DOCUMENTS_RATE_LIMIT: "tuple[int, float]" = _rate_limit("DOCUMENTS", 20)
# Minimum FTS / retrieval score to short-circuit chat to document answers.
DOCUMENTS_RETRIEVAL_MIN_SCORE: float = float(
    os.getenv("MISA_DOCUMENTS_RETRIEVAL_MIN_SCORE", "0.12")
)
DOCUMENTS_RETRIEVAL_TOP_K: int = max(1, int(os.getenv("MISA_DOCUMENTS_RETRIEVAL_TOP_K", "6")))
# How document hits interact with live web search during chat:
#   hybrid    — document-primary answer + web complement (default)
#   docs_first — strong doc hit short-circuits; no internet (legacy)
#   docs_only — never consult the internet when docs (or at all for exec augment)
_DOCUMENTS_WEB_MODE_RAW = (os.getenv("MISA_DOCUMENTS_WEB_MODE") or "hybrid").strip().lower()
DOCUMENTS_WEB_MODE: str = (
    _DOCUMENTS_WEB_MODE_RAW
    if _DOCUMENTS_WEB_MODE_RAW in ("hybrid", "docs_first", "docs_only")
    else "hybrid"
)

# ---------------------------------------------------------------------------
# Chat sessions (persistent history per user)
# ---------------------------------------------------------------------------
SESSIONS_ENABLED: bool = _env_bool("MISA_SESSIONS_ENABLED", True)
SESSIONS_BACKEND: str = (
    os.getenv("MISA_SESSIONS_BACKEND") or "postgres"
).strip().lower()
SESSIONS_RATE_LIMIT: "tuple[int, float]" = _rate_limit("SESSIONS", 60)
# How many prior user turns enter the model prompt (archive can be longer).
SESSIONS_PROMPT_HISTORY_TURNS: int = max(
    0, int(os.getenv("MISA_SESSIONS_PROMPT_HISTORY_TURNS", "6"))
)
# Cap each prior assistant message when building prompt history.
SESSIONS_PROMPT_ASSISTANT_CHARS: int = max(
    120, int(os.getenv("MISA_SESSIONS_PROMPT_ASSISTANT_CHARS", "400"))
)
# Soft-archive sessions idle longer than this many days (0 = disabled).
SESSIONS_IDLE_ARCHIVE_DAYS: int = max(
    0, int(os.getenv("MISA_SESSIONS_IDLE_ARCHIVE_DAYS", "90"))
)
# Hard-delete archived sessions older than this many days (0 = disabled).
SESSIONS_HARD_DELETE_DAYS: int = max(
    0, int(os.getenv("MISA_SESSIONS_HARD_DELETE_DAYS", "365"))
)


# ---------------------------------------------------------------------------
# LinkedIn profile resolution (business card reader enrichment)
# ---------------------------------------------------------------------------
# Three pluggable search backends discover the LinkedIn profile URL(s) for an
# extracted business card:
#   - "ddg"        → DuckDuckGo via the `ddgs` library (free, no key, best-effort)
#   - "serp"       → Serper.dev Google Search API (paid, reliable, needs a key)
#   - "playwright" → headless Chromium scraping Bing results (free, more
#                    profiles + richer snippets; never touches LinkedIn directly)
# The provider is chosen per-request (?provider=); this is just the default.
LINKEDIN_DEFAULT_PROVIDER: str = (os.getenv("LINKEDIN_DEFAULT_PROVIDER") or "ddg").strip().lower()
# Serper.dev credentials/endpoint (only needed when provider=serp is used).
SERPER_API_KEY: str = (os.getenv("SERPER_API_KEY") or "").strip()
SERPER_BASE_URL: str = (os.getenv("SERPER_BASE_URL") or "https://google.serper.dev/search").strip()
# Per-query result cap and overall network timeout for profile search.
LINKEDIN_MAX_RESULTS: int = max(1, int(os.getenv("LINKEDIN_MAX_RESULTS", "50")))
LINKEDIN_SEARCH_TIMEOUT_SEC: float = float(os.getenv("LINKEDIN_SEARCH_TIMEOUT_SEC", "12"))
# Max distinct query variations issued per resolution (name+company, name+title,
# name+location, name+email-brand, …). Higher = more candidates, more cost/time.
LINKEDIN_MAX_QUERIES: int = max(1, int(os.getenv("LINKEDIN_MAX_QUERIES", "6")))
# Bing endpoint used by the Playwright provider.
BING_SEARCH_URL: str = (os.getenv("BING_SEARCH_URL") or "https://www.bing.com/search").strip()
# Score (0-1) at/above which a single candidate is treated as an EXACT match
# (returns one URL instead of an array of candidates).
LINKEDIN_EXACT_NAME_THRESHOLD: float = float(os.getenv("LINKEDIN_EXACT_NAME_THRESHOLD", "0.95"))
LINKEDIN_EXACT_COMPANY_THRESHOLD: float = float(os.getenv("LINKEDIN_EXACT_COMPANY_THRESHOLD", "0.85"))
