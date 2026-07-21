"""
Live introspection of public.* tables with safety denylists.

The chat pipeline used to be hardcoded to a single table (`company_profiles`).
This module discovers what's actually in Postgres so the chat layer can ask
questions about ANY business table, while denying Django/Laravel auth/infra
tables and sensitive columns (passwords, tokens, secrets).

Two layers of denial are applied:
  * Table-level denylist (Django auth, oauth, sessions, jobs, cache, etc.).
  * Column-level denylist applied even on allowed tables (password, token,
    secret, hash, salt, otp, api_key) — these columns are excluded from
    filterable AND from rows returned to the chat/curation pipeline.

Environment overrides:
  MISA_CHAT_TABLES_ALLOW   comma-separated table allowlist (still respects
                           the deny rules); if unset, all non-denied tables
                           in `public` are exposed.
  MISA_CHAT_TABLES_DENY    comma-separated *extra* tables to deny.
"""

from __future__ import annotations

import os
import re
from threading import Lock
from typing import Any

# ---------------------------------------------------------------------------
# Safety lists
# ---------------------------------------------------------------------------

# Exact name OR prefix matches (compared case-insensitively). Covers Django,
# Laravel/Sanctum, OAuth2 provider, Spatie permissions, sessions, jobs, cache.
_DENY_TABLE_PREFIXES: tuple[str, ...] = (
    "auth_", "authtoken_",
    "django_",
    "oauth2_",
    "model_has_", "role_has_",
)
_DENY_TABLE_EXACT: frozenset[str] = frozenset({
    "users", "permissions", "roles",
    "personal_access_tokens", "password_reset_tokens", "device_tokens",
    "nafath_callbacks",
    "sessions", "session",
    "cache", "cache_locks",
    "failed_jobs", "jobs", "job_batches",
    "migrations", "django_migrations",
    "import_progress",
    "media",
})

# Table prefixes belonging to the OTHER applications that share this Postgres
_DENY_TABLE_PREFIXES = _DENY_TABLE_PREFIXES + (
    "wealth_", "family_", "client_", "crm_", "kyc_", "aml_", "ir_", "portfolio_",
    "document_", "documents_", "audit_", "log_", "logs_",
)

# Columns matching these substrings are dropped from filterable AND from
# rows returned to the chat/curation pipeline (defense in depth). Covers
# credential-style fields plus government-ID / banking PII that must never
# surface via the query tool.
_DENY_COLUMN_SUBSTR: tuple[str, ...] = (
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "hash", "salt", "otp", "ssn", "private_key",
    "passport", "national_id", "nationalid", "iban", "swift",
    "credential", "credit_card", "card_number", "cvv", "pin_code",
)

# ---------------------------------------------------------------------------
# Least-privilege table allowlist (Risk-20-5)
# ---------------------------------------------------------------------------
_DEFAULT_CHAT_TABLES_ALLOW: frozenset[str] = frozenset({
    # ── company intelligence ────────────────────────────────────────
    "company_profiles", "company_executives", "company_news",
    "company_ai_insights", "company_business_units", "company_competitors",
    "company_financial_performances", "company_geographic_revenues",
    "company_global_presences", "company_corporate_activities",
    # ── country intelligence ────────────────────────────────────────
    "country_profiles", "country_vision_outlooks",
    "country_strategic_opportunities", "country_free_zones",
    "country_human_capitals", "country_infrastructures", "country_insights",
    "country_key_indicators", "country_policy_incentives",
    "country_recent_reforms", "country_risk_stabilities",
    "country_top_commodities", "country_trade_partners",
    "country_associated_companies", "country_footprints", "countries",
    # ── sectors ─────────────────────────────────────────────────────
    "sectors", "sub_sectors", "sector_key_numbers", "sector_license",
    "focused_sectors",
    # ── investment / engagement core ────────────────────────────────
    "opportunities", "opportunity_data", "suggested_opportunities",
    "strategic_investors", "fdi_data", "executives", "deals",
    "rhq_licenses", "rhq_new_data",
    "meetings", "engagements", "meeting_notes", "misa_contact_details",
    "latest_interactions",
    "bus_data",            # misa_entity_id, sector, revenue_bn, licence dates
    "leads",               # investment-opportunity pipeline (stage/status)
    "reports", "report_types",   # MISA report catalogue + its lookup
    "match_outputs",       # company ↔ opportunity matching / AI scores
    "company_contact_records",   # company-side contacts (company_profile_id)
})

# Columns that look like a human-meaningful name (used for fuzzy entity search).
_NAME_COL_EXACT: frozenset[str] = frozenset({
    "name", "full_name", "fullname", "display_name", "title", "label",
    "legal_name", "trade_name", "common_name",
})
_NAME_COL_SUFFIX: tuple[str, ...] = ("_name", "_title", "_label")

# Column data types that text-substring search can apply to.
_TEXT_TYPES: frozenset[str] = frozenset({
    "text", "character varying", "varchar", "character", "char", "citext",
})
_NUMERIC_TYPES: frozenset[str] = frozenset({
    "smallint", "integer", "bigint", "decimal", "numeric", "real",
    "double precision", "money",
})
_DATE_TYPES: frozenset[str] = frozenset({
    "date", "timestamp without time zone", "timestamp with time zone",
    "timestamp", "timestamptz",
})
_BOOLEAN_TYPES: frozenset[str] = frozenset({"boolean", "bool"})

_MAX_FILTERABLE_COLS = 24
_MAX_SORTABLE_COLS = 8

# Column-name patterns whose distinct values we sample into the catalog so
# the model can map natural-language phrases ("late stage", "won deal",
# "active license") to the actual literal values it must filter on.
_ENUM_COL_PATTERNS: tuple[str, ...] = (
    "stage", "status", "type", "category", "phase", "level",
    "classification", "kind", "tier",
)
_ENUM_MAX_DISTINCT = 30
_ENUM_MAX_PER_TABLE = 4

# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------

def _env_set(name: str) -> set[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return set()
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def _is_denied_table(table: str, extra_deny: set[str]) -> bool:
    tl = table.lower()
    if tl in _DENY_TABLE_EXACT or tl in extra_deny:
        return True
    return any(tl.startswith(p) for p in _DENY_TABLE_PREFIXES)


def _is_denied_column(col: str) -> bool:
    cl = col.lower()
    return any(s in cl for s in _DENY_COLUMN_SUBSTR)


def _is_name_col(col: str) -> bool:
    cl = col.lower()
    if cl in _NAME_COL_EXACT:
        return True
    return any(cl.endswith(s) for s in _NAME_COL_SUFFIX)

# ---------------------------------------------------------------------------
# Discovery (cached)
# ---------------------------------------------------------------------------

_cache: dict[str, dict[str, Any]] | None = None
_cache_lock = Lock()


def _fetch_columns(conn) -> dict[str, list[tuple[str, str]]]:
    """Returns {table_name: [(column_name, data_type), ...]} for public.*."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
            """
        )
        out: dict[str, list[tuple[str, str]]] = {}
        for tname, cname, dtype in cur.fetchall():
            out.setdefault(tname, []).append((cname, dtype))
        return out


def _fetch_enum_samples(conn, table: str, cols: list[str]) -> dict[str, list[str]]:
    """For up to _ENUM_MAX_PER_TABLE candidate enum-like text columns on
    `table`, fetch up to _ENUM_MAX_DISTINCT distinct values. Used to make
    catalog hints concrete so the model maps natural language ('late
    stage', 'won deal') to the actual literal values it must filter on.

    Cheap to call: COUNT(DISTINCT) ... LIMIT is fast even on big tables
    because we early-out when too many distincts."""
    out: dict[str, list[str]] = {}
    if not cols:
        return out
    picked = 0
    safe_table = re.sub(r"[^A-Za-z0-9_]", "", table)
    if not safe_table:
        return out
    for col in cols:
        if picked >= _ENUM_MAX_PER_TABLE:
            break
        safe_col = re.sub(r"[^A-Za-z0-9_]", "", col)
        if not safe_col:
            continue
        try:
            with conn.cursor() as cur:
                # Bound the scan; we don't need to scan a 250k-row table to
                # know it's high-cardinality. Look at the first 5k rows.
                cur.execute(
                    f"""
                    SELECT DISTINCT "{safe_col}"::text AS v
                    FROM (SELECT "{safe_col}" FROM public."{safe_table}"
                          LIMIT 5000) sub
                    WHERE "{safe_col}" IS NOT NULL
                      AND length("{safe_col}"::text) <= 80
                    LIMIT {_ENUM_MAX_DISTINCT + 1}
                    """
                )
                vals = [r[0] for r in cur.fetchall() if r and r[0] is not None]
        except Exception:
            continue
        if not vals or len(vals) > _ENUM_MAX_DISTINCT:
            continue  # too high-cardinality to be useful
        out[col] = sorted(set(vals))[:_ENUM_MAX_DISTINCT]
        picked += 1
    return out


def _is_enum_candidate(col: str, dtype: str) -> bool:
    """Pick low-cardinality-likely text columns to sample distinct values."""
    if dtype not in _TEXT_TYPES:
        return False
    cl = col.lower()
    return any(p in cl for p in _ENUM_COL_PATTERNS)


def _build_table_info(table: str, cols: list[tuple[str, str]]) -> dict[str, Any]:
    safe_cols = [(c, t) for c, t in cols if not _is_denied_column(c)]
    column_types = {c: t for c, t in safe_cols}
    name_cols = [c for c, _ in safe_cols if _is_name_col(c)]
    text_cols = [c for c, t in safe_cols if t in _TEXT_TYPES]
    numeric_cols = [c for c, t in safe_cols if t in _NUMERIC_TYPES]
    date_cols = [c for c, t in safe_cols if t in _DATE_TYPES]
    bool_cols = [c for c, t in safe_cols if t in _BOOLEAN_TYPES]
    enum_candidate_cols = [c for c, t in safe_cols if _is_enum_candidate(c, t)]

    # filterable: ALWAYS include every boolean column (they're high-signal
    # flag fields — rhq_license_status, rhq_status, is_rhq, licensed,
    # active — typically only 5–15 per table and absolutely critical for
    # questions like "which companies are licensed"). Then fill remaining
    # slots with name/text/numeric/date columns up to the cap. Without
    # this, wide tables (like rhq_company with 70+ cols) exhausted the
    # cap on text columns and silently dropped the boolean flags.
    filterable_order: list[str] = list(bool_cols)
    seen: set[str] = set(bool_cols)
    for c in name_cols + text_cols + numeric_cols + date_cols:
        if c not in seen:
            seen.add(c)
            filterable_order.append(c)
        if len(filterable_order) >= _MAX_FILTERABLE_COLS:
            break
    filterable = filterable_order[:_MAX_FILTERABLE_COLS]

    sortable_pool = ["id"] if "id" in column_types else []
    for c in numeric_cols + date_cols:
        if c not in sortable_pool:
            sortable_pool.append(c)
    sortable = sortable_pool[:_MAX_SORTABLE_COLS]

    return {
        "columns": column_types,
        "all_columns": [c for c, _ in safe_cols],
        "name_cols": name_cols,
        "text_cols": text_cols,
        "filterable": filterable,
        "sortable": sortable,
        "enum_candidate_cols": enum_candidate_cols,
        # `enum_samples` is filled in by discover_tables() in a second pass
        # so we batch one DB hit per table for distinct values.
        "enum_samples": {},
    }


def discover_tables(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Discover queryable public.* tables. Cached; thread-safe."""
    global _cache
    with _cache_lock:
        if _cache is not None and not force_refresh:
            return _cache

        from app.database import _run_with_db_transient_retry  # local import to avoid cycle

        extra_deny = _env_set("MISA_CHAT_TABLES_DENY")
        # Risk-20-5: env var overrides the built-in least-privilege default
        # entirely; when unset, restrict the catalog to the
        # investment-intelligence tables this assistant serves. Empty env →
        # the default, NOT "all non-denied tables".
        allow = _env_set("MISA_CHAT_TABLES_ALLOW") or set(_DEFAULT_CHAT_TABLES_ALLOW)

        per_table_cols = _run_with_db_transient_retry(_fetch_columns)

        result: dict[str, dict[str, Any]] = {}
        for tname, cols in per_table_cols.items():
            if _is_denied_table(tname, extra_deny):
                continue
            if allow and tname.lower() not in allow:
                continue
            if not cols:
                continue
            safe_cols = [c for c, _ in cols if not _is_denied_column(c)]
            if not safe_cols:
                continue
            result[tname] = _build_table_info(tname, cols)

        # Second pass: sample distinct values for enum-candidate columns so
        # the catalog can tell the model "stage values: D1, D2, … D10" and
        # let it map natural-language phrases like "late stage" correctly.
        def _samples(conn):
            for tname, info in result.items():
                cand = info.get("enum_candidate_cols") or []
                if cand:
                    info["enum_samples"] = _fetch_enum_samples(conn, tname, cand)
            return result
        try:
            _run_with_db_transient_retry(_samples)
        except Exception:
            pass  # samples are an optimisation; missing them is non-fatal

        _cache = result
        return _cache


def invalidate_cache() -> None:
    global _cache
    with _cache_lock:
        _cache = None


def is_allowed_table(table: str) -> bool:
    return table in discover_tables()


def get_table_info(table: str) -> dict[str, Any] | None:
    return discover_tables().get(table)


def safe_columns(table: str) -> list[str]:
    info = get_table_info(table)
    return list(info["all_columns"]) if info else []


def name_columns(table: str) -> list[str]:
    info = get_table_info(table)
    return list(info["name_cols"]) if info else []


def text_columns(table: str) -> list[str]:
    info = get_table_info(table)
    return list(info["text_cols"]) if info else []


def is_denied_column_name(col: str) -> bool:
    """Exposed so other modules can apply the same column-level deny."""
    return _is_denied_column(col)


_VALID_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_identifier(name: str) -> bool:
    """Reject anything that isn't a plain SQL identifier, regardless of allowlist."""
    return bool(name and _VALID_IDENT_RE.match(name))
