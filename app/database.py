"""
Database layer: Postgres connection management, JSONB projection SQL,
structured query execution, and 4-tier company smart-search.

Arabic-script detection helpers live here because they are needed directly
inside the smart-search ranking logic.
"""

from __future__ import annotations

import re
import time
import threading
from threading import Lock
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import sql as psql

from app.config import (
    DB_CONFIG, DB_READONLY, PG_CONNECT_RETRIES, PG_RETRY_DELAY_SEC,
    QUERY_MAX_LIMIT,
)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

COMPANY_TABLE = "company_profiles"

SCHEMA_HINTS: dict[str, Any] = {
    COMPANY_TABLE: {
        "description": (
            "Company intelligence from public.company_profiles.misa_details JSON: "
            "legal entities, profiles, HQ location, MENA/Saudi presence, revenues, "
            "employee counts, and RHQ licensing fields."
        ),
        "columns": {
            "id": "bigint primary key",
            "internal_code": "internal reference code when available",
            "company_name": "legal or trade name (text)",
            "year_founded": "year founded (varchar)",
            "company_profile": "main narrative company description (text)",
            "team_comments": "internal team comments (text)",
            "company_notes": "additional company notes (text)",
            "legal_structure": "legal structure (text)",
            "type_of_entity": "entity type label (varchar)",
            "status": "operational status (text)",
            "control_structure": "control / ownership structure (text)",
            "ultimate_parent_company": "parent / holding company name (text)",
            "global_headquarters": "global HQ description (text)",
            "sector": "primary sector (text)",
            "number_of_employees": "global employee count (integer)",
            "number_of_locations": "location count (integer)",
            "fiscal_year_end_date": "fiscal year end (date)",
            "revenue_local_currency": "revenue in local currency (numeric)",
            "currency": "currency code (varchar)",
            "exchange_rate": "FX rate used (numeric)",
            "revenue_usd": "revenue in USD (numeric)",
            "logo": "logo asset path (varchar)",
            "presence_of_parent_company_in_mena": "parent present in MENA (boolean)",
            "presence_of_company_in_mena": "company present in MENA (boolean)",
            "type_of_presence": "type of MENA presence (text)",
            "mena_revenue_local_currency": "MENA revenue local (numeric)",
            "ksa_revenue_local_currency": "KSA revenue local (numeric)",
            "history_in_mena": "history narrative in MENA (text)",
            "presence_in_saudi": "presence in Saudi (boolean)",
            "type_of_presence_saudi": "type of Saudi presence (varchar)",
            "companies_name_in_mena": "related entity names in MENA (text)",
            "companies_name_in_ksa": "related entity names in KSA (text)",
            "number_of_employees_parent": "employees at parent (integer)",
            "number_of_employees_ksa": "employees in KSA (integer)",
            "number_of_employees_mena": "employees in MENA (integer)",
            "mena_locations": "MENA locations description (text)",
            "mena_notes": "MENA notes (text)",
            "rhq_status": "RHQ status flag (boolean)",
            "rhq_license_status": "RHQ license status (boolean)",
            "rhq_country": "RHQ country (varchar)",
            "rhq_city": "RHQ city (varchar)",
            "rhq_country_coverage": "RHQ country coverage (varchar)",
            "rhq_entity_name": "RHQ entity legal name (varchar)",
            "rhq_in_mena": "RHQ in MENA flag (boolean)",
            "rhq_number_of_employees": "RHQ employee count (integer)",
            "rhq_mandatory_activities": "mandatory RHQ activities (text)",
            "rhq_optional_activities": "optional RHQ activities (text)",
            "confidence_level": "data confidence label (varchar)",
            "creation_date": "row creation time (timestamptz)",
            "update_date": "last update time (timestamptz)",
            "review_date": "review timestamp (timestamptz)",
            "review_status": "review status (text)",
            "misa_review_status": "MISA review status (text)",
            "misa_comments": "MISA internal comments (text)",
            "reviewer_comments": "reviewer comments (text)",
        },
        "filterable": [
            "company_name", "internal_code", "sector", "status", "type_of_entity",
            "global_headquarters", "ultimate_parent_company", "rhq_country", "rhq_city",
            "company_profile", "history_in_mena", "mena_notes", "review_status",
            "type_of_presence", "rhq_entity_name",
        ],
        "sortable": ["id", "revenue_usd", "number_of_employees", "creation_date", "update_date"],
        "access": "open",
    },
}

REDACTED_FIELDS_FOR_NON_LEADERSHIP: dict = {}

_SUBSTRING_MATCH_ON_EQUALS = frozenset({
    "company_name", "ultimate_parent_company", "global_headquarters",
    "company_profile", "history_in_mena", "mena_notes",
    "rhq_entity_name", "rhq_country", "rhq_city",
})

_ROLE_NAME_SEARCH_COLS = frozenset({
    "company_name", "company_profile", "ultimate_parent_company",
})

_RHQ_Q_STOPWORDS = frozenset({
    "what", "whats", "who", "whos", "where", "when", "why", "how", "tell", "give",
    "show", "list", "find", "search", "lookup", "about", "please", "kindly",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "for", "from", "with", "without", "and", "or", "but", "not", "this", "that",
    "these", "those", "into", "onto", "per", "via", "company", "companies",
    "firm", "firms", "organization", "business", "details", "information", "info",
    "describe", "definition", "meaning", "name", "called", "known",
    "mentioning", "mentioned", "holding", "holdings", "matches", "match",
    "null", "cmpany", "campany", "comapny",
})

# ---------------------------------------------------------------------------
# Arabic-script helpers (needed for smart-search ranking)
# ---------------------------------------------------------------------------

_ARABIC_SCRIPT_RE = re.compile(
    r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)

_AR_ENTITY_LATIN_ALIASES: dict[str, tuple[str, ...]] = {
    "ألفابت": ("Alphabet",),
    "الفابت": ("Alphabet",),
    "جوجل": ("Google", "Alphabet"),
    "ألفا بيت": ("Alphabet",),
    "مايكروسوفت": ("Microsoft",),
    "أبل": ("Apple",),
    "أمازون": ("Amazon",),
    "ميتا": ("Meta", "Facebook"),
    "فيسبوك": ("Facebook", "Meta"),
    "إنفيديا": ("Nvidia", "NVIDIA"),
    "نتفليكس": ("Netflix",),
}


def _text_has_arabic(s: str | None) -> bool:
    return bool(s and _ARABIC_SCRIPT_RE.search(s))


def _latin_company_aliases_for_arabic_entity(entity: str | None) -> list[str]:
    if not entity or not str(entity).strip():
        return []
    key = re.sub(r"\s+", " ", str(entity).strip())
    if not _text_has_arabic(key):
        return []
    al = _AR_ENTITY_LATIN_ALIASES.get(key)
    return list(al) if al else []


def _pg_whole_word_regex(term: str) -> str:
    return r"\m" + re.escape((term or "").strip()) + r"\M"


def _primary_latin_for_ranking(terms: list[str]) -> str | None:
    for t in terms or []:
        ts = (t or "").strip()
        if ts and not _text_has_arabic(ts):
            return ts
    return None


def _smart_search_row_relevance(row: dict, primary_latin: str | None) -> int:
    if not primary_latin:
        return 0
    pl = primary_latin.lower()
    cn = (row.get("company_name") or "").lower()
    head = cn.split(",")[0].strip()
    up = (row.get("ultimate_parent_company") or "").lower()
    prof = (row.get("company_profile") or "").lower()
    if head == pl or head.startswith(pl + ","):
        return 100
    if head.startswith(pl + " ") or head.startswith(pl + "'"):
        return 95
    if pl in head:
        return 75
    if pl in up:
        return 45
    if pl in prof:
        return 22
    return 0


def _filter_rank_smart_search_df(df: pd.DataFrame, terms: list[str], lim: int) -> pd.DataFrame:
    if df.empty:
        return df
    primary = _primary_latin_for_ranking(terms)
    if not primary:
        return df.head(lim)
    records = df.to_dict(orient="records")
    scored = [(_smart_search_row_relevance(r, primary), r) for r in records]
    best = max((s for s, _ in scored), default=0)

    # If the full-entity primary scored 0 across the board (e.g. "Dentons UK
    # And Middle East LLP" isn't a contiguous substring of any returned row),
    # re-rank using the longest distinctive single-word term so we still pick
    # the right entity rather than the longest-named noise row.
    if best == 0:
        distinctive = _most_distinctive_term(terms, primary)
        if distinctive and distinctive.lower() != primary.lower():
            scored = [(_smart_search_row_relevance(r, distinctive), r) for r in records]
            best = max((s for s, _ in scored), default=0)

    if best >= 75:
        floor = max(55, best - 20)
        scored = [(s, r) for s, r in scored if s >= floor]
    elif best >= 45:
        floor = max(38, best - 10)
        scored = [(s, r) for s, r in scored if s >= floor]
    scored.sort(key=lambda x: (-x[0], -len(((x[1].get("company_name") or "").split(",")[0]).strip())))
    out = [r for _, r in scored[:lim]]
    if not out and records:
        return df.head(lim)
    return pd.DataFrame(out) if out else pd.DataFrame()


def _most_distinctive_term(terms: list[str], skip: str | None) -> str | None:
    """Pick the longest single-word, non-stopword Latin term — the one most
    likely to identify the entity (e.g. 'Dentons' from
    ['Dentons UK And Middle East LLP', 'Dentons', 'Middle', 'East', 'LLP'])."""
    skip_lower = (skip or "").lower()
    best_t = None
    best_len = 0
    for t in terms or []:
        if not t:
            continue
        s = str(t).strip()
        if not s or s.lower() == skip_lower:
            continue
        if " " in s:  # we want single-word terms
            continue
        if s.lower() in _RHQ_Q_STOPWORDS:
            continue
        if _text_has_arabic(s):
            continue
        if len(s) > best_len:
            best_len = len(s); best_t = s
    return best_t

# ---------------------------------------------------------------------------
# Filter normalization helpers
# ---------------------------------------------------------------------------

def _normalize_filter_condition(condition: Any) -> dict:
    if condition is None:
        return {"op": "=", "value": None}
    if isinstance(condition, dict):
        if "value" in condition or "op" in condition:
            return {"op": condition.get("op", "="), "value": condition.get("value")}
        if len(condition) == 1:
            return _normalize_filter_condition(next(iter(condition.values())))
        for k in ("val", "query", "matches"):
            if k in condition:
                return {"op": condition.get("op", "="), "value": condition[k]}
        return {"op": "=", "value": str(condition)}
    if isinstance(condition, list):
        return {"op": "IN", "value": condition}
    return {"op": "=", "value": condition}


def _coerce_filters_mapping(filters: Any) -> dict:
    return dict(filters) if isinstance(filters, dict) else {}


def _rhq_filters_include_name_search(filters: dict) -> bool:
    for col, raw in _coerce_filters_mapping(filters).items():
        if col not in _ROLE_NAME_SEARCH_COLS:
            continue
        spec = _normalize_filter_condition(raw)
        val = spec.get("value")
        if isinstance(val, str) and val.strip():
            return True
    return False

# ---------------------------------------------------------------------------
# JSONB projection SQL
# ---------------------------------------------------------------------------

def _json_text(path: str) -> str:
    return f"misa_details::jsonb #>> '{{{path}}}'"


def _json_numeric(path: str) -> str:
    raw = _json_text(path)
    cleaned = f"regexp_replace(COALESCE({raw}, ''), '[^0-9.\\-]', '', 'g')"
    return f"NULLIF({cleaned}, '')::numeric"


def _json_array_text(path: str) -> str:
    raw = f"misa_details::jsonb #> '{{{path}}}'"
    return (
        "CASE "
        f"WHEN jsonb_typeof({raw}) = 'array' THEN "
        f"(SELECT string_agg(value, ', ') FROM jsonb_array_elements_text({raw}) AS value) "
        f"ELSE misa_details::jsonb #>> '{{{path}}}' "
        "END"
    )


def _company_profiles_projection_sql() -> str:
    return f"""
        (
            SELECT
                id,
                NULL::text AS internal_code,
                COALESCE({_json_text("company_details,name")}, {_json_text("general,ultimate_parent")}) AS company_name,
                {_json_text("general,year_founded")} AS year_founded,
                {_json_text("general,company_profile")} AS company_profile,
                NULL::text AS team_comments,
                NULL::text AS company_notes,
                {_json_text("general,legal_structure")} AS legal_structure,
                {_json_text("general,type_of_entity")} AS type_of_entity,
                COALESCE({_json_text("general,status")}, {_json_text("company_details,status")}) AS status,
                {_json_text("general,control_structure")} AS control_structure,
                {_json_text("general,ultimate_parent")} AS ultimate_parent_company,
                {_json_text("general,global_headquarters")} AS global_headquarters,
                NULL::text AS sector,
                {_json_numeric("general,global_employees")} AS number_of_employees,
                {_json_numeric("general,number_of_locations")} AS number_of_locations,
                {_json_text("general,fiscal_year_end_date")} AS fiscal_year_end_date,
                {_json_numeric("general,revenue_local_currency")} AS revenue_local_currency,
                {_json_text("general,currency")} AS currency,
                NULL::numeric AS exchange_rate,
                {_json_numeric("general,revenue_usd")} AS revenue_usd,
                NULL::text AS logo,
                {_json_text("mena_details,presence_of_parent_in_mena")} AS presence_of_parent_company_in_mena,
                {_json_text("mena_details,presence_of_company_in_mena")} AS presence_of_company_in_mena,
                {_json_text("mena_details,presence_in_mena")} AS type_of_presence,
                {_json_numeric("mena_details,mena_revenue_local_currency")} AS mena_revenue_local_currency,
                {_json_numeric("mena_details,ksa_revenue_local_currency")} AS ksa_revenue_local_currency,
                {_json_text("mena_details,history_in_mena")} AS history_in_mena,
                {_json_text("mena_details,presence_in_saudi")} AS presence_in_saudi,
                {_json_text("mena_details,type_of_presence_saudi")} AS type_of_presence_saudi,
                {_json_text("mena_details,companies_name_in_mena")} AS companies_name_in_mena,
                {_json_text("mena_details,companies_name_in_ksa")} AS companies_name_in_ksa,
                {_json_numeric("mena_details,parent_employees")} AS number_of_employees_parent,
                {_json_numeric("mena_details,ksa_employees")} AS number_of_employees_ksa,
                {_json_numeric("mena_details,mena_employees")} AS number_of_employees_mena,
                {_json_array_text("mena_details,mena_locations")} AS mena_locations,
                {_json_text("mena_details,mena_notes")} AS mena_notes,
                {_json_text("rhq_details,rhq_status")} AS rhq_status,
                {_json_text("rhq_details,rhq_license_status")} AS rhq_license_status,
                {_json_text("rhq_details,rhq_country")} AS rhq_country,
                {_json_text("rhq_details,rhq_city")} AS rhq_city,
                {_json_text("rhq_details,rhq_country_coverage")} AS rhq_country_coverage,
                {_json_text("rhq_details,rhq_entity_name")} AS rhq_entity_name,
                {_json_text("rhq_details,rhq_in_mena")} AS rhq_in_mena,
                {_json_numeric("rhq_details,rhq_employees")} AS rhq_number_of_employees,
                {_json_array_text("rhq_details,mandatory_activities")} AS rhq_mandatory_activities,
                {_json_array_text("rhq_details,optional_activities")} AS rhq_optional_activities,
                NULL::text AS confidence_level,
                NULL::timestamptz AS creation_date,
                NULL::timestamptz AS update_date,
                NULL::timestamptz AS review_date,
                NULL::text AS review_status,
                NULL::text AS misa_review_status,
                NULL::text AS misa_comments,
                NULL::text AS reviewer_comments
            FROM public.company_profiles
            WHERE misa_details IS NOT NULL
        ) AS company_profiles_flat
    """


def _table_source_sql(table: str) -> str:
    if table == COMPANY_TABLE:
        return _company_profiles_projection_sql()
    return table

# ---------------------------------------------------------------------------
# Postgres connection (thread-safe module-level cache)
# ---------------------------------------------------------------------------

_PG_TRANSIENT_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)

# Per-thread Postgres connections.
#
# The app dispatches every synchronous DB operation onto a bounded worker-
# thread pool (see app/main.py lifespan, sized by MISA_DB_MAX_CONCURRENCY).
# Giving each thread its OWN autocommit connection means concurrent requests
# run on different threads and no longer serialize on one shared connection
# behind a global lock — which was the previous design's throughput ceiling.
#
# The total number of open connections stays bounded by the worker-pool size
# (one connection per pool thread, reused across requests). get_db()'s
# signature is unchanged, so none of the ~25 call sites need to change.
_tls = threading.local()
_pg_trgm_result: bool | None = None
# Registry of every live connection so shutdown (and a full invalidate) can
# close them deterministically instead of leaking sockets.
_all_conns: "set" = set()
_all_conns_lock = Lock()


def _connect_pg_with_retry():
    last_exc: BaseException | None = None
    for attempt in range(PG_CONNECT_RETRIES):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            # Autocommit on: we only run SELECTs and want each query to be
            # independent. Without this, a failed query leaves the shared
            # cached connection in "current transaction is aborted, commands
            # ignored" state and every subsequent query in the same turn
            # (e.g. compare two companies in one chat call) fails.
            conn.autocommit = True
            if DB_READONLY:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SET SESSION default_transaction_read_only = on")
                except Exception:
                    pass
            return conn
        except _PG_TRANSIENT_ERRORS as e:
            last_exc = e
            if attempt + 1 < PG_CONNECT_RETRIES:
                time.sleep(PG_RETRY_DELAY_SEC * (1.6 ** attempt))
    assert last_exc is not None
    raise last_exc


def _validate_cached_pg_connection(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")


def get_db():
    """Return this thread's Postgres connection, creating it on first use.

    Autocommit is on (each SELECT is independent) and the session is set
    read-only when MISA_DB_READONLY is enabled. One connection per worker
    thread: concurrent requests run on separate threads and therefore do
    not contend for a single shared connection."""
    conn = getattr(_tls, "conn", None)
    if conn is None:
        conn = _connect_pg_with_retry()
        _tls.conn = conn
        with _all_conns_lock:
            _all_conns.add(conn)
    return conn


def _invalidate_db_cache() -> None:
    """Drop THIS thread's cached connection (e.g. after a transient
    OperationalError/InterfaceError) so the next get_db() reconnects, and
    reset the process-wide pg_trgm probe. Only the calling thread's
    connection is affected — other threads keep their healthy ones."""
    global _pg_trgm_result
    conn = getattr(_tls, "conn", None)
    if conn is not None:
        _tls.conn = None
        with _all_conns_lock:
            _all_conns.discard(conn)
        try:
            conn.close()
        except Exception:
            pass
    _pg_trgm_result = None


def close_all_db_connections() -> None:
    """Close every open pooled connection. Called on server shutdown so we
    don't leave sockets dangling; safe to call multiple times."""
    with _all_conns_lock:
        conns = list(_all_conns)
        _all_conns.clear()
    for c in conns:
        try:
            c.close()
        except Exception:
            pass
    _tls.conn = None


def _run_with_db_transient_retry(conn_work):
    def _once():
        conn = get_db()
        try:
            _validate_cached_pg_connection(conn)
        except _PG_TRANSIENT_ERRORS:
            _invalidate_db_cache()
            conn = get_db()
            _validate_cached_pg_connection(conn)
        return conn_work(conn)

    try:
        return _once()
    except _PG_TRANSIENT_ERRORS:
        _invalidate_db_cache()
        return _once()


def _pg_trgm_available() -> bool:
    global _pg_trgm_result
    if _pg_trgm_result is not None:
        return _pg_trgm_result

    def _check(conn) -> bool:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')"
                )
                return bool(cur.fetchone()[0])
        except Exception:
            return False

    try:
        _pg_trgm_result = _run_with_db_transient_retry(_check)
    except Exception:
        _pg_trgm_result = False
    return _pg_trgm_result

# ---------------------------------------------------------------------------
# OpenAI client (lazy singleton)
# ---------------------------------------------------------------------------

from openai import OpenAI, AzureOpenAI

from app.config import (
    OPENAI_API_KEY,
    USE_AZURE_OPENAI,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ADVISORY_DEPLOYMENT,
    DATA_LLM_BASE_URL,
    DATA_LLM_API_KEY,
    DATA_LLM_MODEL,
    DATA_LLM_MAX_TOKENS,
    RESIDENCY_STRICT,
)

# Lazy singleton — typed as the OpenAI base since AzureOpenAI inherits
# from it and exposes the same chat.completions / responses surface.
_openai_client_inst: OpenAI | None = None
_public_openai_client_inst: OpenAI | None = None
_data_llm_client_inst: OpenAI | None = None
_data_llm_client_failed: bool = False


def _patch_azure_model_routing(
    client: AzureOpenAI,
    chat_deployment: str,
    advisory_deployment: str | None = None,
) -> None:
    """Route chat.completions.create() to the right Azure deployment.

    Call sites pass public model family names (`gpt-4o-mini`, `gpt-4o`).
    On Azure those must be deployment names. We map:
      - ADVISORY_MODEL / gpt-4o*  → advisory_deployment (deep writing)
      - everything else           → chat_deployment (fast routing / simple)

    When advisory_deployment is unset or equal to chat_deployment, behaviour
    matches the legacy single-deployment patch.
    """
    from app import config as _cfg

    advisory_deployment = (advisory_deployment or chat_deployment).strip()
    advisory_aliases = {
        (getattr(_cfg, "ADVISORY_MODEL", None) or "gpt-4o").strip(),
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4.1",
    }
    # Prefer advisory deployment for known strong models when distinct.
    original_create = client.chat.completions.create

    def routed_create(*args, **kwargs):
        requested = kwargs.get("model")
        if not requested and args:
            requested = args[0]
        req = (requested or "").strip()
        if advisory_deployment != chat_deployment and (
            req in advisory_aliases or req.startswith("gpt-4o")
        ):
            kwargs["model"] = advisory_deployment
        else:
            kwargs["model"] = chat_deployment
        return original_create(*args, **kwargs)

    client.chat.completions.create = routed_create  # type: ignore[assignment]

    if hasattr(client, "responses") and hasattr(client.responses, "create"):
        original_responses_create = client.responses.create

        def routed_responses_create(*args, **kwargs):
            requested = kwargs.get("model")
            if not requested and args:
                requested = args[0]
            req = (requested or "").strip()
            if advisory_deployment != chat_deployment and (
                req in advisory_aliases or req.startswith("gpt-4o")
            ):
                kwargs["model"] = advisory_deployment
            else:
                kwargs["model"] = chat_deployment
            return original_responses_create(*args, **kwargs)

        client.responses.create = routed_responses_create  # type: ignore[assignment]


def get_openai_client() -> OpenAI | None:
    """Returns the singleton OpenAI client.

    When MISA_USE_AZURE_OPENAI=true AND the Azure endpoint + key are
    configured, returns an AzureOpenAI client routed at the customer's
    Azure deployment (data stays inside the Azure tenant — no training,
    no retention under Microsoft's enterprise DPA). The deployment-name
    translation is patched in at client level so call sites can keep
    passing `model="gpt-4o-mini"` unchanged.

    Otherwise returns the public-API OpenAI client (existing behaviour).

    Returns None when neither path has usable credentials, so callers
    can degrade gracefully.
    """
    global _openai_client_inst
    if _openai_client_inst is not None:
        return _openai_client_inst

    # Azure path — only taken when explicitly enabled AND both
    # endpoint + key are present. Falls back silently to the public
    # API path if Azure config is partial (so a half-configured .env
    # doesn't break local dev).
    if USE_AZURE_OPENAI and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        azure_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        # Route every chat/responses call to the configured deployment,
        # so call sites that pass model="gpt-4o-mini" still hit the
        # right Azure deployment regardless of its actual name.
        _patch_azure_model_routing(
            azure_client,
            AZURE_OPENAI_DEPLOYMENT,
            AZURE_OPENAI_ADVISORY_DEPLOYMENT,
        )
        _openai_client_inst = azure_client
        return _openai_client_inst

    # Public OpenAI API path (existing behaviour).
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-REPLACE"):
        _openai_client_inst = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client_inst


def get_question_llm_client() -> OpenAI | None:
    """LLM for QUESTION-ONLY calls (intent, SQL routing). Never pass DB rows."""
    return get_openai_client()


def get_data_llm_client() -> OpenAI | None:
    """LLM for DATA-GROUNDED calls (curation, docs, deep profile).

    Under MISA_RESIDENCY_MODE=strict + MISA_DATA_LLM_BACKEND=ollama this is
    a local OpenAI-compatible client. Postgres rows stay on the machine.
    """
    global _data_llm_client_inst, _data_llm_client_failed

    from app.services.llm_residency import (
        assert_can_send_payload,
        audit_llm_call,
        data_backend,
        data_model_name,
    )

    assert_can_send_payload("data_grounded", path="get_data_llm_client")
    backend = data_backend()
    if backend != "ollama":
        client = get_openai_client()
        if client is not None:
            audit_llm_call(
                path="get_data_llm_client",
                payload_class="data_grounded",
                backend=backend,
                model=data_model_name(),
            )
        return client

    if _data_llm_client_failed:
        return None
    if _data_llm_client_inst is not None:
        return _data_llm_client_inst

    try:
        client = OpenAI(
            base_url=DATA_LLM_BASE_URL,
            api_key=DATA_LLM_API_KEY or "ollama",
        )
        original_create = client.chat.completions.create
        local_model = DATA_LLM_MODEL or "llama3.1"

        def routed_create(*args, **kwargs):
            kwargs["model"] = local_model
            kwargs.pop("store", None)
            kwargs.pop("seed", None)
            # Ollama's OpenAI-compat layer expects max_tokens, not
            # max_completion_tokens (Azure/newer OpenAI).
            if "max_completion_tokens" in kwargs and "max_tokens" not in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            else:
                kwargs.pop("max_completion_tokens", None)
            # Hard cap — high budgets make llama3.1 loop the same
            # Strategic Read / Live Web block for pages.
            cap = int(DATA_LLM_MAX_TOKENS or 1024)
            cur = kwargs.get("max_tokens")
            try:
                cur_n = int(cur) if cur is not None else cap
            except (TypeError, ValueError):
                cur_n = cap
            kwargs["max_tokens"] = min(cur_n, cap)
            # Local compose: deterministic + stop early on footer.
            kwargs["temperature"] = 0
            stop = list(kwargs.get("stop") or [])
            for s in (
                "\n## From your documents",
                "\n## From the web",
                "\n## From MISA data",
            ):
                if s not in stop:
                    stop.append(s)
            kwargs["stop"] = stop
            return original_create(*args, **kwargs)

        client.chat.completions.create = routed_create  # type: ignore[assignment]
        _data_llm_client_inst = client
        audit_llm_call(
            path="get_data_llm_client",
            payload_class="data_grounded",
            backend="ollama",
            model=local_model,
        )
        return _data_llm_client_inst
    except Exception as e:
        from app.logger import logger
        logger.error(f"data LLM (Ollama) client init failed: {e}")
        _data_llm_client_failed = True
        if RESIDENCY_STRICT:
            raise
        return None


def get_public_openai_client() -> OpenAI | None:
    """Public api.openai.com client for web-search-preview only.

    Under residency strict with MISA_RESIDENCY_ALLOW_PUBLIC_WEB=false
    this returns None (public egress sealed). Web search never sends
    DB rows — only question / entity text.
    """
    global _public_openai_client_inst
    from app.services.llm_residency import public_web_allowed
    if not public_web_allowed():
        return None
    if _public_openai_client_inst is not None:
        return _public_openai_client_inst
    if not OPENAI_API_KEY.startswith("sk-REPLACE") and OPENAI_API_KEY:
        _public_openai_client_inst = OpenAI(api_key=OPENAI_API_KEY)
    return _public_openai_client_inst


# ---------------------------------------------------------------------------
# Structured query execution
# ---------------------------------------------------------------------------

def _table_hints(table: str) -> dict | None:
    """Return the chat schema for a table: the special JSONB-projection schema
    for company_profiles, or introspected info for any other allowed table."""
    if table in SCHEMA_HINTS:
        return SCHEMA_HINTS[table]
    # Lazy import to avoid circular dependency at module load.
    from app.db_introspect import get_table_info
    return get_table_info(table)


def _is_text_col_for_substring(table: str, col: str) -> bool:
    """Whether `=` on this column should map to ILIKE substring search."""
    if table == COMPANY_TABLE:
        return col in _SUBSTRING_MATCH_ON_EQUALS
    from app.db_introspect import text_columns, name_columns
    return col in set(text_columns(table)) or col in set(name_columns(table))


def _build_where_clauses(
    table: str, filters: dict, allowed_filters: set
) -> tuple[list[str], list, list[str]]:
    """Translate the {col: {op, value}} filter dict into parameterized
    WHERE-clause fragments + the matching params list.

    Returns ``(where_clauses, params, dropped_columns)``. Unknown columns
    and invalid identifiers are listed in ``dropped_columns`` so callers
    can surface INVALID_QUERY / PARTIAL_RESULT instead of silently
    querying an unfiltered universe.
    """
    from app.db_introspect import is_valid_identifier
    where_clauses: list[str] = []
    params: list = []
    dropped: list[str] = []
    for col, raw in _coerce_filters_mapping(filters).items():
        if str(col).startswith("_"):
            continue  # internal trace markers
        if col not in allowed_filters or not is_valid_identifier(col):
            dropped.append(str(col))
            continue
        condition = _normalize_filter_condition(raw)
        op = condition.get("op", "=")
        val = condition.get("value")
        if op == "=":
            if _is_text_col_for_substring(table, col) and isinstance(val, str) and val.strip():
                where_clauses.append(f"{col} ILIKE %s")
                params.append(f"%{val.strip()}%")
            else:
                where_clauses.append(f"{col} = %s")
                params.append(val)
        elif op == "ILIKE":
            where_clauses.append(f"{col} ILIKE %s")
            params.append(f"%{val}%")
        elif op in (">", ">=", "<", "<="):
            where_clauses.append(f"{col} {op} %s")
            params.append(val)
        elif op == "IN" and isinstance(val, list):
            placeholders = ", ".join(["%s"] * len(val))
            where_clauses.append(f"{col} IN ({placeholders})")
            params.extend(val)
        else:
            dropped.append(str(col))
    return where_clauses, params, dropped


def count_table_rows(table: str, filters: dict) -> tuple[int, str, list]:
    """Return (count, sql, params) — the COUNT(*) for the table given
    the filter set. Used to answer "how many" / "total number of" /
    "count of" questions without the LIMIT 100 cap that throttles
    list-style queries. Unknown filter columns are dropped; if all
    user-provided filter columns are dropped (none exist on the
    table), raises ValueError so the caller can surface a useful
    error instead of returning the total table size."""
    from app.db_introspect import is_valid_identifier
    hints = _table_hints(table)
    if hints is None:
        raise ValueError(f"Unknown or denied table: {table}")
    if not is_valid_identifier(table):
        raise ValueError(f"Invalid table identifier: {table}")
    allowed_filters = set(hints["filterable"])
    where_clauses, params, dropped = _build_where_clauses(table, filters or {}, allowed_filters)
    user_facing_cols = [k for k in (filters or {}).keys() if not k.startswith("_")]
    if user_facing_cols and not where_clauses:
        raise ValueError(
            f"none of the requested filter columns "
            f"({user_facing_cols}) are valid on `{table}`"
            + (f"; dropped={dropped}" if dropped else "")
        )
    sql = f"SELECT COUNT(*) AS n FROM {_table_source_sql(table)}"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    def _exec(conn):
        with conn.cursor() as cur:
            cur.execute(sql, params)
            n = int(cur.fetchone()[0] or 0)
        return n, sql, params

    return _run_with_db_transient_retry(_exec)


_DISALLOWED_SQL_KEYWORDS: tuple[str, ...] = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "merge", "copy", "call", "do", "vacuum",
)


def _assert_select_only(sql: str) -> None:
    """Risk-20-5 belt-and-suspenders: refuse to execute anything that isn't a
    single read-only SELECT/WITH statement. The builder only ever constructs
    SELECTs, so this should never trip in practice — it exists so a future
    refactor (or an un-parameterised identifier path) can't silently emit a
    write. Rejects statement chaining and any DML/DDL leading keyword."""
    stripped = (sql or "").strip().lstrip("(").lstrip()
    low = stripped.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("Refusing to execute non-SELECT statement.")
    # Block stacked statements: a semicolon anywhere but a lone trailing one.
    if ";" in stripped.rstrip().rstrip(";"):
        raise ValueError("Refusing to execute chained SQL statements.")
    first = low.split(None, 1)[0] if low.split() else ""
    if first in _DISALLOWED_SQL_KEYWORDS:
        raise ValueError(f"Refusing to execute {first.upper()} statement.")


def generate_query_and_run_query(
    table: str,
    filters: dict,
    order_by: str | None = None,
    descending: bool = True,
    limit: int = 25,
) -> tuple[pd.DataFrame, str, list]:
    from app.db_introspect import is_valid_identifier

    hints = _table_hints(table)
    if hints is None:
        raise ValueError(f"Unknown or denied table: {table}")
    if not is_valid_identifier(table):
        raise ValueError(f"Invalid table identifier: {table}")

    allowed_filters = set(hints["filterable"])
    allowed_sort = set(hints["sortable"])
    where_clauses, params, dropped = _build_where_clauses(table, filters, allowed_filters)
    user_facing_cols = [k for k in (filters or {}).keys() if not str(k).startswith("_")]
    # If the caller supplied filters and ALL were dropped, refuse to run
    # an unfiltered query (would look like authoritative data for the
    # wrong universe).
    if user_facing_cols and not where_clauses and dropped:
        raise ValueError(
            f"INVALID_QUERY: all filters dropped on `{table}` "
            f"(dropped={dropped}). Refusing unfiltered result set."
        )

    sql = f"SELECT * FROM {_table_source_sql(table)}"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    if order_by and order_by in allowed_sort and is_valid_identifier(order_by):
        sql += f" ORDER BY {order_by} {'DESC' if descending else 'ASC'} NULLS LAST"
    sql += f" LIMIT {min(int(limit), QUERY_MAX_LIMIT)}"
    _assert_select_only(sql)

    def _exec(conn):
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        df = pd.DataFrame(rows)
        if not df.empty:
            df = _drop_sensitive_columns(df)
        if table in REDACTED_FIELDS_FOR_NON_LEADERSHIP and not df.empty:
            for fld in REDACTED_FIELDS_FOR_NON_LEADERSHIP[table]:
                if fld in df.columns:
                    df[fld] = "[REDACTED — PDPL]"
        if dropped:
            try:
                df.attrs["dropped_filters"] = list(dropped)
                df.attrs["retrieval_status"] = "PARTIAL_RESULT"
            except Exception:
                pass
        return df, sql, params

    return _run_with_db_transient_retry(_exec)


def _drop_sensitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Defense-in-depth: drop password/token/secret-style columns even if a
    projection happened to include them."""
    from app.db_introspect import is_denied_column_name
    drop = [c for c in df.columns if is_denied_column_name(str(c))]
    return df.drop(columns=drop) if drop else df

# ---------------------------------------------------------------------------
# Smart (fuzzy) company search
# ---------------------------------------------------------------------------

def run_rhq_company_smart_search(terms: list[str], limit: int = 25):
    """Fuzzy company search. On success returns (df, sql, params).

    On failure returns an empty DataFrame with SQL prefixed by
    ``-- RETRIEVAL_FAILED`` so callers can distinguish outage from a
    verified empty match via ``smart_search_retrieval_failed(sql)``.
    Never silently pretend DB failure is "no companies found".
    """
    try:
        return _run_rhq_company_smart_search_impl(terms, limit)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        return (
            pd.DataFrame(),
            f"-- RETRIEVAL_FAILED company_profiles smart_search: {err}",
            [],
        )


def smart_search_retrieval_failed(sql: str | None) -> bool:
    """True when ``run_rhq_company_smart_search`` returned a failure marker."""
    return bool(sql) and "-- RETRIEVAL_FAILED" in str(sql)


def smart_search_failure_message(sql: str | None) -> str:
    if not smart_search_retrieval_failed(sql):
        return ""
    return str(sql).replace("-- RETRIEVAL_FAILED company_profiles smart_search: ", "").strip()


def _run_rhq_company_smart_search_impl(terms: list[str], limit: int = 25):
    try:
        return _smart_search_with_conn(terms, limit, get_db())
    except _PG_TRANSIENT_ERRORS:
        _invalidate_db_cache()
        return _smart_search_with_conn(terms, limit, get_db())


def _smart_search_with_conn(terms: list[str], limit: int, conn) -> tuple:
    import difflib

    lim = min(int(limit), QUERY_MAX_LIMIT)
    if not terms:
        return pd.DataFrame(), "-- company_profiles smart_search: no terms", []

    fetch_cap = min(max(lim * 4, 32), QUERY_MAX_LIMIT)
    source_sql = _table_source_sql(COMPANY_TABLE)

    def _search_cols(include_profile: bool) -> list[str]:
        # Core name columns always; broaden to profile + location on the 2nd pass
        # so place-based queries (e.g. "Pakistan") surface matches across the DB.
        cols = ["company_name", "ultimate_parent_company"]
        if include_profile:
            cols += ["company_profile", "global_headquarters", "rhq_country"]
        return cols

    def latin_clause(t: str, include_profile: bool) -> tuple[str, list]:
        cols = _search_cols(include_profile)
        if " " in t or len(t) > 24:
            pat = f"%{t}%"
            frag = "(" + " OR ".join(f"{c} ILIKE %s" for c in cols) + ")"
            return frag, [pat] * len(cols)
        wb = _pg_whole_word_regex(t)
        frag = "(" + " OR ".join(f"{c} ~* %s" for c in cols) + ")"
        return frag, [wb] * len(cols)

    def build_where(include_profile: bool) -> tuple[str, list]:
        blocks: list[str] = []
        prms: list = []
        for raw in terms:
            t = (raw or "").strip()
            if not t:
                continue
            if _text_has_arabic(t):
                cols = _search_cols(include_profile)
                pat = f"%{t}%"
                blocks.append("(" + " OR ".join(f"{c} ILIKE %s" for c in cols) + ")")
                prms.extend([pat] * len(cols))
            else:
                frag, ps = latin_clause(t, include_profile)
                blocks.append(frag)
                prms.extend(ps)
        return " OR ".join(blocks), prms

    last_sql = ""
    last_params: list = []

    for include_profile in (False, True):
        where_sql, params = build_where(include_profile)
        if not where_sql:
            continue
        sql = f"SELECT * FROM {source_sql} WHERE ({where_sql}) LIMIT {fetch_cap}"
        last_sql, last_params = sql, list(params)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        df = pd.DataFrame(rows)
        df = _filter_rank_smart_search_df(df, terms, lim)
        if not df.empty:
            return df, sql, params

    primary = (_primary_latin_for_ranking(terms) or "").strip()
    if not primary:
        stripped = [(x or "").strip() for x in terms if (x or "").strip()]
        primary = max(stripped, key=len) if stripped else ""
    if not primary:
        return (
            pd.DataFrame(),
            last_sql or "-- company_profiles smart_search: no usable terms",
            last_params,
        )

    if _pg_trgm_available():
        thr = 0.18
        sql2 = f"""
            SELECT * FROM {source_sql}
            WHERE GREATEST(
                similarity(COALESCE(company_name, ''), %s),
                similarity(COALESCE(ultimate_parent_company, ''), %s)
            ) > {thr}
            ORDER BY GREATEST(
                similarity(COALESCE(company_name, ''), %s),
                similarity(COALESCE(ultimate_parent_company, ''), %s)
            ) DESC NULLS LAST
            LIMIT {fetch_cap}
        """
        trgm_params = [primary, primary, primary, primary]
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql2, trgm_params)
            rows = cur.fetchall()
        df = pd.DataFrame(rows)
        df = _filter_rank_smart_search_df(df, terms, lim)
        if not df.empty:
            return df, sql2.strip(), trgm_params

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, company_name FROM {source_sql}
            WHERE company_name IS NOT NULL AND company_name <> ''
            """
        )
        id_names = cur.fetchall()

    term_l = primary.lower()
    scored_ids: list[tuple[float, int]] = []
    for cid, name in id_names:
        head = name.split(",")[0].strip().lower()
        r_head = difflib.SequenceMatcher(None, term_l, head).ratio()
        r_full = difflib.SequenceMatcher(None, term_l, name.lower()).ratio()
        scored_ids.append((max(r_head, r_full), cid))
    scored_ids.sort(key=lambda x: x[0], reverse=True)
    best_ids = [cid for score, cid in scored_ids if score >= 0.52][:8]
    if not best_ids:
        return (
            pd.DataFrame(),
            (last_sql or "") + "\n-- smart_search: no ilike/trgm/difflib matches",
            last_params,
        )

    placeholders = ", ".join(["%s"] * len(best_ids))
    sql3 = f"SELECT * FROM {source_sql} WHERE id IN ({placeholders})"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql3, best_ids)
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    df = _filter_rank_smart_search_df(df, terms, lim)
    return df, sql3, best_ids


# ---------------------------------------------------------------------------
# Generic smart search (any introspected table)
# ---------------------------------------------------------------------------

def smart_search(table: str, terms: list[str], limit: int = 25):
    """Dispatch fuzzy keyword search to the right backend for the given table.

    For `company_profiles` we use the existing 4-tier search over the JSONB
    projection. For any other introspected table we fall back to a generic
    ILIKE search across its name-like and text columns.
    """
    if table == COMPANY_TABLE:
        return run_rhq_company_smart_search(terms, limit)
    return _generic_smart_search(table, terms, limit)


def _generic_smart_search(
    table: str, terms: list[str], limit: int = 25
) -> tuple[pd.DataFrame, str, list]:
    from app.db_introspect import (
        get_table_info, is_valid_identifier, name_columns, text_columns,
    )

    if not is_valid_identifier(table) or get_table_info(table) is None:
        return (
            pd.DataFrame(),
            f"-- smart_search blocked: unknown/denied table {table!r}",
            [],
        )
    clean_terms = [str(t).strip() for t in (terms or []) if str(t or "").strip()]
    if not clean_terms:
        return pd.DataFrame(), f"-- smart_search ({table}): no terms", []

    cols = name_columns(table) or text_columns(table)[:5]
    if not cols:
        return (
            pd.DataFrame(),
            f"-- smart_search ({table}): no searchable text columns",
            [],
        )

    safe_cols = [c for c in cols if is_valid_identifier(c)]
    if not safe_cols:
        return pd.DataFrame(), f"-- smart_search ({table}): no safe columns", []

    lim = min(int(limit), QUERY_MAX_LIMIT)
    where_blocks: list[str] = []
    params: list = []
    for t in clean_terms:
        pat = f"%{t}%"
        block = "(" + " OR ".join(f"{c} ILIKE %s" for c in safe_cols) + ")"
        where_blocks.append(block)
        params.extend([pat] * len(safe_cols))

    table_ident = psql.Identifier(table).as_string(get_db())
    sql = (
        f"SELECT * FROM {table_ident} "
        f"WHERE ({' OR '.join(where_blocks)}) LIMIT {lim}"
    )
    _assert_select_only(sql)

    def _exec(conn):
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        df = pd.DataFrame(rows)
        if not df.empty:
            df = _drop_sensitive_columns(df)
        return df, sql, params

    try:
        df, sql_used, params_used = _run_with_db_transient_retry(_exec)
    except Exception as e:
        return (
            pd.DataFrame(),
            f"-- smart_search ({table}) failed: {type(e).__name__}: {e}",
            [],
        )

    # Typo-tolerant fallback: if exact ILIKE found nothing AND pg_trgm is
    # installed, rank rows by trigram similarity against the longest single-
    # word term (e.g. "paksitan" → "Pakistan" in country tables). This is
    # the same idea as the company_profiles smart search's 4-tier flow,
    # adapted to whichever name/text columns the table has.
    if df.empty and _pg_trgm_available():
        primary = max((t for t in clean_terms if " " not in t), key=len, default="")
        if primary and len(primary) >= 4:
            sim_expr = ", ".join(
                [f"similarity(COALESCE({c}, ''), %s)" for c in safe_cols]
            )
            trgm_sql = (
                f"SELECT * FROM {table_ident} "
                f"WHERE GREATEST({sim_expr}) > 0.30 "
                f"ORDER BY GREATEST({sim_expr}) DESC NULLS LAST "
                f"LIMIT {lim}"
            )
            trgm_params = [primary] * (len(safe_cols) * 2)

            def _exec_trgm(conn):
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(trgm_sql, trgm_params)
                    rows = cur.fetchall()
                df2 = pd.DataFrame(rows)
                if not df2.empty:
                    df2 = _drop_sensitive_columns(df2)
                return df2, trgm_sql, trgm_params

            try:
                df, sql_used, params_used = _run_with_db_transient_retry(_exec_trgm)
            except Exception:
                pass  # keep original empty df

    return df, sql_used, params_used
