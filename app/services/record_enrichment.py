"""
Server-side automatic FK-enrichment for primary chat records.

When the chat returns a row from a primary table (company_profiles,
country_profiles), the curation layer should also see all the FK-
linked supplementary data (AI insights, executives, business units,
competitors, key indicators, trade partners, recent reforms, …).

Without this, large blocks of useful pre-computed content sit unused
in the DB while the curation hallucinates around the sparse parent row.

This module fetches the supplementary tables by FK and attaches them
to the parent row under a `_related` key as a mapping of human label
→ list of cleaned row dicts. Curation prompts know about the `_related`
key and are instructed to use it.

Best-effort: any per-table fetch failure is silently skipped so a
broken sub-query never blocks the primary answer.
"""

from __future__ import annotations

import psycopg2
import psycopg2.extras
from typing import Any

from app.database import _drop_sensitive_columns, get_db
from app.db_introspect import is_allowed_table, is_valid_identifier

# Per-primary-table FK-linked tables to auto-pull.
# Each entry: (related_table, fk_column_in_related_table, human_label, max_rows).
# Caps are set per table by how useful the volume is — long-form insight
# tables get more rows; one-row-per-parent tables (human_capitals etc.)
# only need 1-5.
ENRICHMENT_MAP: dict[str, list[tuple[str, str, str, int]]] = {
    "company_profiles": [
        ("company_ai_insights",            "company_profile_id", "AI insights",            30),
        ("company_executives",             "company_profile_id", "executives",             15),
        ("company_business_units",         "company_profile_id", "business units",         15),
        ("company_competitors",            "company_profile_id", "competitors",            15),
        ("company_global_presences",       "company_profile_id", "global presence",        20),
        ("company_geographic_revenues",    "company_profile_id", "geographic revenues",    15),
        ("company_financial_performances", "company_profile_id", "financial performance",  10),
        ("company_corporate_activities",   "company_profile_id", "corporate activities",   20),
        ("company_news",                   "company_profile_id", "news",                   10),
        ("misa_contact_details",           "company_profile_id", "MISA contacts",          10),
    ],
    "country_profiles": [
        ("country_insights",                "country_profile_id", "insights",               30),
        ("country_key_indicators",          "country_profile_id", "key indicators",         30),
        ("country_top_commodities",         "country_profile_id", "top commodities",        20),
        ("country_trade_partners",          "country_profile_id", "trade partners",         30),
        ("country_recent_reforms",          "country_profile_id", "recent reforms",         10),
        ("country_human_capitals",          "country_profile_id", "human capital",          5),
        ("country_infrastructures",         "country_profile_id", "infrastructure",         5),
        ("country_policy_incentives",       "country_profile_id", "policy incentives",      5),
        ("country_risk_stabilities",        "country_profile_id", "risk & stability",       5),
        ("country_strategic_opportunities", "country_profile_id", "strategic opportunities", 15),
        ("country_vision_outlooks",         "country_profile_id", "vision outlooks",        5),
        ("country_associated_companies",    "country_profile_id", "associated companies",   15),
        ("country_free_zones",              "country_profile_id", "free zones",             10),
        ("fdi_data",                        "country_profile_id", "FDI data",               10),
    ],
}

# Columns to drop from enrichment rows — pure housekeeping noise that
# wastes tokens and contributes nothing to the curation answer.
_ENRICHMENT_DROP_COLS: tuple[str, ...] = (
    "created_at", "updated_at", "search_vector", "creation_date",
    "update_date", "review_date", "created_by", "reviewed_by",
    "updated_by", "logo", "icon", "profile_image",
)


def supports_enrichment(primary_table: str) -> bool:
    """Is FK-enrichment defined for this primary table?"""
    return primary_table in ENRICHMENT_MAP


def _clean_enrichment_rows(rows: list[dict]) -> list[dict]:
    """Strip housekeeping/audit columns and apply the standard sensitive-
    column denylist before the rows go into the curation payload."""
    if not rows:
        return rows
    import pandas as _pd
    df = _pd.DataFrame(rows)
    df = _drop_sensitive_columns(df)
    drop = [c for c in df.columns if c.lower() in _ENRICHMENT_DROP_COLS]
    if drop:
        df = df.drop(columns=drop)
    return df.to_dict(orient="records")


def _fetch_related(conn, rel_table: str, fk_col: str, parent_id: Any, cap: int) -> list[dict]:
    """One related-table fetch. Identifier-safe; never SQL-injectable
    because rel_table / fk_col are validated identifiers."""
    if not is_valid_identifier(rel_table) or not is_valid_identifier(fk_col):
        return []
    sql = (
        f'SELECT * FROM public."{rel_table}" '
        f'WHERE "{fk_col}" = %s LIMIT %s'
    )
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, [parent_id, int(cap)])
        return list(cur.fetchall())


def enrich_records(primary_table: str, primary_rows: list[dict]) -> list[dict]:
    """Attach a `_related` dict to each primary row containing all FK-
    linked supplementary data. Mutates rows in place and returns them.

    Best-effort — any per-table failure is silently swallowed so a single
    broken sub-query never blocks the primary answer.
    """
    spec = ENRICHMENT_MAP.get(primary_table)
    if not spec or not primary_rows:
        return primary_rows

    try:
        conn = get_db()
    except Exception:
        return primary_rows

    for row in primary_rows:
        pid = row.get("id")
        if pid is None:
            continue
        related: dict[str, list[dict]] = {}
        for rel_table, fk_col, label, cap in spec:
            if not is_allowed_table(rel_table):
                continue
            try:
                rows = _fetch_related(conn, rel_table, fk_col, pid, cap)
            except Exception:
                rows = []
            if not rows:
                continue
            cleaned = _clean_enrichment_rows(rows)
            if cleaned:
                related[label] = cleaned
        if related:
            row["_related"] = related
    return primary_rows
