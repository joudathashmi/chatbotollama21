"""
Entity correlation layer.

Given a resolved entity (company / person / country) this module runs
PARALLEL DB queries across all FK-related tables and returns ONE
consolidated bundle. The curator then has rich, cross-related data
to weave into a single answer instead of multiple disconnected
chunks.

Before this module existed, each direct path queried only its own
1-2 tables. So a `company_profile` answer never referenced the
company's executives, prior MISA engagements, or open opportunities
— even when that data was sitting one FK away. This is the root
cause Joudat flagged as "answers too high-level / lack useful detail".

DESIGN:
  - Pure read-only SQL. No LLM calls inside this module.
  - All queries fan out via ThreadPoolExecutor — total wall-clock
    cost is the slowest individual query (~50-150ms), not the sum.
  - Per-table row caps so the curation prompt stays bounded.
  - Returns a typed dict with stable section names. Curator templates
    can reference them as `bundle["executives"]`, `bundle["meetings"]`,
    etc.
  - Caches results in memory with a short TTL (60s) so repeat
    questions in the same session don't re-fetch.
  - Failure-tolerant: if any one query fails, the bundle still
    returns with that section empty + an entry in `_errors`.

PUBLIC API:
  correlate_company(company_ids: list[int]) -> dict
  correlate_country(country_id: int) -> dict
  correlate_person(person_name, parent_company_ids=None) -> dict
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Optional

import psycopg2.extras

from app.database import get_db


# ─── In-memory cache (per-process, short TTL) ─────────────────────────

_CACHE: dict[tuple, tuple[float, dict]] = {}  # key → (expires_at, bundle)
_CACHE_LOCK = Lock()
_CACHE_TTL_SEC = 60.0
_CACHE_MAX_ENTRIES = 200


def _cache_get(key: tuple) -> dict | None:
    """Return cached bundle if fresh, else None. Thread-safe."""
    now = time.time()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        expires_at, bundle = entry
        if now >= expires_at:
            _CACHE.pop(key, None)
            return None
        return bundle


def _cache_put(key: tuple, bundle: dict) -> None:
    """Insert into cache with TTL. Evicts oldest if over cap."""
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX_ENTRIES:
            # Drop the oldest expiry — rough LRU.
            oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])
            _CACHE.pop(oldest[0], None)
        _CACHE[key] = (time.time() + _CACHE_TTL_SEC, bundle)


def clear_cache() -> None:
    """For tests / admin: drop everything."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ─── Generic query helper ─────────────────────────────────────────────

def _run_query(sql: str, params: tuple) -> list[dict]:
    """Execute a parameterised SELECT and return list-of-dicts.
    Each correlator job dispatches several of these in parallel via
    ThreadPoolExecutor. Each gets its own cursor; psycopg2 connection
    is autocommit so concurrent cursors on the same connection are
    safe for SELECTs."""
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def _fan_out(jobs: dict[str, tuple[str, tuple]], max_workers: int = 8
             ) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Run a dict of {section_name: (sql, params)} jobs in parallel.
    Returns (results_dict, errors_dict). Sections that error get
    `[]` in results and the error text in errors so the curator
    sees a partially-populated bundle rather than a 500."""
    results: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}

    def _one(name: str, sql: str, params: tuple) -> tuple[str, list[dict] | str]:
        try:
            return name, _run_query(sql, params)
        except Exception as e:
            return name, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, name, sql, params)
                   for name, (sql, params) in jobs.items()]
        for f in futures:
            name, out = f.result()
            if isinstance(out, list):
                results[name] = out
            else:
                results[name] = []
                errors[name] = out
    return results, errors


# ─── Company correlation ──────────────────────────────────────────────

def correlate_company(company_ids: list[int]) -> dict:
    """Fan out across all tables FK-related to a list of company IDs
    (multiple IDs cover the duplicate-rows-per-logical-company case).
    Returns a bundle dict with stable keys; missing data → `[]`.

    Tables queried (all in parallel):
      - company_profiles    (primary row, picked by largest revenue)
      - company_executives  (leadership)
      - company_news        (recent corporate news)
      - company_ai_insights (AI-generated insights tied to company)
      - company_business_units
      - company_competitors
      - company_financial_performances
      - company_geographic_revenues
      - company_global_presences
      - misa_contact_details (MISA-side contacts)
      - opportunities       (commercial pipeline)
      - meetings + engagements + meeting_notes (engagement history,
                                                joined in-Python)
      - strategic_investors (investor relationships)
    """
    if not company_ids:
        return {"kind": "company", "company_ids": [], "_meta": {}}
    cache_key = ("company", tuple(sorted(company_ids)))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    t0 = time.time()
    jobs = {
        "primary": (
            "SELECT * FROM company_profiles WHERE id = ANY(%s) "
            "ORDER BY annual_revenue DESC NULLS LAST LIMIT 1",
            (company_ids,),
        ),
        "executives": (
            "SELECT id, name, position, tenure, key_contribution "
            "FROM company_executives "
            "WHERE company_profile_id = ANY(%s) LIMIT 15",
            (company_ids,),
        ),
        "news": (
            "SELECT id, title, description, source, url, published_date, "
            "category, impact "
            "FROM company_news WHERE company_profile_id = ANY(%s) "
            "ORDER BY published_date DESC NULLS LAST LIMIT 8",
            (company_ids,),
        ),
        "ai_insights": (
            # company_ai_insights has 37 columns of LLM-generated text;
            # we select the most curation-useful ones to keep the prompt
            # bounded. Reviewer can extend this list per use case.
            "SELECT id, insight_type, strategic_analysis, "
            "competitive_positioning, growth_opportunities, "
            "risk_assessment, strategic_recommendations, market_dynamics "
            "FROM company_ai_insights WHERE company_profile_id = ANY(%s) "
            "ORDER BY created_at DESC NULLS LAST LIMIT 3",
            (company_ids,),
        ),
        "business_units": (
            "SELECT * FROM company_business_units "
            "WHERE company_profile_id = ANY(%s) LIMIT 8",
            (company_ids,),
        ),
        "competitors": (
            "SELECT * FROM company_competitors "
            "WHERE company_profile_id = ANY(%s) LIMIT 8",
            (company_ids,),
        ),
        "financial_performances": (
            "SELECT * FROM company_financial_performances "
            "WHERE company_profile_id = ANY(%s) "
            "ORDER BY year DESC NULLS LAST LIMIT 5",
            (company_ids,),
        ),
        "geographic_revenues": (
            "SELECT * FROM company_geographic_revenues "
            "WHERE company_profile_id = ANY(%s) LIMIT 8",
            (company_ids,),
        ),
        "global_presences": (
            "SELECT * FROM company_global_presences "
            "WHERE company_profile_id = ANY(%s) LIMIT 8",
            (company_ids,),
        ),
        "misa_contacts": (
            "SELECT id, name, position, email, phone, type "
            "FROM misa_contact_details WHERE company_profile_id = ANY(%s) "
            "ORDER BY id DESC LIMIT 8",
            (company_ids,),
        ),
        "opportunities": (
            "SELECT id, title, description, type, value, currency, "
            "stage, status, sector_name "
            "FROM opportunities WHERE company_profile_id = ANY(%s) "
            "ORDER BY value DESC NULLS LAST LIMIT 8",
            (company_ids,),
        ),
        "strategic_investors": (
            "SELECT id, organization, name, title, email, country_name, "
            "sector_name, investor_suitability "
            "FROM strategic_investors WHERE company_profile_id = ANY(%s) "
            "LIMIT 8",
            (company_ids,),
        ),
        # Engagement history — meetings keyed by company_id (different
        # FK column to most others; this is a real schema quirk).
        "meetings": (
            "SELECT id, title, agenda, description, start_date, "
            "meeting_type, classification, outcomes, discussion_point "
            "FROM meetings WHERE company_id = ANY(%s) "
            "ORDER BY start_date DESC NULLS LAST LIMIT 5",
            (company_ids,),
        ),
    }
    results, errors = _fan_out(jobs)

    # Resolve primary row (single dict, not a list)
    primary_rows = results.get("primary") or []
    primary = primary_rows[0] if primary_rows else None

    # Inline engagement rows under their meetings (so the curator
    # sees one chronological record per meeting with full context).
    meetings = results.get("meetings") or []
    meeting_ids = [m["id"] for m in meetings]
    if meeting_ids:
        sub_jobs = {
            "engagements": (
                "SELECT id, meeting_id, type, date, topic, description, "
                "discussions, deliverables, next_steps, action_points "
                "FROM engagements WHERE meeting_id = ANY(%s) "
                "ORDER BY date DESC NULLS LAST",
                (meeting_ids,),
            ),
            "meeting_notes": (
                "SELECT id, meeting_id, note, action_items, next_meeting "
                "FROM meeting_notes WHERE meeting_id = ANY(%s) "
                "ORDER BY created_at DESC NULLS LAST",
                (meeting_ids,),
            ),
        }
        sub_results, sub_errors = _fan_out(sub_jobs, max_workers=2)
        errors.update(sub_errors)
        eng_by_mid: dict[int, list[dict]] = {}
        notes_by_mid: dict[int, list[dict]] = {}
        for e in sub_results.get("engagements") or []:
            eng_by_mid.setdefault(e["meeting_id"], []).append(e)
        for n in sub_results.get("meeting_notes") or []:
            notes_by_mid.setdefault(n["meeting_id"], []).append(n)
        for m in meetings:
            m["_engagements"] = eng_by_mid.get(m["id"], [])
            m["_notes"] = notes_by_mid.get(m["id"], [])

    bundle = {
        "kind": "company",
        "company_ids": company_ids,
        "primary": primary,
        "executives": results.get("executives") or [],
        "news": results.get("news") or [],
        "ai_insights": results.get("ai_insights") or [],
        "business_units": results.get("business_units") or [],
        "competitors": results.get("competitors") or [],
        "financial_performances": results.get("financial_performances") or [],
        "geographic_revenues": results.get("geographic_revenues") or [],
        "global_presences": results.get("global_presences") or [],
        "misa_contacts": results.get("misa_contacts") or [],
        "opportunities": results.get("opportunities") or [],
        "strategic_investors": results.get("strategic_investors") or [],
        "meetings": meetings,
        "_meta": {
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "query_count": len(jobs) + (2 if meeting_ids else 0),
            "errors": errors,
            "from_cache": False,
        },
    }
    _cache_put(cache_key, bundle)
    return bundle


# ─── Country correlation ──────────────────────────────────────────────

def correlate_country(country_id: int) -> dict:
    """Fan out all country-keyed tables for a sovereign profile."""
    if not country_id:
        return {"kind": "country", "country_id": None, "_meta": {}}
    cache_key = ("country", int(country_id))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    t0 = time.time()
    jobs = {
        "primary": (
            "SELECT * FROM country_profiles WHERE id = %s LIMIT 1",
            (country_id,),
        ),
        "vision_outlook": (
            "SELECT national_vision, diversification_goals, five_year_outlook "
            "FROM country_vision_outlooks WHERE country_profile_id = %s LIMIT 1",
            (country_id,),
        ),
        "strategic_opportunities": (
            "SELECT id, category, description "
            "FROM country_strategic_opportunities "
            "WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
        "free_zones": (
            "SELECT * FROM country_free_zones "
            "WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
        "human_capitals": (
            "SELECT * FROM country_human_capitals "
            "WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
        "infrastructures": (
            "SELECT * FROM country_infrastructures "
            "WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
        "insights": (
            "SELECT * FROM country_insights "
            "WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
        "key_indicators": (
            "SELECT * FROM country_key_indicators "
            "WHERE country_profile_id = %s LIMIT 8",
            (country_id,),
        ),
        "policy_incentives": (
            "SELECT * FROM country_policy_incentives "
            "WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
        "recent_reforms": (
            "SELECT * FROM country_recent_reforms "
            "WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
        "risk_stabilities": (
            "SELECT * FROM country_risk_stabilities "
            "WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
        "top_commodities": (
            "SELECT * FROM country_top_commodities "
            "WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
        "trade_partners": (
            "SELECT * FROM country_trade_partners "
            "WHERE country_profile_id = %s LIMIT 8",
            (country_id,),
        ),
        "fdi_data": (
            "SELECT * FROM fdi_data WHERE country_profile_id = %s LIMIT 6",
            (country_id,),
        ),
    }
    results, errors = _fan_out(jobs, max_workers=10)

    primary_rows = results.get("primary") or []
    vision_rows = results.get("vision_outlook") or []
    bundle = {
        "kind": "country",
        "country_id": country_id,
        "primary": primary_rows[0] if primary_rows else None,
        "vision_outlook": vision_rows[0] if vision_rows else None,
        "strategic_opportunities": results.get("strategic_opportunities") or [],
        "free_zones": results.get("free_zones") or [],
        "human_capitals": results.get("human_capitals") or [],
        "infrastructures": results.get("infrastructures") or [],
        "insights": results.get("insights") or [],
        "key_indicators": results.get("key_indicators") or [],
        "policy_incentives": results.get("policy_incentives") or [],
        "recent_reforms": results.get("recent_reforms") or [],
        "risk_stabilities": results.get("risk_stabilities") or [],
        "top_commodities": results.get("top_commodities") or [],
        "trade_partners": results.get("trade_partners") or [],
        "fdi_data": results.get("fdi_data") or [],
        "_meta": {
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "query_count": len(jobs),
            "errors": errors,
            "from_cache": False,
        },
    }
    _cache_put(cache_key, bundle)
    return bundle


# ─── Person correlation ───────────────────────────────────────────────

def correlate_person(person_name: str,
                     parent_company_ids: Optional[list[int]] = None
                     ) -> dict:
    """Person bundle: executive_lookup-style query against
    company_executives, plus the FULL company correlation for the
    parent company (so the curator can reference engagement history,
    opportunities, etc., relating to the same company).

    The parent_company_ids argument lets the caller pre-resolve the
    company (which the chat engine already does via the
    executive-target LLM extractor); when None, we search the
    executive tables by name and try to back-derive the company."""
    if not person_name:
        return {"kind": "person", "_meta": {}}
    cache_key = ("person", person_name.lower().strip(),
                 tuple(sorted(parent_company_ids or [])))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    t0 = time.time()
    # 1. Find the person's records across executive tables.
    person_jobs = {
        "company_executives": (
            "SELECT id, company_profile_id, name, position, tenure, "
            "key_contribution FROM company_executives "
            "WHERE name ILIKE %s LIMIT 5",
            (f"%{person_name.strip()}%",),
        ),
        "executives": (
            "SELECT * FROM executives WHERE name ILIKE %s LIMIT 5",
            (f"%{person_name.strip()}%",),
        ),
        "rhq_topexecutives": (
            "SELECT * FROM rhq_topexecutives WHERE name ILIKE %s LIMIT 5",
            (f"%{person_name.strip()}%",),
        ),
    }
    p_results, p_errors = _fan_out(person_jobs, max_workers=3)

    # 2. If we don't have parent company IDs, derive them from the
    #    company_profile_id on company_executives rows.
    if not parent_company_ids:
        derived: list[int] = []
        for row in p_results.get("company_executives") or []:
            cid = row.get("company_profile_id")
            if cid is not None:
                derived.append(int(cid))
        parent_company_ids = list({*derived})

    # 3. Run the company correlator on the parent company.
    company_bundle = (
        correlate_company(parent_company_ids) if parent_company_ids else None
    )

    bundle = {
        "kind": "person",
        "person_name": person_name,
        "parent_company_ids": parent_company_ids or [],
        "company_executives_rows": p_results.get("company_executives") or [],
        "executives_rows": p_results.get("executives") or [],
        "rhq_topexecutives_rows": p_results.get("rhq_topexecutives") or [],
        "company_context": company_bundle,
        "_meta": {
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "errors": p_errors,
            "from_cache": False,
        },
    }
    _cache_put(cache_key, bundle)
    return bundle


# ─── Bundle summarisation helpers (used by curator prompt) ───────────

def bundle_summary_for_prompt(bundle: dict, max_chars: int = 12000) -> dict:
    """Strip a correlator bundle down to a curation-prompt-friendly
    payload: drop _meta and internal flags, cap any oversize text
    fields, keep the structural shape. Returns a new dict — never
    mutates the input."""
    if not bundle:
        return {}
    out: dict[str, Any] = {}
    for k, v in bundle.items():
        if k.startswith("_"):
            continue
        if isinstance(v, list):
            # Truncate long string fields in each row.
            out[k] = [_clip_row(r) for r in v]
        elif isinstance(v, dict):
            out[k] = _clip_row(v)
        else:
            out[k] = v
    return out


def _clip_row(row: Any, field_max: int = 600) -> Any:
    """Clip string fields to a max char count in-place style (returns
    new dict). Numbers / nulls / nested structures unchanged."""
    if not isinstance(row, dict):
        return row
    clipped: dict = {}
    for k, v in row.items():
        if isinstance(v, str) and len(v) > field_max:
            clipped[k] = v[: field_max - 1] + "…"
        else:
            clipped[k] = v
    return clipped
