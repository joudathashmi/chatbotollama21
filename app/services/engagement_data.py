"""
Engagement-history + opportunity-alignment data paths.

Two intents that the classifier resolves but had no dedicated DB
query path until now. This module provides direct SQL helpers that
the chat engine calls when intent ∈ {relationship_intelligence,
opportunity_alignment}, replacing the generic company_profiles
fallback.

Data model (verified against the live schema):

  relationship_intelligence:
    - meetings.company_id   → company_profiles.id     (FK)
    - engagements.meeting_id → meetings.id            (FK)
    - meeting_notes.meeting_id → meetings.id          (FK)
    - misa_contact_details.company_profile_id → company_profiles.id
    - meeting_members.meeting_id → meetings.id        (FK)

  opportunity_alignment:
    - opportunities.company_profile_id → company_profiles.id (FK)
    - opportunities.country_id, opportunities.sector_id
    - focused_sectors.profile_id, suggested_opportunities.profile_id
    - country_vision_outlooks.country_profile_id → country_profiles.id
    - country_strategic_opportunities.country_profile_id → ditto

All queries are SELECT-only, parameterised, capped to reasonable
limits. Returns lists of plain dicts (psycopg2 RealDictCursor rows)
that the curator can serialise directly into the LLM prompt.
"""

from __future__ import annotations


import psycopg2.extras

from app.database import get_db

# ─── Canonical "active licence" predicates ───────────────────────────
# PLATFORM SOURCE OF TRUTH (all origins — not country-specific):
#   company_profiles.licensed IS TRUE
#   company_profiles.is_rhq IS TRUE
# Confirmed live scale (national, not origin-filtered):
#   SELECT COUNT(*) FROM company_profiles WHERE licensed = true;  -- ~95,671
#   SELECT COUNT(*) FROM company_profiles WHERE is_rhq = true;    -- ~727
# Never inflate with role-code ORs (ZLA/ZRHQ) — that produced the
# false 96,283 / 1,645 totals. Legacy schemas without the booleans fall
# back to role + lifecycle_status + registration_type ONLY when columns
# are absent.
#
# Jul21-era PDFs that cited role+lifecycle origin totals (e.g. inflated
# corridor figures) are NOT the SoR. Every origin corridor, footprint
# inject, targeting seed, and prompt must use `_licensing_predicates`.

LICENSING_SOR = "company_profiles.licensed / company_profiles.is_rhq"

CANONICAL_LICENSED = "licensed IS TRUE"
CANONICAL_RHQ = "is_rhq IS TRUE"

ACTIVE_CLAUSE = "lifecycle_status = 'Active'"
LICENSED_ENTITY = "role = 'Licensed Entity'"
# Module-level aliases: prefer canonical booleans. Call sites that need
# schema adaptation must use `_licensing_predicates(cur)`.
ACTIVE_LICENSED = CANONICAL_LICENSED
ACTIVE_RHQ = CANONICAL_RHQ
NON_LICENSED = "licensed IS NOT TRUE"
NON_LICENSED_RHQ = "is_rhq IS TRUE AND licensed IS NOT TRUE"

# Legacy-only fragments (used when licensed/is_rhq columns are absent)
_LEGACY_LICENSED = f"{LICENSED_ENTITY} AND {ACTIVE_CLAUSE}"
_LEGACY_RHQ = (
    f"{LICENSED_ENTITY} AND registration_type = 'RHQ' AND {ACTIVE_CLAUSE}"
)


def _column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s "
        "AND column_name = %s LIMIT 1",
        (table, column),
    )
    return cur.fetchone() is not None


def _licensing_predicates(cur, *, alias: str = "") -> dict[str, str]:
    """Return SQL fragments for licensed / RHQ matching the connected schema.

    Prefer the boolean columns `licensed` / `is_rhq` whenever present —
    these are MISA's canonical markers for licence and RHQ counts.

    `alias` — optional table alias prefix (e.g. 'cp') for JOIN queries.
    """
    p = f"{alias}." if alias else ""
    has_lic = _column_exists(cur, "company_profiles", "licensed")
    has_rhq = _column_exists(cur, "company_profiles", "is_rhq")
    if has_lic and has_rhq:
        return {
            "licensed": f"{p}licensed IS TRUE",
            "rhq": f"{p}is_rhq IS TRUE",
            "non_licensed": f"{p}licensed IS NOT TRUE",
            "non_licensed_rhq": (
                f"{p}is_rhq IS TRUE AND {p}licensed IS NOT TRUE"
            ),
            "legacy": False,
            "source": "company_profiles.licensed / company_profiles.is_rhq",
        }
    has_life = _column_exists(cur, "company_profiles", "lifecycle_status")
    has_reg = _column_exists(cur, "company_profiles", "registration_type")
    if has_life and has_reg:
        return {
            "licensed": (
                f"{p}role = 'Licensed Entity' AND {p}lifecycle_status = 'Active'"
            ),
            "rhq": (
                f"{p}role = 'Licensed Entity' AND {p}registration_type = 'RHQ' "
                f"AND {p}lifecycle_status = 'Active'"
            ),
            "non_licensed": f"{p}role IS DISTINCT FROM 'Licensed Entity'",
            "non_licensed_rhq": f"{p}role = 'RHQ Entity'",
            "legacy": True,
            "source": (
                "company_profiles.role + lifecycle_status + registration_type"
            ),
        }
    # Last-resort role codes only when booleans and legacy cols are absent.
    return {
        "licensed": f"{p}role IN ('Licensed Entity', 'ZLA')",
        "rhq": f"{p}role IN ('ZRHQ', 'RHQ Entity')",
        "non_licensed": f"{p}role IS DISTINCT FROM 'Licensed Entity'",
        "non_licensed_rhq": f"{p}role = 'ZRHQ'",
        "legacy": False,
        "source": "company_profiles.role (fallback)",
    }


def _resolve_iso_codes(cur, cn: str) -> list[str]:
    """ISO alpha-2 code(s) for a country, read from country_profiles."""
    try:
        cur.execute(
            "SELECT DISTINCT UPPER(TRIM(country_code)) AS cc "
            "FROM country_profiles "
            "WHERE country_code IS NOT NULL AND TRIM(country_code) <> '' "
            "AND (country_name ILIKE %s OR country_name ILIKE %s)",
            (cn, f"%{cn}%"),
        )
        rows = cur.fetchall()
        codes = []
        for r in rows:
            if isinstance(r, dict):
                cc = r.get("cc")
            else:
                cc = r[0]
            if cc:
                codes.append(str(cc).upper())
        return codes
    except Exception:
        return []


def _build_origin_filter(cur, cn: str) -> tuple[str, tuple]:
    """SQL predicate matching a company's origin / shareholder nationality.

    Legacy: shareholder_country_name / shareholder_country_code JSONB on
    company_profiles.
    Live: ir_shareholders.shareholder_country (ISO) → contracts.c4c_id →
    company_profiles.entity_id, plus bus_data.nationality when present.
    """
    cn = (cn or "").strip()
    if _column_exists(cur, "company_profiles", "shareholder_country_name"):
        patterns = (f"%{cn}%",)
        name_cond = "sc ILIKE %s"
        codes = _resolve_iso_codes(cur, cn)
        if codes:
            code_cond = " OR ".join(["UPPER(scc) = %s"] * len(codes))
            code_exists = (
                " OR EXISTS (SELECT 1 FROM "
                "jsonb_array_elements_text(shareholder_country_code::jsonb) "
                f"AS scc WHERE {code_cond})"
            )
            params = patterns + tuple(codes)
        else:
            code_exists = ""
            params = patterns
        sql = (
            "(EXISTS (SELECT 1 FROM "
            "jsonb_array_elements_text(shareholder_country_name::jsonb) AS sc "
            f"WHERE {name_cond}){code_exists})"
        )
        return sql, params

    # Live schema path
    codes = _resolve_iso_codes(cur, cn)
    parts: list[str] = []
    params: list = []
    if codes and _column_exists(cur, "ir_shareholders", "shareholder_country"):
        parts.append(
            "EXISTS ("
            "SELECT 1 FROM ir_shareholders s "
            "JOIN contracts ct ON ct.contract_id::text = s.contract_id::text "
            "WHERE company_profiles.entity_id::text = ct.c4c_id::text "
            "AND UPPER(TRIM(s.shareholder_country)) = ANY(%s)"
            ")"
        )
        params.append(codes)
    if _column_exists(cur, "bus_data", "nationality"):
        parts.append(
            "EXISTS ("
            "SELECT 1 FROM bus_data b "
            "WHERE TRIM(b.misa_entity_id) = TRIM(company_profiles.entity_id::text) "
            "AND b.nationality ILIKE %s"
            ")"
        )
        params.append(cn)
    # HQ-country fallback (weaker — parent sitting in origin country).
    if _column_exists(cur, "company_profiles", "country_id") and _column_exists(
        cur, "countries", "name"
    ):
        parts.append(
            "EXISTS ("
            "SELECT 1 FROM countries c "
            "WHERE c.id = company_profiles.country_id AND c.name ILIKE %s"
            ")"
        )
        params.append(cn)

    if not parts:
        # Last resort — never match silently as "everything".
        return ("FALSE", tuple())
    return "(" + " OR ".join(parts) + ")", tuple(params)


def _hq_country_filter(cur, cn: str) -> tuple[str, tuple]:
    """Match companies whose HQ country_profile / countries row is cn."""
    if _column_exists(cur, "company_profiles", "country_profile_name"):
        return "country_profile_name ILIKE %s", (f"%{cn}%",)
    if _column_exists(cur, "company_profiles", "country_id"):
        return (
            "EXISTS (SELECT 1 FROM countries c "
            "WHERE c.id = company_profiles.country_id AND c.name ILIKE %s)",
            (cn,),
        )
    return "FALSE", tuple()


# ─── Shared helpers ─────────────────────────────────────────────────

_RESOLVER_STOPWORDS = frozenset({
    "the", "and", "of", "for", "in", "on", "with", "&", "or", "to",
    "inc", "ltd", "llc", "plc", "corp", "co", "group", "holdings",
    "company", "limited", "incorporated", "corporation", "international",
    "global",
})


def _content_words(s: str) -> set[str]:
    """Lowercase content tokens (3+ chars, not company-suffix stopwords).
    Used as the 'does this name actually match' signal — keeps the
    resolver from accepting a smart-search top hit that shares zero
    meaningful word with the user's typed entity."""
    import re as _re
    return {
        w for w in _re.findall(r"[a-z0-9]{3,}", (s or "").lower())
        if w not in _RESOLVER_STOPWORDS
    }


# ─── Helper: resolve entity name → company_profile.id ──────────────

def resolve_company_ids(name: str, limit: int = 10) -> tuple[list[int], str | None]:
    """Resolve an entity name to ALL matching company_profiles.id values
    plus a canonical display name. The DB has multiple rows per logical
    company (e.g., 4 variants of "Franklin Templeton"), and engagement
    tables link to whichever variant was used at meeting-creation time.
    To find engagement records we need ANY of the matching IDs, not
    just smart_search's top pick.

    Strategy:
      1. Smart-search top hits (the user's typed name + aliases).
      2. Word-anchored ILIKE on alias terms (uses POSIX `~*` with
         word boundaries to avoid sub-word fuzz like "lulu" matching
         "Al Lulu Assatea").
      3. If smart-search's top hit has ZERO content-word overlap with
         the typed name, demote it (don't make it the canonical).
         Prevents the "Lulu Group" → "Corporate Research And
         Investigations Ltd" misfire where smart_search returns a
         spurious top hit.

    Returns ([id1, id2, ...], canonical_name) — empty list when no match.
    """
    import re as _re
    from app.database import run_rhq_company_smart_search
    from app.services.alias_resolver import expand_aliases
    if not name:
        return [], None
    terms = expand_aliases(name) or [name]
    typed_words = _content_words(name)
    # Also accept words from aliases as "matching" — when user types
    # "Google", an alias like "Alphabet" counts as a real match.
    alias_words: set[str] = set(typed_words)
    for t in terms:
        alias_words.update(_content_words(t))

    # Smart-search pass — best ranked.
    ranked_ids: list[int] = []
    canon: str | None = None
    smart_top_name: str | None = None
    try:
        df, _, _ = run_rhq_company_smart_search(terms, limit)
        if df is not None and not df.empty:
            ranked_ids = [int(x) for x in df["id"].tolist()]
            smart_top_name = str(df.iloc[0]["company_name"])
            # Only adopt as canonical if the top hit shares a content
            # word with the typed term (or an alias). Otherwise it's
            # noise from trigram/fuzzy ranking.
            if alias_words & _content_words(smart_top_name):
                canon = smart_top_name
    except Exception:
        pass

    # ILIKE pass — catches duplicate rows smart-search may have ranked
    # below the cap. Limited so we never pull a thousand junk matches.
    # Uses POSIX word-boundary regex `~*` instead of bare ILIKE so
    # "lulu" doesn't sub-word-match "Al Lulu Assatea" type noise.
    conn = get_db()
    extra_ids: list[int] = []
    extra_names: dict[int, str] = {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for t in terms:
            t_clean = t.strip()
            if len(t_clean) < 3:
                continue
            # First try whole-word match via POSIX word-boundary.
            cur.execute(
                "SELECT id, company_name FROM company_profiles "
                "WHERE company_name ~* %s LIMIT 20",
                (rf"\m{_re.escape(t_clean)}\M",),
            )
            rows = cur.fetchall()
            # Fall back to substring if word-boundary returned nothing
            # — preserves recall on partial matches like "lulu" → "Lulu
            # Group International" when there's no boundary mismatch.
            if not rows:
                cur.execute(
                    "SELECT id, company_name FROM company_profiles "
                    "WHERE company_name ILIKE %s LIMIT 20",
                    (f"%{t_clean}%",),
                )
                rows = cur.fetchall()
            for r in rows:
                extra_ids.append(int(r["id"]))
                extra_names[int(r["id"])] = r["company_name"]

    # If smart-search canon was dropped (no overlap) but ILIKE found
    # word-anchored matches, adopt the first ILIKE match as canon.
    if canon is None and extra_ids:
        canon = extra_names.get(extra_ids[0])

    # ILIKE OVERRIDE — when smart_search picked a canon whose match
    # is ONLY on common geography/legal-form words ("saudi", "arabia",
    # "ltd", "inc"), AND the ILIKE pass found a row containing the
    # DISTINCTIVE typed word (e.g. "aramco"), the ILIKE row is the
    # better canonical. This handles the Aramco → Daqing Saudi Arabia
    # Ltd misfire where Aramco rows are filtered out of smart_search's
    # projection by misa_details IS NOT NULL.
    #
    # Distinctive = typed word that's NOT a common geography or legal
    # token. We don't override on common-only matches because brand→
    # legal-name redirects (Google → Alphabet) intentionally produce
    # canons that don't contain the typed word.
    if canon is not None and typed_words:
        _COMMON_TOKENS = frozenset({
            "saudi", "arabia", "arabian", "ksa", "kingdom",
            "uae", "emirates", "qatar", "kuwait", "bahrain",
            "saudi", "company", "corp", "corporation", "inc", "ltd",
            "limited", "llc", "international", "group", "holding",
            "holdings", "co", "the", "of", "and", "for",
        })
        distinctive_typed = {w for w in typed_words if w not in _COMMON_TOKENS}
        canon_words = _content_words(canon)
        canon_has_distinctive = bool(distinctive_typed & canon_words)
        if distinctive_typed and not canon_has_distinctive:
            for eid in extra_ids:
                ename = extra_names.get(eid, "")
                ew = _content_words(ename)
                if distinctive_typed & ew:
                    canon = ename
                    extra_ids = [eid] + [x for x in extra_ids if x != eid]
                    break

    # Dedup, preserve smart-search ranking order.
    # When ILIKE override fired, put extra_ids FIRST so the chosen
    # canonical's id is at position 0 (downstream queries that pick
    # ids[0] for the primary row will then anchor on the right entity).
    seen: set[int] = set()
    merged: list[int] = []
    # If canon came from extra_ids (override or smart_search-was-None
    # case), put extra_ids first so canon's id leads.
    if canon and (not ranked_ids or canon != smart_top_name):
        order = extra_ids + ranked_ids
    else:
        order = ranked_ids + extra_ids
    for x in order:
        if x not in seen:
            seen.add(x); merged.append(x)
    return merged[:limit], canon


# Back-compat alias for callers that only need the top match.
def resolve_company_id(name: str) -> tuple[int | None, str | None]:
    ids, canon = resolve_company_ids(name, limit=1)
    return (ids[0] if ids else None), canon


def resolve_country_id(name: str) -> tuple[int | None, str | None]:
    """Look up a country_profiles row by name. Plain ILIKE — country
    list is small enough that smart-search overhead isn't needed."""
    if not name:
        return None, None
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, country_name FROM country_profiles "
            "WHERE country_name ILIKE %s OR country_name ILIKE %s "
            "LIMIT 1",
            (name.strip(), f"%{name.strip()}%"),
        )
        r = cur.fetchone()
    if not r:
        return None, None
    return int(r["id"]), r["country_name"]


# ─── relationship_intelligence ──────────────────────────────────────

def fetch_engagement_history(company_ids: list[int]) -> dict[str, list[dict]]:
    """All engagement-related data for one logical company, queried
    across ALL its duplicate company_profiles rows (since meetings /
    contacts may link to any variant). Aggregates:
       - meetings (with linked engagements + notes inlined)
       - misa_contact_details
       - latest_interactions
    Returns the dict shape the curator's prompt expects."""
    out: dict[str, list[dict]] = {
        "meetings": [], "contacts": [], "interactions": [],
    }
    if not company_ids:
        return out
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # 1. Meetings — curated columns; full meetings row has 50+ fields
        #    most of which would just clutter the curation prompt.
        cur.execute("""
            SELECT id, title, agenda, description, start_date,
                   meeting_type, classification, outcomes,
                   discussion_point, previous_meeting_outcome,
                   sector_recommendation, additional_notes,
                   he_office_comments
            FROM meetings
            WHERE company_id = ANY(%s)
            ORDER BY start_date DESC NULLS LAST
            LIMIT 10
        """, (company_ids,))
        meetings = [dict(r) for r in cur.fetchall()]
        meeting_ids = [m["id"] for m in meetings]

        # 2. Engagements + meeting_notes attached to those meetings.
        engagements_by_mid: dict[int, list[dict]] = {}
        notes_by_mid: dict[int, list[dict]] = {}
        if meeting_ids:
            cur.execute("""
                SELECT id, meeting_id, type, date, topic, counterpart,
                       description, highlights, discussions, deliverables,
                       next_steps, action_points, tasks, challenges
                FROM engagements WHERE meeting_id = ANY(%s)
                ORDER BY date DESC NULLS LAST
            """, (meeting_ids,))
            for r in cur.fetchall():
                engagements_by_mid.setdefault(r["meeting_id"], []).append(dict(r))
            cur.execute("""
                SELECT id, meeting_id, note, agenda, action_items, next_meeting
                FROM meeting_notes WHERE meeting_id = ANY(%s)
                ORDER BY created_at DESC NULLS LAST
            """, (meeting_ids,))
            for r in cur.fetchall():
                notes_by_mid.setdefault(r["meeting_id"], []).append(dict(r))
        for m in meetings:
            m["_engagements"] = engagements_by_mid.get(m["id"], [])
            m["_notes"] = notes_by_mid.get(m["id"], [])
        out["meetings"] = meetings

        # 3. MISA contacts for this company.
        cur.execute("""
            SELECT id, name, position, email, phone, type
            FROM misa_contact_details
            WHERE company_profile_id = ANY(%s)
            ORDER BY id DESC LIMIT 10
        """, (company_ids,))
        out["contacts"] = [dict(r) for r in cur.fetchall()]

        # 4. Latest interactions (profile-keyed).
        cur.execute("""
            SELECT id, date, title, description, location, document
            FROM latest_interactions
            WHERE profile_id = ANY(%s)
            ORDER BY date DESC NULLS LAST LIMIT 5
        """, (company_ids,))
        out["interactions"] = [dict(r) for r in cur.fetchall()]

    return out


# ─── opportunity_alignment ──────────────────────────────────────────

def fetch_opportunity_alignment(
    company_ids: list[int] | None, country_id: int | None,
) -> dict[str, list[dict]]:
    """All opportunity / sector-fit / Vision-2030 data tied to a
    company (across all dup rows) OR country. Either may be None.

    Returns: {opportunities, focused_sectors, suggested_opportunities,
              country_strategic_opportunities, country_vision_outlooks}."""
    out: dict[str, list[dict]] = {
        "opportunities": [], "focused_sectors": [],
        "suggested_opportunities": [],
        "country_strategic_opportunities": [],
        "country_vision_outlooks": [],
    }
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if company_ids:
            cur.execute("""
                SELECT id, title, description, type, value, currency,
                       stage, status, expected_close_date, sector_name,
                       opportunity_source_name
                FROM opportunities
                WHERE company_profile_id = ANY(%s)
                ORDER BY value DESC NULLS LAST, expected_close_date DESC NULLS LAST
                LIMIT 8
            """, (company_ids,))
            out["opportunities"] = [dict(r) for r in cur.fetchall()]
            cur.execute("""
                SELECT id, sector, description, category
                FROM focused_sectors
                WHERE profile_id = ANY(%s)
                ORDER BY id DESC LIMIT 5
            """, (company_ids,))
            out["focused_sectors"] = [dict(r) for r in cur.fetchall()]
            cur.execute("""
                SELECT id, category, description, opportunities, estimated_value
                FROM suggested_opportunities
                WHERE profile_id = ANY(%s)
                ORDER BY id DESC LIMIT 5
            """, (company_ids,))
            out["suggested_opportunities"] = [dict(r) for r in cur.fetchall()]
        if country_id:
            cur.execute("""
                SELECT id, category, description
                FROM country_strategic_opportunities
                WHERE country_profile_id = %s
                ORDER BY id DESC LIMIT 6
            """, (country_id,))
            out["country_strategic_opportunities"] = [dict(r) for r in cur.fetchall()]
            cur.execute("""
                SELECT id, national_vision, diversification_goals, five_year_outlook
                FROM country_vision_outlooks
                WHERE country_profile_id = %s
                LIMIT 1
            """, (country_id,))
            out["country_vision_outlooks"] = [dict(r) for r in cur.fetchall()]
            if not out["opportunities"]:
                cur.execute("""
                    SELECT id, title, description, type, value, currency,
                           stage, status, sector_name
                    FROM opportunities
                    WHERE country_id = %s
                    ORDER BY value DESC NULLS LAST LIMIT 8
                """, (country_id,))
                out["opportunities"] = [dict(r) for r in cur.fetchall()]
    return out


def has_any_engagement_data(data: dict) -> bool:
    """True iff at least one of the engagement-record buckets has rows."""
    return any(data.get(k) for k in ("meetings", "contacts", "interactions"))


# ─── country_profile: companies from a country investing in Saudi ──

def fetch_country_saudi_investors(country_name: str) -> dict:
    """List companies from a given origin nationality with Saudi presence
    (licensed and/or RHQ).

    Origin matching is schema-adaptive (see `_build_origin_filter`).
    Licensing/RHQ predicates are schema-adaptive (see
    `_licensing_predicates`).

    Returns a dict with rhq / licensed_only / non_licensed lists and
    totals. On DB/schema failure sets `_db_error` — callers MUST treat
    that as "data unavailable", never as real zeros.
    """
    out: dict = {
        "rhq": [], "licensed_only": [], "non_licensed": [],
        "total_rhq": 0, "total_licensed": 0,
        "total_non_licensed": 0, "total_non_licensed_rhq": 0,
    }
    if not country_name:
        return out
    cn = country_name.strip()

    try:
        conn = get_db()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            preds = _licensing_predicates(cur)
            lic_sql = preds["licensed"]
            rhq_sql = preds["rhq"]
            non_lic_sql = preds["non_licensed"]
            non_rhq_sql = preds["non_licensed_rhq"]

            _origin_filter, _origin_params = _build_origin_filter(cur, cn)
            if _origin_filter == "FALSE":
                out["_db_error"] = (
                    f"no origin-matching path for country={cn!r} "
                    "on this schema"
                )
                out["retrieval_status"] = "SOURCE_UNAVAILABLE"
                out["retrieval"] = {
                    "retrieval_status": "SOURCE_UNAVAILABLE",
                    "source_name": "company_profiles",
                    "counts_unavailable": True,
                    "do_not_claim_zero": True,
                    "filters": {"origin_country": cn},
                }
                return out

            cur.execute(
                f"SELECT COUNT(*) c FROM company_profiles "
                f"WHERE {_origin_filter} AND {lic_sql}",
                _origin_params,
            )
            out["total_licensed"] = int(cur.fetchone()["c"])
            cur.execute(
                f"SELECT COUNT(*) c FROM company_profiles "
                f"WHERE {_origin_filter} AND {rhq_sql}",
                _origin_params,
            )
            out["total_rhq"] = int(cur.fetchone()["c"])

            select_cols = (
                "id, company_name, headquarters, "
                "annual_revenue, employee_count, "
                "industry, founded, ceo, role"
            )
            if not preds.get("legacy"):
                if _column_exists(cur, "company_profiles", "licensed"):
                    select_cols += ", licensed"
                if _column_exists(cur, "company_profiles", "is_rhq"):
                    select_cols += ", is_rhq"
            else:
                select_cols += ", registration_type"

            cur.execute(f"""
                SELECT {select_cols}
                FROM company_profiles
                WHERE {_origin_filter}
                  AND {rhq_sql}
                ORDER BY NULLIF(regexp_replace(annual_revenue::text, '[^0-9.]', '', 'g'), '')::numeric DESC NULLS LAST
                LIMIT 15
            """, _origin_params)
            out["rhq"] = [dict(r) for r in cur.fetchall()]

            cur.execute(f"""
                SELECT {select_cols}
                FROM company_profiles
                WHERE {_origin_filter}
                  AND {lic_sql}
                  AND NOT ({rhq_sql})
                ORDER BY NULLIF(regexp_replace(annual_revenue::text, '[^0-9.]', '', 'g'), '')::numeric DESC NULLS LAST
                LIMIT 15
            """, _origin_params)
            out["licensed_only"] = [dict(r) for r in cur.fetchall()]

            hq_filter, hq_params = _hq_country_filter(cur, cn)
            cur.execute(
                f"SELECT COUNT(*) c FROM company_profiles "
                f"WHERE {hq_filter} AND {non_lic_sql}",
                hq_params,
            )
            out["total_non_licensed"] = int(cur.fetchone()["c"])

            cur.execute(
                f"SELECT COUNT(*) c FROM company_profiles "
                f"WHERE {hq_filter} AND {non_rhq_sql}",
                hq_params,
            )
            out["total_non_licensed_rhq"] = int(cur.fetchone()["c"])

            cur.execute(f"""
                SELECT id, company_name, headquarters, annual_revenue,
                       employee_count, industry, founded, ceo, role
                FROM company_profiles
                WHERE {hq_filter} AND {non_lic_sql}
                ORDER BY NULLIF(regexp_replace(annual_revenue::text, '[^0-9.]', '', 'g'), '')::numeric DESC NULLS LAST LIMIT 10
            """, hq_params)
            out["non_licensed"] = [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        out["_db_error"] = str(exc)
        # Keep totals at 0 but always signal error — never "real zero".
        out["rhq"] = []
        out["licensed_only"] = []
        out["non_licensed"] = []
        try:
            from app.services.retrieval_status import (
                classify_exception, failure,
            )
            rr = failure(
                classify_exception(exc),
                source_name="company_profiles",
                error=str(exc),
                filters={"origin_country": cn},
            )
            out["retrieval_status"] = rr.status.value
            out["retrieval"] = rr.to_context_dict()
        except Exception:
            out["retrieval_status"] = "UNKNOWN_ERROR"
            out["retrieval"] = {
                "counts_unavailable": True,
                "do_not_claim_zero": True,
            }
        return out

    try:
        from app.services.retrieval_status import (
            RetrievalStatus, success_counts,
        )
        n = int(out.get("total_licensed") or 0) + int(out.get("total_rhq") or 0)
        rr = success_counts(
            source_name="company_profiles.licensed/is_rhq",
            count=n,
            filters={"origin_country": cn},
            query="COUNT where licensed/is_rhq + origin filter",
            metadata={
                "total_licensed": out.get("total_licensed"),
                "total_rhq": out.get("total_rhq"),
            },
        )
        if n == 0:
            rr.status = RetrievalStatus.SUCCESS_EMPTY
        out["retrieval_status"] = rr.status.value
        out["retrieval"] = rr.to_context_dict()
    except Exception:
        out["retrieval_status"] = (
            "SUCCESS_EMPTY"
            if int(out.get("total_licensed") or 0) == 0
            and int(out.get("total_rhq") or 0) == 0
            else "SUCCESS_WITH_RESULTS"
        )
    return out


def fetch_country_sector_distribution(country_name: str) -> dict:
    """Sector breakdown of the licensed companies from a given origin
    country — the DB evidence for 'which sectors convert'.

    Returns ``{"sectors": [...], "_db_error": None|str, ...}``.
    Empty ``sectors`` with no ``_db_error`` means a successful empty
    result; ``_db_error`` set means counts are unavailable (never
    interpret as 'no sectors').
    """
    out: dict = {
        "sectors": [],
        "_db_error": None,
        "retrieval_status": None,
    }
    if not country_name:
        return out
    cn = country_name.strip()
    try:
        conn = get_db()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            preds = _licensing_predicates(cur)
            _origin_filter, _origin_params = _build_origin_filter(cur, cn)
            if _origin_filter == "FALSE":
                out["retrieval_status"] = "SUCCESS_EMPTY"
                return out
            cur.execute(f"""
                SELECT COALESCE(NULLIF(TRIM(industry), ''), 'Unclassified')
                           AS industry,
                       COUNT(*) AS licensed_count,
                       COUNT(*) FILTER (WHERE {preds['rhq']}) AS rhq_count
                FROM company_profiles
                WHERE {_origin_filter}
                  AND {preds['licensed']}
                GROUP BY 1
                ORDER BY licensed_count DESC
                LIMIT 12
            """, _origin_params)
            out["sectors"] = [dict(r) for r in cur.fetchall()]
            out["retrieval_status"] = (
                "SUCCESS" if out["sectors"] else "SUCCESS_EMPTY"
            )
            return out
    except Exception as exc:
        out["_db_error"] = str(exc)
        out["retrieval_status"] = "SOURCE_UNAVAILABLE"
        try:
            from app.services.retrieval_status import (
                classify_exception, failure,
            )
            rr = failure(
                classify_exception(exc),
                source_name="company_profiles.sector_distribution",
                error=str(exc),
                filters={"origin_country": cn},
            )
            out["retrieval"] = rr.to_context_dict()
            out["retrieval_status"] = rr.status.value
        except Exception:
            out["retrieval"] = {
                "counts_unavailable": True,
                "do_not_claim_zero": True,
            }
        return out


def fetch_country_profile_bundle(country_name: str) -> dict:
    """Aggregate all the data a country_profile briefing needs in
    one round-trip: country macros, Vision 2030-style outlook, and
    the actual list of companies from that country with Saudi RHQ
    licences. Replaces the LLM's tendency to duplicate
    'Notable Companies' sections by giving it ONE consolidated source.
    """
    out = {
        "country_profile": None,
        "vision_outlook": None,
        "strategic_opportunities": [],
        "saudi_investors": [],
    }
    if not country_name:
        return out
    country_id, canon = resolve_country_id(country_name)
    out["_canonical_name"] = canon or country_name
    out["_country_id"] = country_id
    if country_id:
        conn = get_db()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM country_profiles WHERE id = %s", (country_id,))
            r = cur.fetchone()
            out["country_profile"] = dict(r) if r else None
            cur.execute(
                "SELECT national_vision, diversification_goals, five_year_outlook "
                "FROM country_vision_outlooks WHERE country_profile_id = %s LIMIT 1",
                (country_id,))
            r = cur.fetchone()
            out["vision_outlook"] = dict(r) if r else None
            cur.execute(
                "SELECT category, description FROM country_strategic_opportunities "
                "WHERE country_profile_id = %s LIMIT 5", (country_id,))
            out["strategic_opportunities"] = [dict(r) for r in cur.fetchall()]
    # Pull the Saudi presence bundle (canonical source:
    # company_profiles.licensed / is_rhq, keyed by HQ country name).
    out["saudi_investors"] = fetch_country_saudi_investors(
        out["_canonical_name"]
    )
    return out


# ─── Saudi licensing aggregate (for "how many RHQ / licensed?") ────

def fetch_saudi_licensing_summary() -> dict:
    """Aggregate totals + breakdown for Saudi-licensing counts.

    Totals always use canonical `licensed` / `is_rhq` when present.
    Country breakdown prefers shareholder nationality (origin), not HQ
    city — HQ country_id is often Saudi for every RHQ and misleads.

    On any exception returns ``_db_error`` / ``do_not_claim_zero`` —
    never a silent zero census.
    """
    out = {
        "total_licensed": 0,
        "total_rhq": 0,
        "rhq_by_country": [],
        "licensed_by_country": [],
        "predicate_source": "",
    }
    try:
        return _fetch_saudi_licensing_summary_inner(out)
    except Exception as exc:
        out["_db_error"] = str(exc)
        out["counts_unavailable"] = True
        out["do_not_claim_zero"] = True
        out["footprint_data_unavailable"] = True
        try:
            from app.services.retrieval_status import (
                classify_exception, failure,
            )
            rr = failure(
                classify_exception(exc),
                source_name="company_profiles.licensed/is_rhq",
                error=str(exc),
            )
            out["retrieval_status"] = rr.status.value
            out["retrieval"] = rr.to_context_dict()
        except Exception:
            out["retrieval_status"] = "UNKNOWN_ERROR"
            out["retrieval"] = {
                "counts_unavailable": True,
                "do_not_claim_zero": True,
            }
        return out


def _fetch_saudi_licensing_summary_inner(out: dict) -> dict:
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        preds = _licensing_predicates(cur)
        out["predicate_source"] = preds.get("source") or ""
        cur.execute(
            f"SELECT COUNT(*) c FROM company_profiles WHERE {preds['licensed']}"
        )
        out["total_licensed"] = int(cur.fetchone()["c"])
        cur.execute(
            f"SELECT COUNT(*) c FROM company_profiles WHERE {preds['rhq']}"
        )
        out["total_rhq"] = int(cur.fetchone()["c"])

        if _column_exists(cur, "company_profiles", "shareholder_country_name"):
            cur.execute(f"""
                SELECT sc AS country, COUNT(*) AS n
                FROM company_profiles,
                     jsonb_array_elements_text(shareholder_country_name::jsonb) AS sc
                WHERE {preds['rhq']}
                GROUP BY 1
                ORDER BY n DESC LIMIT 12
            """)
            out["rhq_by_country"] = [dict(r) for r in cur.fetchall()]
            cur.execute(f"""
                SELECT sc AS country, COUNT(*) AS n
                FROM company_profiles,
                     jsonb_array_elements_text(shareholder_country_name::jsonb) AS sc
                WHERE {preds['licensed']}
                GROUP BY 1
                ORDER BY n DESC LIMIT 12
            """)
            out["licensed_by_country"] = [dict(r) for r in cur.fetchall()]
        elif _column_exists(cur, "bus_data", "nationality") or (
            _column_exists(cur, "ir_shareholders", "shareholder_country")
            and _column_exists(cur, "contracts", "c4c_id")
        ):
            # Live: attribute each company to an origin nationality.
            # Prefer bus_data.nationality (best RHQ coverage); fall back
            # to ir_shareholders ISO → country_profiles name.
            has_bus = _column_exists(cur, "bus_data", "nationality")
            has_share = (
                _column_exists(cur, "ir_shareholders", "shareholder_country")
                and _column_exists(cur, "contracts", "c4c_id")
            )
            has_cpn = _column_exists(cur, "country_profiles", "country_code")
            origin_parts: list[str] = []
            if has_bus:
                origin_parts.append(
                    "(SELECT NULLIF(TRIM(b.nationality), '') "
                    "FROM bus_data b "
                    "WHERE TRIM(b.misa_entity_id) = TRIM(cp.entity_id::text) "
                    "AND b.nationality IS NOT NULL LIMIT 1)"
                )
            if has_share:
                share_sel = (
                    "COALESCE(cpn.country_name, s.shareholder_country)"
                    if has_cpn else "s.shareholder_country"
                )
                share_join = (
                    "LEFT JOIN country_profiles cpn ON "
                    "UPPER(TRIM(cpn.country_code)) = "
                    "UPPER(TRIM(s.shareholder_country)) "
                    if has_cpn else ""
                )
                origin_parts.append(
                    "(SELECT " + share_sel + " "
                    "FROM contracts ct "
                    "JOIN ir_shareholders s "
                    "ON s.contract_id::text = ct.contract_id::text "
                    + share_join +
                    "WHERE ct.c4c_id::text = cp.entity_id::text "
                    "AND s.shareholder_country IS NOT NULL "
                    "AND TRIM(s.shareholder_country) <> '' "
                    "LIMIT 1)"
                )
            origin_expr = (
                "COALESCE(" + ", ".join(origin_parts)
                + ", 'Other / Unspecified')"
                if origin_parts else "'Other / Unspecified'"
            )
            preds_cp = _licensing_predicates(cur, alias="cp")
            # RHQ breakdown only (~727 rows — correlated origin is fine)
            cur.execute(f"""
                SELECT country, COUNT(*) AS n FROM (
                    SELECT cp.id, {origin_expr} AS country
                    FROM company_profiles cp
                    WHERE {preds_cp['rhq']}
                ) attributed
                GROUP BY country
                ORDER BY n DESC
                LIMIT 12
            """)
            out["rhq_by_country"] = [dict(r) for r in cur.fetchall()]

            # Licensed breakdown — cheap LEFT JOIN on bus_data only
            if has_bus:
                cur.execute(f"""
                    SELECT COALESCE(NULLIF(TRIM(b.nationality), ''),
                                    'Other / Unspecified') AS country,
                           COUNT(DISTINCT cp.id) AS n
                    FROM company_profiles cp
                    LEFT JOIN bus_data b
                      ON TRIM(b.misa_entity_id) = TRIM(cp.entity_id::text)
                    WHERE {preds_cp['licensed']}
                    GROUP BY 1
                    ORDER BY n DESC
                    LIMIT 12
                """)
                out["licensed_by_country"] = [dict(r) for r in cur.fetchall()]
        elif _column_exists(cur, "countries", "name"):
            preds_cp = _licensing_predicates(cur, alias="cp")
            cur.execute(f"""
                SELECT c.name AS country, COUNT(*) AS n
                FROM company_profiles cp
                JOIN countries c ON c.id = cp.country_id
                WHERE {preds_cp['rhq']}
                GROUP BY 1
                ORDER BY n DESC LIMIT 12
            """)
            out["rhq_by_country"] = [dict(r) for r in cur.fetchall()]
            cur.execute(f"""
                SELECT c.name AS country, COUNT(*) AS n
                FROM company_profiles cp
                JOIN countries c ON c.id = cp.country_id
                WHERE {preds_cp['licensed']}
                GROUP BY 1
                ORDER BY n DESC LIMIT 12
            """)
            out["licensed_by_country"] = [dict(r) for r in cur.fetchall()]
    try:
        from app.services.retrieval_status import success_counts
        rr = success_counts(
            source_name="company_profiles.licensed/is_rhq",
            count=int(out.get("total_licensed") or 0),
            filters={"predicate": out.get("predicate_source")},
            metadata={
                "total_licensed": out.get("total_licensed"),
                "total_rhq": out.get("total_rhq"),
            },
        )
        out["retrieval_status"] = rr.status.value
        out["retrieval"] = rr.to_context_dict()
    except Exception:
        out["retrieval_status"] = "SUCCESS_WITH_RESULTS"
    return out



def has_any_saudi_investors(bundle: dict) -> bool:
    """True iff the saudi_investors bundle has at least one entry
    in either bucket. Convenience for callers deciding whether to
    render the section at all."""
    si = (bundle or {}).get("saudi_investors") or {}
    return bool(si.get("rhq") or si.get("licensed_only"))


def has_any_opportunity_data(data: dict) -> bool:
    """True iff at least one of the opportunity buckets has rows."""
    return any(data.get(k) for k in (
        "opportunities", "focused_sectors", "suggested_opportunities",
        "country_strategic_opportunities", "country_vision_outlooks",
    ))
