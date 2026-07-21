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

# ─── Canonical "active licence" predicates (single source of truth) ──
# These are hardcoded SQL fragments — safe to inline in f-strings because
# they contain no user input. User-supplied values stay as %s bind params.
ACTIVE_CLAUSE   = "lifecycle_status = 'Active'"
# is-Licensed → the entity's ROLE marks it as a licensed company. The legacy
# `licensed` boolean is unreliable (7,273 rows with role='Licensed Entity'
# carry licensed=false) and the `is_rhq` boolean is entirely unpopulated in
# the current data — so we key off role / registration_type instead.
LICENSED_ENTITY  = "role = 'Licensed Entity'"
ACTIVE_LICENSED  = f"{LICENSED_ENTITY} AND {ACTIVE_CLAUSE}"
# is-RHQ for a LICENSED company → registration_type = 'RHQ'.
ACTIVE_RHQ       = f"{LICENSED_ENTITY} AND registration_type = 'RHQ' AND {ACTIVE_CLAUSE}"
# Non-licensed company → any entity that is not a Licensed Entity. Its country
# is taken from country_profile_name (not shareholder nationality).
NON_LICENSED     = "role IS DISTINCT FROM 'Licensed Entity'"
# is-RHQ for a NON-LICENSED company → role = 'RHQ Entity'.
NON_LICENSED_RHQ = "role = 'RHQ Entity'"

def _resolve_iso_codes(cur, cn: str) -> list[str]:
    """ISO alpha-2 code(s) for a country, read from country_profiles.
    Used to also match shareholder_country_code — the JSON array of
    codes that runs parallel to shareholder_country_name."""
    try:
        cur.execute(
            "SELECT DISTINCT UPPER(TRIM(country_code)) AS cc "
            "FROM country_profiles "
            "WHERE country_code IS NOT NULL AND TRIM(country_code) <> '' "
            "AND (country_name ILIKE %s OR country_name ILIKE %s)",
            (cn, f"%{cn}%"),
        )
        return [r["cc"] for r in cur.fetchall() if r.get("cc")]
    except Exception:
        return []


def _build_origin_filter(cur, cn: str) -> tuple[str, tuple]:
    """SQL predicate matching a LICENSED company's shareholder NATIONALITY
    on BOTH columns, using the SAME exact-name logic for every country
    (no per-country alias list):
      • shareholder_country_name  (exact country name, ILIKE '%<name>%')
      • shareholder_country_code  (ISO alpha-2 code, resolved dynamically
        from country_profiles — this is what safely catches spelling
        variants like 'USA' without hardcoding them)
    Returns (sql_fragment, params) — params bind positionally in order."""
    patterns = (f"%{cn}%",)
    name_cond = "sc ILIKE %s"
    codes = _resolve_iso_codes(cur, cn)
    if codes:
        code_cond = " OR ".join(["UPPER(scc) = %s"] * len(codes))
        code_exists = (
            " OR EXISTS (SELECT 1 FROM "
            "jsonb_array_elements_text(shareholder_country_code::jsonb) AS scc "
            f"WHERE {code_cond})"
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
    """List companies from a given HQ country with a presence in Saudi
    Arabia (licensed and/or RHQ).

    CANONICAL SOURCE: `company_profiles`. Licensing/RHQ status is
    derived from role / registration_type (NOT the legacy `licensed`
    / `is_rhq` booleans, which are unreliable / unpopulated):

      • Licensed company   → role = 'Licensed Entity'
      • Licensed RHQ       → role = 'Licensed Entity'
                             AND registration_type = 'RHQ'
      • Non-licensed       → role <> 'Licensed Entity'
      • Non-licensed RHQ   → role = 'RHQ Entity'

    Country attribution differs by group:
      • Licensed     → nationality of shareholders (shareholder_country_name)
      • Non-licensed → country_profile_name / country_profile_id

    Returns a dict:
      {
        'rhq': [...]           # licensed RHQ rows (registration_type='RHQ')
        'licensed_only': [...] # licensed, non-RHQ rows
        'non_licensed': [...]  # non-licensed rows (country_profile_name match)
        'total_rhq': int,              # licensed RHQ count
        'total_licensed': int,         # licensed count
        'total_non_licensed': int,     # non-licensed count
        'total_non_licensed_rhq': int, # non-licensed RHQ (role='RHQ Entity')
      }
    Licensed lists are ordered by annual_revenue DESC, capped at 15 rows;
    the non-licensed list at 10, so the curation prompt stays bounded.
    """
    # _db_error is set to a message string when the DB is unreachable.
    # Callers MUST check this key — a missing key means the query ran
    # cleanly; zeros are real zeros, not "DB down".
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
            # Licensed nationality matches BOTH shareholder_country_name
            # (spellings) AND shareholder_country_code (ISO alpha-2).
            _origin_filter, _origin_params = _build_origin_filter(cur, cn)

            cur.execute(
                f"SELECT COUNT(*) c FROM company_profiles "
                f"WHERE {_origin_filter} AND {ACTIVE_LICENSED}",
                _origin_params,
            )
            out["total_licensed"] = int(cur.fetchone()["c"])
            cur.execute(
                f"SELECT COUNT(*) c FROM company_profiles "
                f"WHERE {_origin_filter} AND {ACTIVE_RHQ}",
                _origin_params,
            )
            out["total_rhq"] = int(cur.fetchone()["c"])

            cur.execute(f"""
                SELECT id, company_name, headquarters,
                       annual_revenue, employee_count,
                       industry, founded, ceo,
                       role, registration_type
                FROM company_profiles
                WHERE {_origin_filter}
                  AND {ACTIVE_RHQ}
                ORDER BY annual_revenue DESC NULLS LAST
                LIMIT 15
            """, _origin_params)
            out["rhq"] = [dict(r) for r in cur.fetchall()]

            cur.execute(f"""
                SELECT id, company_name, headquarters,
                       annual_revenue, employee_count,
                       industry, founded, ceo,
                       role, registration_type
                FROM company_profiles
                WHERE {_origin_filter}
                  AND {ACTIVE_LICENSED}
                  AND registration_type IS DISTINCT FROM 'RHQ'
                ORDER BY annual_revenue DESC NULLS LAST
                LIMIT 15
            """, _origin_params)
            out["licensed_only"] = [dict(r) for r in cur.fetchall()]

            # Non-licensed companies — keyed by country_profile_name (plain
            # varchar / country_profile_id), NOT shareholder nationality.
            # "Non-licensed" = any entity whose role is not 'Licensed Entity'.
            cur.execute(
                f"SELECT COUNT(*) c FROM company_profiles "
                f"WHERE country_profile_name ILIKE %s AND {NON_LICENSED}",
                (f"%{cn}%",),
            )
            out["total_non_licensed"] = int(cur.fetchone()["c"])

            # is-RHQ for a non-licensed company → role = 'RHQ Entity'.
            cur.execute(
                f"SELECT COUNT(*) c FROM company_profiles "
                f"WHERE country_profile_name ILIKE %s AND {NON_LICENSED_RHQ}",
                (f"%{cn}%",),
            )
            out["total_non_licensed_rhq"] = int(cur.fetchone()["c"])

            cur.execute(f"""
                SELECT id, company_name, headquarters, annual_revenue,
                       employee_count, industry, founded, ceo,
                       role, registration_type, country_profile_name
                FROM company_profiles
                WHERE country_profile_name ILIKE %s AND {NON_LICENSED}
                ORDER BY annual_revenue DESC NULLS LAST LIMIT 10
            """, (f"%{cn}%",))
            out["non_licensed"] = [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        out["_db_error"] = str(exc)
    return out


def fetch_country_sector_distribution(country_name: str) -> list[dict]:
    """Sector breakdown of the licensed companies from a given HQ
    country — the DB evidence for 'which sectors convert'. Returns
    [{industry, licensed_count, rhq_count}, ...] ordered by licensed
    volume, capped at 12 sectors."""
    if not country_name:
        return []
    cn = country_name.strip()
    try:
        conn = get_db()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Same both-column nationality match as fetch_country_saudi_investors.
            _origin_filter, _origin_params = _build_origin_filter(cur, cn)
            cur.execute(f"""
                SELECT COALESCE(NULLIF(TRIM(industry), ''), 'Unclassified')
                           AS industry,
                       COUNT(*) AS licensed_count,
                       COUNT(*) FILTER (WHERE {ACTIVE_RHQ}) AS rhq_count
                FROM company_profiles
                WHERE {_origin_filter}
                  AND {ACTIVE_LICENSED}
                GROUP BY 1
                ORDER BY licensed_count DESC
                LIMIT 12
            """, _origin_params)
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


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
    """Aggregate totals + breakdown for the canonical Saudi-licensing
    counts on company_profiles. Powers the dedicated path that
    answers "how many RHQ licences do we have?" / "how many licensed
    companies in Saudi?" with executive-grade context, not a raw
    SELECT COUNT(*) on the auxiliary rhq_licenses table (only 661
    rows vs the canonical 727 RHQ / 95k+ licensed)."""
    out = {
        "total_licensed": 0,
        "total_rhq": 0,
        "rhq_by_country": [],     # [{country, n}] top 10
        "licensed_by_country": [], # [{country, n}] top 10
    }
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"SELECT COUNT(*) c FROM company_profiles WHERE {ACTIVE_LICENSED}"
        )
        out["total_licensed"] = int(cur.fetchone()["c"])
        cur.execute(
            f"SELECT COUNT(*) c FROM company_profiles WHERE {ACTIVE_RHQ}"
        )
        out["total_rhq"] = int(cur.fetchone()["c"])

        # Top origin countries for RHQ companies via shareholder_country_name JSONB.
        cur.execute(f"""
            SELECT sc AS country, COUNT(*) AS n
            FROM company_profiles,
                 jsonb_array_elements_text(shareholder_country_name::jsonb) AS sc
            WHERE {ACTIVE_RHQ}
            GROUP BY 1
            ORDER BY n DESC LIMIT 12
        """)
        out["rhq_by_country"] = [dict(r) for r in cur.fetchall()]

        # Top origin countries for ALL licensed companies (incl RHQ).
        cur.execute(f"""
            SELECT sc AS country, COUNT(*) AS n
            FROM company_profiles,
                 jsonb_array_elements_text(shareholder_country_name::jsonb) AS sc
            WHERE {ACTIVE_LICENSED}
            GROUP BY 1
            ORDER BY n DESC LIMIT 12
        """)
        out["licensed_by_country"] = [dict(r) for r in cur.fetchall()]
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
