"""
Country name → id resolver used by the chat engine.

Many tables in the DB reference a country by integer FK rather than name:
  - `country_id`         (FK to public.countries.id)
  - `country_profile_id` (FK to public.country_profiles.id)
The chat layer's LLM passes country names ("Pakistan", "Saudi Arabia") in
filters, so we resolve them server-side to the right integer id, taking
fuzzy matches into account (via pg_trgm if available, then difflib).

Caches both forward (name → ids) and reverse (id → name) lookups at module
level. Invalidate via `invalidate_cache()` if the underlying tables change
during a session.
"""

from __future__ import annotations

import difflib
import re
from threading import Lock

from app.database import _run_with_db_transient_retry

_lock = Lock()
_forward_cache: dict[str, dict[str, int]] = {}   # normalised name → {"countries": id, "country_profiles": id}
_table_to_country_table: dict[str, str] = {}     # FK column name → target table for resolution


def _norm(name: str) -> str:
    """Normalise a country name for matching."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


# Semantic aliases that token matching can't derive: the common name
# shares no tokens with the official one. Normalised alias → list of
# normalised official-name keys to try.
_SEMANTIC_ALIASES: dict[str, list[str]] = {
    "south korea": ["korea, republic of", "korea, south"],
    "north korea": ["korea, democratic people's republic of",
                    "korea, north"],
    "russia": ["russian federation"],
    "syria": ["syrian arab republic"],
    "iran": ["iran, islamic republic of"],
    "vietnam": ["viet nam"],
    "laos": ["lao people's democratic republic"],
    "uae": ["united arab emirates"],
    "usa": ["united states", "united states of america"],
    "uk": ["united kingdom"],
    "czechia": ["czech republic"],
    "turkey": ["turkiye", "türkiye"],
    "ivory coast": ["cote d'ivoire", "côte d'ivoire"],
}


def _key_variants(name: str) -> list[str]:
    """All lookup keys a canonical DB name should answer to. The DB
    stores comma-inverted names ('Korea, South', 'Congo, Democratic
    Republic of the'); users type natural order ('South Korea'). We
    index: the name as-is, the comma-reversed natural form, and a
    token-sorted form (commas stripped, words alphabetised) that makes
    'South Korea' and 'Korea, South' collide on the same key."""
    base = _norm(name)
    if not base:
        return []
    variants = [base]
    if "," in base:
        parts = [p.strip() for p in base.split(",") if p.strip()]
        if len(parts) == 2:
            variants.append(_norm(f"{parts[1]} {parts[0]}"))
    tokens = sorted(re.findall(r"[a-z0-9]+", base))
    if tokens:
        variants.append(" ".join(tokens))
    return variants


def _fetch_country_index(conn) -> dict[str, dict[str, int]]:
    """Build {normalised name → {table → id}} for `countries` + `country_profiles`."""
    out: dict[str, dict[str, int]] = {}
    with conn.cursor() as cur:
        # Pull canonical names from both tables and accept either the name
        # column or ISO code as a key.
        cur.execute("""
            SELECT id, name, code, iso3 FROM public.countries
            WHERE name IS NOT NULL AND btrim(name) <> ''
        """)
        for cid, name, code, iso3 in cur.fetchall():
            for k in filter(None, (name, code, iso3)):
                for variant in _key_variants(k):
                    out.setdefault(variant, {}).setdefault("countries", cid)

        cur.execute("""
            SELECT id, country_name, country_code, country_name_ar
            FROM public.country_profiles
            WHERE country_name IS NOT NULL AND btrim(country_name) <> ''
        """)
        for cid, name, code, name_ar in cur.fetchall():
            for k in filter(None, (name, code, name_ar)):
                for variant in _key_variants(k):
                    out.setdefault(variant, {}).setdefault(
                        "country_profiles", cid)
    return out


def _ensure_loaded() -> dict[str, dict[str, int]]:
    global _forward_cache
    with _lock:
        if _forward_cache:
            return _forward_cache
        try:
            _forward_cache = _run_with_db_transient_retry(_fetch_country_index)
        except Exception:
            _forward_cache = {}
        return _forward_cache


def invalidate_cache() -> None:
    global _forward_cache
    with _lock:
        _forward_cache = {}


# Map FK column name → which canonical country table it points to.
# Default: a column named `country_id` → `countries`; `country_profile_id`
# → `country_profiles`.
def _fk_target_table(col: str) -> str | None:
    cl = (col or "").lower()
    if cl == "country_profile_id":
        return "country_profiles"
    if cl == "country_id" or cl.endswith("_country_id"):
        return "countries"
    return None


def resolve_country(name: str) -> dict[str, int] | None:
    """Returns {table_name: id} for a country name. Tries exact match first,
    then trigram-style ratio match via difflib over all known canonical
    names. Returns None if no acceptable match."""
    if not name or not str(name).strip():
        return None
    idx = _ensure_loaded()
    if not idx:
        return None
    key = _norm(name)

    # 1) exact match (covers full names, ISO codes, Arabic names) —
    #    every variant of the user's input is tried, so 'South Korea'
    #    hits the token-sorted key of 'Korea, South'. Semantic aliases
    #    ('South Korea' → 'Korea, Republic of') fill in whichever
    #    target table the direct variants missed — the two country
    #    tables use different official spellings for the same country.
    merged: dict[str, int] = {}
    for variant in _key_variants(name):
        for tbl, cid in (idx.get(variant) or {}).items():
            merged.setdefault(tbl, cid)
    if key in _SEMANTIC_ALIASES:
        for official in _SEMANTIC_ALIASES[key]:
            for variant in _key_variants(official):
                for tbl, cid in (idx.get(variant) or {}).items():
                    merged.setdefault(tbl, cid)
    if merged:
        return merged

    # 2) fuzzy match using difflib ratio across known canonical names
    keys = list(idx.keys())
    if not keys:
        return None
    scored = sorted(
        ((difflib.SequenceMatcher(None, key, k).ratio(), k) for k in keys),
        reverse=True,
    )
    best_ratio, best_key = scored[0]
    if best_ratio >= 0.70:
        return dict(idx[best_key])
    return None


_reverse_cache: dict[str, dict[int, str]] = {}


def _fetch_reverse_index(conn) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {"countries": {}, "country_profiles": {}}
    with conn.cursor() as cur:
        cur.execute("SELECT id, name FROM public.countries "
                    "WHERE name IS NOT NULL")
        out["countries"] = {cid: n for cid, n in cur.fetchall()}
        cur.execute("SELECT id, country_name FROM public.country_profiles "
                    "WHERE country_name IS NOT NULL")
        out["country_profiles"] = {cid: n for cid, n in cur.fetchall()}
    return out


def country_name_for_fk(fk_column: str, country_id: int) -> str | None:
    """Reverse lookup: integer FK value → canonical country name.
    Used to annotate rows sent to the curation LLM so it can see WHICH
    country a country_id / country_profile_id actually points to —
    without this, a row from Saudi Arabia's aggregate series can get
    silently attributed to whatever country the question mentioned."""
    global _reverse_cache
    target = _fk_target_table(fk_column)
    if target is None:
        return None
    with _lock:
        if not _reverse_cache:
            try:
                _reverse_cache = _run_with_db_transient_retry(
                    _fetch_reverse_index)
            except Exception:
                _reverse_cache = {}
    try:
        return (_reverse_cache.get(target) or {}).get(int(country_id))
    except (TypeError, ValueError):
        return None


def resolve_country_id_for_fk(name: str, fk_column: str) -> int | None:
    """Resolve a country name to the integer ID expected by the given FK
    column (`country_id` or `country_profile_id` — case-insensitive).
    Returns None if the column isn't a known FK or if no acceptable match."""
    target = _fk_target_table(fk_column)
    if target is None:
        return None
    bag = resolve_country(name)
    if not bag:
        return None
    return bag.get(target)
