"""
Deterministic candidate discovery + ambiguity detection.

Runs BEFORE the LLM routing call, so the clarification decision does
NOT depend on which table the model happens to choose on a given run.
For short entity-lookup questions ("alpha", "apple", "tesla"), this
module:

  1. directly hits the DB smart-search for the entity
  2. deduplicates same-company name variants
  3. scores candidates against the query
  4. applies deterministic ambiguity rules
  5. returns a clarification message OR `None` (proceed with normal flow)

Threshold constants are at the top of the file and can be tuned without
touching the algorithm.
"""

from __future__ import annotations

import difflib
import re
from typing import Optional

from app.database import run_rhq_company_smart_search


# ─── Tunable thresholds ───────────────────────────────────────────────

# Top candidate must score at least this against the user's entity for
# the result set to even be considered ambiguous. Below this, smart-
# search is just returning noise and we should let the normal path
# handle the "no match" message.
TOP_SCORE_MIN = 0.55

# If the gap between #1 and #2 is bigger than this, #1 dominates and
# we should NOT ask for clarification — there's a clear best match.
DOMINANCE_GAP = 0.10

# Minimum score for a candidate to be SHOWN in the clarification list.
SHOW_THRESHOLD = 0.50

# How many candidates to surface in the clarification message.
MAX_CANDIDATES_SHOWN = 5

# Maximum tokens in the user entity for ambiguity to even be considered.
# Long queries like "Dentons UK And Middle East LLP" already pin down
# one entity.
MAX_ENTITY_TOKENS = 2

# Smart-search row cap for candidate discovery.
DISCOVERY_LIMIT = 12


# ─── Helpers ──────────────────────────────────────────────────────────

_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|ltd|limited|llc|llp|plc|corp|"
    r"corporation|co|group|holdings|company)\b",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalise_company_name(name: str) -> str:
    """Normalise for dedup + exact-match comparison.
    'Alphabet, Inc.' / 'Alphabet Inc.' / 'Alphabet Inc' → 'alphabet'."""
    if not name:
        return ""
    n = _NON_ALNUM_RE.sub(" ", name.lower())
    n = _COMPANY_SUFFIX_RE.sub(" ", n)
    n = _WHITESPACE_RE.sub(" ", n).strip()
    return n


_CONTENT_WORD_STOP = {
    "the", "and", "for", "inc", "ltd", "llc", "plc", "corp", "co",
    "group", "holdings", "company", "limited", "incorporated",
    "corporation", "international", "global", "saudi", "arabia",
}


def _shares_content_word(entity: str, name: str) -> bool:
    """True if the user's entity and the candidate name share at least
    one meaningful alphanumeric token (length >= 3, not a stopword /
    company-suffix / generic geography), OR if a 4+-char entity token
    is a substring of the candidate name.

    Guards against pure letter-noise fuzzy matches like
    'Sundar Pichai' → 'Standard Chartered' (no shared word — purely
    s/n/a/r letter overlap) being surfaced as clarification candidates.
    """
    ent_words = {
        w for w in re.findall(r"[a-z0-9]{3,}", (entity or "").lower())
        if w not in _CONTENT_WORD_STOP
    }
    name_words = {
        w for w in re.findall(r"[a-z0-9]{3,}", (name or "").lower())
        if w not in _CONTENT_WORD_STOP
    }
    if ent_words & name_words:
        return True
    name_lc = (name or "").lower()
    for w in ent_words:
        if len(w) >= 4 and w in name_lc:
            return True
    return False


def _score_against_entity(name: str, entity_lc: str) -> float:
    """Best of: full-string ratio, head-only ratio, per-word ratio."""
    nl = (name or "").lower()
    if not nl or not entity_lc:
        return 0.0
    head = nl.split(",", 1)[0].strip()
    best = max(
        difflib.SequenceMatcher(None, entity_lc, nl).ratio(),
        difflib.SequenceMatcher(None, entity_lc, head).ratio(),
    )
    for word in re.findall(r"[a-z0-9]+", head):
        if len(word) < 3:
            continue
        r = difflib.SequenceMatcher(None, entity_lc, word).ratio()
        if r > best:
            best = r
    return best


# ─── Public API ───────────────────────────────────────────────────────

def discover_candidates(entity: str) -> list[dict]:
    """Run DB smart-search for the entity and return a deduplicated,
    scored candidate list. Pure DB call — no LLM, no routing variance.

    Each candidate dict has:
      name      — original company_name from the DB row
      norm_key  — normalised dedup key (suffix-stripped)
      hq        — global_headquarters if present
      sector    — sector if present
      score     — fuzzy score [0..1] against the entity
    """
    if not entity or len(entity.strip()) < 3:
        return []
    e = entity.strip().lower()

    try:
        df, _, _ = run_rhq_company_smart_search([entity], DISCOVERY_LIMIT)
    except Exception:
        return []
    if df is None or df.empty:
        return []

    by_key: dict[str, dict] = {}
    for r in df.to_dict(orient="records"):
        name = (r.get("company_name") or "").strip()
        if not name:
            continue
        key = normalise_company_name(name)
        if not key:
            continue
        # GUARD: drop pure letter-noise matches. If 'Sundar Pichai'
        # produced 'Standard Chartered' / 'SNAM SpA' / 'SNAP Inc.',
        # none share a content word with the entity — those are
        # trigram noise, not real candidates, and must never reach
        # the clarification card.
        if not _shares_content_word(e, name):
            continue
        score = _score_against_entity(name, e)
        existing = by_key.get(key)
        if existing is None or existing["score"] < score:
            by_key[key] = {
                "name":     name,
                "norm_key": key,
                "hq":       (r.get("global_headquarters") or r.get("rhq_country") or "") or None,
                "sector":   r.get("sector") or None,
                "score":    score,
            }
    return sorted(by_key.values(), key=lambda c: -c["score"])


def detect_ambiguity(entity: str, candidates: list[dict]) -> Optional[list[dict]]:
    """Return list of candidates to surface in a clarification, or None
    if the set is unambiguous (either a clear winner, no good matches,
    or only same-company duplicates).

    DETERMINISTIC RULES (in order):
      1. Need at least 2 distinct (normalised) candidates.
      2. Top candidate must score >= TOP_SCORE_MIN. Otherwise smart-
         search is returning noise — let the normal no-match path
         handle it.
      3. If user's normalised entity EXACTLY equals any candidate's
         normalised name → that candidate dominates → no ambiguity.
      4. If top score - second score > DOMINANCE_GAP → top dominates
         → no ambiguity.
      5. Otherwise → AMBIGUOUS. Return top candidates with score
         >= SHOW_THRESHOLD, capped at MAX_CANDIDATES_SHOWN.
    """
    if not candidates or len(candidates) < 2:
        return None
    if not entity:
        return None

    e_norm = normalise_company_name(entity)
    if not e_norm:
        return None

    # Rule 3: exact-match dominance
    for c in candidates:
        if c["norm_key"] == e_norm:
            return None

    top = candidates[0]
    second = candidates[1]

    # Rule 2: top must be reasonably high
    if top["score"] < TOP_SCORE_MIN:
        return None

    # Rule 4: dominance gap
    if (top["score"] - second["score"]) > DOMINANCE_GAP:
        return None

    shown = [c for c in candidates[:MAX_CANDIDATES_SHOWN]
             if c["score"] >= SHOW_THRESHOLD]
    if len(shown) < 2:
        return None
    return shown


def is_short_entity_query(entity: str | None) -> bool:
    """Should we run the pre-LLM ambiguity check on this query?
    Yes if the entity is a short single-token (or two-token) string
    that looks like a name fragment ('alpha', 'apple', 'tata')."""
    if not entity:
        return False
    e = entity.strip()
    if not e:
        return False
    if len(e) < 3:
        return False
    tokens = e.split()
    if len(tokens) > MAX_ENTITY_TOKENS:
        return False
    return any(c.isalpha() for c in e)


def format_clarification(entity: str, candidates: list[dict]) -> str:
    """Build the markdown clarification message the user sees."""
    ent_clean = entity.strip()
    lines: list[str] = []
    for c in candidates:
        bits = [f"**{c['name']}**"]
        if c.get("hq"):
            bits.append(f"(HQ {c['hq']})")
        if c.get("sector"):
            bits.append(f"— {c['sector']}")
        lines.append("  - " + " ".join(bits))
    return (
        f"## Multiple possible matches for \"{ent_clean}\"\n\n"
        f"I found several distinct records that could match your "
        f"query. Please tell me which one you meant:\n\n"
        + "\n".join(lines) +
        "\n\n_Reply with the exact name and I'll pull the full "
        "profile, including FK-linked AI insights, executives, "
        "and MENA presence._"
    )
