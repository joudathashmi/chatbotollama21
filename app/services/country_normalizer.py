"""
Business-card country normalizer.

Maps the free-text country string OpenAI extracts from a business card
(e.g. "USA", "U.A.E.", "Pakstan") to the client's canonical country name
so the API response lines up with the 265 names stored in their database.

Design: extract first (LLM), normalize second (code). The OpenAI prompt is
NOT changed — it never sees the 265-name list (that would bloat every call's
token cost and the model paraphrases anyway). Instead we resolve server-side:

    1. exact / alias match (free, deterministic)  →  canonical name
    2. fuzzy match via rapidfuzz (>= cutoff)       →  canonical name
    3. no confident match                          →  original value, unchanged

This module is self-contained and does NOT touch Postgres. It is dedicated
to the business-card reader only.
"""

from __future__ import annotations

import re

from app.data.countries import CANONICAL_COUNTRIES, COUNTRY_ALIASES

# rapidfuzz is the intended matcher. Fall back to stdlib difflib if it is
# not installed, so a forgotten `pip install` degrades gracefully (exact +
# alias hits still work; fuzzy gets a slightly weaker scorer) instead of
# crashing the whole business-card endpoint at import time.
try:
    from rapidfuzz import fuzz, process

    _HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised only when dep missing
    import difflib

    _HAVE_RAPIDFUZZ = False

# Minimum score (0–100) for a fuzzy match to be accepted. Below this we keep
# the original value rather than risk a wrong canonical name. Tune against
# real cards.
MATCH_SCORE_CUTOFF: int = 85


def _norm(s: str | None) -> str:
    """Lowercase, strip, and collapse internal whitespace for comparison."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# Normalized-key → canonical-name index, built once at import. Covers both the
# canonical names themselves and the alias table.
_EXACT_INDEX: dict[str, str] = {_norm(c): c for c in CANONICAL_COUNTRIES}
for _alias, _canonical in COUNTRY_ALIASES.items():
    _EXACT_INDEX[_norm(_alias)] = _canonical


def resolve_country_name(raw: str | None) -> str:
    """Return the canonical country name for `raw`, or `raw` unchanged.

    - Empty / whitespace input is returned as-is.
    - Exact or alias hits return immediately (no fuzzy cost).
    - Otherwise a fuzzy match >= MATCH_SCORE_CUTOFF returns the canonical name.
    - No confident match returns the original string (never loses data).
    """
    if not raw or not str(raw).strip():
        return raw or ""

    key = _norm(raw)

    # 1) exact / alias match
    canonical = _EXACT_INDEX.get(key)
    if canonical is not None:
        return canonical

    # 2) fuzzy match against the canonical list
    if _HAVE_RAPIDFUZZ:
        match = process.extractOne(
            str(raw),
            CANONICAL_COUNTRIES,
            scorer=fuzz.WRatio,
            score_cutoff=MATCH_SCORE_CUTOFF,
        )
        if match:
            return match[0]
    else:  # pragma: no cover - difflib fallback path
        best = difflib.get_close_matches(
            str(raw), CANONICAL_COUNTRIES, n=1, cutoff=MATCH_SCORE_CUTOFF / 100
        )
        if best:
            return best[0]

    # 3) no confident match — keep what OpenAI gave us
    return raw
