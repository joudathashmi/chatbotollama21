"""
Company alias resolver.

The DB stores companies under their legal names (Alphabet Inc.,
Apple Inc., Meta Platforms Inc., …). Users type brand or trade names
(Google, Apple, Facebook) and expect to find the legal record.

This module maintains a curated alias map and helps the chat engine
(a) expand smart-search terms to include all known aliases of a
typed entity, and (b) confirm a returned row really IS about the
entity the user named (alias match counts).

The map is hand-maintained because the only fully reliable list of
aliases for top global firms is curated knowledge. New entries
should land here whenever an alias-miss is observed in production —
see `tests/golden_cases.json` for the failure-mode-to-test pattern.

If the DB ever grows its own aliases column, this module can be
extended to merge DB aliases with the curated map. For now the
curated map is the source.
"""

from __future__ import annotations


# canonical_key (lowercase) → list of alias strings (any case)
# Symmetric: if X is an alias of Y, Y is also an alias of X via
# expand_aliases().
COMPANY_ALIASES: dict[str, list[str]] = {
    # ─── Big tech ───
    "apple":       ["Apple Inc", "Apple Computer", "Apple Operations",
                    "Apple Distribution", "Apple Middle East",
                    "Apple Saudi Arabia"],
    "alphabet":    ["Alphabet Inc", "Google", "Google LLC",
                    "Google Regional Office", "Google Cloud",
                    "Google Saudi", "Google MENA"],
    "google":      ["Alphabet", "Alphabet Inc", "Google LLC",
                    "Google Regional Office", "Google Cloud"],
    "microsoft":   ["Microsoft Corporation", "Microsoft Arabia",
                    "Microsoft Saudi", "Microsoft MENA"],
    "meta":        ["Meta Platforms", "Facebook", "Instagram",
                    "WhatsApp", "Meta Inc"],
    "facebook":    ["Meta", "Meta Platforms", "Meta Inc"],
    "instagram":   ["Meta", "Meta Platforms"],
    "whatsapp":    ["Meta", "Meta Platforms"],
    "amazon":      ["Amazon.com", "Amazon Web Services", "AWS",
                    "Amazon Saudi", "Amazon MENA"],
    "aws":         ["Amazon Web Services", "Amazon", "Amazon.com"],
    "tesla":       ["Tesla Motors", "Tesla Inc"],
    "spacex":      ["Space Exploration Technologies"],
    "openai":      ["OpenAI Inc"],
    "anthropic":   ["Anthropic PBC"],
    "nvidia":      ["NVIDIA Corporation", "Nvidia"],
    "ibm":         ["International Business Machines"],
    "uber":        ["Uber Technologies"],
    "netflix":     ["Netflix Inc"],
    "twitter":     ["X", "X Corp", "X Holdings"],
    "x":           ["Twitter", "X Corp"],
    # ─── Consulting / professional services ───
    "deloitte":    ["Deloitte Touche Tohmatsu", "Deloitte Saudi",
                    "Deloitte & Touche"],
    "pwc":         ["PricewaterhouseCoopers", "PwC Saudi"],
    "ey":          ["Ernst & Young", "EY Saudi"],
    "kpmg":        ["KPMG International"],
    "mckinsey":    ["McKinsey & Company"],
    "bcg":         ["Boston Consulting Group"],
    "bain":        ["Bain & Company"],
    # ─── Energy ───
    "aramco":      ["Saudi Aramco", "Saudi Arabian Oil Company"],
    "shell":       ["Royal Dutch Shell", "Shell plc"],
    "bp":          ["British Petroleum", "BP plc"],
    "totalenergies": ["Total", "TotalEnergies SE"],
}


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def expand_aliases(entity: str) -> list[str]:
    """Return [entity, …all known aliases…], dedup-preserving order.

    Symmetric: passing 'Google' returns Google + Alphabet + variants;
    passing 'Alphabet' returns the same combined set. Comparison is
    case-insensitive.
    """
    if not entity or not _normalize(entity):
        return []
    original = entity.strip()
    e_norm = _normalize(original)
    collected: list[str] = [original]

    # Direct hit on the canonical key
    if e_norm in COMPANY_ALIASES:
        collected.extend(COMPANY_ALIASES[e_norm])

    # Reverse: entity matches some alias — pull canonical + its alias list
    for canonical, alias_list in COMPANY_ALIASES.items():
        if e_norm == canonical:
            continue
        if any(_normalize(a) == e_norm for a in alias_list):
            collected.append(canonical)
            collected.extend(alias_list)

    # Dedup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for s in collected:
        k = _normalize(s)
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def matches_any_alias(entity: str, text: str) -> bool:
    """True if `text` contains `entity` OR any of its aliases as a
    case-insensitive whole-word substring. Used by the row-sanity
    guard so 'Google' → row 'Alphabet, Inc.' is accepted as a real
    match instead of being rejected as noise."""
    import re
    if not entity or not text:
        return False
    text_lc = text.lower()
    for alias in expand_aliases(entity):
        a = alias.lower().strip()
        if not a:
            continue
        if re.search(r"\b" + re.escape(a) + r"\b", text_lc):
            return True
    return False
