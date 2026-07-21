"""
Lightweight intent classifier for chat questions.

Categorises the user's question into one of a small set of intent labels
that the chat engine and logging layer use to (a) make routing decisions
explicit and (b) record what kind of query was attempted on each turn.

The classification is heuristic, not ML — same approach as the rest of
the input-cleaning pipeline. It's fast, deterministic, debuggable, and
good enough at the granularity the chat layer needs.

Intent labels:
  - "off_topic"      → general-knowledge question; bypass DB
  - "person_lookup"  → named individual (Tim Cook, Sundar Pichai)
  - "count"          → "how many", "total number", "count of"
  - "browse"         → "show me X", "list X", "top N X"
  - "comparison"     → "compare X and Y", "X vs Y"
  - "followup"       → pronominal / command following prior turn
  - "entity_lookup"  → about a specific named company / country
  - "broad_topic"    → topic / concept query without a named entity
  - "unknown"        → couldn't classify
"""

from __future__ import annotations

import re

from app.services.input_cleaner import (
    clean_user_question,
    detect_pure_browse,
    looks_like_general_knowledge_question,
)


_COUNT_RE = re.compile(
    r"\b(how\s+many|total\s+(number|count)\s+of|number\s+of|count\s+of|"
    r"count\s+(the\s+)?\w+|how\s+much\s+(of\s+|are\s+|is\s+)?)\b",
    re.I,
)
_COMPARE_RE = re.compile(
    r"\b(compare|vs\.?|versus|difference\s+between|"
    r"\w+\s+against\s+\w+)\b",
    re.I,
)
_FOLLOWUP_HINT_RE = re.compile(
    r"\b(more|further|continue|elaborate|expand|deeper|"
    r"research|investigate|analyze|analyse|plan|strategy|brief|"
    r"this|that|them|it|its|their|tell\s+me\s+more|go\s+on)\b",
    re.I,
)
_PERSON_NAME_RE = re.compile(
    r"\b(who\s+is|who\s+was|ceo\s+of|chairman\s+of|founder\s+of|"
    r"executives?\s+of|leader(?:ship)?\s+of|board\s+of)\b",
    re.I,
)


def _looks_like_person_name(s: str) -> bool:
    """Two-token Capitalized phrase where neither token is a known
    company suffix → almost certainly a person name (Tim Cook,
    Sundar Pichai, Mohammed Al-Rajhi)."""
    if not s:
        return False
    tokens = s.strip().split()
    if len(tokens) != 2:
        return False
    company_suffixes = {"inc", "ltd", "llc", "plc", "corp", "co",
                        "group", "holdings", "company", "limited",
                        "incorporated", "corporation"}
    for t in tokens:
        if t.lower().rstrip(",.;") in company_suffixes:
            return False
        # Must be alpha and Capitalized
        if not t[:1].isalpha() or not t[0].isupper():
            return False
    return True


def detect_intent(user_question: str, history: list | None = None) -> str:
    """Return a single intent label. See module docstring for the set."""
    q = (user_question or "").strip()
    if not q:
        return "unknown"

    # 1) Off-topic short-circuit (matches the chat-engine short-circuit)
    if looks_like_general_knowledge_question(q):
        return "off_topic"

    # 2) Pure browse: "show me companies", "list deals" — exact phrase match
    if detect_pure_browse(q) is not None:
        return "browse"

    # 3) Count / aggregation
    if _COUNT_RE.search(q):
        return "count"

    # 4) Comparison
    if _COMPARE_RE.search(q):
        return "comparison"

    # 5) Person lookup — explicit phrasing or extracted entity looks
    #    like a person name (two Capitalized tokens, no company suffix)
    pack = clean_user_question(q)
    entity = (pack.get("entity_candidate") or "").strip()
    if _PERSON_NAME_RE.search(q):
        return "person_lookup"
    if _looks_like_person_name(entity):
        return "person_lookup"

    # 6) Follow-up — pronominal / command words AND history present
    if history and _FOLLOWUP_HINT_RE.search(q):
        # Only count as follow-up if entity is weak or absent
        if not entity or len(entity.split()) > 4:
            return "followup"

    # 7) Entity lookup — has a usable named entity in the question
    if entity and len(entity.split()) <= 4:
        return "entity_lookup"

    # 8) Broad / topic
    if entity:
        return "broad_topic"

    return "unknown"
