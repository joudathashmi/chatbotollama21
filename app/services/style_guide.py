"""
MISA Executive Briefing — Style Guide.

ONE source of truth for how every answer looks. Imported by every
prompt and every deterministic renderer in the system so the output
is consistent regardless of which path produced it.

When a new path / template is added it MUST import from here. When
the user reports a style inconsistency, the fix lives here and
propagates everywhere.

Two kinds of content live in this module:

  1. STRING CONSTANTS — text blocks injected into LLM prompts so
     the curator follows the same rules everywhere.

  2. PYTHON HELPERS — `format_currency()`, `make_footer()`, etc. used
     by deterministic (no-LLM) renderers to match the same style.

Read the comments. They explain the rationale so future editors
understand why each decision is here.
"""

from __future__ import annotations

import re
from typing import Iterable


# ═══════════════════════════════════════════════════════════════════
# 1. SECTION HEADER VOCABULARY
# ═══════════════════════════════════════════════════════════════════
# One canonical header per concept. Drift like
# "Strategic Read for MISA" vs "Strategic Read & Alignment" vs
# "Engagement Read for MISA" was a real bug — the same idea appeared
# under three different headings depending on which path fired.

HEADERS = {
    # The "Snapshot" concept — first-section summary of the entity.
    "snapshot":          "## Snapshot",
    "executive_briefing": "Executive Briefing",  # used in `## {name} — Executive Briefing`
    # Person concept — for executive_lookup intent.
    "person_role":       "## Role",
    "person_background": "## Background",
    # Country / company sub-headings.
    # Saudi / MENA Position
    "saudi_position":    "## Saudi / MENA Position",
    "economic_outlook":  "## Economic Outlook",
    "policy_regulatory": "## Policy & Regulatory",
    # Strategic / engagement framing — ONE name, everywhere.
    "strategic_read":    "## 🇸🇦 Strategic Read",
    "engagement_recommendation": "## Engagement Recommendation",
    "companies_investors": "## Companies & Investors",
    "from_documents":    "## From your documents",
    "from_web":          "## From the web",
    "from_misa_data":    "## From MISA data",
    # Engagement history (relationship_intelligence intent).
    "engagement_history": "## Engagement History",
    "misa_contacts":     "## MISA Contacts",
    "open_actions":      "## Open Action Items",
    # Web grounding.
    "whats_reported":    "## What's Reported (Live Web)",
    "supporting_reporting": "### Supporting reporting",
    # Sources footer (rendered as `_Sources: ..._` line, not a heading).
    "sources_footer":    "_Sources: {sources}_",
    # Deep-profile (/profile X) 3-pillar headings — kept distinct
    # because /profile is an explicit opt-in mode with provenance tags.
    "dp_footprint":      "### 📊 Corporate Profile & Regional Footprint",
    "dp_leadership":     "### 🇸🇦 Strategic Read & Alignment",
}


# ═══════════════════════════════════════════════════════════════════
# 2. EMOJI POLICY
# ═══════════════════════════════════════════════════════════════════
# Emoji use is intentional and minimal. Two emojis only:
#   📊 — quantitative-table sub-headers
#   🇸🇦 — Saudi-context sub-headers (strategic read, MENA position)
# No emoji in the headline (## ...). No flag/decorative emoji.
# This is enforced via the VALIDATOR (style_validator.py).

APPROVED_EMOJIS = {"📊", "🇸🇦"}


# ═══════════════════════════════════════════════════════════════════
# 3. FORBIDDEN STRINGS (consolidated)
# ═══════════════════════════════════════════════════════════════════
# These get stripped by answer_finalize.finalize_answer (every composer
# exit) AND checked by the style validator on the curation path.
# Adding one here must propagate to BOTH — never only to prompts.

FORBIDDEN_STRINGS = (
    # Confidence tags
    "(High)", "(Medium)", "(Low)", "(Unknown)",
    # Provenance markers (kept in /profile mode only; stripped from
    # the normal-flow output where the user sees the answer directly).
    "[DB]", "[gk]", "[inferred]",
    # Source: DB debug strings
    "Source: DB", "**Source:** DB", "Source: web",
    # General-knowledge italic marker (left over from a transitional
    # phase; user explicitly asked to remove these from output).
    "_(general knowledge)_",
    # Placeholder strings (we OMIT empty fields, never display these).
    "Not available in the current database",
    # Legacy section names (old "Notable Companies & Investors" /
    # "Other notable companies" produced the dup-companies bug).
    "MISA company records",
    "Other notable companies",
    "Notable Companies & Investors",
    # Strategy-filler phrases. Prompt-level bans alone don't stop the
    # model from emitting these; listing them here makes the style
    # validator flag them and regenerate with the violation named.
    "explore opportunities",
    "explore further",
    "explore partnership",
    "explore collaborations",
    "leverage",
    "strengthen bilateral relations",
    "facilitate knowledge exchange",
    "foster a collaborative environment",
    # Trust-killing meta / schema leaks
    "From the MISA Record",
    "Background (general knowledge)",
    "Internal records do not currently show",
    "Company: Multiple",
)


# ═══════════════════════════════════════════════════════════════════
# 4. NUMBER + CURRENCY FORMATTING
# ═══════════════════════════════════════════════════════════════════
# ONE format everywhere. Used by deterministic renderers; the LLM
# is also instructed (via STYLE_GUIDE_PROMPT below) to match this.

def format_currency(value, default: str = "—") -> str:
    """Render a numeric value as `$X.XB` / `$X.XM` / `$XK` / `$X`.

    Examples:
      391_000_000_000 → "$391.0B"
      62_600_000      → "$62.6M"
      846_910_000     → "$846.9M"
      None / non-numeric → default ("—" so the bullet doesn't break)
    """
    if value is None:
        return default
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    if n <= 0:
        return "$0"
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"${n / 1_000:.0f}K"
    return f"${n:.0f}"


def format_count(n) -> str:
    """Render an integer count with thousands separators. `None` → `—`."""
    if n is None:
        return "—"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def make_footer(sources: Iterable[str]) -> str:
    """Render the canonical footer line.
    Example:
      make_footer(["company_profiles.is_rhq", "country_vision_outlooks"])
      → "_Sources: company_profiles.is_rhq, country_vision_outlooks._"
    """
    parts = [s.strip() for s in sources if s and s.strip()]
    if not parts:
        return ""
    return f"_Sources: {', '.join(parts)}._"


# ═══════════════════════════════════════════════════════════════════
# 5. STYLE GUIDE PROMPT BLOCK
# ═══════════════════════════════════════════════════════════════════
# Injected into EVERY curation prompt + every per-intent directive.
# The single point where prompt-side style is defined. Update here →
# every answer's style updates.

STYLE_GUIDE_PROMPT = """
═══════════ MISA EXECUTIVE BRIEFING — STYLE GUIDE ═══════════
(These rules apply to EVERY answer regardless of intent. They make
the system feel like one voice instead of ten different templates.)

SECTION HEADERS — use ONLY these (verbatim):
  ## Snapshot                            (entity overview)
  ## Role                                (executive_lookup — person facts)
  ## Background                          (person public context under Role)
  ## Saudi / MENA Position               (Saudi-presence section)
  ## Economic Outlook                    (country economy)
  ## Policy & Regulatory                 (regulatory section)
  ## Engagement History                  (prior meetings/interactions)
  ## MISA Contacts                       (named people at MISA / partner)
  ## Open Action Items                   (action tracking)
  ## Engagement Recommendation           (engagement_strategy intent)
  ## Companies & Investors               (country / market lists)
  ## From your documents                 (document-library answers)
  ## From the web                        (live-web complement)
  ## From MISA data                      (optional DB lane in hybrid)
  ## What's Reported (Live Web)          (web-augmented succession etc.)
  ### Supporting reporting               (sub-heading under What's Reported)
  ## 🇸🇦 Strategic Read                  (closing strategic-fit analysis)
  ### 📊 Top HQ Countries — RHQ Licence Holders   (only in licensing briefings)
  ### 🇸🇦 Licensed Pool (Broader)        (only in licensing briefings)
For executive briefings of companies/countries, the H1 is the entity
name with " — Executive Briefing":
  ## <Entity Name> — Executive Briefing
NEVER invent new headings. NEVER use synonyms ("Strategic Read for
MISA" ≠ "🇸🇦 Strategic Read" — pick the latter).

EMOJI — only two emojis allowed, only in these positions:
  📊 in `### 📊 Top HQ Countries...` and `### 📊 Corporate Profile...`
  🇸🇦 in `## 🇸🇦 Strategic Read` and `### 🇸🇦 Licensed Pool...`
No other emoji anywhere. No emoji in the H1 heading.

HORIZONTAL RULES (`---`) — use sparingly, ONLY between top-level
sections in a multi-pillar briefing. Never between bullets.

SOURCE LANES — `## From your documents`, `## From the web`, and
`## From MISA data` are ONLY for real content from that source.
If a lane has nothing useful: OMIT the heading entirely. Never write
"No relevant documents…", "No relevant information from the web…",
or similar empty stubs.

BOLDING ECONOMY — at most ONE **bolded** phrase per bullet, on the
critical number or strategic trigger. Bullets with three bolds read
as noise.

NUMBER FORMAT — currency is `$391.0B` / `$62.6M` / `$799K`.
NEVER write `$391 billion` or `$391,000,000,000` or `391B`.
Counts use thousands separators: `**164,000 employees**`.
Percentages: `**25%**` (no decimal when whole).

FOOTER — every answer ends with a single italic source line:
  _Sources: <table_or_column>, <table_or_column>._
Examples:
  _Sources: company_profiles, company_executives._
  _Sources: company_profiles.licensed / company_profiles.is_rhq._

FORBIDDEN IN OUTPUT (these are stripped by the post-processor; just
don't write them in the first place):
  (High) (Medium) (Low) (Unknown)
  [DB] [gk] [inferred]
  Source: DB
  _(general knowledge)_
  Not available in the current database
  MISA company records  /  Other notable companies
For unknown values: OMIT the bullet entirely. Never display a
placeholder.

BANNED VAGUE WORDS — never write these hollow adjectives/nouns; they
signal generalisation, which is exactly what a MISA analyst distrusts.
Replace each with a specific, named fact:
  synergies / synergistic, holistic, cutting-edge, best-in-class,
  world-class, game-changer / game-changing, paradigm, seamless(ly),
  robust ecosystem, foster innovation, unlock potential, drive growth,
  leverage synergies, tap into, moving forward.
Instead of "world-class engineering capabilities" write "engineering
capabilities proven on <named project>"; instead of "strong synergies"
name the specific sector overlap and the Saudi programme it maps to.

OPENING DISCIPLINE — for any intent that has a "primary answer"
(executive_lookup, succession, financial_lookup, country_profile
when the question is "how many"), the FIRST non-heading line must
be the direct answer. Never lead with a preamble or premise.
═══════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════
# 6. STYLE VALIDATOR — programmatic checks
# ═══════════════════════════════════════════════════════════════════

# Compiled patterns reused by the validator. Kept here so the
# enforcement of each rule lives next to its definition.

_FORBIDDEN_PATTERNS = [
    re.compile(re.escape(s), re.IGNORECASE)
    for s in FORBIDDEN_STRINGS
]


def find_style_violations(answer: str) -> list[str]:
    """Return a list of human-readable violation strings for any
    style-guide infringements found in `answer`. Empty list when
    fully compliant. Used by the auto-validator AND by the corpus
    auto-eval — same definition, same enforcement.
    """
    if not answer:
        return []
    violations: list[str] = []
    # 1. Forbidden strings present
    for pat, raw in zip(_FORBIDDEN_PATTERNS, FORBIDDEN_STRINGS):
        if pat.search(answer):
            violations.append(f"forbidden_string:{raw}")
    # 2. Disallowed emoji (only 📊 and 🇸🇦 allowed)
    emoji_chars = re.findall(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+",
        answer,
    )
    for ch in emoji_chars:
        if ch.strip() and ch not in APPROVED_EMOJIS:
            # 🇸🇦 is two codepoints (regional indicators); allow it
            if "🇸" not in ch and "🇦" not in ch:
                violations.append(f"unapproved_emoji:{ch}")
    # 3. Old-style "Strategic Read for MISA" (canonical is "Strategic Read")
    if "Strategic Read for MISA" in answer:
        violations.append("legacy_heading:Strategic Read for MISA")
    if "Engagement Read for MISA" in answer:
        violations.append("legacy_heading:Engagement Read for MISA")
    # 4. Number formatting — flag the long-form ones we explicitly banned
    if re.search(r"\$\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", answer):
        violations.append("number_format:long_currency_with_commas")
    if re.search(r"\$\d+(?:\.\d+)?\s+billion\b", answer, re.I):
        violations.append("number_format:dollars_billion_spelled_out")
    return violations


# Convenience for callers who just want yes/no.
def is_style_compliant(answer: str) -> bool:
    return not find_style_violations(answer)
