"""
Answer-depth detector.

Intent tells us WHAT the user wants (executive_lookup, company_profile,
etc.). Depth tells us HOW MUCH the user wants.

The same intent can take three forms:
  - "Where is Apple's RHQ?"            → simple_fact (1-3 line answer)
  - "Tell me about Apple's market"     → operational_detail (rich sub-sections)
  - "Give me a strategic briefing on Apple" → executive_briefing (10 sections)

Before this module, all three got the same curator template. The
"strategic briefing" request got the same length as the "where"
question, and vice versa — over- and under-served simultaneously.

DESIGN:
  - Pure regex; no LLM call, no extra latency.
  - Returns ONE depth label + the matched trigger (for debug visibility).
  - Defaults to `operational_detail` when no triggers match — middle
    of the road, won't over- or under-serve.
  - The curation prompt gets a depth-specific block appended via
    intent_router.depth_note_for_curation().

Triggers were taken verbatim from Joudat's spec items 17-22 + the
IPA executive intelligence requirements.
"""

from __future__ import annotations

import re


DEPTH_LABELS = (
    "simple_fact",
    "operational_detail",
    "executive_briefing",
    "strategic_recommendation",
)


# Ordered: most specific first. The first regex that matches wins so
# that "strategic briefing" hits executive_briefing before it could
# trip strategic_recommendation.
_DEPTH_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("executive_briefing", re.compile(
        r"\b("
        r"executive\s+(briefing|profile|summary)|"
        r"investment\s+case|"
        r"strategic\s+read|"
        r"deep\s+(analysis|dive)|"
        r"full\s+profile|"
        r"comprehensive\s+(profile|briefing|overview)|"
        r"briefing\b|"
        r"profile\b"
        r")", re.I,
    )),
    ("strategic_recommendation", re.compile(
        r"\b("
        r"how\s+(should|do|can)\s+(we|misa|i)|"
        r"engage(?:ment)?\s+(plan|strategy)|"
        r"suggest\s+(an?\s+)?(engagement|plan|strategy|outreach)|"
        r"recommend(ation)?|"
        r"next\s+(action|step|best\s+action)|"
        r"action\s+plan|"
        r"outreach\s+plan|"
        r"talking\s+points|"
        r"approach\s+(this|the)"
        r")", re.I,
    )),
    # "how many" / "how much" are SIMPLE facts (count answers).
    # Match them BEFORE the operational_detail "how" trigger.
    ("simple_fact_count", re.compile(
        r"\bhow\s+(many|much)\b", re.I,
    )),
    ("operational_detail", re.compile(
        r"\b("
        r"market(s)?\b|"
        r"operations?\b|"
        r"business\s+(line|model|unit|segments?)|"
        r"presence\b|"
        r"footprint\b|"
        r"strategy\b|"
        r"opportunit(y|ies)\b|"
        r"expansion\b|"
        r"compare\b|"
        r"analyze\b|"
        r"why\b|"
        r"how\s+(does|did|will|do|is|are)\b|"  # specific "how" + verb forms
        r"give\s+me\s+details?|"
        r"detail(ed|s)?\b|"
        r"breakdown\b|"
        r"competitors?\b|"
        r"customers?\b|"
        r"sectors?\b|"
        r"revenue\s+by|"
        r"geograph(y|ic)\b"
        r")", re.I,
    )),
    ("simple_fact", re.compile(
        r"^\s*("
        r"where\b|"
        r"when\b|"
        # Short attribute / titled-role asks only — NOT full person bios.
        # "who is Elon Musk" is operational_detail (see _PERSON_BIO_RE).
        r"(?:who|whi|woo|hwo)\s+(?:is|was)\s+the\s+"
        r"(?:ceo|cfo|coo|chairman|chairperson|president|founder|minister|"
        r"head|director|chief)\b|"
        r"(?:who|whi)\s+(?:chairs|runs|leads)\b|"
        r"is\s+(it|he|she|they|this|that|\w+)\s+(in|the|a|an)|"
        r"does\s+\w+\s+(have|own|operate|hold|run)|"
        r"how\s+many\b|"
        r"how\s+much\b|"
        r"what\s+(is|are)\s+(the|its)\s+"
        r"(revenue|turnover|sales|hq|headquarters|head\s+office|"
        r"ceo|cfo|coo|chairman|chairperson|president|founder|"
        r"parent|owner|sector|industry|status|city|country|"
        r"location|headcount|employees?|employee\s+count|size|"
        r"address|website|url|year\s+founded|founding\s+year)\b"
        r")", re.I,
    )),
]

# Full person biographies — richer briefing, same shape for typos.
_PERSON_BIO_RE = re.compile(
    r"(?ix)^\s*"
    r"(?:"
    r"(?:who|whi|woo|hwo)\s+(?:is|was)\s+(?!the\s+(?:ceo|cfo|coo|chairman|"
    r"chairperson|president|founder|minister|head|director|chief)\b)"
    r"|"
    r"tell\s+me\s+about\s+"
    r"|"
    r"what\s+(?:do\s+you\s+know|can\s+you\s+tell\s+me)\s+about\s+"
    r")"
)


def detect_depth(user_question: str) -> tuple[str, str | None]:
    """Return (depth_label, matched_trigger_text). When no pattern
    matches, default to `operational_detail` (sensible middle ground).
    The matched trigger is None for the default.
    `simple_fact_count` is an internal alias for `simple_fact` matched
    by the "how many" pattern — normalised before returning.
    """
    if not user_question:
        return "operational_detail", None
    q = user_question.strip()
    # Person bios first so "who is Elon Musk" / "whi is …" get the
    # fuller briefing users prefer (Role + Background + Strategic Read).
    if _PERSON_BIO_RE.search(q):
        return "operational_detail", "person biography"
    for label, pattern in _DEPTH_PATTERNS:
        m = pattern.search(user_question)
        if m:
            label = "simple_fact" if label == "simple_fact_count" else label
            if label == "simple_fact" and user_question.count("?") >= 2:
                return "operational_detail", "multi-part question"
            return label, m.group(0)
    return "operational_detail", None


# ─── Depth note for curation prompt ──────────────────────────────────
# Same shape as intent_router.intent_note_for_curation — injected
# into the user-side of the curation prompt so the model adjusts its
# breadth/length to match the detected depth.

_DEPTH_NOTE = {
    "simple_fact": (
        "DEPTH: simple_fact — keep it SHORT (1-3 lines for attribute\n"
        "facts like where / how many / who is the CEO of X).\n"
        "Skip multi-section briefing. Strategic Read NOT required.\n"
    ),
    "operational_detail": (
        "DEPTH: operational_detail — richer detail than a one-liner.\n"
        "For PERSON bios (who is / tell me about <Person>): use the full\n"
        "canonical trio — ## Role, ## Background, AND ## 🇸🇦 Strategic Read\n"
        "(2-4 concrete MISA angles). Name companies (never 'Multiple').\n"
        "Same shape for typos as for clean spelling.\n"
        "For companies/topics: include the intent's standard sections and\n"
        "weave operational sub-sections when relevant (~5-10 bullets).\n"
        "MANDATORY: close with '🇸🇦 Strategic Read' (2-4 bullets) on\n"
        "investment-attraction implications — never omit it.\n"
    ),
    "executive_briefing": (
        "DEPTH: executive_briefing — the user wants a FULL ministerial\n"
        "briefing at Jul21 narrative depth (typically 800–1,500 words).\n"
        "Render all 10 sections (only skip a section when\n"
        "the data is genuinely absent — say so per the missing-data\n"
        "transparency rule):\n"
        "  1. Executive Summary — 2-3 sentence BLUF\n"
        "  2. Company Overview — sector, size, history (terse)\n"
        "  3. Saudi Presence — RHQ, licence, headcount, activities\n"
        "  4. Regional Presence — MENA revenue, hubs, operating cities\n"
        "  5. Strategic Relevance — Vision 2030 / sector priorities\n"
        "  6. Investment Activity — opportunities from `opportunities`\n"
        "     table, named MISA opportunities, recent investments\n"
        "  7. Leadership — CEO + key execs from `company_executives`\n"
        "  8. Engagement Opportunities — recommended approach, sectors\n"
        "  9. Risks / Watch Items — succession, regulatory, market\n"
        " 10. Recommended Actions — concrete next-best actions\n"
        "Every Strategic / Engagement / Recommended Actions bullet must\n"
        "name a concrete anchor (programme, giga-project, agency, or\n"
        "counterpart). Bullets dense but no padding. Bold the critical\n"
        "numbers. Never paste another country's IPA into the brief.\n"
    ),
    "strategic_recommendation": (
        "DEPTH: strategic_recommendation — the user wants ACTIONABLE\n"
        "advice, not a profile dump. Lead with the recommendation\n"
        "(one bold sentence). Then:\n"
        "  - Engagement objective (specific, one line)\n"
        "  - Saudi opportunity angle (named sector / programme)\n"
        "  - Relevant business units (from company_business_units\n"
        "    when available)\n"
        "  - Decision-makers / contacts (from misa_contact_details\n"
        "    + company_executives)\n"
        "  - Proposed talking points (3-5 bullets, concrete)\n"
        "  - Next action (one bullet, named owner / date if known)\n"
        "Skip generic profile sections. The answer is the plan.\n"
    ),
}


def depth_note_for_curation(depth: str) -> str:
    """Returns the curation-prompt directive for a given depth label.
    Empty string for unrecognised labels (no-op)."""
    return _DEPTH_NOTE.get(depth or "", "")
