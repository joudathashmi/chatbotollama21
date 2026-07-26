"""Bulletproof advisory answer safety contract.

Failure class this module exists to prevent
--------------------------------------------
Post-compose guards (validator / quality gate / polish) used to treat
*absence of a company-targeting section* as truncation, then wipe the
whole answer and replace it with a ranking rebuild — or hard-block PDF
export into "Response withheld". That hit market-fit, engagement plans,
sector priorities, and any typo'd attraction ask.

Hard rules (enforced in code, not prompts)
------------------------------------------
1. Never full-replace an answer with company-targeting markdown unless
   the deliverable is ``company_targeting`` (or the answer is already
   unmistakably that shape AND deliverable is unknown).
2. Truncation = structural corruption of an *existing* ranking table
   (mid-row pipe cut / finish_reason=length). Missing sections alone
   are NOT truncation.
3. Hard withhold is only for unrepaired false-zero / count contradiction.
   Truncation → soft repair (trim last broken row / DB rebuild for
   targeting only) → keep long draft.
4. Footprint repair only touches ``Current (MISA|Saudi) Footprint``.
   Never ``The Evidence Base`` / market-fit / roadmap headings.
5. Collapse / window-cut polish is skipped for advisory deliverables and
   for any answer that already looks like a structured strategy doc.
"""

from __future__ import annotations

import re
from typing import Optional


# Deliverables produced by ``_detect_advisory_deliverable``.
ADVISORY_DELIVERABLES = frozenset({
    "market_fit",
    "engagement_plan",
    "sector_priorities",
    "company_targeting",
    "strategy_analysis",
})

# Never overwrite these with a company-targeting rebuild.
NON_TARGETING_DELIVERABLES = frozenset({
    "market_fit",
    "engagement_plan",
    "sector_priorities",
    "strategy_analysis",
})

_FACTUAL_CRITICALS = frozenset({
    "false_zero_on_retrieval_failure",
    "contradicts_internal_licensed_count",
    "contradicts_internal_rhq_count",
})

_TRUNCATION_CODES = frozenset({
    "truncated_ranking_table",
})

# True footprint headings only — NOT "The Evidence Base" (sector spine).
_FOOTPRINT_HEADING_RE = re.compile(
    r"^#{0,3}\s*current\s+(misa|saudi)\s+footprint\s*$",
    re.I,
)

_TARGETING_SHAPE_RE = re.compile(
    r"(?im)^##\s*Priority Company Ranking\b|"
    r"^##\s*Target Companies and Investment Thesis Matrix\b|"
    r"^##\s*Top .{0,40}RHQ Companies in Saudi Arabia\b|"
    r"^#\s*Targeting .+ Companies\b|"
    r"^#\s*.{0,80}Strategic List and Investment Thesis\b|"
    r"^#\s*.{0,80}Strategic Prioriti[sz]ation and Investment Thesis\b",
)

_ADVISORY_DOC_MARKERS_RE = re.compile(
    r"(?im)^#+\s*("
    r"Market Fit Assessment|"
    r"Overall Market Fit|"
    r"Strategic Context|"
    r"Engagement Plan|"
    r"Phased Roadmap|"
    r"Priority Target Segments|"
    r"The Evidence Base|"
    r"Sector Priorit|"
    r"Investment & Trade Bodies|"
    r"Cross-Cutting Investment Themes|"
    r"Strategic Targeting Recommendations|"
    r"Priority Company Ranking|"
    r"Detailed Investment Theses|"
    r"Target Companies and Investment Thesis Matrix|"
    r"Recommendations to MISA"
    r")\b"
)


def normalize_deliverable(deliverable: Optional[str]) -> Optional[str]:
    d = (deliverable or "").strip().lower() or None
    if d and d not in ADVISORY_DELIVERABLES:
        # Unknown labels still flow through — treat as non-targeting for
        # rebuild purposes (fail safe: don't wipe).
        return d
    return d


def is_footprint_heading(heading: Optional[str]) -> bool:
    if not heading:
        return False
    return bool(_FOOTPRINT_HEADING_RE.match(heading.strip()))


def looks_like_company_targeting(answer: str) -> bool:
    """Strong shape check — requires ranking heading or targeting title."""
    if not answer:
        return False
    return bool(_TARGETING_SHAPE_RE.search(answer))


def looks_like_advisory_document(answer: str) -> bool:
    if not answer:
        return False
    if _ADVISORY_DOC_MARKERS_RE.search(answer):
        return True
    # Dense markdown tables are almost always strategy docs here.
    return answer.count("\n|") >= 6


def may_rebuild_as_company_targeting(
    *,
    deliverable: Optional[str],
    answer: str,
) -> bool:
    """Full DB ranking rebuild is allowed ONLY for targeting docs."""
    d = normalize_deliverable(deliverable)
    if d in NON_TARGETING_DELIVERABLES:
        return False
    if d == "company_targeting":
        return True
    # deliverable unknown / None — only if answer is already targeting shape
    return looks_like_company_targeting(answer)


def may_hard_withhold(issue_codes: set[str]) -> bool:
    """Withhold stub only for unrepaired factual integrity failures."""
    return bool(issue_codes & _FACTUAL_CRITICALS)


def is_truncation_only(issue_codes: set[str]) -> bool:
    return bool(issue_codes) and issue_codes <= _TRUNCATION_CODES


def should_skip_aggressive_collapse(
    *,
    deliverable: Optional[str] = None,
    answer: str = "",
    answer_source: Optional[str] = None,
) -> bool:
    """Polish must never window-cut strategy / ranking documents."""
    d = normalize_deliverable(deliverable)
    if d in ADVISORY_DELIVERABLES:
        return True
    if (answer_source or "").strip() == "strategic_advisory":
        return True
    return looks_like_advisory_document(answer)


def ranking_midrow_truncated(answer: str) -> bool:
    """True only when a Priority Company Ranking table is cut mid-row.

    Missing ``## Detailed…`` alone is NOT truncation — theses may use
    alternate headings. Too-few rows alone is NOT truncation unless the
    last pipe row is also structurally incomplete.
    """
    if not answer:
        return False
    m = re.search(
        r"(?is)##\s*(?:Priority Company Ranking|Target Companies and Investment Thesis Matrix)\s*(.*?)(?=\n##\s|\Z)",
        answer,
    )
    if not m:
        return False
    block = m.group(1)
    rows = [
        ln for ln in block.splitlines()
        if ln.strip().startswith("|") and "---" not in ln
    ]
    if not rows:
        return True  # heading present, zero table rows → cut before table
    # Drop header row if present
    data = rows[1:] if re.search(r"(?i)rank|company", rows[0]) else rows
    if not data:
        return True
    last = data[-1].rstrip()
    # Compare against THIS table's header width (5-col executive ranking
    # is valid; do not hard-require the old 8-col layout).
    header_pipes = rows[0].count("|")
    min_pipes = header_pipes if header_pipes >= 4 else 6
    if last.count("|") < min_pipes:
        return True
    # Cut mid-cell: last non-empty line is a pipe row that doesn't end with |
    if last.startswith("|") and not last.endswith("|"):
        return True
    return False
