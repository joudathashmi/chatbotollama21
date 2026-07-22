"""Global source-precedence and claim-verification policy.

Single source of truth for prompts + validators. Do not duplicate
conflicting hierarchy text across prompt files.
"""

from __future__ import annotations

from enum import Enum


class SourceTier(str, Enum):
    INTERNAL = "1_internal"       # MISA DB / approved APIs / internal docs
    OFFICIAL = "2_official"       # Gov / regulator / company filings
    EXTERNAL = "3_external"       # Reliable research / market
    MODEL_KNOWLEDGE = "4_model"   # Explanatory colour only


class VerificationStatus(str, Enum):
    VERIFIED_INTERNAL = "VERIFIED_INTERNAL"
    VERIFIED_OFFICIAL = "VERIFIED_OFFICIAL"
    VERIFIED_EXTERNAL = "VERIFIED_EXTERNAL"
    ANALYTICAL_INFERENCE = "ANALYTICAL_INFERENCE"
    PROPOSAL = "PROPOSAL"
    REQUIRES_VALIDATION = "REQUIRES_VALIDATION"
    UNAVAILABLE = "UNAVAILABLE"


SOURCE_HIERARCHY_PROMPT = """
SOURCE HIERARCHY (mandatory — never invert):
1. Approved internal MISA databases, APIs, structured datasets, and
   approved internal documents — system of record when present.
2. Official Saudi government / regulatory / company primary sources.
3. Reliable external research and market sources.
4. General model knowledge — explanatory context ONLY; never override
   tiers 1–3; never invent internal figures.

RETRIEVAL STATUS RULES:
- Claim zero / none / "no companies" ONLY when retrieval_status is
  SUCCESS_EMPTY (or explicit zero_records) AND the source + filters
  are named.
- If retrieval_status is an ERROR / UNAVAILABLE / TIMEOUT: say data
  could not be retrieved. NEVER say the count is zero.
- If PARTIAL_RESULT: say the result is partial; do not present as a
  complete census.
- If counts_unavailable / do_not_claim_zero / footprint_data_unavailable
  is set: omit inventing footprint numbers.
- Label proposals as recommendations, not existing arrangements.
- Mark unsupported external claims as Requires validation.
""".strip()


CLAIM_POLICY_PROMPT = """
CLAIM LABELS (use in Sources / Data Limitations when material):
- VERIFIED_INTERNAL — taken from MISA DATABASE CONTEXT / tool rows.
- VERIFIED_OFFICIAL — named government or company primary source.
- VERIFIED_EXTERNAL — named reliable external source.
- ANALYTICAL_INFERENCE — reasoned from evidence; not a raw field.
- PROPOSAL — recommended action; does not exist yet.
- REQUIRES_VALIDATION — plausible but not evidenced in context.
- UNAVAILABLE — asked for, but retrieval failed or was empty of proof.
""".strip()


def source_policy_system_addon() -> str:
    parts = [SOURCE_HIERARCHY_PROMPT, CLAIM_POLICY_PROMPT]
    try:
        from app.services.recommendation_quality import (
            RECOMMENDATION_PROMPT_ADDON,
        )
        parts.append(RECOMMENDATION_PROMPT_ADDON)
    except Exception:
        pass
    return "\n\n".join(parts)


# ─── Canonical licensing source (code enforcer) ───────────────────────
# Prompt text alone is not enough: tool-calling and browse maps used to
# hit the auxiliary `rhq_licenses` table (~661 rows) for country/totals,
# inflating or deflating vs company_profiles.licensed / is_rhq (~95k / ~727).

CANONICAL_LICENSING_TABLE = "company_profiles"
AUXILIARY_LICENSE_DETAIL_TABLE = "rhq_licenses"

_FOOTPRINT_EXPECTED_DELIVERABLES = frozenset({
    "market_fit",
    "engagement_plan",
    "company_targeting",
    "strategy_analysis",
})


def may_inject_missing_footprint(deliverable: str | None) -> bool:
    """Auto-inject Current Saudi Footprint only for shapes that expect it.

    Sector-priorities uses ``## The Evidence Base`` as its spine — do not
    force a footprint section into that deliverable.
    """
    d = (deliverable or "").strip().lower()
    if not d:
        # Unknown: allow inject (safer for grounding than inventing zeros).
        return True
    if d == "sector_priorities":
        return False
    return d in _FOOTPRINT_EXPECTED_DELIVERABLES


def rewrite_aggregate_licensing_query(
    table: str,
    *,
    count_only: bool = False,
    filters: dict | None = None,
    question: str = "",
) -> tuple[str, dict, list[str]]:
    """Rewrite ``rhq_licenses`` aggregate/count queries to company_profiles.

    Returns ``(table, filters, notes)``. Detail-row lookups against
    ``rhq_licenses`` with entity-specific filters are left alone; bare
    counts and empty/country-only aggregates are forced onto the
    canonical ``licensed`` / ``is_rhq`` flags.
    """
    import re
    notes: list[str] = []
    tbl = (table or "").strip()
    flt = dict(filters or {})
    if tbl != AUXILIARY_LICENSE_DETAIL_TABLE:
        return tbl, flt, notes

    q = (question or "").lower()
    # Entity-specific detail list (e.g. license_number / company_id) —
    # allow auxiliary table.
    detail_keys = {
        "id", "license_number", "licence_number", "company_id",
        "cr_number", "commercial_registration", "name", "company_name",
    }
    has_detail = any(k in flt for k in detail_keys)
    if has_detail and not count_only:
        return tbl, flt, notes

    # Aggregate / count / browse → canonical source
    notes.append("rewrote_rhq_licenses_aggregate_to_company_profiles")
    tbl = CANONICAL_LICENSING_TABLE
    wants_rhq = bool(re.search(r"\brhq\b|regional\s+headquarter", q))
    if wants_rhq and "is_rhq" not in flt and "licensed" not in flt:
        flt["is_rhq"] = True
    elif "licensed" not in flt and "is_rhq" not in flt:
        flt["licensed"] = True
    return tbl, flt, notes

