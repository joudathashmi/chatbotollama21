"""Jul21 answer-shape contracts — single place that defines "good".

These are the durable invariants that case-by-case patches kept breaking.
Import in tests (hard fail) and in compose/finalize (soft log / gate).

Architecture invariant:
  Postgres/doc rows → privacy-filtered fact cards locally
  → Azure/OpenAI narrative when MISA_NARRATIVE_CLOUD=true
  → deterministic templates ONLY as fallback
"""

from __future__ import annotations

import re
from typing import Iterable


class ContractViolation(ValueError):
    """Answer failed a Jul21 shape / accuracy contract."""


def _has_header(text: str, *needles: str) -> bool:
    low = text or ""
    return all(n in low for n in needles)


def company_brief_violations(text: str) -> list[str]:
    """Executive company briefing must match the rich Jul21 body shape."""
    t = text or ""
    v: list[str] = []
    if "Executive Briefing" not in t:
        v.append("missing 'Executive Briefing' headline")
    if "Corporate Profile" not in t and "Regional Footprint" not in t:
        v.append("missing Corporate Profile / Regional Footprint table")
    # Prefer Snapshot of Operations; Operational Detail is the template alias.
    if (
        "Snapshot of Operations" not in t
        and "Operational Detail" not in t
    ):
        v.append(
            "missing ops body "
            "('Snapshot of Operations and Market Position' or 'Operational Detail')"
        )
    if "Strategic Read" not in t:
        v.append("missing Strategic Read")
    # Invent-by-percentage MENA/Saudi dollars (the $39.1B failure mode).
    if re.search(
        r"(?i)\$\s?[\d,.]+\s*(?:B|M|bn|m)?\s*"
        r"(?:estimated\s+)?(?:MENA|Saudi|KSA)\s+revenue"
        r"(?:\s*\([^)]*%\s*of\s*global[^)]*\))?",
        t,
    ) or re.search(
        r"(?i)(?:MENA|Saudi|KSA)\s+revenue.{0,40}\(\s*\d+(\.\d+)?\s*%\s+of\s+global",
        t,
    ):
        v.append("invented MENA/Saudi revenue from global × percentage")
    if "From the MISA Record" in t or "Background (general knowledge)" in t:
        v.append("forbidden legacy provenance headers")
    return v


def person_brief_violations(text: str) -> list[str]:
    """Person / CEO briefs: named Role + Background + Jul21 depth sections."""
    t = text or ""
    v: list[str] = []
    if not re.search(r"(?m)^##+\s*Role\b", t):
        v.append("missing ## Role")
    else:
        m = re.search(
            r"(?ms)^##+\s*Role\s*\n+(.*?)(?=^##+|\Z)",
            t,
        )
        role_body = (m.group(1) if m else "").strip()
        if not role_body:
            v.append("empty ## Role body")
        elif not re.search(
            r"(\*\*[^*]{2,80}\*\*|[A-Z][a-z]+(?:\s+[A-Z][a-z'.-]+){1,3})",
            role_body,
        ):
            v.append("## Role missing named person")
    if not re.search(r"(?m)^##+\s*Background\b", t):
        v.append("missing ## Background (Role-only is a regression)")
    if not re.search(r"(?m)^##+\s*.*Strategic Read\b", t):
        v.append("missing Strategic Read")
    if "Operational Detail" in t or "Corporate Profile" in t:
        v.append("person brief leaked company dump sections")
    if "From the MISA Record" in t or "Background (general knowledge)" in t:
        v.append("forbidden legacy provenance headers")
    return v


def engagement_brief_violations(text: str) -> list[str]:
    """ai_response_6 shape: Recommendation → Snapshot → MENA → Strategic Read."""
    t = text or ""
    v: list[str] = []
    for needle in (
        "Engagement Recommendation",
        "Snapshot",
        "Saudi / MENA Position",
        "Strategic Read",
    ):
        if needle not in t:
            v.append(f"missing '{needle}'")
    # Order
    positions = []
    for needle in (
        "Engagement Recommendation",
        "Snapshot",
        "Saudi / MENA Position",
        "Strategic Read",
    ):
        i = t.find(needle)
        if i >= 0:
            positions.append((needle, i))
    for (a, ia), (b, ib) in zip(positions, positions[1:]):
        if ia > ib:
            v.append(f"section order wrong: {a!r} must precede {b!r}")
    # Must not be the company Ops dump
    if "Operational Detail" in t:
        v.append("engagement plan must not include Operational Detail")
    if "Corporate Profile" in t:
        v.append("engagement plan must not include Corporate Profile table")
    if re.search(r"(?m)^##+\s+.+\s*—\s*Executive Briefing\s*$", t):
        v.append("engagement plan must not use Executive Briefing headline")
    return v


def advisory_deliverable_violations(
    text: str,
    *,
    deliverable: str | None = None,
) -> list[str]:
    """Required Jul21 sections for corridor advisories (any origin)."""
    t = text or ""
    d = (deliverable or "").strip().lower()
    v: list[str] = []
    if not d or d == "company_targeting":
        # Targeting uses structured renderer; only require trade bodies / recs
        # when freeform prose is present.
        if d == "company_targeting":
            if not re.search(r"(?im)trade\s*bod|Invest |IPA|chamber", t):
                v.append("company_targeting missing trade-body / IPA signal")
        return v

    if not re.search(r"(?im)^#{1,3}\s+Strategic Context\b", t):
        v.append("missing Strategic Context")
    if not re.search(
        r"(?im)^#{1,3}\s+(Recommended Next|Strategic Targeting Recommendations|"
        r"Closing Recommendations|Strategic Conclusion)\b",
        t,
    ):
        v.append("missing recommendations / conclusion")

    if d in ("market_fit", "sector_priorities"):
        if not re.search(r"(?im)^#{1,3}\s+\d+\.\s+\S+", t):
            v.append("missing numbered sector deep-dives")
        if not re.search(r"(?im)Investment\s*(&|and)\s*Trade\s*Bodies", t):
            v.append("missing Investment & Trade Bodies")

    if d in ("engagement_plan", "strategy_analysis"):
        if not (
            re.search(r"(?im)^#{1,3}\s*Phased\s+Roadmap\b", t)
            or re.search(r"(?im)^#{2,3}\s*Phase\s*1\b", t)
        ):
            v.append("missing Phased Roadmap / Phase 1")
        if not re.search(r"(?im)^#{1,3}\s*KPIs?\s*(&|and)?\s*Governance\b", t):
            v.append("missing KPIs & Governance")

    return v


def assert_company_brief(text: str) -> None:
    v = company_brief_violations(text)
    if v:
        raise ContractViolation("; ".join(v))


def assert_person_brief(text: str) -> None:
    v = person_brief_violations(text)
    if v:
        raise ContractViolation("; ".join(v))


def assert_engagement_brief(text: str) -> None:
    v = engagement_brief_violations(text)
    if v:
        raise ContractViolation("; ".join(v))


def classify_brief_kind(
    text: str,
    *,
    intent: str | None = None,
    user_question: str = "",
) -> str:
    """Best-effort kind for soft gating."""
    q = (user_question or "").lower()
    intent = (intent or "").strip().lower()
    if intent == "engagement_strategy" or "engagement plan" in q or (
        "how should" in q and "engage" in q
    ):
        return "engagement"
    # Executive / person intents always gate as person — even when the
    # draft is a mis-routed company Executive Briefing without ## Role.
    if intent in ("executive_lookup", "person_lookup", "executive_succession"):
        return "person"
    if re.search(r"(?m)^##+\s*Role\b", text or ""):
        # Role header alone is not enough when the ask is clearly a company
        # briefing ("company briefing for Pfizer") — that Role is often a
        # mis-inject from a prior polish bug.
        try:
            from app.services.db_briefing import _question_looks_like_person
            person_q = _question_looks_like_person(user_question)
        except Exception:
            person_q = False
        company_q = bool(re.search(
            r"(?i)\b(company\s+(?:profile|briefing)|briefing\s+on|"
            r"profile\s+of|tell\s+me\s+about)\b",
            user_question or "",
        )) and not person_q
        if person_q or not company_q:
            return "person"
    try:
        from app.services.db_briefing import _question_looks_like_person
        if _question_looks_like_person(user_question):
            return "person"
    except Exception:
        pass
    if re.search(
        r"(?i)\b(company\s+(?:profile|briefing)|briefing\s+on|profile\s+of)\b",
        user_question or "",
    ):
        return "company"
    # "tell me about X" is company only when it is NOT a person/CEO ask.
    if re.search(r"(?i)\btell\s+me\s+about\b", user_question or ""):
        try:
            from app.services.db_briefing import _question_looks_like_person
            if not _question_looks_like_person(user_question):
                return "company"
        except Exception:
            return "company"
    if "Executive Briefing" in (text or "") or intent in (
        "company_profile", "saudi_presence", "entity_lookup",
    ):
        return "company"
    return "unknown"


def soft_check_answer(
    text: str,
    *,
    intent: str | None = None,
    user_question: str = "",
    db_context: dict | None = None,
    hard_on_false_zero: bool = True,
    deliverable: str | None = None,
) -> list[str]:
    """Return violations for logging — never raises.

    When ``hard_on_false_zero`` and db_context indicates unavailable /
    contradictory counts, critical quality_gate codes are included so
    callers can refuse to ship the curated draft.
    """
    kind = classify_brief_kind(text, intent=intent, user_question=user_question)
    viol: list[str] = []
    if kind == "company":
        viol = company_brief_violations(text)
    elif kind == "person":
        viol = person_brief_violations(text)
    elif kind == "engagement":
        viol = engagement_brief_violations(text)
    if deliverable:
        viol.extend(
            advisory_deliverable_violations(text, deliverable=deliverable)
        )
    if hard_on_false_zero and db_context is not None:
        try:
            from app.services.quality_gate import detect_quality_issues
            for issue in detect_quality_issues(
                text, db_context=db_context, question=user_question,
            ):
                if issue.get("severity") == "critical":
                    viol.append(f"quality:{issue.get('code')}")
        except Exception:
            pass
    return viol


def intent_prompt_must_contain(intent_note: str, required: Iterable[str]) -> list[str]:
    missing = [r for r in required if r not in (intent_note or "")]
    return [f"intent note missing {m!r}" for m in missing]
