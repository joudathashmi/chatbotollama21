"""Cross-path Jul21 surface polish.

Advisory deliverables already run ``enrich_advisory_deliverable``. Every
other answer (company briefs, country licensing, sector ranks, GK,
engagement templates) still needs the same accuracy bar: no foreign-IPA
bleed, PDF-safe tables, origin-aware framing, and Jul21 section depth
even when Azure curation undershoots the deterministic template.
"""

from __future__ import annotations

import re
from typing import Any


_INVESTMENT_INTEL_RE = re.compile(
    r"(?i)\b("
    r"invest(?:ment|or|ors)?|fdi|attract|outbound|inbound|"
    r"opportunit(?:y|ies)|priorit(?:y|ies|ise|ize)|strateg(?:y|ic)|"
    r"corridor|market\s+entry|soft.?landing|trade\s+body|"
    r"engagement\s+plan|market\s+fit|sector|rhq|licen[cs]"
    r")\b"
)

_NAMED_COMPANY_ENGAGE_RE = re.compile(
    r"(?i)\b(how\s+should\s+(?:misa|we)\s+engage|engagement\s+plan\s+for|"
    r"engage\s+)\s+[A-Z]"
)

_COMPANY_BRIEF_RE = re.compile(
    r"(?im)^#{1,3}\s+.+\s+—\s+Executive Briefing\b|^#{1,3}\s*.*Corporate Profile"
)
_PERSON_BRIEF_RE = re.compile(r"(?im)^##\s+Role\b")
_ENGAGEMENT_BRIEF_RE = re.compile(r"(?im)^##\s+Engagement Recommendation\b")
_SECTOR_BRIEF_RE = re.compile(
    r"(?im)"
    r"^#{1,3}\s+.+\s+—\s+Sector\s+Brief\b|"
    r"^#{1,3}\s+Sector\s+(?:Opportunity|Briefing|Overview)\b|"
    r"^#{1,3}\s+.+\s+Sector\s+(?:in\s+)?(?:Saudi|KSA)\b|"
    r"^#\s+Sector\s+Opportunity"
)
_STRATEGIC_CTX_RE = re.compile(r"(?im)^#{1,3}\s+Strategic Context\b")
_RECS_RE = re.compile(
    r"(?im)^#{1,3}\s+("
    r"Recommended Next Actions(?:\s+for\s+MISA)?"
    r"|Recommended Next Moves(?:\s+for\s+MISA)?"
    r"|Strategic Targeting Recommendations"
    r")"
)
_STRATEGIC_READ_RE = re.compile(r"(?im)^#{1,3}\s+.*Strategic Read\b")


def looks_like_corridor_investment_ask(question: str) -> bool:
    """True for origin-market investment intel that should not be thin GK."""
    q = (question or "").strip()
    if not q or len(q) < 12:
        return False
    if not _INVESTMENT_INTEL_RE.search(q):
        return False
    if _NAMED_COMPANY_ENGAGE_RE.search(q):
        return False
    try:
        from app.services.chat_engine import _detect_origin_country
        return bool(_detect_origin_country(q))
    except Exception:
        return False


def _entity_name_from_brief(answer: str) -> str:
    m = re.search(
        r"(?im)^##\s+(.+?)\s+—\s+Executive Briefing\b",
        answer or "",
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"(?im)^#{1,3}\s+(.+?)\s+—\s+Sector\s+Brief\b",
        answer or "",
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"(?im)^\*\*(.+?)\s+is\s+", answer or "")
    if m:
        return m.group(1).strip()
    return "this account"


def _inject_after_h1_or_lead(answer: str, block: str) -> str:
    text = answer or ""
    m = re.search(r"(?m)^#{1,2}\s+[^\n]+\n+", text)
    if m:
        return text[: m.end()] + block.rstrip() + "\n\n" + text[m.end():]
    return block.rstrip() + "\n\n" + text


def _inject_before_footer(answer: str, block: str) -> str:
    text = (answer or "").rstrip()
    m = re.search(r"(?im)^_Sources:|^Sources:|^\*Strategic analysis", text)
    if m:
        return (
            text[: m.start()].rstrip()
            + "\n\n"
            + block.rstrip()
            + "\n\n"
            + text[m.start():]
        )
    return text + "\n\n" + block.rstrip() + "\n"


def _demand_anchors_for_text(answer: str) -> str:
    """Pick Vision anchors from sector cues — never universal paste when unknown."""
    blob = (answer or "").casefold()
    anchors: list[str] = []
    if any(k in blob for k in ("health", "pharma", "hospital", "life science", "biotech")):
        anchors.append("healthcare localisation (NUPCO)")
    if any(k in blob for k in (
        "ict", "software", "cloud", "digital", "telecom", " technol",
        "ai ", "semiconductor", "gaming", "esports",
    )):
        anchors.append("digital economy (SDAIA, LEAP)")
    if any(k in blob for k in (
        "energy", "oil", "gas", "power", "renew", "water", "petro",
        "industrial", "manufactur", "mining", "logistic", "construct",
    )):
        anchors.append("industrial / energy programmes (NIDLP, NEOM)")
    if any(k in blob for k in ("tourism", "hospitality", "entertain", "real estate")):
        anchors.append("giga-projects and tourism (NEOM, Red Sea Global, Qiddiya)")
    if not anchors:
        # Weak sector cue — do NOT paste NEOM/SDAIA/NUPCO/NIDLP together.
        return (
            "a sector-qualified Vision 2030 demand corridor "
            "(confirm ICT / health / industrial / energy fit first)"
        )
    return ", ".join(anchors[:2])


def enrich_entity_brief_depth(
    answer: str,
    *,
    intent: str | None = None,
    db_context: dict[str, Any] | None = None,
    user_question: str = "",
) -> tuple[str, list[str]]:
    """Bring curated/template company+person+engagement+sector briefs to Jul21 depth."""
    if not answer:
        return answer or "", []
    fixes: list[str] = []
    text = answer
    intent_l = (intent or "").strip().lower()
    q = (user_question or "").strip()
    db_context = db_context if isinstance(db_context, dict) else {}
    is_company = bool(_COMPANY_BRIEF_RE.search(text))
    is_person = bool(_PERSON_BRIEF_RE.search(text)) or intent_l in {
        "executive_lookup", "person_lookup", "executive_succession",
    }
    is_engagement = bool(_ENGAGEMENT_BRIEF_RE.search(text)) or intent_l in {
        "engagement_strategy",
    }
    is_sector = bool(_SECTOR_BRIEF_RE.search(text)) or intent_l in {
        "sector_lookup",
    } or bool(db_context.get("_single_sector_opp")) or bool(
        db_context.get("_single_sector_opp_mode")
    )

    looks_person_q = False
    looks_company_q = False
    try:
        from app.services.db_briefing import _question_looks_like_person
        looks_person_q = bool(q) and _question_looks_like_person(q)
    except Exception:
        looks_person_q = False
    if q and re.search(
        r"(?i)\b(company\s+(?:profile|briefing)|briefing\s+on|profile\s+of|"
        r"tell\s+me\s+about|brief\s+me\s+on)\b",
        q,
    ) and not looks_person_q:
        looks_company_q = True

    if looks_person_q and not is_company:
        is_person = True
    if looks_company_q:
        is_person = False
        # Don't promote raw multi-hit listings into company briefs.
        if re.search(
            r"(?i)Multiple possible matches|Your search matched|"
            r"Retrieval trace",
            text,
        ):
            is_company = False
        else:
            is_company = True

    # Person briefs sometimes omit a clean ## Role header after hybrid
    # layering — still treat Strategic Read + Background as person ONLY
    # when the ask is person-shaped (never promote company briefs).
    if (
        not is_person and not is_company and not is_engagement and not is_sector
        and (looks_person_q or intent_l in {
            "executive_lookup", "person_lookup", "executive_succession",
        })
    ):
        if re.search(r"(?im)^#{1,3}\s+.*Strategic Read\b", text) and re.search(
            r"(?im)^#{1,3}\s+Background\b", text,
        ):
            is_person = True
    # Heuristic: sector-ish headlines without the exact Sector Brief marker.
    if not is_sector and not is_company and not is_person and not is_engagement:
        if re.search(
            r"(?im)^#{1,3}\s+.+\b(sector|industry)\b",
            text,
        ) and re.search(
            r"(?im)\b(Saudi|KSA|Vision\s*2030|opportunit)\b",
            text,
        ):
            is_sector = True
    if not (is_company or is_person or is_engagement or is_sector):
        return text, fixes

    name = _entity_name_from_brief(text)
    if name == "this account":
        # Prefer bold lead name: **Tim Cook is CEO...
        m = re.search(r"(?m)\*\*([A-Z][^*]{1,60}?)\*\*", text)
        if m:
            name = m.group(1).split(" is ")[0].strip()
        elif is_sector:
            m = re.search(r"(?im)^#{1,3}\s+([^\n]{3,80})", text)
            if m:
                name = re.sub(
                    r"\s*[—-].*$", "", m.group(1)
                ).strip() or name

    # Person safety net: always restore ## Role when intent/shape says person.
    if is_person and not _PERSON_BRIEF_RE.search(text):
        lead = None
        m = re.search(r"(?m)\*\*([^*]+?)\*\*[^\n]*", text)
        if m:
            lead = m.group(0).strip()
        if not lead:
            lead = (
                f"**{name}** holds a senior leadership role at this account."
            )
        role_block = f"## Role\n\n{lead}\n\n"
        text = role_block + text.lstrip()
        fixes.append("injected_person_role")

    anchors = _demand_anchors_for_text(text)
    # For a person brief the engagement subject is the COMPANY they lead,
    # never the person. Pull the employer from the Role lead
    # ("**Tim Cook is CEO at Apple Inc.**") so the strategic sections talk
    # about the organisation, not "Tim Cook is a priority account".
    employer = None
    if is_person:
        me = re.search(
            r"(?i)\b(?:is|as)\s+[\w /&-]*?\b(?:at|of|for|with)\s+"
            r"\*{0,2}([A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,4})",
            text,
        )
        if me:
            employer = re.sub(r"[.,'*]+$", "", me.group(1)).strip()
    try:
        from app.services.recommendation_quality import saudi_counterpart_for_sector
        # Prefer sector cue from brief text for a single named counterpart.
        sector_hint = None
        for cue in ("ICT", "healthcare", "industrial", "energy", "finance",
                    "tourism", "software", "pharma", "manufactur", "mining",
                    "construct", "education", "renewable", "logistics"):
            if cue.casefold() in (text or "").casefold():
                sector_hint = cue
                break
        counterpart = saudi_counterpart_for_sector(sector_hint)
    except Exception:
        counterpart = "SDAIA / LEAP"

    if not _STRATEGIC_CTX_RE.search(text):
        if is_sector:
            ctx = (
                "## Strategic Context\n\n"
                f"**{name}** is a priority Saudi demand corridor for MISA "
                f"investment attraction. Lead with sector-qualified Vision "
                f"2030 programmes — {anchors} — and a dated capability offer "
                f"mapped to **{counterpart}**, not a generic partnership "
                f"pitch. Prioritise accounts with licence / RHQ upside and "
                f"clear localisation commitments.\n"
            )
        elif is_person:
            org = employer or "the organisation they lead"
            ctx = (
                "## Strategic Context\n\n"
                f"**{name}** is the senior leadership contact for engaging "
                f"**{org}** on Saudi investment. Route engagement through "
                f"{org}'s installed Saudi / MENA footprint — {anchors} — "
                f"with a dated capability offer to **{counterpart}**, not a "
                f"generic outreach, and convert warm licence / RHQ presence "
                f"into expansion commitments.\n"
            )
        else:
            ctx = (
                "## Strategic Context\n\n"
                f"**{name}** is a priority account for Saudi investment "
                f"attraction. Lead with demand corridors that match this "
                f"company — {anchors} — and a dated capability offer to "
                f"**{counterpart}**, not a generic partnership pitch. MISA "
                f"should deepen the installed Saudi / MENA footprint and "
                f"convert warm licence / RHQ presence into expansion "
                f"commitments.\n"
            )
        if is_company:
            m = re.search(r"(?m)^###\s*.*Corporate Profile|^---\s*$", text)
            if m:
                text = text[: m.start()] + ctx + "\n" + text[m.start():]
            else:
                text = _inject_after_h1_or_lead(text, ctx)
        elif is_person:
            # Insert AFTER the Role section body — never between ## Role
            # and its lead sentence (that split was emptying Role).
            m = re.search(
                r"(?im)^#{1,3}\s+(Background|Strategic Context|"
                r".*Strategic Read|Recommended Next)\b",
                text,
            )
            if m:
                text = text[: m.start()] + ctx + "\n" + text[m.start():]
            else:
                text = _inject_after_h1_or_lead(text, ctx)
        else:
            text = _inject_after_h1_or_lead(text, ctx)
        fixes.append("injected_entity_strategic_context")

    if not _RECS_RE.search(text):
        weak_sector = "sector-qualified" in anchors
        if is_person:
            org = employer or "the organisation they lead"
            recs = (
                "## Recommended Next Actions for MISA\n\n"
                f"- Open a MISA relationship track with **{name}** as the "
                f"executive sponsor for engaging **{org}** within 90 days.\n"
                f"- Map **{org}**'s Saudi / MENA footprint to {anchors} and "
                f"table a written RHQ / localisation offer to **{counterpart}**.\n"
                f"- Secure a LEAP / FII calendar slot with **{org}** anchored "
                f"to {name}'s office and a named Saudi counterpart.\n"
            )
        elif is_sector:
            recs = (
                "## Recommended Next Actions for MISA\n\n"
                f"- Run a sector desk review for **{name}** within 90 days — "
                f"rank licensed / RHQ-ready accounts and map demand to "
                f"**{counterpart}**.\n"
                f"- Table a written localisation / RHQ incentive offer for "
                f"the top three **{name}** accounts within 90 days.\n"
                f"- Align a LEAP / FII calendar slot for **{name}** with a "
                f"named Saudi counterpart under **{counterpart}**.\n"
            )
        elif weak_sector:
            recs = (
                "## Recommended Next Actions for MISA\n\n"
                f"- Qualify the primary sector fit for **{name}** within "
                f"90 days (ICT / health / industrial / energy) before "
                f"naming a Vision 2030 counterpart.\n"
                f"- Schedule a MISA account review with **{name}** within "
                f"90 days — table a written RHQ / localisation offer once "
                f"sector fit is confirmed.\n"
                f"- Align **{name}** to the next LEAP / FII slot only after "
                f"the sector desk names a Saudi counterpart.\n"
            )
        else:
            recs = (
                "## Recommended Next Actions for MISA\n\n"
                f"- Run an account review with **{name}** within 90 days — map "
                f"demand to {anchors} and table a written capability offer for "
                f"**{counterpart}**.\n"
                f"- Qualify **{name}** for RHQ conversion / licence deepening — "
                f"schedule a MISA-led incentive walkthrough within 90 days.\n"
                f"- Align **{name}** to a LEAP / FII calendar slot with a named "
                f"Saudi counterpart under **{counterpart}**.\n"
            )
        text = _inject_before_footer(text, recs)
        fixes.append("injected_entity_recommended_actions")

    if _STRATEGIC_READ_RE.search(text):
        read_blob = text[text.lower().find("strategic read"):]
        if not re.search(
            r"(?i)\b(NEOM|SDAIA|NUPCO|LEAP|RHQ Program|NIDLP)\b",
            read_blob,
        ):
            anchor_line = (
                f"\n- Map **{name}** to a named Vision 2030 demand corridor "
                f"(NEOM, SDAIA, NUPCO, or NIDLP) before the next outreach.\n"
            )
            text = re.sub(
                r"(?im)(^#{1,3}\s+.*Strategic Read\b[^\n]*\n)",
                r"\1" + anchor_line,
                text,
                count=1,
            )
            fixes.append("injected_vision_anchor_in_strategic_read")

    return text, fixes


def apply_jul21_surface_polish(
    answer: str,
    *,
    question: str = "",
    pack: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Country-accurate scrub + table slim + entity depth for ANY path."""
    if not answer or not str(answer).strip():
        return answer or "", []
    pack = pack or {}
    fixes: list[str] = []
    text = str(answer)

    if pack.get("_answer_source") == "strategic_advisory" or pack.get(
        "_short_circuit"
    ) == "strategic_advisory":
        return text, fixes

    if pack.get("_answer_source") != "sector_aggregation":
        intent = (
            pack.get("_intent")
            or (pack.get("_query_intent") or {}).get("legacy_intent_label")
            or (pack.get("_query_intent") or {}).get("task_type")
        )
        db_ctx = (
            pack.get("_advisory_db_context")
            or pack.get("_db_context")
            or {}
        )
        if isinstance(db_ctx, dict):
            db_ctx = dict(db_ctx)
        else:
            db_ctx = {}
        if pack.get("_single_sector_opp_mode") or pack.get("_single_sector_opp"):
            db_ctx["_single_sector_opp_mode"] = True
        # sector_lookup / briefing asks: force sector enrich even if the
        # curated draft lacks a recognisable Sector Brief headline.
        if intent in ("sector_lookup",) or re.search(
            r"(?i)\bsector\s+(?:briefing|overview)\b",
            question or "",
        ):
            db_ctx.setdefault("_single_sector_opp_mode", True)
            if not intent:
                intent = "sector_lookup"
        text, f = enrich_entity_brief_depth(
            text,
            intent=intent,
            db_context=db_ctx,
            user_question=question or "",
        )
        fixes.extend(f)
        try:
            from app.services.recommendation_quality import scrub_recommendation_section
            text, sf = scrub_recommendation_section(text)
            fixes.extend(sf)
        except Exception:
            pass

    country = ""
    db_ctx = (
        pack.get("_advisory_db_context")
        or pack.get("_db_context")
        or {}
    )
    if isinstance(db_ctx, dict):
        country = str(db_ctx.get("origin_country") or "").strip()
    if not country:
        try:
            from app.services.chat_engine import _detect_origin_country
            country = _detect_origin_country(question) or ""
        except Exception:
            country = ""

    try:
        from app.services.advisory_enrichment import (
            _ensure_trade_bodies,
            _scrub_foreign_ipa_prose,
            _slim_wide_advisory_tables,
        )
        text, f = _scrub_foreign_ipa_prose(text, country)
        fixes.extend(f)
        if country and (
            re.search(
                r"(?im)^#{1,3}\s*Investment\s*(&|and)\s*Trade\s*Bodies",
                text,
            )
            or looks_like_corridor_investment_ask(question)
        ):
            text, f = _ensure_trade_bodies(text, country)
            fixes.extend(f)
        text, f = _slim_wide_advisory_tables(text)
        fixes.extend(f)
    except Exception:
        pass

    return text, fixes
