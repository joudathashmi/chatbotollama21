"""Post-compose enrichment for freeform advisory deliverables.

Enforces Jul21-class depth and country accuracy after generation for
market-fit, engagement-plan, sector-priorities, strategy_analysis, and
(lightly) company_targeting — every origin market, not India-only.
"""

from __future__ import annotations

import re

from app.services.advisory_structured import (
    _default_trade_bodies,
    foreign_ipa_markers_for_scrub,
    primary_trade_body_name,
)


_TRADE_HEADING_RE = re.compile(
    r"(?im)^#{1,3}\s*Investment\s*(&|and)\s*Trade\s*Bodies"
)
_STRATEGIC_CTX_RE = re.compile(
    r"(?im)^#{1,3}\s*Strategic\s+Context\b"
)
_RECS_HEADING_RE = re.compile(
    r"(?im)^#{1,3}\s*(Strategic\s+Targeting\s+Recommendations|"
    r"Recommended\s+Next\s+(Moves|Actions)|"
    r"Recommendations\s+for\s+MISA|"
    r"Closing\s+Recommendations|"
    r"Strategic\s+Conclusion)\b"
)
_FOOTPRINT_RE = re.compile(
    r"(?im)^#{1,3}\s*Current\s+(MISA|Saudi)\s+Footprint\b"
)
_SECTION_UNTIL_NEXT_H = re.compile(
    r"(?ims)(^#{1,3}\s*Investment\s*(?:&|and)\s*Trade\s*Bodies[^\n]*\n)"
    r"(.*?)(?=^#{1,3}\s|\Z)"
)


def _country_from_ctx(db_context: dict | None) -> str:
    if not db_context:
        return ""
    return str(db_context.get("origin_country") or "").strip()


def _render_trade_bodies_md(country: str) -> str:
    bodies = _default_trade_bodies(country or "priority markets")
    lines = [
        "## Investment & Trade Bodies to Engage",
        "",
        "| Organisation | Type | Role in engagement |",
        "|---|---|---|",
    ]
    for b in bodies:
        lines.append(
            f"| {b.get('organisation')} | {b.get('type')} | {b.get('role')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _default_strategic_context(country: str, db_context: dict | None) -> str:
    """Payload-grounded corridor framing — named accounts + sectors first."""
    c = country or "the source market"
    licensed = None
    rhq = None
    named_bits: list[str] = []
    sector_bits: list[str] = []
    if db_context:
        licensed = db_context.get("companies_from_origin_licensed_in_saudi")
        rhq = db_context.get("companies_from_origin_with_rhq")
        for row in (db_context.get("expansion_targets") or [])[:4]:
            if not isinstance(row, dict):
                continue
            n = row.get("company") or row.get("name")
            sec = row.get("sector") or row.get("industry")
            if n:
                if sec:
                    named_bits.append(f"**{n}** ({sec})")
                else:
                    named_bits.append(f"**{n}**")
        if not named_bits:
            for key in ("top_rhq_companies", "top_licensed_companies"):
                for row in db_context.get(key) or []:
                    if not isinstance(row, dict):
                        continue
                    n = row.get("name") or row.get("company")
                    sec = row.get("industry") or row.get("sector")
                    if n and f"**{n}**" not in " ".join(named_bits):
                        named_bits.append(
                            f"**{n}**" + (f" ({sec})" if sec else "")
                        )
                    if len(named_bits) >= 4:
                        break
                if len(named_bits) >= 4:
                    break
        dist = db_context.get("licensed_sector_distribution") or []
        if isinstance(dist, list):
            for item in dist[:3]:
                if isinstance(item, dict):
                    sec = (
                        item.get("sector")
                        or item.get("sector_name")
                        or item.get("name")
                    )
                    if sec:
                        sector_bits.append(str(sec))
    count_bit = ""
    if licensed is not None:
        count_bit = (
            f" MISA records show **{licensed}** licensed companies from "
            f"{c} in Saudi Arabia"
            + (f", of which **{rhq}** hold RHQ status" if rhq is not None else "")
            + "."
        )
    lead_bit = ""
    if named_bits:
        lead_bit = (
            f" Priority account targets on file include "
            f"{', '.join(named_bits[:4])}."
        )
    sector_bit = ""
    if sector_bits:
        from app.services.recommendation_quality import saudi_counterpart_for_sector
        anchors = [
            f"{sec} → {saudi_counterpart_for_sector(sec)}"
            for sec in sector_bits[:3]
        ]
        sector_bit = (
            f" Lead with sector corridors already converting in the "
            f"footprint: {'; '.join(anchors)}."
        )
    else:
        sector_bit = (
            f" Map {c} champions to the demand corridors that match their "
            f"capabilities — digital (SDAIA / LEAP), industrial (NIDLP), "
            f"healthcare (NUPCO), or energy (NEOM) — not a generic pitch."
        )
    ipa = primary_trade_body_name(c) or f"the national IPA of {c}"
    return (
        f"{c} is a priority outbound-investment source for Saudi Arabia."
        f"{count_bit}{lead_bit}{sector_bit} "
        f"MISA should deepen the installed base and convert warm licence / "
        f"RHQ presence through **{ipa}** and peer industry / export-finance "
        f"bodies — with dated capability offers, not MoU theatre."
    )


def _named_actions_from_footprint(db_context: dict | None) -> list[str]:
    from app.services.recommendation_quality import saudi_counterpart_for_sector

    country = _country_from_ctx(db_context) or "the source market"
    ipa = primary_trade_body_name(country) or f"the national IPA of {country}"
    actions: list[str] = []
    if db_context:
        for row in (db_context.get("expansion_targets") or [])[:5]:
            if not isinstance(row, dict):
                continue
            name = row.get("company") or row.get("name")
            if not name:
                continue
            sector = row.get("sector") or "priority sector"
            presence = row.get("current_saudi_presence") or "licensed"
            counterpart = saudi_counterpart_for_sector(str(sector))
            if str(presence).upper() == "RHQ":
                actions.append(
                    f"Run an RHQ expansion account review with **{name}** "
                    f"({sector}) within 90 days — table a written capability "
                    f"offer mapped to **{counterpart}**, not a generic MoU."
                )
            else:
                actions.append(
                    f"Qualify **{name}** ({sector}) for licence deepening or "
                    f"RHQ conversion — schedule a MISA-led incentive and site "
                    f"walkthrough with **{counterpart}** within 90 days."
                )
    if not actions and db_context:
        dist = db_context.get("licensed_sector_distribution") or []
        if isinstance(dist, list):
            for item in dist[:3]:
                if not isinstance(item, dict):
                    continue
                sec = (
                    item.get("sector")
                    or item.get("sector_name")
                    or item.get("name")
                )
                if not sec:
                    continue
                counterpart = saudi_counterpart_for_sector(str(sec))
                n = item.get("licensed") or item.get("count") or item.get("companies")
                count_bit = f" ({n} licensed on file)" if n is not None else ""
                actions.append(
                    f"Stand up a **{sec}** desk sprint{count_bit} within "
                    f"90 days — pair top accounts with **{counterpart}** and "
                    f"a LEAP / FII calendar slot."
                )
    if not actions:
        actions = [
            f"Brief **{ipa}** within 90 days on a Tier-1 sector corridor "
            f"offer tied to **SDAIA / LEAP** or **NIDLP** — with three named "
            f"account targets and dated follow-ups.",
            f"Publish a one-pager of {country} licensed / RHQ footprint for "
            f"desk targeting ahead of **LEAP / FII**, naming the top five "
            f"accounts and their Saudi counterparts.",
            f"Align a joint mission calendar with **{ipa}** (LEAP / FII / "
            f"sector exhibitions) and assign owners for each Tier-1 account "
            f"within 90 days.",
        ]
    else:
        actions.append(
            f"Align the top-account roadmap with **{ipa}** for a joint "
            f"mission calendar (LEAP / FII / sector exhibitions) within "
            f"90 days — each account owns a named Saudi counterpart."
        )
    return actions


def _inject_before_footer(answer: str, block: str) -> str:
    """Insert block before the italic source footer if present."""
    text = (answer or "").rstrip()
    m = re.search(
        r"(?im)^\*Strategic analysis synthesised[^\n]*\*\s*$",
        text,
    )
    if m:
        return (
            text[: m.start()].rstrip()
            + "\n\n"
            + block.rstrip()
            + "\n\n"
            + text[m.start():]
        )
    return text + "\n\n" + block.rstrip() + "\n"


def _trade_section_contaminated(section: str, country: str) -> bool:
    blob = (section or "").casefold()
    markers = foreign_ipa_markers_for_scrub(country)
    hits = [m for m in markers if len(m) >= 6 and m in blob]
    return len(hits) >= 1


def _ensure_trade_bodies(answer: str, country: str) -> tuple[str, list[str]]:
    fixes: list[str] = []
    text = answer or ""
    canonical = _render_trade_bodies_md(country)
    primary = primary_trade_body_name(country)
    m = _SECTION_UNTIL_NEXT_H.search(text)
    if m:
        section = m.group(0)
        needs_replace = _trade_section_contaminated(section, country)
        if primary and primary.casefold() not in section.casefold():
            # Missing lead IPA — replace rather than append a stray row
            # into a possibly wrong-country table.
            needs_replace = True
        # Legacy thin placeholder (not "{Country} national investment…").
        if re.search(
            r"(?i)\|\s*National investment promotion agency\s*\|",
            section,
        ) or "requires validation for this origin" in section.casefold():
            needs_replace = True
        if needs_replace:
            text = _SECTION_UNTIL_NEXT_H.sub(
                canonical.rstrip() + "\n\n", text, count=1,
            )
            fixes.append("replaced_trade_bodies_section")
            return text, fixes
        return text, fixes
    return _inject_before_footer(text, canonical), ["injected_trade_bodies_section"]


def _ensure_strategic_context(
    answer: str, country: str, db_context: dict | None,
) -> tuple[str, list[str]]:
    if _STRATEGIC_CTX_RE.search(answer or ""):
        return answer, []
    body = _default_strategic_context(country, db_context)
    block = f"## Strategic Context\n\n{body}\n"
    text = answer or ""
    parts = text.split("\n", 1)
    if parts and parts[0].startswith("#"):
        rest = parts[1] if len(parts) > 1 else ""
        return parts[0] + "\n\n" + block + "\n" + rest.lstrip("\n"), [
            "injected_strategic_context"
        ]
    return block + "\n" + text, ["injected_strategic_context"]


def _ensure_footprint(
    answer: str, country: str, db_context: dict | None,
) -> tuple[str, list[str]]:
    if not db_context:
        return answer, []
    if _FOOTPRINT_RE.search(answer or ""):
        return answer, []
    licensed = db_context.get("companies_from_origin_licensed_in_saudi")
    rhq = db_context.get("companies_from_origin_with_rhq")
    if licensed is None and rhq is None:
        return answer, []
    c = country or "this origin"
    leads: list[str] = []
    for key in ("top_rhq_companies", "top_licensed_companies",
                "expansion_targets"):
        for row in db_context.get(key) or []:
            if isinstance(row, dict):
                n = row.get("name") or row.get("company")
                if n and str(n) not in leads:
                    leads.append(str(n))
            if len(leads) >= 5:
                break
        if len(leads) >= 5:
            break
    body = (
        f"According to MISA's database, **{licensed}** companies from "
        f"{c} are licensed in Saudi Arabia"
        + (f", of which **{rhq}** hold RHQ status" if rhq is not None else "")
        + "."
    )
    if leads:
        body += f" Leading footprint anchors: {', '.join(leads)}."
    block = f"## Current MISA Footprint\n\n{body}\n"
    text = answer or ""
    # Place after Strategic Context if present, else after H1.
    m = _STRATEGIC_CTX_RE.search(text)
    if m:
        # Find end of strategic context section
        after = text[m.start():]
        nxt = re.search(r"(?m)^#{1,3}\s", after[1:])
        if nxt:
            insert_at = m.start() + 1 + nxt.start()
            return (
                text[:insert_at] + block + "\n" + text[insert_at:],
                ["injected_footprint_section"],
            )
    parts = text.split("\n", 1)
    if parts and parts[0].startswith("#"):
        rest = parts[1] if len(parts) > 1 else ""
        return parts[0] + "\n\n" + block + "\n" + rest.lstrip("\n"), [
            "injected_footprint_section"
        ]
    return block + "\n" + text, ["injected_footprint_section"]


def _scrub_foreign_ipa_prose(answer: str, country: str) -> tuple[str, list[str]]:
    """Remove table rows / obvious bleed lines naming another country's IPA."""
    markers = [m for m in foreign_ipa_markers_for_scrub(country) if len(m) >= 6]
    if not markers or not answer:
        return answer, []
    lines = answer.splitlines()
    out: list[str] = []
    removed = 0
    for ln in lines:
        low = ln.casefold()
        # Only scrub markdown table rows and short bullets that are
        # clearly foreign-IPA lines — avoid gutting narrative that
        # mentions a peer market comparatively.
        is_row = ln.strip().startswith("|")
        is_bullet = ln.strip().startswith(("-", "*"))
        if (is_row or is_bullet) and any(m in low for m in markers):
            removed += 1
            continue
        out.append(ln)
    if removed:
        return "\n".join(out), [f"scrubbed_foreign_ipa_lines:{removed}"]
    return answer, []


def _ensure_actionable_recommendations(
    answer: str, db_context: dict | None,
) -> tuple[str, list[str]]:
    from app.services.recommendation_quality import (
        is_generic_recommendation,
        scrub_recommendation_section,
    )

    fixes: list[str] = []
    text = answer or ""
    actions = _named_actions_from_footprint(db_context)

    # First pass: drop soft bullets already in the answer.
    text, scrub_fixes = scrub_recommendation_section(
        text, replacement_actions=actions,
    )
    fixes.extend(scrub_fixes)

    has_recs = bool(_RECS_HEADING_RE.search(text))
    rec_idx = text.lower().find("recommend")
    rec_tail = text[rec_idx:] if rec_idx >= 0 else ""
    named_company = bool(
        re.search(
            r"(?i)\*\*[A-Za-z][^*]{2,40}\*\*|NEOM|SDAIA|NUPCO|GTAI|JETRO|KOTRA|"
            r"SelectUSA|Invest |GIPC|BOI|ApexBrasil",
            rec_tail,
        )
    )
    soft = is_generic_recommendation(rec_tail[:400]) if rec_tail else True
    # Also detect soft phrases anywhere in the rec section.
    soft = soft or bool(
        re.search(
            r"(?i)\b(engage stakeholders|leverage networks|"
            r"develop a framework|identify and map|showcase opportunities|"
            r"strengthen bilateral|focus on ict|explore opportunities)\b",
            rec_tail,
        )
    )
    if has_recs and named_company and not soft:
        return text, fixes

    block_lines = [
        "## Strategic Targeting Recommendations for MISA",
        "",
    ] + [f"- {a}" for a in actions] + [""]
    block = "\n".join(block_lines)
    if has_recs:
        pattern = re.compile(
            r"(?is)(^#{1,3}\s*(?:Strategic\s+Targeting\s+Recommendations|"
            r"Recommended\s+Next\s+(?:Moves|Actions)|"
            r"Recommendations\s+for\s+MISA|"
            r"Closing\s+Recommendations(?:\s+to\s+MISA)?)\b[^\n]*\n)"
            r"(.*?)(?=^#{1,3}\s|\Z)",
            re.M,
        )

        def _repl(m: re.Match) -> str:
            return m.group(1) + "\n".join(f"- {a}" for a in actions) + "\n\n"

        text2, n = pattern.subn(_repl, text, count=1)
        if n:
            return text2, fixes + ["rebuilt_actionable_recommendations"]
    return _inject_before_footer(text, block), fixes + [
        "injected_actionable_recommendations"
    ]


_PRIORITY_TABLE_RE = re.compile(
    r"(?im)^\|.+\bPriority\b.+\|\s*$"
)


def _ensure_priority_ranking_table(
    answer: str, db_context: dict | None,
) -> tuple[str, list[str]]:
    """If a freeform strategy brief has no Priority table, inject one."""
    if _PRIORITY_TABLE_RE.search(answer or ""):
        return answer, []
    dist = []
    if db_context:
        dist = db_context.get("licensed_sector_distribution") or []
    rows: list[tuple[str, str]] = []
    if isinstance(dist, list):
        for item in dist[:8]:
            if isinstance(item, dict):
                sec = item.get("sector") or item.get("sector_name") or item.get("name")
                n = item.get("licensed") or item.get("count") or item.get("companies")
                if sec:
                    rows.append((str(sec), str(n) if n is not None else "—"))
    if not rows:
        # Generic Vision 2030 tiers so the brief never ships table-less.
        rows = [
            ("Information Technology", "Tier 1"),
            ("Healthcare & Life Sciences", "Tier 1"),
            ("Industrial / Manufacturing", "Tier 1"),
            ("Energy & Water", "Tier 2"),
            ("Financial Services", "Tier 2"),
            ("Tourism & Entertainment", "Tier 2"),
        ]
        block_lines = [
            "## Priority Sectors",
            "",
            "| Sector | Priority | Saudi demand anchor |",
            "|---|---|---|",
        ]
        for sec, pri in rows:
            anchor = "Vision 2030 programme"
            s = sec.casefold()
            if "tech" in s or "information" in s:
                anchor = "SDAIA / LEAP"
            elif "health" in s:
                anchor = "NUPCO"
            elif "energy" in s:
                anchor = "NEOM / energy transition"
            elif "industrial" in s or "manufactur" in s:
                anchor = "NIDLP / PIF zones"
            elif "financ" in s:
                anchor = "Financial sector development"
            elif "tourism" in s:
                anchor = "Tourism / entertainment vision"
            block_lines.append(f"| {sec} | {pri} | {anchor} |")
        block_lines.append("")
        return _inject_before_footer(answer, "\n".join(block_lines)), [
            "injected_priority_sectors_table"
        ]

    block_lines = [
        "## Priority Sectors",
        "",
        "| Sector | MISA licensed | Priority | Saudi demand anchor |",
        "|---|---|---|---|",
    ]
    for i, (sec, n) in enumerate(rows):
        pri = "Tier 1" if i < 3 else ("Tier 2" if i < 6 else "Tier 3")
        block_lines.append(
            f"| {sec} | {n} | {pri} | Vision 2030 programme |"
        )
    block_lines.append("")
    return _inject_before_footer(answer, "\n".join(block_lines)), [
        "injected_priority_sectors_from_db"
    ]


def _slim_wide_advisory_tables(answer: str) -> tuple[str, list[str]]:
    """Keep sector/fit tables at ≤5 columns so PDF stays tabular not cards."""
    fixes: list[str] = []
    lines = (answer or "").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if (
            ln.strip().startswith("|")
            and i + 1 < len(lines)
            and set(lines[i + 1].replace("|", "").strip()) <= set("-: ")
        ):
            header = ln
            cols = [c.strip() for c in header.strip("|").split("|")]
            if len(cols) > 5:
                keep_idx = list(range(min(4, len(cols))))
                if len(cols) > 4 and (len(cols) - 1) not in keep_idx:
                    keep_idx.append(len(cols) - 1)
                keep_idx = sorted(set(keep_idx))[:5]

                def _project(row: str, idxs: list[int] = keep_idx) -> str:
                    cells = [c.strip() for c in row.strip().strip("|").split("|")]
                    picked = [cells[j] if j < len(cells) else "" for j in idxs]
                    return "| " + " | ".join(picked) + " |"

                out.append(_project(header))
                i += 1
                if i < len(lines) and lines[i].strip().startswith("|"):
                    out.append("|" + "|".join(["---"] * len(keep_idx)) + "|")
                    i += 1
                while i < len(lines) and lines[i].strip().startswith("|"):
                    out.append(_project(lines[i]))
                    i += 1
                fixes.append("slimmed_wide_advisory_table")
                continue
        out.append(ln)
        i += 1
    return "\n".join(out), fixes


def _ensure_phased_roadmap(
    answer: str, deliverable: str, db_context: dict | None,
) -> tuple[str, list[str]]:
    """Engagement / strategy briefs need Phase 1–3 scaffolding when missing."""
    d = (deliverable or "").strip().lower()
    if d not in ("engagement_plan", "strategy_analysis"):
        return answer, []
    if re.search(r"(?im)^#{1,3}\s*Phased\s+Roadmap\b", answer or ""):
        return answer, []
    if re.search(r"(?im)^#{2,3}\s*Phase\s*1\b", answer or ""):
        return answer, []
    country = _country_from_ctx(db_context) or "the source market"
    ipa = primary_trade_body_name(country) or f"the national IPA of {country}"
    leads: list[str] = []
    if db_context:
        for key in ("top_rhq_companies", "top_licensed_companies",
                    "expansion_targets"):
            for row in db_context.get(key) or []:
                if isinstance(row, dict):
                    n = row.get("name") or row.get("company")
                    if n and str(n) not in leads:
                        leads.append(str(n))
                if len(leads) >= 3:
                    break
            if len(leads) >= 3:
                break
    lead_bit = (
        f" Named anchors: {', '.join(leads)}."
        if leads else ""
    )
    block = (
        "## Phased Roadmap\n\n"
        f"### Phase 1 — Foundation (months 0–3)\n"
        f"- Confirm live footprint and open opportunities for {country} "
        f"accounts; brief the desk on {ipa}.{lead_bit}\n"
        f"- Agree one Tier-1 sector pilot with a dated capability offer.\n\n"
        f"### Phase 2 — Outreach & Activation (months 3–9)\n"
        f"- Run a roadshow / IPA calendar slot with named Saudi "
        f"counterparts (SDAIA / NEOM / NUPCO as sector-relevant).\n"
        f"- Convert warm licence / RHQ conversations into written "
        f"localisation commitments.\n\n"
        f"### Phase 3 — Conversion & Aftercare (months 9–18)\n"
        f"- Close at least one expansion / RHQ conversion and hand to "
        f"aftercare with KPI owners.\n"
        f"- Publish a corridor scorecard (pipeline, RHQ, localisation).\n"
    )
    return _inject_before_footer(answer, block), ["injected_phased_roadmap"]


def _ensure_kpi_governance(
    answer: str, deliverable: str,
) -> tuple[str, list[str]]:
    d = (deliverable or "").strip().lower()
    if d not in ("engagement_plan", "strategy_analysis"):
        return answer, []
    if re.search(r"(?im)^#{1,3}\s*KPIs?\s*(&|and)?\s*Governance\b", answer or ""):
        return answer, []
    block = (
        "## KPIs & Governance\n\n"
        "| KPI | Target (12 months) | Owner |\n"
        "|---|---|---|\n"
        "| Qualified account meetings | ≥ 12 Tier-1 meetings | Sector desk |\n"
        "| Written localisation / RHQ offers tabled | ≥ 4 | Account lead |\n"
        "| Pipeline conversions (licence / RHQ / expansion) | ≥ 2 | Corridor lead |\n"
        "| Aftercare NPS / open actions closed | 100% of open actions < 90 days | Relationship owner |\n"
    )
    return _inject_before_footer(answer, block), ["injected_kpi_governance"]


def _ensure_sector_deep_dive_scaffold(
    answer: str, deliverable: str, db_context: dict | None,
) -> tuple[str, list[str]]:
    """Market-fit / sector-priorities: ensure at least one numbered deep-dive."""
    d = (deliverable or "").strip().lower()
    if d not in ("market_fit", "sector_priorities"):
        return answer, []
    if re.search(r"(?im)^#\s+\d+\.\s+\S+", answer or ""):
        return answer, []
    if re.search(r"(?im)^##\s+\d+\.\s+\S+", answer or ""):
        return answer, []
    # Prefer sectors from DB distribution; else Vision defaults.
    sectors: list[str] = []
    if db_context:
        dist = db_context.get("licensed_sector_distribution") or []
        if isinstance(dist, list):
            for item in dist[:3]:
                if isinstance(item, dict):
                    sec = (
                        item.get("sector")
                        or item.get("sector_name")
                        or item.get("name")
                    )
                    if sec:
                        sectors.append(str(sec))
    if not sectors:
        sectors = [
            "Information Technology",
            "Healthcare & Life Sciences",
            "Industrial / Manufacturing",
        ]
    country = _country_from_ctx(db_context) or "the source market"
    lines = ["## Tier-1 Sector Deep-Dives", ""]
    for i, sec in enumerate(sectors, 1):
        lines += [
            f"# {i}. {sec}",
            "",
            f"**Why it matters for {country}.** Align outreach to a named "
            f"Saudi demand programme and a dated capability offer — not a "
            f"generic MoU.",
            "",
            f"**MISA play.** Desk sprint on top accounts within 90 days; "
            f"pair each with a Saudi counterpart and a LEAP / FII slot.",
            "",
        ]
    return _inject_before_footer(answer, "\n".join(lines)), [
        "injected_sector_deep_dive_scaffold"
    ]


def enrich_advisory_deliverable(
    answer: str,
    *,
    deliverable: str | None = None,
    db_context: dict | None = None,
) -> tuple[str, list[str]]:
    """Enforce Jul21-class depth and country-accurate trade bodies."""
    if not answer or not str(answer).strip():
        return answer or "", []
    d = (deliverable or "").strip().lower()
    country = _country_from_ctx(db_context)
    text = answer
    fixes: list[str] = []

    # Company targeting has its own structured renderer — only scrub bleed
    # and ensure trade bodies are country-correct.
    if d == "company_targeting":
        text, f = _scrub_foreign_ipa_prose(text, country)
        fixes.extend(f)
        text, f = _ensure_trade_bodies(text, country)
        fixes.extend(f)
        return text, fixes

    text, f = _scrub_foreign_ipa_prose(text, country)
    fixes.extend(f)

    text, f = _ensure_strategic_context(text, country, db_context)
    fixes.extend(f)

    text, f = _ensure_footprint(text, country, db_context)
    fixes.extend(f)

    text, f = _ensure_priority_ranking_table(text, db_context)
    fixes.extend(f)

    text, f = _ensure_sector_deep_dive_scaffold(text, d, db_context)
    fixes.extend(f)

    text, f = _ensure_trade_bodies(text, country)
    fixes.extend(f)

    text, f = _ensure_phased_roadmap(text, d, db_context)
    fixes.extend(f)

    text, f = _ensure_kpi_governance(text, d)
    fixes.extend(f)

    text, f = _ensure_actionable_recommendations(text, db_context)
    fixes.extend(f)

    text, f = _slim_wide_advisory_tables(text)
    fixes.extend(f)

    if "Strategic analysis synthesised" not in text:
        text = text.rstrip() + (
            "\n\n*Strategic analysis synthesised from market knowledge; "
            "MISA database figures cited where noted.*\n"
        )
        fixes.append("injected_source_footer")

    return text, fixes
