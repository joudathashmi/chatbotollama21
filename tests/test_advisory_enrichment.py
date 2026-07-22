"""Advisory enrichment — restore Jul21-class depth on freeform deliverables."""

from __future__ import annotations

import re

from app.services.advisory_enrichment import enrich_advisory_deliverable
from app.services.advisory_structured import (
    _default_trade_bodies,
    primary_trade_body_name,
)


_CTX = {
    "origin_country": "India",
    "companies_from_origin_licensed_in_saudi": 2437,
    "companies_from_origin_with_rhq": 14,
    "expansion_targets": [
        {"company": "Tech Mahindra", "sector": "ICT",
         "current_saudi_presence": "RHQ"},
        {"company": "TATA CONSULTANCY SERVICES", "sector": "ICT",
         "current_saudi_presence": "RHQ"},
    ],
}

_DE_CTX = {
    "origin_country": "Germany",
    "companies_from_origin_licensed_in_saudi": 400,
    "companies_from_origin_with_rhq": 12,
    "expansion_targets": [
        {"company": "ROBERT BOSCH", "sector": "Industrial",
         "current_saudi_presence": "RHQ"},
    ],
}


def test_enrich_injects_trade_bodies_and_context_for_thin_market_fit():
    thin = (
        "# Market Fit Assessment: Attracting Indian Companies\n\n"
        "## Overall Market Fit\n"
        "| Sector | Fit | Potential | Priority | Extra | Extra2 |\n"
        "|---|---|---|---|---|---|\n"
        "| ICT | High | High | Tier 1 | x | y |\n"
        "| Health | High | Med | Tier 1 | x | y |\n\n"
        "## Strategic Conclusion\n"
        "Complementarity holds.\n"
    )
    out, fixes = enrich_advisory_deliverable(
        thin, deliverable="market_fit", db_context=_CTX,
    )
    assert "Strategic Context" in out
    assert "Invest India" in out
    assert "Investment & Trade Bodies" in out
    assert "Tech Mahindra" in out or "TATA" in out
    assert "Current MISA Footprint" in out
    assert any("trade_bodies" in f or "strategic_context" in f for f in fixes)
    # Wide table slimmed to ≤5 cols
    for ln in out.splitlines():
        if ln.startswith("|") and "Sector" in ln and "---" not in ln:
            assert ln.count("|") <= 6  # ≤5 cells


def test_enrich_scrubs_india_bleed_from_german_brief():
    polluted = (
        "# Market Fit Assessment: Attracting German Companies\n\n"
        "## Strategic Context\n"
        "Germany is a priority source.\n\n"
        "## Investment & Trade Bodies to Engage\n"
        "| Organisation | Type | Role |\n"
        "|---|---|---|\n"
        "| Invest India | IPA | Wrong country bleed |\n"
        "| NASSCOM | Sector | Wrong |\n"
        "| BDI | Industry | Correct-ish |\n\n"
        "## Strategic Conclusion\n"
        "Done.\n"
    )
    out, fixes = enrich_advisory_deliverable(
        polluted, deliverable="market_fit", db_context=_DE_CTX,
    )
    assert "Invest India" not in out
    assert "NASSCOM" not in out
    assert "GTAI" in out or "Germany Trade" in out
    assert "BDI" in out or "Federation of German" in out
    assert any("trade_bodies" in f or "scrubbed" in f for f in fixes)


def test_enrich_strategy_analysis_gets_jul21_sections():
    thin = (
        "# Investment Attraction Strategy for Indian Companies\n\n"
        "## Priority Sectors\n"
        "| Sector | Priority |\n|---|---|\n| ICT | Tier 1 |\n"
    )
    out, _ = enrich_advisory_deliverable(
        thin, deliverable="strategy_analysis", db_context=_CTX,
    )
    assert "Strategic Context" in out
    assert "Invest India" in out
    assert "Current MISA Footprint" in out
    assert "2437" in out or "2,437" in out or "2437" in out.replace(",", "")


def test_enrich_company_targeting_scrubs_bleed_only():
    doc = (
        "# Targeting German Companies\n\n"
        "## Priority Company Ranking\n\n"
        "| Rank | Company |\n|---|---|\n| 1 | Bosch |\n\n"
        "## Investment and Trade Bodies\n"
        "| Organisation | Type | Role |\n"
        "|---|---|---|\n"
        "| Invest India | IPA | bleed |\n"
    )
    out, fixes = enrich_advisory_deliverable(
        doc, deliverable="company_targeting", db_context=_DE_CTX,
    )
    assert "Invest India" not in out
    assert "GTAI" in out or "Germany Trade" in out
    assert fixes  # scrub or replace happened


def test_default_trade_bodies_cover_major_origins():
    assert primary_trade_body_name("India") == "Invest India"
    assert "GTAI" in primary_trade_body_name("Germany")
    assert "SelectUSA" in primary_trade_body_name("United States")
    assert "JETRO" in primary_trade_body_name("Japan")
    assert "KOTRA" in primary_trade_body_name("South Korea")
    assert "Business France" in primary_trade_body_name("France")
    # Unknown origin still gets a structured fallback, not a blank.
    fallback = _default_trade_bodies("Finland")
    assert len(fallback) >= 3
    assert "Finland" in fallback[0]["organisation"]


def test_enrich_engagement_plan_gets_invest_india():
    plan = (
        "# Engagement Plan: Attracting Investment from India\n\n"
        "## Objectives & Success Metrics\n"
        "- 50 qualified leads\n\n"
        "## Phased Roadmap\n"
        "### Phase 1 — Foundation (months 0–3)\n"
        "- Map accounts\n"
    )
    out, _fixes = enrich_advisory_deliverable(
        plan, deliverable="engagement_plan", db_context=_CTX,
    )
    assert "Invest India" in out
    assert "Strategic Context" in out


def test_sector_scoped_no_rhq_does_not_withhold():
    """'No RHQ in textiles yet' must not wipe a sector priorities brief."""
    from app.services.quality_gate import run_quality_gate
    doc = (
        "# Sector Priorities: Attracting Indian Investment\n\n"
        + ("Strategic framing. " * 40)
        + "\n\n## The Evidence Base\n"
        "| Sector | Licensed | RHQ |\n|---|---|---|\n"
        "| ICT | 500 | 8 |\n| Textiles | 20 | 0 |\n\n"
        "## Sector Ranking\n"
        "Textiles has **no RHQ in this sector** yet, but remains a "
        "build bet under NIDLP localisation.\n\n"
        "## Investment & Trade Bodies to Engage\n"
        "| Organisation | Type | Role |\n|---|---|---|\n"
        "| Invest India | IPA | Pipeline |\n\n"
        "## Recommended Next Moves for MISA\n"
        "- Brief Invest India on ICT RHQ expansion with Tech Mahindra.\n"
    )
    text, issues, fixes = run_quality_gate(
        doc,
        question="which sectors should we prioritise for India",
        db_context={
            "origin_country": "India",
            "companies_from_origin_licensed_in_saudi": 2437,
            "companies_from_origin_with_rhq": 14,
        },
        hard_block=True,
        deliverable="sector_priorities",
    )
    assert "Response withheld" not in text
    assert "Sector Priorities" in text
    assert not any(
        i["code"] == "contradicts_internal_rhq_count" for i in issues
    ) or "kept" in " ".join(fixes) or "scrubbed" in " ".join(fixes)


def test_enrich_injects_phases_kpis_for_thin_engagement_plan():
    thin = (
        "# Engagement Plan: India → Saudi Arabia\n\n"
        "## Strategic Context\n"
        "India is a priority source market.\n\n"
        "## Recommended Next Moves for MISA\n"
        "- Brief Invest India with Tech Mahindra on ICT RHQ.\n"
    )
    out, fixes = enrich_advisory_deliverable(
        thin, deliverable="engagement_plan", db_context=_CTX,
    )
    assert "Phased Roadmap" in out
    assert "Phase 1" in out and "Phase 2" in out and "Phase 3" in out
    assert "KPIs & Governance" in out
    assert any("phased_roadmap" in f or "kpi" in f for f in fixes)


def test_enrich_injects_deep_dives_for_thin_market_fit():
    thin = (
        "# Market Fit Assessment: Germany\n\n"
        "## Strategic Context\n"
        "Germany is a priority source market.\n\n"
        "## Priority Sectors\n"
        "| Sector | Priority |\n|---|---|\n| ICT | Tier 1 |\n\n"
        "## Recommended Next Moves for MISA\n"
        "- Brief GTAI on industrial localisation.\n"
    )
    out, fixes = enrich_advisory_deliverable(
        thin, deliverable="market_fit", db_context=_DE_CTX,
    )
    assert "Deep-Dive" in out or re.search(r"(?m)^#\s+1\.\s+", out)
    assert any("deep_dive" in f for f in fixes)
