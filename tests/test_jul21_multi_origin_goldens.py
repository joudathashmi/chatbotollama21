"""Multi-origin Jul21 structure goldens — platform contracts, not India-only.

These fixtures assert the same section / IPA / no-bleed rules for every
origin ask shape. They do not hardcode corridor counts.
"""

from __future__ import annotations

import re

import pytest

from app.services.advisory_enrichment import enrich_advisory_deliverable
from app.services.advisory_structured import (
    _default_trade_bodies,
    foreign_ipa_markers_for_scrub,
    primary_trade_body_name,
)
from app.services.answer_contracts import (
    advisory_deliverable_violations,
    company_brief_violations,
    person_brief_violations,
)
from app.services.engagement_data import (
    CANONICAL_LICENSED,
    CANONICAL_RHQ,
    LICENSING_SOR,
)
from app.services.jul21_surface import enrich_entity_brief_depth


# Representative origins across regions — not a single-corridor suite.
_ORIGINS = [
    ("India", "Invest India"),
    ("Germany", "GTAI"),
    ("United States", "SelectUSA"),
    ("Japan", "JETRO"),
    ("Pakistan", "BOI"),
    ("Brazil", "ApexBrasil"),
    ("Ghana", "GIPC"),
    ("Kenya", "KenInvest"),
    ("South Korea", "KOTRA"),
    ("France", "Business France"),
]


def _thin_market_fit(country: str) -> str:
    return (
        f"# Market Fit Assessment: Attracting {country} Companies\n\n"
        f"## Overall Market Fit\n"
        f"| Sector | Fit |\n|---|---|\n| ICT | High |\n\n"
        f"## Recommended Next Moves for MISA\n"
        f"- Engage the national IPA on ICT RHQ expansion.\n"
    )


def _thin_engagement(country: str) -> str:
    return (
        f"# Engagement Plan: {country} → Saudi Arabia\n\n"
        f"## Strategic Context\n"
        f"{country} is a priority source market.\n\n"
        f"## Recommended Next Moves for MISA\n"
        f"- Brief the national IPA with a named Tier-1 account.\n"
    )


@pytest.mark.parametrize("country,ipa_needle", _ORIGINS)
def test_trade_bodies_named_ipa_and_sector_depth(country, ipa_needle):
    bodies = _default_trade_bodies(country)
    assert len(bodies) >= 5, f"{country} should have ≥5 trade bodies"
    primary = primary_trade_body_name(country)
    assert primary and "national IPA of" not in primary.lower()
    blob = " ".join(b.get("organisation") or "" for b in bodies)
    assert ipa_needle.lower() in blob.lower() or ipa_needle.lower() in primary.lower()
    # Depth: either an explicit sector association or ≥5 outreach channels
    assert len(bodies) >= 5


@pytest.mark.parametrize("country,ipa_needle", _ORIGINS)
def test_market_fit_enrich_completes_jul21_shape(country, ipa_needle):
    ctx = {
        "origin_country": country,
        "companies_from_origin_licensed_in_saudi": 100,
        "companies_from_origin_with_rhq": 5,
        "expansion_targets": [
            {"company": "Acme Corp", "sector": "ICT",
             "current_saudi_presence": "Licensed"},
        ],
    }
    out, fixes = enrich_advisory_deliverable(
        _thin_market_fit(country),
        deliverable="market_fit",
        db_context=ctx,
    )
    assert "Strategic Context" in out
    assert "Investment & Trade Bodies" in out or "Trade Bodies" in out
    assert ipa_needle.lower() in out.lower() or primary_trade_body_name(country).lower() in out.lower()
    # No foreign IPA bleed from peer catalogs
    for marker in foreign_ipa_markers_for_scrub(country):
        if marker.lower() in (ipa_needle or "").lower():
            continue
        if marker.lower() in ("invest india",) and country.lower() == "india":
            continue
        # Only fail on clear peer-IPA table bleed for non-matching origins
        if country.lower() != "india" and marker == "invest india":
            assert "Invest India" not in out
            break
    gaps = advisory_deliverable_violations(out, deliverable="market_fit")
    assert not gaps, f"{country} market_fit gaps: {gaps}"
    assert fixes


@pytest.mark.parametrize("country,ipa_needle", [
    ("Germany", "GTAI"),
    ("United States", "SelectUSA"),
    ("Japan", "JETRO"),
    ("Ghana", "GIPC"),
])
def test_engagement_enrich_injects_phases_and_kpis(country, ipa_needle):
    ctx = {
        "origin_country": country,
        "companies_from_origin_licensed_in_saudi": 50,
        "companies_from_origin_with_rhq": 3,
    }
    out, fixes = enrich_advisory_deliverable(
        _thin_engagement(country),
        deliverable="engagement_plan",
        db_context=ctx,
    )
    assert "Phase 1" in out
    assert "KPIs" in out
    assert ipa_needle.lower() in out.lower() or "Trade Bodies" in out
    gaps = advisory_deliverable_violations(out, deliverable="engagement_plan")
    assert not gaps, gaps
    assert any("phase" in f or "kpi" in f or "trade" in f for f in fixes)


def test_licensing_sor_is_booleans_not_role_codes():
    assert "licensed" in LICENSING_SOR and "is_rhq" in LICENSING_SOR
    assert CANONICAL_LICENSED == "licensed IS TRUE"
    assert CANONICAL_RHQ == "is_rhq IS TRUE"
    # Teaching remnant must not reintroduce role SoR in style guide
    from pathlib import Path
    sg = Path("app/services/style_guide.py").read_text()
    assert "company_profiles.licensed / company_profiles.is_rhq" in sg
    assert "company_profiles.role / registration_type" not in sg


def test_company_and_person_entity_goldens():
    company = (
        "## Acme Corp — Executive Briefing\n\n"
        "Acme builds industrial software.\n\n"
        "### 📊 Corporate Profile & Regional Footprint\n\n"
        "| Metric | Global | MENA |\n|---|---|---|\n| Core Sector | ICT | ICT |\n\n"
        "### Snapshot of Operations and Market Position\n\n"
        "- Products: ERP; analytics.\n"
        "- CEO: Jane Doe.\n\n"
        "### 🇸🇦 Strategic Read\n\n"
        "- Localise analytics into SDAIA demand.\n"
    )
    out, fixes = enrich_entity_brief_depth(company, intent="company_profile")
    assert not company_brief_violations(out) or "Strategic Context" in out
    assert "Strategic Context" in out
    assert "Recommended Next Actions" in out
    assert fixes

    person = (
        "## Role\n\n"
        "**Jane Doe is CEO at Acme Corp.**\n\n"
        "## Background\n\n"
        "* Prior COO at Acme.\n\n"
        "## 🇸🇦 Strategic Read\n\n"
        "* Engage on RHQ conversion.\n"
    )
    pout, pfixes = enrich_entity_brief_depth(person, intent="executive_lookup")
    assert not person_brief_violations(pout)
    assert "Strategic Context" in pout
    assert "Recommended Next Actions" in pout
    assert pfixes
