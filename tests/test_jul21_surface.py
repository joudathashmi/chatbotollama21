"""Cross-path Jul21 surface polish — country accuracy on every answer."""

from __future__ import annotations

from app.services.jul21_surface import (
    apply_jul21_surface_polish,
    looks_like_corridor_investment_ask,
)


def test_corridor_ask_detection():
    assert looks_like_corridor_investment_ask(
        "investment opportunities from Japan in Saudi Arabia"
    )
    assert looks_like_corridor_investment_ask(
        "how can Saudi attract French manufacturers"
    )
    assert not looks_like_corridor_investment_ask("what is the capital of France")
    assert not looks_like_corridor_investment_ask("tell me about Apple")


def test_surface_scrubs_india_bleed_on_german_company_brief():
    polluted = (
        "# Bosch — Executive Briefing\n\n"
        "## Strategic Read\n"
        "- Brief Invest India on Bosch localisation.\n"
        "| Organisation | Type | Role |\n"
        "|---|---|---|\n"
        "| Invest India | IPA | bleed |\n"
    )
    out, fixes = apply_jul21_surface_polish(
        polluted,
        question="how should MISA engage Bosch from Germany",
        pack={"_db_context": {"origin_country": "Germany"}},
    )
    assert "Invest India" not in out
    assert fixes


def test_enrich_entity_brief_injects_jul21_sections_on_thin_curated():
    from app.services.jul21_surface import enrich_entity_brief_depth
    thin = (
        "## Apple Inc — Executive Briefing\n\n"
        "Apple makes phones.\n\n"
        "### Corporate Profile & Regional Footprint\n\n"
        "| Metric | Global | MENA |\n|---|---|---|\n| Sector | ICT | ICT |\n\n"
        "## 🇸🇦 Strategic Read\n\n"
        "- Expand in the region.\n"
    )
    out, fixes = enrich_entity_brief_depth(thin)
    assert "Strategic Context" in out
    assert "Recommended Next Actions for MISA" in out
    assert "NEOM" in out or "SDAIA" in out
    assert fixes


def test_person_scrub_keeps_jul21_strategic_sections():
    """Router polish must not delete finalize-injected Jul21 person depth."""
    from app.services.curation import _strip_company_sections_from_person_brief

    brief = (
        "## Role\n\n"
        "**Tim Cook is CEO at Apple Inc.**\n\n"
        "## Strategic Context\n\n"
        "Priority executive contact for Vision 2030.\n\n"
        "## Background\n\n"
        "* Joined Apple in 1998.\n\n"
        "## 🇸🇦 Strategic Read\n\n"
        "* Engage on RHQ.\n\n"
        "## Recommended Next Actions for MISA\n\n"
        "* Brief within 90 days.\n"
    )
    out = _strip_company_sections_from_person_brief(brief)
    assert "## Strategic Context" in out
    assert "## Recommended Next Actions for MISA" in out
    assert "**Tim Cook is CEO at Apple Inc.**" in out
    with_dump = brief + "\n## Corporate Profile & Regional Footprint\n\n| a | b |\n"
    out2 = _strip_company_sections_from_person_brief(with_dump)
    assert "Corporate Profile" not in out2


def test_person_enrich_does_not_split_role_lead():
    from app.services.jul21_surface import enrich_entity_brief_depth

    thin = (
        "## Role\n\n"
        "**Tim Cook is CEO at Apple Inc.**\n"
        "- Current tenure as CEO.\n\n"
        "## Background\n\n"
        "* Joined Apple in 1998.\n\n"
        "## 🇸🇦 Strategic Read\n\n"
        "* Engage on RHQ.\n"
    )
    out, fixes = enrich_entity_brief_depth(thin, intent="executive_lookup")
    assert "injected_entity_strategic_context" in fixes
    # Lead must stay under Role, before Strategic Context
    role_i = out.index("## Role")
    lead_i = out.index("**Tim Cook is CEO at Apple Inc.**")
    ctx_i = out.index("## Strategic Context")
    bg_i = out.index("## Background")
    assert role_i < lead_i < ctx_i < bg_i
    assert "Recommended Next Actions for MISA" in out
