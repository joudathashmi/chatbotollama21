"""World-class quality battery — named, dated, counterpart-grounded output.

Platform-wide (any origin / any account). Fails CI if soft recs ship or
Strategic Context ignores payload names.
"""

from __future__ import annotations

import re

from app.services.advisory_enrichment import (
    _default_strategic_context,
    _named_actions_from_footprint,
    enrich_advisory_deliverable,
)
from app.services.answer_finalize import finalize_answer
from app.services.jul21_surface import enrich_entity_brief_depth
from app.services.recommendation_quality import (
    filter_recommendations,
    is_generic_recommendation,
    is_world_class_recommendation,
    saudi_counterpart_for_sector,
    scrub_recommendation_section,
    score_recommendation,
)


_CTX = {
    "origin_country": "Germany",
    "companies_from_origin_licensed_in_saudi": 400,
    "companies_from_origin_with_rhq": 12,
    "expansion_targets": [
        {"company": "ROBERT BOSCH", "sector": "Industrial",
         "current_saudi_presence": "RHQ"},
        {"company": "SIEMENS", "sector": "Industrial",
         "current_saudi_presence": "Licensed"},
        {"company": "SAP", "sector": "ICT",
         "current_saudi_presence": "RHQ"},
    ],
    "licensed_sector_distribution": [
        {"sector": "Industrial", "licensed": 80},
        {"sector": "ICT", "licensed": 40},
    ],
}


def test_soft_phrases_rejected():
    assert is_generic_recommendation("Engage stakeholders to unlock value")
    assert is_generic_recommendation("Explore opportunities and leverage synergies")
    kept, rejected = filter_recommendations([
        "Engage stakeholders across the ecosystem",
        "Qualify **SIEMENS** (Industrial) for RHQ conversion within 90 days "
        "with **NIDLP / PIF industrial zones**.",
    ])
    assert len(kept) == 1
    assert len(rejected) == 1
    assert "SIEMENS" in str(kept[0])


def test_world_class_rec_requires_named_dated_counterpart():
    soft = "Deepen the bilateral relationship with industry"
    hard = (
        "Run an RHQ expansion account review with **ROBERT BOSCH** "
        "(Industrial) within 90 days — table a written capability offer "
        "mapped to **NIDLP / PIF industrial zones**."
    )
    assert not is_world_class_recommendation(soft)
    assert is_world_class_recommendation(hard)
    assert score_recommendation(hard)["ok"]
    assert score_recommendation(soft)["ok"] is False


def test_counterpart_is_sector_specific():
    assert "SDAIA" in saudi_counterpart_for_sector("ICT / Software")
    assert "NUPCO" in saudi_counterpart_for_sector("Healthcare & Pharma")
    assert "NIDLP" in saudi_counterpart_for_sector("Industrial Manufacturing")
    assert "NEOM" in saudi_counterpart_for_sector("Energy & Water")


def test_strategic_context_cites_named_expansion_targets():
    ctx = _default_strategic_context("Germany", _CTX)
    assert "**ROBERT BOSCH**" in ctx or "ROBERT BOSCH" in ctx
    assert "**SIEMENS**" in ctx or "SIEMENS" in ctx
    assert "400" in ctx
    assert "GTAI" in ctx or "Germany Trade" in ctx


def test_named_actions_are_world_class():
    actions = _named_actions_from_footprint(_CTX)
    assert len(actions) >= 3
    for a in actions[:3]:
        assert is_world_class_recommendation(a) or (
            "**" in a and "within 90 days" in a.lower()
        )
    blob = " ".join(actions)
    assert "ROBERT BOSCH" in blob
    assert "SIEMENS" in blob or "SAP" in blob
    assert re.search(r"NIDLP|SDAIA|NEOM|NUPCO", blob)


def test_scrub_drops_soft_bullets_and_rebuilds():
    thin = (
        "# Engagement Plan: Germany\n\n"
        "## Recommended Next Moves for MISA\n"
        "- Engage stakeholders to unlock value across the corridor.\n"
        "- Explore opportunities and leverage synergies.\n"
    )
    actions = _named_actions_from_footprint(_CTX)
    out, fixes = scrub_recommendation_section(
        thin, replacement_actions=actions,
    )
    assert "Engage stakeholders" not in out
    assert "ROBERT BOSCH" in out or "SIEMENS" in out
    assert fixes


def test_enrich_market_fit_grounds_recs_in_payload():
    thin = (
        "# Market Fit Assessment: Germany\n\n"
        "## Overall Market Fit\n"
        "| Sector | Fit |\n|---|---|\n| Industrial | High |\n\n"
        "## Recommended Next Moves for MISA\n"
        "- Engage stakeholders and strengthen bilateral ties.\n"
    )
    out, fixes = enrich_advisory_deliverable(
        thin, deliverable="market_fit", db_context=_CTX,
    )
    assert "Engage stakeholders" not in out
    assert "ROBERT BOSCH" in out or "SIEMENS" in out
    assert "GTAI" in out
    assert "Strategic Context" in out
    assert any("actionable" in f or "rec" in f or "scrub" in f or "rebuild" in f
               for f in fixes) or "ROBERT BOSCH" in out


def test_entity_enrich_uses_named_counterpart():
    thin = (
        "## Acme Soft — Executive Briefing\n\n"
        "Acme Soft builds cloud software.\n\n"
        "### Corporate Profile & Regional Footprint\n\n"
        "| Metric | Global | MENA |\n|---|---|---|\n| Sector | ICT | ICT |\n\n"
        "### Snapshot of Operations and Market Position\n\n"
        "- Cloud ERP.\n\n"
        "### 🇸🇦 Strategic Read\n\n"
        "- Expand digital footprint.\n"
    )
    out, fixes = enrich_entity_brief_depth(thin, intent="company_profile")
    assert "Strategic Context" in out
    assert "Recommended Next Actions" in out
    assert "**Acme Soft**" in out or "Acme Soft" in out
    assert "within 90 days" in out.lower()
    assert re.search(r"SDAIA|LEAP|NIDLP|NEOM|NUPCO", out)
    assert fixes


def test_finalize_scrubs_soft_recs_with_db_context():
    soft = (
        "## Germany — Engagement\n\n"
        "## Recommended Next Actions for MISA\n"
        "- Engage stakeholders to drive growth.\n"
        "- Explore opportunities in the region.\n"
    )
    pack = {"_advisory_db_context": _CTX, "_answer_source": "strategic_advisory"}
    # strategic_advisory skips entity polish — scrub still runs in finalize
    out = finalize_answer(soft, user_question="engagement plan Germany", pack=pack)
    assert "Engage stakeholders" not in out
    assert "ROBERT BOSCH" in out or "SIEMENS" in out or "GTAI" in out


def test_multi_origin_actions_name_local_ipa():
    for country, needle in (
        ("Japan", "JETRO"),
        ("United States", "SelectUSA"),
        ("Ghana", "GIPC"),
        ("Pakistan", "BOI"),
    ):
        ctx = {
            "origin_country": country,
            "companies_from_origin_licensed_in_saudi": 20,
            "companies_from_origin_with_rhq": 2,
            "licensed_sector_distribution": [
                {"sector": "ICT", "licensed": 8},
            ],
        }
        actions = _named_actions_from_footprint(ctx)
        blob = " ".join(actions)
        assert needle in blob or country.split()[0] in blob
        assert "within 90 days" in blob.lower()


def test_weak_sector_does_not_paste_universal_vision():
    from app.services.jul21_surface import (
        _demand_anchors_for_text,
        enrich_entity_brief_depth,
    )

    anchors = _demand_anchors_for_text(
        "## Role\n\n**Pat Lee is Chair at ObscureCo.**\n"
    )
    assert "sector-qualified" in anchors

    thin = (
        "## Role\n\n"
        "**Pat Lee is Chair at ObscureCo.**\n\n"
        "## Background\n\n"
        "* Career in conglomerates.\n\n"
        "## 🇸🇦 Strategic Read\n\n"
        "* Engage carefully.\n"
    )
    out, _fixes = enrich_entity_brief_depth(thin, intent="executive_lookup")
    assert (
        "Qualify the primary sector fit" in out
        or "sector-qualified" in out.lower()
    )
