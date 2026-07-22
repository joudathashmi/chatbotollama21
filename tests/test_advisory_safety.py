"""Bulletproof advisory safety — cross-deliverable wipe/withhold regressions."""

from __future__ import annotations

from app.services.advisory_safety import (
    is_footprint_heading,
    may_hard_withhold,
    may_rebuild_as_company_targeting,
    ranking_midrow_truncated,
    should_skip_aggressive_collapse,
)
from app.services.curation import collapse_repetitive_briefing
from app.services.quality_gate import detect_quality_issues, run_quality_gate
from app.services.response_validator import validate_advisory_answer


_CTX = {
    "origin_country": "India",
    "companies_from_origin_licensed_in_saudi": 2437,
    "companies_from_origin_with_rhq": 14,
    "retrieval_status": "SUCCESS_WITH_RESULTS",
    "expansion_targets": [
        {"company": "Tech Mahindra", "sector": "IT",
         "current_saudi_presence": "RHQ"},
        {"company": "Indian Oil", "sector": "Energy",
         "current_saudi_presence": "RHQ"},
        {"company": "State Bank of India", "sector": "Banking",
         "current_saudi_presence": "RHQ"},
        {"company": "Ramco Systems", "sector": "IT",
         "current_saudi_presence": "RHQ"},
    ],
}


def _market_fit_doc() -> str:
    return (
        "# Market Fit Assessment: Attracting Indian Companies to "
        "Saudi Arabia\n\n"
        "## Strategic Context\n"
        "India is a major outbound investor. Saudi Arabia is a "
        "regional growth platform for Vision 2030 sectors.\n\n"
        "## Overall Market Fit\n"
        "| Sector | Strategic Fit | Investment Potential | Priority |\n"
        "|---|---|---|---|\n"
        "| ICT | High | High | Tier 1 |\n"
        "| Healthcare | High | Medium | Tier 1 |\n"
        "| Manufacturing | High | High | Tier 1 |\n\n"
        "# Investment & Trade Bodies to Engage\n"
        "| Organisation | Type | Role |\n"
        "|---|---|---|\n"
        "| Invest India | IPA | National pipeline |\n"
        "| CII | Industry body | Manufacturing outreach |\n"
        "| FICCI | Industry body | Multi-sector |\n\n"
        "## Strategic Conclusion\n"
        "Complementarity thesis holds for Indian capital and Saudi "
        "localisation demand.\n"
    )


def _engagement_plan_doc() -> str:
    return (
        "# Engagement Plan: Attracting Investment from India\n\n"
        "## Objectives & Success Metrics\n"
        "- 50 qualified leads in 12 months\n\n"
        "## Priority Target Segments\n"
        "| Segment | Why | Archetypes | Priority |\n"
        "|---|---|---|---|\n"
        "| ICT | Digital push | SaaS vendors | Tier 1 |\n\n"
        "## Phased Roadmap\n"
        "### Phase 1 — Foundation (months 0–3)\n"
        "- Map accounts\n"
        "### Phase 2 — Activation (months 3–9)\n"
        "- Roadshows\n"
        "### Phase 3 — Conversion (months 9–18)\n"
        "- Licence conversion\n"
    )


def _sector_priorities_doc() -> str:
    return (
        "# Sector Priorities: India → Saudi Arabia\n\n"
        "## The Evidence Base\n"
        "| Sector | Share | Companies |\n"
        "|---|---|---|\n"
        "| ICT | 22% | 540 |\n"
        "| Construction | 18% | 430 |\n\n"
        "## Priority Sectors\n"
        "1. ICT\n2. Healthcare\n3. Energy\n"
    )


def test_evidence_base_is_not_a_footprint_heading():
    assert is_footprint_heading("## The Evidence Base") is False
    assert is_footprint_heading("## Current MISA Footprint") is True
    assert is_footprint_heading("## Current Saudi Footprint") is True


def test_sector_evidence_base_not_stripped_or_replaced():
    doc = _sector_priorities_doc()
    fixed, fixes = validate_advisory_answer(
        doc, _CTX, deliverable="sector_priorities",
    )
    assert "The Evidence Base" in fixed
    assert "22%" in fixed
    assert "rebuilt_footprint_from_db_counts" not in fixes
    assert "stripped_fabricated_footprint_section" not in fixes
    assert "rebuilt_truncated_company_targeting_from_db" not in fixes


def test_non_targeting_deliverables_never_full_replaced():
    for deliverable, doc in (
        ("market_fit", _market_fit_doc()),
        ("engagement_plan", _engagement_plan_doc()),
        ("sector_priorities", _sector_priorities_doc()),
        ("strategy_analysis", _market_fit_doc()),
    ):
        assert may_rebuild_as_company_targeting(
            deliverable=deliverable, answer=doc,
        ) is False
        fixed, fixes = validate_advisory_answer(
            doc, _CTX, deliverable=deliverable,
        )
        assert "rebuilt_truncated_company_targeting_from_db" not in fixes
        assert "Priority Company Ranking" not in fixed
        assert "Response withheld" not in fixed


def test_missing_detailed_section_is_not_truncation():
    completeish = (
        "# Targeting India Companies\n\n"
        "## Priority Company Ranking\n\n"
        "| Rank | Company | Sector | Saudi Presence | Target Type | "
        "Proposed Investment | Thesis | MISA Action |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 1 | A | IT | RHQ | expansion | Expand | Thesis | Act |\n"
        "| 2 | B | Energy | RHQ | expansion | Expand | Thesis | Act |\n"
        "| 3 | C | Banking | Licensed | expansion | Expand | Thesis | Act |\n\n"
        "## Next Steps\n- Follow up\n"
    )
    assert ranking_midrow_truncated(completeish) is False
    issues = detect_quality_issues(
        completeish, deliverable="company_targeting",
    )
    assert not any(i["code"] == "truncated_ranking_table" for i in issues)


def test_midrow_cut_is_truncation_for_targeting_only():
    cut = (
        "# Targeting India Companies\n\n"
        "## Priority Company Ranking\n\n"
        "| Rank | Company | Sector | Saudi Presence | Target Type | "
        "Proposed Investment | Thesis | MISA Action |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 1 | A | IT | RHQ | expansion | Expand | Thesis | Act |\n"
        "| 2 | B | Energy |"
    )
    assert ranking_midrow_truncated(cut) is True
    assert may_rebuild_as_company_targeting(
        deliverable="company_targeting", answer=cut,
    ) is True
    assert may_rebuild_as_company_targeting(
        deliverable="market_fit", answer=cut,
    ) is False


def test_quality_gate_never_withholds_truncation_alone():
    cut = (
        "# Targeting India Companies for Investment Attraction: "
        "Strategic List and Investment Thesis\n\n"
        + ("Context. " * 80)
        + "\n\n## Priority Company Ranking\n\n"
        "| Rank | Company | Sector | Saudi Presence | Target Type | "
        "Proposed Investment | Thesis | MISA Action |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 1 | A | IT | RHQ | expansion | Expand | Thesis | Act |\n"
        "| 2 | B | Energy |"
    )
    text, _issues, fixes = run_quality_gate(
        cut,
        question="best Indian companies to target",
        db_context=_CTX,
        hard_block=True,
        deliverable="company_targeting",
    )
    assert "Response withheld" not in text
    assert len(text) > 400
    assert may_hard_withhold({"truncated_ranking_table"}) is False


def test_quality_gate_withholds_only_factual_criticals():
    text, _issues, fixes = run_quality_gate(
        "There are zero Indian companies licensed in Saudi Arabia.",
        question="how many Indian companies",
        db_context={
            "footprint_data_unavailable": True,
            "origin_country": "India",
            "retrieval_status": "error",
        },
        hard_block=True,
        deliverable="market_fit",
    )
    assert (
        "hard_block" in " ".join(fixes)
        or "not a verified zero" in text.lower()
        or "unavailable" in text.lower()
    )


def test_collapse_never_chops_advisory_tables():
    # Template-ish repeated thesis wording across ranking rows used to
    # trigger the window cutter and cut mid-pipe.
    rows = "\n".join(
        f"| {i} | Company {i} | ICT | RHQ | expansion | Expand | "
        f"Already present in the MISA Saudi footprint (RHQ). Existing "
        f"licence/RHQ is an expansion base for Vision 2030. | Act |"
        for i in range(1, 12)
    )
    body = (
        "# Targeting India Companies\n\n"
        "## Priority Company Ranking\n\n"
        "| Rank | Company | Sector | Saudi Presence | Target Type | "
        "Proposed Investment | Thesis | MISA Action |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + rows
        + "\n\n## Detailed Investment Theses\n\n### Company 1\nThesis.\n"
    )
    assert should_skip_aggressive_collapse(
        deliverable="company_targeting", answer=body,
    ) is True
    out = collapse_repetitive_briefing(
        body, deliverable="company_targeting",
    )
    assert "| 11 |" in out
    assert len(out) > len(body) * 0.9


def test_pdf_export_path_keeps_market_fit_on_truncation_noise():
    """Export without db_context must not withhold a market-fit doc."""
    doc = _market_fit_doc() + "\n| orphan incomplete"
    text, issues, fixes = run_quality_gate(
        doc,
        question="make me a market for to atrract indian companies",
        hard_block=True,
        deliverable="market_fit",
    )
    assert "Response withheld" not in text
    assert "Market Fit Assessment" in text
    assert not any(i["code"] == "truncated_ranking_table" for i in issues)


def test_sector_priorities_does_not_get_forced_footprint_inject():
    doc = _sector_priorities_doc()
    fixed, fixes = validate_advisory_answer(
        doc, _CTX, deliverable="sector_priorities",
    )
    assert "injected_missing_footprint_section" not in fixes
    assert "The Evidence Base" in fixed
    assert fixed.index("The Evidence Base") < fixed.find("Priority Sectors")


def test_rhq_licenses_aggregate_rewrites_to_company_profiles():
    from app.services.source_policy import rewrite_aggregate_licensing_query
    tbl, flt, notes = rewrite_aggregate_licensing_query(
        "rhq_licenses", count_only=True, filters={}, question="how many licenses",
    )
    assert tbl == "company_profiles"
    assert flt.get("licensed") is True
    assert notes and "rewrote" in notes[0]

    tbl2, flt2, notes2 = rewrite_aggregate_licensing_query(
        "rhq_licenses",
        count_only=True,
        filters={},
        question="how many RHQ licenses",
    )
    assert tbl2 == "company_profiles"
    assert flt2.get("is_rhq") is True

    # Detail lookup by id stays on auxiliary table
    tbl3, flt3, notes3 = rewrite_aggregate_licensing_query(
        "rhq_licenses",
        count_only=False,
        filters={"license_number": "123"},
        question="show license 123",
    )
    assert tbl3 == "rhq_licenses"
    assert not notes3


def test_browse_licenses_maps_to_company_profiles_not_rhq_licenses():
    from app.services.input_cleaner import _BROWSE_TABLE_MAP
    assert _BROWSE_TABLE_MAP["licenses"][0] == "company_profiles"
    assert _BROWSE_TABLE_MAP["licences"][0] == "company_profiles"
