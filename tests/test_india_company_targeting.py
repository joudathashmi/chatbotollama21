"""Regression tests for India company-targeting quality + PDF tables."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from app.prompts.chat_system import advisory_system_prompt
from app.services import chat_engine as mc
from app.services.advisory_structured import (
    extract_json_object,
    ranking_table_is_truncated,
    render_company_targeting_markdown,
    validate_company_targeting_payload,
)
from app.services.pdf_export import (
    normalize_answer_markdown_for_pdf,
    render_pdf,
)
from app.services.response_validator import validate_advisory_answer
from app.services.target_ranking import rank_expansion_targets


def test_company_targeting_deliverable_detected():
    q = (
        "Give me the best companies to target from India with an "
        "investment thesis for each."
    )
    assert mc._detect_advisory_deliverable(q) == "company_targeting"
    p = advisory_system_prompt("en", "company_targeting")
    assert "Priority Company Ranking" in p
    assert "expansion" in p and "new_entry" in p


def test_failed_retrieval_not_represented_as_zero():
    with patch(
        "app.services.engagement_data.fetch_country_saudi_investors",
        return_value={"_db_error": "relation missing", "total_licensed": 0,
                      "total_rhq": 0},
    ):
        ctx = mc._advisory_country_context(
            "best companies to target from India with investment thesis"
        )
    assert ctx["footprint_data_unavailable"] is True
    assert ctx.get("retrieval_status") in (
        "error", "SOURCE_UNAVAILABLE", "UNKNOWN_ERROR", "CONNECTION_ERROR",
    )
    assert "companies_from_origin_licensed_in_saudi" not in ctx

    bad = (
        "# Targeting\n\n## Current MISA Footprint\n"
        "There are zero Indian companies licensed.\n"
    )
    fixed, fixes = validate_advisory_answer(bad, ctx)
    assert "rebuilt_footprint_unavailable_notice" in fixes
    assert "could not be retrieved" in fixed.lower()
    assert not re.search(r"\bzero Indian companies licensed\b", fixed, re.I)


def test_actual_zero_labelled_with_source_and_filters():
    ctx = {
        "origin_country": "India",
        "companies_from_origin_licensed_in_saudi": 0,
        "companies_from_origin_with_rhq": 0,
        "retrieval_status": "zero_records",
        "retrieval_filters": {
            "origin_country": "India",
            "source": "company_profiles + nationality/origin join",
        },
        "expansion_targets": [],
    }
    answer = "# T\n\n## Current Saudi Footprint\nSomething wrong.\n"
    fixed, fixes = validate_advisory_answer(answer, ctx)
    assert "rebuilt_footprint_from_db_counts" in fixes or "0" in fixed
    assert "source:" in fixed.lower() or "filters" in fixed.lower()
    assert "zero-result" in fixed.lower() or "0** verified" in fixed


def test_internal_counts_override_model_zeros():
    ctx = {
        "origin_country": "India",
        "companies_from_origin_licensed_in_saudi": 2438,
        "companies_from_origin_with_rhq": 39,
        "retrieval_status": "ok",
        "top_rhq_companies": [{"name": "Tech Mahindra"}],
        "top_licensed_companies": [{"name": "Biocon"}],
        "expansion_targets": [
            {"company": "Tech Mahindra", "sector": "IT",
             "current_saudi_presence": "RHQ", "target_type": "expansion"},
            {"company": "Tata Consultancy Services", "sector": "IT",
             "current_saudi_presence": "Licensed", "target_type": "expansion"},
            {"company": "Meril", "sector": "Medtech",
             "current_saudi_presence": "RHQ", "target_type": "expansion"},
        ],
    }
    answer = (
        "# India Targeting\n\n## Executive Summary\n"
        "- Indian companies licensed: 0\n"
        "- RHQ: 0\n\n"
        "## Priority Company Ranking\n"
        "Only Infosys and Reliance (generic).\n"
    )
    fixed, fixes = validate_advisory_answer(answer, ctx)
    assert any(
        f in fixes
        for f in (
            "injected_missing_footprint_section",
            "rebuilt_footprint_from_db_counts",
            "scrubbed_false_zero_in_section",
            "injected_expansion_targets_from_db",
        )
    )
    assert "**2438**" in fixed or "2438" in fixed
    assert "Tech Mahindra" in fixed
    assert "expansion" in fixed.lower()


def test_rank_expansion_targets_prefers_rhq():
    stats = {
        "rhq": [
            {"company_name": "Tech Mahindra Regional HQ", "industry": "IT",
             "annual_revenue": 5e9},
            {"company_name": "Local Cafe Corner", "industry": "Food",
             "annual_revenue": 1e5},
        ],
        "licensed_only": [
            {"company_name": "Biocon Limited", "industry": "Pharma",
             "annual_revenue": 1e9},
        ],
    }
    ranked = rank_expansion_targets(stats)
    assert ranked[0]["target_type"] == "expansion"
    assert ranked[0]["current_saudi_presence"] == "RHQ"
    assert "Cafe" not in ranked[0]["company"]
    assert any(t["company"] == "Biocon Limited" for t in ranked)


def test_structured_payload_requires_investment_and_separates_types():
    from app.services.advisory_structured import (
        merge_thesis_enrichment,
        seed_company_targeting_payload_from_db,
    )
    db = {
        "origin_country": "India",
        "companies_from_origin_licensed_in_saudi": 2438,
        "companies_from_origin_with_rhq": 39,
        "retrieval_status": "ok",
        "retrieval_filters": {
            "source": "company_profiles + nationality/origin join",
        },
        "expansion_targets": [
            {"company": "Tech Mahindra", "sector": "IT",
             "current_saudi_presence": "RHQ", "evidence_strength": "high"},
            {"company": "Biocon Limited", "sector": "Pharma",
             "current_saudi_presence": "Licensed", "evidence_strength": "medium"},
        ],
        "top_rhq_companies": [{"name": "Tech Mahindra"}],
    }
    seed = seed_company_targeting_payload_from_db(db)
    assert seed is not None
    assert seed["current_footprint"]["licensed_companies"] == 2438
    assert len(seed["targets"]) >= 2
    assert all(t["target_type"] == "expansion" for t in seed["targets"])

    enrichment = {
        "executive_summary": {
            "key_findings": ["Focus on RHQ deepen"],
            "top_recommendation": "Account plan for Tech Mahindra",
        },
        "theses": {
            "Tech Mahindra": {
                "why_company": "Digital scale",
                "why_saudi": "Vision 2030 digital",
                "why_now": "NEOM demand",
                "proposed_investment": "Cloud delivery centre",
                "misa_action": "Brief RHQ MD",
            }
        },
        "new_entry_targets": [
            {
                "company": "NewCo Energy",
                "sector": "Energy",
                "proposed_investment": "Manufacturing JV",
                "why_company": "Capability",
                "why_saudi": "NREP",
                "why_now": "Auction cycle",
                "misa_action": "Intro via Invest India",
                "validation_required": ["Demand proof"],
            }
        ],
        "recommendations": ["Call Tech Mahindra RHQ"],
    }
    payload = merge_thesis_enrichment(seed, enrichment)
    assert payload["current_footprint"]["licensed_companies"] == 2438
    md = render_company_targeting_markdown(payload)
    assert "expansion" in md and "new_entry" in md
    assert "Cloud delivery centre" in md
    assert "Tech Mahindra" in md
    assert "Sources and Data Limitations" in md
    assert "2438" in md
    # Ranking must be complete (not truncated mid-row)
    assert ranking_table_is_truncated(md) is False
    # Lean executive ranking (5 cols) — expansion type lives in theses
    assert "| Rank | Company | Sector | Saudi Presence | Investment Thesis |" in md
    assert "Strategic Context" in md
    assert "Invest India" in md
    assert "Tech Mahindra" in md


def test_validator_rebuilds_truncated_ranking_table():
    ctx = {
        "origin_country": "India",
        "companies_from_origin_licensed_in_saudi": 2438,
        "companies_from_origin_with_rhq": 39,
        "retrieval_status": "ok",
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
    truncated = (
        "# Targeting\n\n## Executive Summary\n- 12 expansion\n\n"
        "## Current Saudi Footprint\n**2438** licensed, **39** RHQ.\n\n"
        "## Priority Company Ranking\n\n"
        "| Rank | Company | Sector | Saudi Presence | Target Type | "
        "Proposed Investment | Thesis | MISA Action |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 1 | Union Bank of India | Banking | Licensed | expansion | "
        "Expand | Already present | Account review |\n"
        "| 2 | Aditya Birla Capital Ltd. | Financial Services |"
    )
    fixed, fixes = validate_advisory_answer(truncated, ctx)
    assert "rebuilt_truncated_company_targeting_from_db" in fixes
    assert "Tech Mahindra" in fixed
    assert "Detailed Investment Theses" in fixed
    assert ranking_table_is_truncated(fixed) is False



def test_pdf_tables_no_raw_pipes_or_hash_headings():
    md = (
        "# India Targeting\n\n"
        "## Priority Company Ranking\n\n"
        "| Rank | Company | Sector | Saudi Presence | Target Type | "
        "Proposed Investment | Thesis | MISA Action |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 1 | Tech Mahindra Regional Headquarters Company | IT Services | "
        "RHQ | expansion | RHQ capability expansion into cloud SSC | "
        "Deepen installed base against Vision 2030 digitalisation | "
        "Brief RHQ MD with NEOM digital RFP map |\n"
        "| 2 | Very Long Pharmaceutical Company Name Private Limited | "
        "Pharma | Licensed | expansion | Local fill-finish plant | "
        "Long thesis text that must wrap without character-by-character "
        "column collapse or overlapping cells in the PDF output | "
        "Arrange SFDA localisation workshop |\n"
    )
    prepared = normalize_answer_markdown_for_pdf(md)
    assert "<table" in prepared or "profile-card" in prepared
    assert prepared.count("| 1 |") == 0  # pipes consumed
    pdf = render_pdf(
        "Give me the best companies to target from India",
        md,
    )
    assert pdf[:4] == b"%PDF"
    # Extract textish content loosely — xhtml2pdf embeds strings.
    # Ensure raw markdown artefacts are not dominant.
    as_ascii = pdf.decode("latin-1", errors="ignore")
    assert "## Priority" not in as_ascii
    # Pipe-table source rows should not appear as a contiguous artefact.
    assert "| Rank | Company |" not in as_ascii


def test_pdf_empty_columns_dropped():
    md = (
        "| Rank | Company | Empty |\n"
        "|---|---|---|\n"
        "| 1 | Acme |  |\n"
        "| 2 | Beta |   |\n"
    )
    html = normalize_answer_markdown_for_pdf(md)
    assert "Empty" not in html


def test_extract_json_object_from_fenced_block():
    raw = '```json\n{"title": "T", "targets": []}\n```'
    data = extract_json_object(raw)
    assert data and data["title"] == "T"


@pytest.mark.integration
def test_live_india_footprint_positive_counts():
    """Live DB: successful retrieval must not be a false zero."""
    from app.services.engagement_data import fetch_country_saudi_investors
    stats = fetch_country_saudi_investors("India")
    if stats.get("_db_error"):
        pytest.skip(f"DB unavailable: {stats['_db_error']}")
    assert int(stats.get("total_licensed") or 0) > 0
    ctx = mc._advisory_country_context(
        "best companies to target from India with an investment thesis"
    )
    assert ctx and not ctx.get("footprint_data_unavailable")
    assert int(ctx.get("companies_from_origin_licensed_in_saudi") or 0) > 0
    assert ctx.get("retrieval_status") in (
        "ok", "SUCCESS_WITH_RESULTS",
    )
    assert ctx.get("expansion_targets")
