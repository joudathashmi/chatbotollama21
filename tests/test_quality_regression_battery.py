"""Offline regression battery — quality shapes that must not regress.

These run without a live server / LLM. They lock the contract that
caused case-by-case user pain: company briefs keep Operational Detail +
Strategic Read; person/CEO briefs keep Role + Background + Strategic
Read; forbidden legacy headers never ship; officeholder gate does not
hijack corporate C-suite asks.
"""

from __future__ import annotations

import pytest

from app.services import db_briefing as db
from app.services.answer_finalize import finalize_answer
from app.services.chat_engine import _is_current_officeholder_question


APPLE = {
    "company_name": "Apple Inc",
    "profit_margin": "24%",
    "roe": "20%",
    "potential_strategic_opportunity": (
        "Saudi gaming & esports hub: $32B sports economy target by 2030."
    ),
    "key_metrics": {
        "annual_revenue_usd": 391_000_000_000,
        "employee_count": 164000,
        "ksa_employees": 1800,
        "mena_employees": 5600,
        "sector": "Information and Communication Technology",
        "headquarters": "California, USA",
    },
    "misa_details": {
        "general": {
            "company_profile": (
                "Apple designs and sells smartphones, computers, and services."
            ),
            "product_services": "iPhone; Mac; iPad; Services",
            "global_headquarters": "Cupertino, California, USA",
        },
        "mena_details": {
            "history_in_mena": "Entered Middle East retail in 2015.",
            "companies_name_in_mena": "Apple M E FZCO",
            "companies_name_in_ksa": "Apple (Saudi Arabia)",
            "mena_notes": (
                "Middle East distribution hub in Saudi logistics zone. "
                "Developer Academy in Riyadh. Arab Business Machine distributor."
            ),
            "presence_in_saudi": "Yes",
        },
        "rhq_details": {
            "rhq_status": "Yes",
            "rhq_license_status": "Inactive",
            "rhq_city": "Dubai",
            "rhq_country": "UAE",
        },
    },
    "geographic_revenue": [
        {"region": "North America", "percentage": "55%", "revenue": "215e9"},
        {"region": "Middle East & Africa", "percentage": "10%", "revenue": "39e9"},
        {"region": "Europe", "percentage": "35%", "revenue": "136e9"},
    ],
}

TIM = {
    "full_name": "Tim Cook",
    "title": "CEO",
    "tenure": "Current",
    "company_profile_id": 1,
}


@pytest.fixture(autouse=True)
def _det(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")


# ─── Company shapes ──────────────────────────────────────────────────

def test_company_brief_never_drops_ops_or_strategy():
    out = db.render_db_briefing(
        [APPLE, TIM], intent="company_profile",
        user_question="tell me about apple",
    )
    assert out and "Snapshot of Operations" in out
    assert "Strategic Read" in out
    assert "55%" in out and "1,800" in out
    assert "Developer Academy" in out or "distribution hub" in out.lower()
    assert "From the MISA Record" not in out
    assert "Background (general knowledge)" not in out


def test_saudi_presence_intent_still_rich():
    out = db.render_db_briefing(
        [APPLE, TIM], intent="saudi_presence",
        user_question="what is Apple's presence in Saudi Arabia?",
    )
    assert out and "Snapshot of Operations" in out
    assert "1,800" in out or "Saudi" in out


def test_flat_company_without_misa_json_still_has_ops():
    row = {
        "company_name": "FlatCo",
        "sector": "ICT",
        "revenue_usd": 5e9,
        "history_in_mena": "Entered in 2012.",
        "mena_notes": "distribution hub in Saudi logistics zone.",
        "product_services": "ERP; Analytics",
        "companies_name_in_mena": "FlatCo FZCO",
        "number_of_employees_ksa": 100,
        "number_of_employees_mena": 400,
        "geographic_revenue": [
            {"region": "MENA", "percentage": "30%", "revenue": "1.5e9"},
        ],
        "potential_strategic_opportunity": "Industrial digitisation in KSA.",
    }
    out = db.render_company_brief([row])
    assert out and "Snapshot of Operations" in out
    assert "Entered in 2012" in out
    assert "Strategic Read" in out


# ─── Person / CEO shapes ─────────────────────────────────────────────

def test_ceo_ask_is_person_not_company_dump():
    out = db.render_db_briefing(
        [APPLE, TIM], intent="company_profile",
        user_question="Who is the CEO of Apple?",
    )
    assert "## Role" in out and "Tim Cook" in out
    assert "Snapshot of Operations" not in out and "Operational Detail" not in out
    assert "## Background" in out
    assert "Strategic Read" in out


def test_named_person_bio_shape():
    out = db.render_db_briefing(
        [APPLE, TIM], intent="executive_lookup",
        user_question="Tell me about Tim Cook",
    )
    assert "## Role" in out and "Tim Cook" in out
    assert "Snapshot of Operations" not in out and "Operational Detail" not in out


def test_corporate_ceo_not_officeholder():
    assert not _is_current_officeholder_question("Who is the CEO of Apple?")
    assert not _is_current_officeholder_question("Who is the CFO of Aramco?")
    assert _is_current_officeholder_question(
        "Who is the Minister of Investment of Saudi Arabia?"
    )


# ─── Finalize gate ───────────────────────────────────────────────────

def test_finalize_strips_forbidden_legacy_headers():
    raw = (
        "## From the MISA Record\n\n* Name: Tim Cook\n\n"
        "## Background (general knowledge)\n\n* Old career fluff\n\n"
        "## Role\n\n**Tim Cook is CEO at Apple Inc.**\n"
    )
    out = finalize_answer(raw, user_question="Who is the CEO of Apple?")
    assert "From the MISA Record" not in out
    assert "Background (general knowledge)" not in out
    assert "## Role" in out
    assert "_Sources:" in out


def test_finalize_strips_forbidden_inline_phrases():
    raw = (
        "## Engagement History — Acme\n\n"
        "Internal records do not currently show anything.\n"
        "_(general knowledge)_\n"
    )
    out = finalize_answer(raw)
    assert "Internal records do not currently show" not in out
    assert "_(general knowledge)_" not in out


def test_engagement_plan_matches_ai_response_6_shape(monkeypatch):
    """Target shape: Recommendation → Snapshot → MENA → Strategic Read."""
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")
    q = "make me an engagement plan for apple for investment into new opportunities in saudi"
    out = db.render_db_briefing(
        [APPLE, TIM], intent="engagement_strategy",
        user_question=q,
    )
    assert out is not None
    assert "## Engagement Recommendation" in out
    assert "Recommended approach" in out
    assert "Priority stakeholders" in out
    assert "Talking points" in out
    assert "## Snapshot" in out
    assert "Saudi / MENA Position" in out
    assert "Strategic Read" in out
    # Must NOT be the company Executive Briefing / Ops table shape
    assert "Snapshot of Operations" not in out and "Operational Detail" not in out
    assert "Executive Briefing" not in out
    assert "Corporate Profile" not in out
    assert out.find("Engagement Recommendation") < out.find("Snapshot")
    assert out.find("Snapshot") < out.find("Saudi / MENA Position")
    assert "distribution hub" in out.lower() or "Developer Academy" in out
