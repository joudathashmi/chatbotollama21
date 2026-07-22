"""Durable Jul21 contracts — must not regress again.

Locks:
  1. Narrative cloud = Azure compose over fact cards (not Ollama as narrator)
  2. Compose prefers curation; templates are fallback only
  3. Company / person / engagement answer shapes
  4. No invent-by-% MENA revenue
  5. Intent prompts still require the rich sections
"""

from __future__ import annotations

import inspect
import re

import pytest

from app.services import answer_contracts as ac
from app.services import db_briefing as db
from app.services.curation import _neutralise_unreliable_regional_revenue
from app.services.intent_router import intent_note_for_curation


# ─── Architecture invariants ─────────────────────────────────────────

def test_narrative_cloud_default_on(monkeypatch):
    """Quality path must stay on unless explicitly disabled."""
    monkeypatch.setattr("app.config.NARRATIVE_CLOUD_ENABLED", True)
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "auto")
    monkeypatch.setattr("app.config.RESIDENCY_STRICT", True)
    assert db.use_deterministic_db_briefing() is False


def test_compose_source_prefers_curation_before_templates():
    """chat_engine must call curate before force-template fallback."""
    from app.services import chat_engine as ce
    src = inspect.getsource(ce._compose_local_commentary_raw)
    curate_at = src.find("curate_company_insights")
    template_at = src.find("force=True")
    assert curate_at > 0 and template_at > 0
    assert curate_at < template_at, "curation must run before template fallback"
    assert "resolve_narrative_completion_client" in inspect.getsource(
        __import__("app.services.curation", fromlist=["curation"]).curate_company_insights
    ) or "resolve_narrative_completion_client" in open(
        "app/services/curation.py"
    ).read()


def test_curation_uses_narrative_client_not_hard_ollama():
    src = open("app/services/curation.py").read()
    assert "resolve_narrative_completion_client" in src
    # Hard data path may still exist for advisory; narrative compose must not
    # be forced through resolve_data_completion_client alone.
    assert src.index("resolve_narrative_completion_client") < src.index(
        "def curate_company_insights"
    ) or "resolve_narrative_completion_client" in inspect.getsource(
        __import__("app.services.curation", fromlist=["c"]).curate_company_insights
    )


def test_compose_falls_back_when_contract_fails(monkeypatch):
    """Thin curated answer missing ops body → templates, not ship regression."""
    from app.services import chat_engine as ce
    from app.config import CHAT_CURATION_ENABLED
    if not CHAT_CURATION_ENABLED:
        pytest.skip("curation disabled")

    thin = (
        "## Apple Inc — Executive Briefing\n\n"
        "Apple is big.\n\n"
        "### 📊 Corporate Profile & Regional Footprint\n\n"
        "| Metric | Global | MENA |\n|---|---|---|\n"
        "| **Financials** | $391B | $39.1B estimated MENA revenue (10% of global) |\n\n"
        "### 🇸🇦 Strategic Read\n\n- Do something.\n"
    )
    monkeypatch.setattr(ce, "CHAT_CURATION_ENABLED", True)
    monkeypatch.setattr(ce, "get_openai_client", lambda: object())
    monkeypatch.setattr(
        ce, "curate_company_insights",
        lambda *a, **k: thin,
    )
    # Force template path available
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "auto")
    monkeypatch.setattr("app.config.NARRATIVE_CLOUD_ENABLED", True)

    apple = {
        "company_name": "Apple Inc",
        "key_metrics": {
            "annual_revenue_usd": 391e9,
            "employee_count": 164000,
            "ksa_employees": 1800,
            "sector": "ICT",
            "headquarters": "Cupertino",
        },
        "misa_details": {
            "general": {"company_profile": "Devices and services.",
                        "global_headquarters": "Cupertino"},
            "mena_details": {
                "mena_notes": "Developer Academy in Riyadh. distribution hub.",
                "companies_name_in_mena": "Apple M E FZCO",
                "presence_in_saudi": "Yes",
            },
            "rhq_details": {"rhq_status": "Yes", "rhq_city": "Dubai",
                            "rhq_country": "UAE", "rhq_license_status": "Inactive"},
        },
    }
    import pandas as pd
    tc = [{
        "table": "company_profiles",
        "error": None,
        "rows_df": pd.DataFrame([apple]),
        "row_count": 1,
        "sql_entity_check_passed": True,
        "row_entity_sanity_passed": True,
        "closest_names": [],
    }]
    pack = {
        "_intent": "company_profile",
        "_depth": "operational_detail",
        "entity_candidate": "Apple",
        "entity_matched": "Apple Inc",
    }
    out = ce._compose_local_commentary_raw(
        tc, "Brief me on Apple", pack, response_locale="en",
    )
    assert out
    # Must not ship the thin curated invent
    assert "39.1" not in out
    # Template or repaired path must still be a company brief
    assert out and (
        "Executive Briefing" in out
        or "Snapshot of Operations" in out
        or "Operational Detail" in out
        or "Strategic Read" in out
    )
    assert pack.get("_contract_violations"), "expected contract gate to fire"


# ─── Shape contracts ─────────────────────────────────────────────────

def test_company_contract_accepts_jul21_shape():
    good = (
        "## Apple Inc — Executive Briefing\n\n"
        "Apple makes devices.\n\n"
        "### 📊 Corporate Profile & Regional Footprint\n\n"
        "| Metric | Global | MENA |\n|---|---|---|\n"
        "| **Financials** | $391.0B | MENA revenue not separately reported |\n\n"
        "### Snapshot of Operations and Market Position\n\n"
        "- Products and hubs.\n\n"
        "### 🇸🇦 Strategic Read\n\n- Engage on academy.\n"
    )
    ac.assert_company_brief(good)


def test_company_contract_rejects_ops_less_and_invented_mena():
    bad = (
        "## Apple Inc — Executive Briefing\n\n"
        "### 📊 Corporate Profile & Regional Footprint\n\n"
        "| **Financials** | $391B | $39.1B estimated MENA revenue (10% of global) |\n\n"
        "### 🇸🇦 Strategic Read\n\n- x\n"
    )
    v = ac.company_brief_violations(bad)
    assert any("ops body" in x for x in v)
    assert any("invented MENA" in x for x in v)


def test_person_contract_rejects_role_only():
    v = ac.person_brief_violations(
        "## Role\n\n**Tim Cook is CEO at Apple Inc.**\n"
    )
    assert any("Background" in x for x in v)


def test_person_violations_catch_empty_role():
    v = ac.person_brief_violations(
        "## Role\n\n## Background\n\n* Career arc.\n\n## Strategic Read\n\n* Engage.\n"
    )
    assert any("empty ## Role" in x for x in v)


def test_engagement_contract_order_and_forbidden_ops():
    good = (
        "## Engagement Recommendation\n\n- **Recommended approach:** talk\n\n"
        "## Snapshot\n\nApple is ICT.\n\n"
        "## Saudi / MENA Position\n\n- Hub\n\n"
        "## 🇸🇦 Strategic Read\n\n- Engage\n"
    )
    ac.assert_engagement_brief(good)
    bad = good + "\n## Operational Detail\n\n- dump\n"
    assert any("Operational Detail" in x for x in ac.engagement_brief_violations(bad))


# ─── Intent prompts locked ───────────────────────────────────────────

def test_company_profile_intent_requires_ops_snapshot():
    note = intent_note_for_curation("company_profile")
    assert "Snapshot of Operations and Market Position" in note
    assert "never invent MENA" in note.lower() or "10% of" in note
    assert "Strategic Read" in note


def test_engagement_intent_forbids_executive_briefing_ops():
    note = intent_note_for_curation("engagement_strategy")
    assert "Engagement Recommendation" in note
    assert "Snapshot" in note
    assert "Saudi / MENA Position" in note
    assert "FORBIDDEN" in note and "Executive Briefing" in note


def test_person_curation_no_longer_role_only_strip():
    src = open("app/services/curation.py").read()
    # Must not call _keep_person_role_only on the simple_fact path anymore
    assert "_keep_person_role_only(t)" not in src
    assert "_strip_company_sections_from_person_brief" in src


# ─── MENA invent scrub ───────────────────────────────────────────────

def test_mena_percent_of_global_scrubbed_from_table():
    sample = (
        "| **Financials** | $391.0B annual revenue | "
        "$39.1B estimated MENA revenue (10% of global) |\n"
    )
    blob = '{"company_name":"Apple Inc","revenue_usd":391000000000}'
    out = _neutralise_unreliable_regional_revenue(sample, blob)
    assert "39.1" not in out
    assert "not separately reported" in out.lower()


# ─── Template battery still green ────────────────────────────────────

APPLE = {
    "company_name": "Apple Inc",
    "key_metrics": {
        "annual_revenue_usd": 391_000_000_000,
        "employee_count": 164000,
        "ksa_employees": 1800,
        "mena_employees": 5600,
        "sector": "ICT",
        "headquarters": "California, USA",
    },
    "misa_details": {
        "general": {
            "company_profile": "Apple designs devices and services.",
            "product_services": "iPhone; Mac",
            "global_headquarters": "Cupertino, California, USA",
        },
        "mena_details": {
            "mena_notes": (
                "Middle East distribution hub in Saudi logistics zone. "
                "Developer Academy in Riyadh."
            ),
            "companies_name_in_mena": "Apple M E FZCO",
            "presence_in_saudi": "Yes",
        },
        "rhq_details": {
            "rhq_status": "Yes",
            "rhq_license_status": "Inactive",
            "rhq_city": "Dubai",
            "rhq_country": "UAE",
        },
    },
}


def test_template_fallback_still_rich(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")
    out = db.render_db_briefing(
        [APPLE], intent="company_profile",
        user_question="tell me about apple", force=True,
    )
    assert out and (
        "Snapshot of Operations" in out or "Operational Detail" in out
    ) and "Strategic Read" in out
