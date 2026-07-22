"""Unit tests for deterministic DB-first briefings (old.pdf parity)."""

from __future__ import annotations

from app.services import db_briefing as db


APPLE_NESTED = {
    "company_name": "Apple Inc",
    "profit_margin": "24%",
    "roe": "20%",
    "potential_strategic_opportunity": (
        "Saudi gaming & esports hub: $32B sports economy target by 2030, "
        "Savvy Games Group (PIF) investing $38B. Opportunities: game "
        "development studios, esports arena operations."
    ),
    "key_metrics": {
        "annual_revenue_usd": 391_000_000_000,
        "employee_count": 164000,
        "ksa_employees": 1800,
        "mena_employees": 5600,
        "sector": "Information and Communication Technology",
        "headquarters": "California, USA",
    },
    "market_intelligence": {
        "market_cap": "$3,500B",
        "pricing_strategy": "Subscription/SaaS models dominant",
        "market_trend": "Global IT spending growing 9-10% CAGR",
    },
    "misa_details": {
        "general": {
            "company_profile": (
                "Apple, Inc. engages in the design, manufacture, and sale of "
                "smartphones, personal computers, tablets, wearables and "
                "accessories, and other varieties of related services."
            ),
            "product_services": (
                "Personal Devices: iPhone, Apple Watch, AirPods; "
                "Computing Devices: Mac, iPad; Services: Apple Music, App Store."
            ),
            "global_headquarters": "Cupertino, California, USA",
        },
        "mena_details": {
            "history_in_mena": (
                "Apple officially entered the Middle East market by opening "
                "its first retail stores in 2015 at the Mall of the Emirates "
                "in Dubai and Yas Mall in Abu Dhabi."
            ),
            "companies_name_in_mena": "Apple M E FZCO",
            "companies_name_in_ksa": "Apple (Saudi Arabia)",
            "mena_locations": ["UAE", "Saudi Arabia"],
            "presence_in_mena": "Company",
            "presence_in_saudi": "Yes",
            "type_of_presence_saudi": "Company",
            "mena_notes": (
                "1) Apple M E FZCO, a Dubai-based branch of Apple Inc., "
                "functions as the regional hub.\n"
                "Apple has agreed to set up its Middle East distribution hub "
                "in Saudi Arabia, in a coup for the kingdom as it rolls out "
                "its first special logistics zone.\n"
                "Apple has chosen Riyadh as the headquarters for its Apple "
                "Developer Academy, making it the first in MENA region.\n"
                "Arab Business Machine (ABM), established in 1986, serves as "
                "the principal distributor for Apple products in the Middle East."
            ),
            "ksa_employees": 1800,
            "mena_employees": 5600,
        },
        "rhq_details": {
            "rhq_status": "Yes",
            "rhq_license_status": "Inactive",
            "rhq_city": "Dubai",
            "rhq_country": "UAE",
        },
    },
    "geographic_revenue": [
        {"region": "North America", "percentage": "55%", "revenue": "215000000000"},
        {"region": "Middle East & Africa", "percentage": "10%", "revenue": "39100000000"},
        {"region": "Europe", "percentage": "35%", "revenue": "136000000000"},
    ],
}


def test_use_deterministic_when_mode_deterministic(monkeypatch):
    monkeypatch.setattr(db, "use_deterministic_db_briefing", lambda: True)
    assert db.use_deterministic_db_briefing() is True


def test_company_brief_from_rows(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")
    rows = [{
        "company_name": "Apple Inc.",
        "sector": "ICT",
        "revenue_usd": 391_000_000_000,
        "number_of_employees": 164000,
        "number_of_employees_mena": 5600,
        "number_of_employees_ksa": 1800,
        "global_headquarters": "Cupertino, California, USA",
        "is_rhq": False,
        "rhq_city": "Dubai",
        "rhq_country": "UAE",
        "company_profile": "Apple designs consumer electronics. It sells iPhones worldwide.",
    }, {
        "full_name": "Tim Cook",
        "title": "CEO",
        "company_profile_id": 1,
        "tenure": "Current",
    }]
    out = db.render_db_briefing(
        rows, intent="company_profile", table="company_profiles",
        user_question="tell me about apple",
    )
    assert out is not None
    assert "Apple Inc." in out
    assert "Executive Briefing" in out
    assert "Strategic Context" in out
    assert "Strategic Read" in out
    assert "Recommended Next Actions for MISA" in out
    assert "NEOM" in out or "SDAIA" in out
    assert "$391.0B" in out
    assert "Tim Cook" in out
    assert out.lower().count("strategic read") == 1
    assert "From your documents" not in out


def test_nested_apple_matches_old_pdf_ops_and_strategy(monkeypatch):
    """Parity with Downloads/old.pdf Operational Detail + Strategic Read."""
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")
    rows = [
        dict(APPLE_NESTED),
        {"full_name": "Tim Cook", "title": "CEO", "company_profile_id": 1, "tenure": "Current"},
    ]
    out = db.render_db_briefing(
        rows, intent="company_profile", table="company_profiles",
        user_question="tell me about apple",
    )
    assert out is not None
    assert "Snapshot of Operations" in out
    assert "55%" in out and "35%" in out and "10%" in out
    assert "1,800" in out and "5,600" in out
    assert "$391.0B" in out or "$391B" in out
    assert "Developer Academy" in out
    assert "distribution hub" in out.lower()
    assert "Arab Business Machine" in out or "ABM" in out
    assert "gaming" in out.lower()
    assert "Inactive" in out
    # Distinct angles — not the old boilerplate thrice.
    assert out.lower().count("engage on localisation / partnership") == 0
    assert "supply chain" in out.lower() or "logistics" in out.lower()
    assert "digital skills" in out.lower() or "Developer Academy" in out


def test_flat_projection_still_builds_ops(monkeypatch):
    """When misa_details is missing, flat columns must still feed Ops."""
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")
    rows = [{
        "company_name": "FlatCo Inc",
        "sector": "ICT",
        "revenue_usd": 10_000_000_000,
        "number_of_employees": 5000,
        "number_of_employees_ksa": 200,
        "number_of_employees_mena": 800,
        "global_headquarters": "Riyadh, Saudi Arabia",
        "is_rhq": True,
        "rhq_city": "Riyadh",
        "rhq_country": "Saudi Arabia",
        "rhq_license_status": "Active",
        "company_profile": "FlatCo builds industrial software for the Gulf.",
        "product_services": "ERP; Field service; Analytics",
        "history_in_mena": "Entered MENA in 2012 via a Dubai free-zone entity.",
        "mena_notes": (
            "FlatCo set up a distribution hub in Saudi logistics zone. "
            "Principal distributor partners cover KSA and UAE."
        ),
        "companies_name_in_mena": "FlatCo MENA FZCO",
        "companies_name_in_ksa": "FlatCo Saudi",
        "presence_in_saudi": "Yes",
        "type_of_presence_saudi": "Company",
        "geographic_revenue": [
            {"region": "MENA", "percentage": "40%", "revenue": "4000000000"},
            {"region": "Europe", "percentage": "60%", "revenue": "6000000000"},
        ],
        "potential_strategic_opportunity": (
            "Saudi industrial digitisation: incentives for localisation of "
            "ERP and field-service delivery centres."
        ),
    }]
    out = db.render_company_brief(rows, user_question="tell me about FlatCo")
    assert out is not None
    assert "Snapshot of Operations" in out
    assert "Entered MENA in 2012" in out
    assert "FlatCo MENA FZCO" in out
    assert "40%" in out
    assert "distribution hub" in out.lower() or "logistics" in out.lower()
    assert "industrial digitisation" in out.lower() or "Strategic opportunity" in out or "digitisation" in out.lower()


def test_related_enrichment_feeds_geo_and_opps(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")
    rows = [{
        "company_name": "RelCo",
        "sector": "Energy",
        "revenue_usd": 50e9,
        "misa_details": {},
        "_related": {
            "geographic revenues": [
                {"region": "Middle East & Africa", "percentage": "25%", "revenue": "12500000000"},
            ],
            "opportunities": [
                {
                    "title": "Green hydrogen corridor",
                    "description": "Anchor offtake for NEOM-linked hydrogen export.",
                    "value": 2_000_000_000,
                },
            ],
        },
    }]
    out = db.render_company_brief(rows)
    assert "25%" in out
    assert "Green hydrogen corridor" in out
    assert "$12.5B" in out or "12.5" in out


def test_rows_from_correlator_summary_folds_sections():
    summary = {
        "primary": {"company_name": "Apple Inc", "sector": "ICT"},
        "geographic_revenues": [
            {"region": "North America", "percentage": "55%", "revenue": "1"},
        ],
        "opportunities": [
            {"title": "Gaming hub", "description": "$32B sports economy"},
        ],
        "financial_performances": [{"year": 2025, "total_revenue": 391e9}],
        "executives": [{"name": "Tim Cook", "position": "CEO"}],
    }
    rows = db.rows_from_correlator_summary(summary)
    assert rows[0]["geographic_revenue"][0]["percentage"] == "55%"
    assert rows[0]["match_opportunities"][0]["title"] == "Gaming hub"
    assert rows[1]["full_name"] == "Tim Cook"
    assert rows[1]["title"] == "CEO"


def test_ceo_brief_stays_person_focused(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")
    rows = [{
        "company_name": "Apple Inc.",
        "sector": "ICT",
        "revenue_usd": 391_000_000_000,
        "number_of_employees": 164000,
        "number_of_employees_ksa": 1800,
        "number_of_employees_mena": 5600,
        "company_description": "Apple designs consumer electronics and services.",
        "is_rhq": False,
        "potential_strategic_opportunity": (
            "Saudi gaming & esports hub: $32B sports economy target by 2030."
        ),
        "key_metrics": {"sector": "ICT", "ksa_employees": 1800, "mena_employees": 5600},
    }, {
        "full_name": "Tim Cook",
        "title": "CEO",
        "company_profile_id": 1,
        "tenure": "Current",
        "key_contribution": "Led Apple through services expansion.",
    }]
    out = db.render_db_briefing(
        rows, intent="executive_lookup", table="company_executives",
        user_question="Who is the CEO of Apple?",
    )
    assert out is not None
    assert "## Role" in out
    assert "## Background" in out
    assert "## Strategic Context" in out
    assert "Tim Cook" in out
    assert "CEO" in out
    assert "1,800" in out or "ICT" in out or "services expansion" in out
    assert "Strategic Read" in out
    assert "Recommended Next Actions for MISA" in out
    assert "Corporate Profile" not in out
    assert "Snapshot of Operations" not in out and "Operational Detail" not in out


def test_hybrid_person_gets_public_background(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")
    from app.services.hybrid_briefing import enrich_db_briefing

    person = (
        "## Role\n\n**Tim Cook is CEO at Apple Inc.**\n\n"
        "* **Position:** CEO\n\n"
        "## Background\n\n* On file as **CEO** at **Apple Inc**.\n"
    )

    monkeypatch.setattr(
        "app.services.hybrid_briefing._person_public_background",
        lambda *_a, **_k: (
            "## Background\n\n"
            "* Became CEO in August 2011, succeeding Steve Jobs.\n"
            "* Previously served as Apple's COO.\n"
            "* Holds an MBA from Duke University.\n"
        ),
    )
    monkeypatch.setattr(
        "app.services.hybrid_briefing._doc_section",
        lambda *_a, **_k: ("", []),
    )
    out = enrich_db_briefing(person, "Who is the CEO of Apple?")
    assert "## Role" in out["answer"]
    assert "Tim Cook" in out["answer"]
    assert "Duke University" in out["answer"]
    assert "August 2011" in out["answer"]
    assert "John Ternus" not in out["answer"]
    assert "From the web" not in out["answer"]


def test_who_is_ceo_overrides_company_intent(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "deterministic")
    rows = [
        dict(APPLE_NESTED),
        {"full_name": "Tim Cook", "title": "CEO", "company_profile_id": 1, "tenure": "Current"},
    ]
    out = db.render_db_briefing(
        rows, intent="company_profile", table="company_profiles",
        user_question="Who is the CEO of Apple?",
    )
    assert "## Role" in out
    assert "Tim Cook" in out
    assert "Snapshot of Operations" not in out and "Operational Detail" not in out

def test_hybrid_strips_empty_source_lanes(monkeypatch):
    from app.services.hybrid_briefing import enrich_db_briefing

    core = (
        "## Acme — Executive Briefing\n\nBlurb.\n\n"
        "## 🇸🇦 Strategic Read\n\n* Do the thing.\n"
    )
    monkeypatch.setattr(
        "app.services.hybrid_briefing._doc_section",
        lambda *_a, **_k: (
            "## From your documents\n\n_No relevant document excerpts._\n",
            [],
        ),
    )
    monkeypatch.setattr(
        "app.services.hybrid_briefing._web_section",
        lambda *_a, **_k: ("", []),
    )
    out = enrich_db_briefing(core, "tell me about Acme", include_web=False)
    assert "From your documents" not in out["answer"]


def test_ollama_mode_disables_renderer(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "ollama")
    monkeypatch.setattr("app.config.RESIDENCY_STRICT", True)
    from app.services.db_briefing import use_deterministic_db_briefing
    assert use_deterministic_db_briefing() is False


def test_auto_prefers_narrative_cloud_over_templates(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "auto")
    monkeypatch.setattr("app.config.RESIDENCY_STRICT", True)
    monkeypatch.setattr("app.config.NARRATIVE_CLOUD_ENABLED", True)
    monkeypatch.setattr("app.config.DATA_LLM_BACKEND", "ollama")
    from app.services.db_briefing import use_deterministic_db_briefing
    assert use_deterministic_db_briefing() is False


def test_auto_templates_when_narrative_cloud_off(monkeypatch):
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "auto")
    monkeypatch.setattr("app.config.RESIDENCY_STRICT", True)
    monkeypatch.setattr("app.config.NARRATIVE_CLOUD_ENABLED", False)
    monkeypatch.setattr("app.config.DATA_LLM_BACKEND", "ollama")
    from app.services.db_briefing import use_deterministic_db_briefing
    assert use_deterministic_db_briefing() is True


def test_force_renders_when_narrative_preferred(monkeypatch):
    """Fallback templates must still work with force=True."""
    monkeypatch.setattr("app.config.DB_BRIEFING_MODE", "auto")
    monkeypatch.setattr("app.config.NARRATIVE_CLOUD_ENABLED", True)
    from app.services import db_briefing as db

    rows = [{
        "company_name": "Acme Corp",
        "misa_details": {
            "general": {"company_name": "Acme Corp", "hq_country": "USA"},
            "mena_details": {},
        },
        "_related": {},
    }]
    assert db.render_db_briefing(rows, user_question="tell me about Acme") is None
    forced = db.render_db_briefing(
        rows, user_question="tell me about Acme", force=True,
    )
    assert forced
    assert "Acme" in forced
