"""Deep-profile triggers must stay explicit — not Jul21 briefing language."""

from __future__ import annotations

from app.services.answer_contracts import classify_brief_kind
from app.services.db_briefing import _question_looks_like_person
from app.services.deep_profile import is_deep_profile_request
from app.services.jul21_surface import (
    apply_jul21_surface_polish,
    enrich_entity_brief_depth,
)


def test_deep_profile_only_explicit_opt_in():
    assert is_deep_profile_request("/profile Apple") == "Apple"
    assert is_deep_profile_request("deep profile of Apple") == "Apple"
    assert is_deep_profile_request("  /profile Siemens") == "Siemens"


def test_natural_briefing_profile_not_deep_profile():
    for q in (
        "give me a briefing on Siemens",
        "company profile for Toyota",
        "company briefing for Pfizer",
        "profile of Unilever",
        "briefing on ABB",
        "company profile for Hitachi",
        "company briefing for Microsoft",
        "briefing on Amazon",
        "tell me about Apple",
        "CEO profile for Pfizer",
        "sector briefing for healthcare and pharma in KSA",
        "executive briefing on Apple",
        "investment case for Microsoft",
    ):
        assert is_deep_profile_request(q) is None, q


def test_person_question_detector_covers_live_fails():
    for q in (
        "CEO profile for Pfizer",
        "who runs Nestle",
        "who is the CEO of Ericsson",
        "CEO of Schneider Electric",
        "tell me about the CEO of Huawei",
    ):
        assert _question_looks_like_person(q), q
        assert classify_brief_kind(
            "Apple Inc — Executive Briefing\n\nthin",
            intent="executive_lookup",
            user_question=q,
        ) == "person"
        assert classify_brief_kind(
            "some prose without role",
            user_question=q,
        ) == "person"


def test_person_enrich_injects_missing_role():
    thin = (
        "## Background\n\n* Prior COO.\n\n"
        "## 🇸🇦 Strategic Read\n\n* Engage on RHQ.\n"
    )
    out, fixes = enrich_entity_brief_depth(
        thin, intent="executive_lookup",
        user_question="who is the CEO of Acme",
    )
    assert "## Role" in out
    assert "injected_person_role" in fixes
    assert "Strategic Context" in out
    assert "Recommended Next Actions" in out


def test_company_briefing_ask_not_person_enriched():
    thin = (
        "## Background\n\nPfizer is a pharma company.\n\n"
        "### Strategic Read\n\n- Expand vaccines.\n"
    )
    out, fixes = enrich_entity_brief_depth(
        thin,
        intent="company_profile",
        user_question="company briefing for Pfizer",
    )
    assert "## Role" not in out
    assert "injected_person_role" not in fixes
    assert classify_brief_kind(
        out,
        intent="company_profile",
        user_question="company briefing for Pfizer",
    ) == "company"


def test_deep_profile_uses_determinism_kw():
    src = open("app/services/deep_profile.py").read()
    assert "openai_determinism_kw" in src
    assert "temperature=0.2" not in src


def test_response_cache_defaults_on():
    import os
    from app.services import chat_engine as ce
    # Default when env unset
    prev = os.environ.pop("MISA_RESPONSE_CACHE", None)
    try:
        enabled, ttl, _ = ce._response_cache_settings()
        assert enabled is True
        assert ttl >= 60
    finally:
        if prev is not None:
            os.environ["MISA_RESPONSE_CACHE"] = prev


def test_sector_enrich_injects_strategic_and_recommended():
    thin = (
        "## Healthcare — Sector Brief\n\n"
        "Saudi pharma demand is rising.\n\n"
        "### Outlook\n\n- Localisation.\n"
    )
    out, fixes = enrich_entity_brief_depth(
        thin, intent="sector_lookup",
    )
    assert "Strategic Context" in out
    assert "Recommended Next Actions" in out
    assert fixes

    polished, pfixes = apply_jul21_surface_polish(
        "## Mining sector overview in Saudi\n\nThin body.\n",
        question="mining and minerals sector briefing Saudi",
        pack={"_intent": "sector_lookup"},
    )
    assert "Strategic Context" in polished
    assert "Recommended" in polished
    assert pfixes
