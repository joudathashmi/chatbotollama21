"""Audit P0 follow-ups: smart-search failure marker, no country→global
licensing fallthrough, web envelope, status vocabulary."""

from __future__ import annotations

import pandas as pd

from app.database import (
    run_rhq_company_smart_search,
    smart_search_failure_message,
    smart_search_retrieval_failed,
)
from app.services import chat_engine as ce
from app.services.web_search import search_with_status
from app.services.quality_gate import detect_quality_issues
from app.services.response_validator import _build_footprint_section
from app.services.retrieval_status import (
    classify_exception,
    failure,
    user_facing_retrieval_message,
)


def test_smart_search_marks_retrieval_failure(monkeypatch):
    import app.database as db

    def boom(*a, **k):
        raise ConnectionError("db down")

    monkeypatch.setattr(db, "_run_rhq_company_smart_search_impl", boom)
    df, sql, params = run_rhq_company_smart_search(["Acme"], 5)
    assert isinstance(df, pd.DataFrame) and df.empty
    assert smart_search_retrieval_failed(sql)
    assert "ConnectionError" in smart_search_failure_message(sql)


def test_forced_smart_search_propagates_failure(monkeypatch):
    def fake_search(terms, limit=25):
        return (
            pd.DataFrame(),
            "-- RETRIEVAL_FAILED company_profiles smart_search: boom",
            [],
        )

    monkeypatch.setattr(ce, "run_rhq_company_smart_search", fake_search)
    pack = {"entity_candidate": "Acme"}
    out = ce._forced_smart_search_tool_result("tell me about Acme", pack)
    assert out.get("_retrieval_failed")
    assert out.get("row_count") is None
    assert pack.get("_retrieval", {}).get("do_not_claim_zero")


def test_country_licensing_unavailable_message():
    rr = failure(
        classify_exception(RuntimeError("db down")),
        source_name="company_profiles.licensed/is_rhq",
        error="db down",
        filters={"origin_country": "India"},
    )
    msg = user_facing_retrieval_message(rr)
    assert "not" in msg.lower() and "zero" in msg.lower()


def test_web_search_with_status_unavailable(monkeypatch):
    import app.services.web_search as ws

    monkeypatch.setattr(ws, "get_public_openai_client", lambda: None)
    env = search_with_status("Saudi RHQ policy")
    assert env["do_not_claim_zero"] is True
    assert env["retrieval_status"] == "SOURCE_UNAVAILABLE"
    assert env["results"] == []


def test_status_vocab_source_unavailable_triggers_gate():
    issues = detect_quality_issues(
        "There are zero licensed companies from India.",
        db_context={
            "footprint_data_unavailable": True,
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "origin_country": "India",
        },
    )
    assert any(i["code"] == "false_zero_on_retrieval_failure" for i in issues)


def test_footprint_section_uses_success_empty():
    text = _build_footprint_section({
        "origin_country": "Atlantis",
        "retrieval_status": "SUCCESS_EMPTY",
        "companies_from_origin_licensed_in_saudi": 0,
        "companies_from_origin_with_rhq": 0,
        "retrieval_filters": {"source": "company_profiles"},
    })
    assert "0" in text
    assert "successful zero-result" in text.lower() or "verified" in text.lower()


def test_footprint_section_source_unavailable():
    text = _build_footprint_section({
        "origin_country": "India",
        "retrieval_status": "SOURCE_UNAVAILABLE",
        "footprint_data_unavailable": True,
        "_db_error": "timeout",
    })
    assert "not" in text.lower() and "zero" in text.lower()
