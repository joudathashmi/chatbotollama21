"""QA-style scenarios: guards, truncation, noise filters, ranking fallbacks."""

import pandas as pd

from app import database as mc
from app.services import chat_engine as ce
from app.services.input_cleaner import clean_user_question, looks_like_schema_browse_question


def test_schema_browse_question_detection():
    assert looks_like_schema_browse_question(
        "Companies with ultimate_parent_company mentioning a known holding name?"
    )
    assert not looks_like_schema_browse_question("What is Alphabet?")


def test_forced_retrieval_skipped_for_schema_preset_without_entity():
    q = "Companies with ultimate_parent_company mentioning a known holding name?"
    pack = clean_user_question(q)
    assert pack["entity_candidate"] is None
    assert not ce._likely_rhq_company_lookup(q, pack)


def test_schema_browse_with_quoted_company_still_allows_forced_lookup():
    q = 'Tell me about "Alphabet" where ultimate_parent_company is not null'
    pack = clean_user_question(q)
    assert pack.get("entity_candidate")
    assert ce._likely_rhq_company_lookup(q, pack)


def test_search_terms_drop_schema_tokens():
    q = "Companies with ultimate_parent_company mentioning holding"
    terms = ce._search_terms_from_question(q)
    assert "ultimate_parent_company" not in terms
    assert "rhq_city" not in ce._search_terms_from_question("filter rhq_city ILIKE riyadh")


def test_truncate_for_llm_keeps_short():
    assert ce._truncate_for_llm("hello") == "hello"


def test_truncate_for_llm_long():
    s = "x" * 20_000
    out = ce._truncate_for_llm(s, max_chars=100)
    assert len(out) <= 100
    assert "truncated" in out.lower()


def test_filter_rank_no_latin_primary_returns_unmodified_head():
    df = pd.DataFrame(
        [
            {"company_name": "شركة محلية", "ultimate_parent_company": "", "company_profile": ""},
        ]
    )
    terms = ["شركة", "محلية"]
    out = mc._filter_rank_smart_search_df(df, terms, 5)
    assert len(out) == 1


def test_parse_tool_arguments_fenced_json():
    raw = '```json\n{"filters": {"company_name": "x"}, "limit": 5}\n```'
    d = ce._parse_tool_arguments(raw)
    assert d["limit"] == 5
    assert d["filters"]["company_name"] == "x"


def test_parse_tool_arguments_invalid_returns_empty():
    assert ce._parse_tool_arguments("not json {") == {}


def test_db_failure_smart_search_returns_empty_df(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(mc, "get_db", boom)
    df, sql, params = mc.run_rhq_company_smart_search(["Alphabet"], 10)
    assert df.empty
    assert "failed" in (sql or "").lower()
    assert params == []
