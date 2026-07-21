"""Unit tests for SQL / row guardrails in the chat engine."""

from unittest.mock import MagicMock, patch

import pandas as pd

from app.database import _filter_rank_smart_search_df
from app.services import chat_engine as mc
from app.services.commentary import generate_commentary


def test_sql_covers_entity_requires_where_and_name_column():
    assert not mc._sql_covers_entity(
        "SELECT * FROM company_profiles ORDER BY id", [], "Dentons Uk"
    )
    assert not mc._sql_covers_entity(
        "SELECT * FROM company_profiles WHERE revenue_usd > %s", [1], "Dentons Uk"
    )


def test_sql_covers_entity_accepts_ilike_on_company_name():
    sql = "SELECT * FROM company_profiles WHERE company_name ILIKE %s LIMIT 25"
    params = ["%Dentons%"]
    assert mc._sql_covers_entity(sql, params, "Dentons Uk And Middle East Llp")


def test_rows_match_entity_substring():
    rows = [
        {"company_name": "Dentons UK LLP", "ultimate_parent_company": "", "company_profile": ""},
    ]
    assert mc._rows_match_entity(rows, "Dentons Uk And Middle East Llp")


def test_search_terms_add_latin_when_entity_is_arabic_alphabet():
    pack = {"entity_candidate": "ألفابت", "cleaned": "ألفابت", "raw": ""}
    terms = mc._search_terms_for_pack(pack, "ما هي شركة ألفابت؟")
    assert "ألفابت" in terms
    assert "Alphabet" in terms


def test_rows_match_arabic_alphabet_entity_against_latin_row():
    rows = [
        {
            "company_name": "Alphabet, Inc.",
            "ultimate_parent_company": "",
            "company_profile": "",
        }
    ]
    assert mc._rows_match_entity(rows, "ألفابت")


def test_sql_covers_entity_accepts_latin_param_for_arabic_entity():
    sql = (
        "SELECT * FROM company_profiles WHERE (company_name ILIKE %s OR "
        "company_profile ILIKE %s) LIMIT 25"
    )
    params = ["%Alphabet%", "%Alphabet%"]
    assert mc._sql_covers_entity(sql, params, "ألفابت")


def test_smart_search_ranking_prefers_legal_name_over_subsidiary():
    df = pd.DataFrame(
        [
            {
                "company_name": "Mandiant, Inc.",
                "ultimate_parent_company": "Some Alphabet mention",
                "company_profile": "",
            },
            {
                "company_name": "Alphabet, Inc.",
                "ultimate_parent_company": "",
                "company_profile": "Technology holding company.",
            },
        ]
    )
    terms = ["ألفابت", "Alphabet"]
    out = _filter_rank_smart_search_df(df, terms, 5)
    assert len(out) == 1
    assert "Alphabet" in (out.iloc[0]["company_name"] or "")


def test_rows_match_entity_false_for_unrelated():
    rows = [
        {
            "company_name": "Climachill Ltd",
            "ultimate_parent_company": "",
            "company_profile": "",
        },
    ]
    assert not mc._rows_match_entity(rows, "Dentons")


def test_entity_requires_sql_constraint_multiword():
    assert mc._entity_requires_sql_constraint("Dentons Uk")
    assert mc._entity_requires_sql_constraint("Alphabetical")
    assert not mc._entity_requires_sql_constraint("Uk")
    assert not mc._entity_requires_sql_constraint(None)


def test_commentary_refuses_unrelated_rows_when_sanity_fails():
    rows = [
        {
            "company_name": "Wrong Co",
            "ultimate_parent_company": "",
            "company_profile": "",
        }
    ]
    out = generate_commentary(
        rows,
        "company_profiles",
        "what about Dentons",
        entity_candidate="Dentons",
        entity_lookup_required=True,
        row_entity_sanity_passed=False,
        closest_names=["Near1", "Near2", "Near3"],
    )
    assert "couldn't find" in out.lower() or "i couldn't find" in out.lower()
    assert "Wrong Co" not in out


def test_commentary_zero_rows_entity_plain():
    out = generate_commentary(
        [],
        "company_profiles",
        "Dentons?",
        entity_candidate="Dentons",
        entity_lookup_required=True,
    )
    assert "No companies matching" in out
    assert "Dentons" in out


@patch.object(mc, "get_db")
def test_top_similar_company_names_uses_difflib(mock_get_db):
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    conn.cursor.return_value.__exit__.return_value = False
    mock_get_db.return_value = conn
    cur.fetchall.return_value = [
        ("Dentens Legal",),
        ("Duntons LLC",),
        ("Other PLC",),
    ]
    got = mc._top_similar_company_names("Dentons", k=3)
    assert len(got) <= 3
    assert any("Dent" in x or "Dunton" in x for x in got)


def test_likely_rhq_lookup_when_entity_present():
    pack = {
        "raw": "q",
        "cleaned": "x",
        "entity_candidate": "Alphabet",
    }
    assert mc._likely_rhq_company_lookup("anything about cos", pack)


def test_not_likely_lookup_for_short_greeting():
    pack = {"raw": "hi", "cleaned": "hi", "entity_candidate": None}
    assert not mc._likely_rhq_company_lookup("Hi", pack)


# Regression: when the full-entity primary fails to match any row exactly,
# the ranker must fall back to the most distinctive single-word term so we
# pick the real entity instead of the longest-named noise row.
def test_filter_rank_falls_back_to_distinctive_term():
    from app.database import _filter_rank_smart_search_df
    df = pd.DataFrame([
        {"company_name": "Some Long Construction And Trade Joint Stock Company",
         "ultimate_parent_company": "", "company_profile": ""},
        {"company_name": "Dentons Uk And Middle East Llp",
         "ultimate_parent_company": "", "company_profile": ""},
    ])
    # The full primary "Dentons UK And Middle East LLP" isn't a contiguous
    # substring of either row, so initial scores are 0. The fallback
    # distinctive term "Dentons" should surface the Dentons row.
    terms = ["Dentons UK And Middle East LLP", "Dentons", "Middle", "East", "LLP"]
    out = _filter_rank_smart_search_df(df, terms, 5)
    assert len(out) >= 1
    assert "Dentons" in (out.iloc[0]["company_name"] or "")


def test_search_terms_drop_english_stopwords_from_entity():
    """Common stopwords like 'and' inside a multi-word entity must NOT be
    issued as standalone search terms — they ILIKE-match tens of thousands
    of unrelated rows and drown out the real target in the fetch window."""
    pack = {"raw": "x", "cleaned": "x",
            "entity_candidate": "Dentons UK And Middle East LLP"}
    terms = mc._search_terms_for_pack(pack, "x")
    assert "And" not in terms and "and" not in terms
    assert "Dentons" in terms
