"""Tests for deterministic commentary."""

from app.services.commentary import generate_commentary


def test_single_row_alphabet_mentions_profile_revenue():
    row = {
        "id": 1320,
        "company_name": "Alphabet, Inc.",
        "internal_code": "ALPH-01",
        "year_founded": "2015",
        "company_profile": (
            "Technology holding company created in 2015 as the parent of Google, "
            "headquartered in Mountain View, California."
        ),
        "global_headquarters": "Mountain View, California",
        "revenue_usd": 307_160_000_000,
        "number_of_employees": 150_000,
        "sector": "Technology",
    }
    out = generate_commentary([row], "company_profiles", "What is Alphabet?")
    assert "Alphabet" in out
    assert "2015" in out
    assert "Mountain View" in out
    assert "USD 307.16 billion" in out
    lowered = out.lower()
    assert "n/a" not in lowered
    assert "null" not in lowered
    assert "unknown" not in lowered


def test_three_rows_lead_and_bullets():
    rows = [
        {"id": 1, "company_name": "Acme A", "sector": "Tech", "revenue_usd": 1.2e9},
        {"id": 2, "company_name": "Acme B", "sector": "Tech", "revenue_usd": 900e6},
        {"id": 3, "company_name": "Acme C", "sector": "Finance", "revenue_usd": None},
    ]
    out = generate_commentary(rows, "company_profiles", "list acme")
    assert "matched" in out and "3" in out and "companies" in out
    assert "1." in out and "2." in out and "3." in out


def test_all_null_row_no_na_words():
    row = {
        "id": 99,
        "company_name": None,
        "internal_code": None,
        "company_profile": None,
        "sector": None,
        "revenue_usd": None,
        "number_of_employees": None,
    }
    out = generate_commentary([row], "company_profiles", "something")
    lowered = out.lower()
    assert "n/a" not in lowered
    assert "null" not in lowered
    assert "unknown" not in lowered


def test_multi_row_omits_placeholder_zero_city():
    rows = [
        {
            "id": 1,
            "company_name": "TestCo",
            "sector": "Tech",
            "rhq_city": "0",
            "global_headquarters": "USA",
            "revenue_usd": 1e6,
        }
    ]
    out = generate_commentary(rows, "company_profiles", "x")
    assert "HQ city **0**" not in out
    assert "TestCo" in out
