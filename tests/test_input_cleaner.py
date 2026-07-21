"""Tests for input_cleaner preprocessing and entity extraction."""

from app.services.input_cleaner import clean_user_question


def test_dentons_messy_question_extracts_entity():
    p = clean_user_question(r"whats tthis \Dentons Uk And Middle East Llp")
    assert p["entity_candidate"] == "Dentons Uk And Middle East Llp"
    assert "Dentons" in p["cleaned"]
    assert p["raw"].startswith("whats")


def test_tell_me_about_alphabet():
    p = clean_user_question("tell me about Alphabet")
    assert p["entity_candidate"] == "Alphabet"
    assert p["cleaned"] == "Alphabet"


def test_raw_preserved():
    p = clean_user_question("  what's  Xyz  ")
    assert p["raw"] == "  what's  Xyz  "


def test_quoted_entity():
    p = clean_user_question('Who is "Acme Corp Holdings" anyway')
    assert p["entity_candidate"] == "Acme Corp Holdings"


def test_company_profiles_says_about_extracts_entity():
    p = clean_user_question("What does company profiles says about Dragos Security Llc?")
    assert p["cleaned"] == "Dragos Security Llc"
    assert p["entity_candidate"] == "Dragos Security Llc"


def test_company_profiles_underscore_say_about_extracts_entity():
    p = clean_user_question("What does company_profiles say about Alphabet, Inc.?")
    assert p["cleaned"] == "Alphabet, Inc"
    assert p["entity_candidate"] == "Alphabet, Inc"


def test_aplhabet_typo_normalizes_for_search():
    p = clean_user_question("WHATS APLHABET CMPANY")
    assert "alphabet" in (p.get("entity_candidate") or "").lower()


def test_schema_browse_preset_is_not_entity():
    p = clean_user_question(
        "Companies with ultimate_parent_company mentioning a known holding name?"
    )
    assert p["entity_candidate"] is None
    assert "ultimate_parent_company" in p["cleaned"]


# Regression: trailing legal/corporate suffixes on SHORT (≤3-token) entities
# must be stripped so the bare name still matches canonical company names
# like "Apple, Inc.". Longer multi-word legal names (Dentons UK And Middle
# East LLP — covered by test_dentons_messy_question_extracts_entity above)
# must keep the suffix.
def test_short_entity_strips_trailing_company():
    assert clean_user_question("what is apple company")["entity_candidate"] == "apple"

def test_short_entity_strips_trailing_inc():
    assert clean_user_question("tell me about apple inc")["entity_candidate"] == "apple"

def test_short_entity_strips_trailing_ltd():
    assert clean_user_question("describe google ltd")["entity_candidate"] == "google"

def test_three_token_strips_trailing_inc():
    assert clean_user_question("Berkshire Hathaway Inc")["entity_candidate"] == "Berkshire Hathaway"


# Regression: leading-filler patterns the old regex missed (the cleaner was
# only stripping "details ON" and "tell me about"; common phrasings like
# "details about X" / "show me X" / "info about X" / "more about X" left the
# entity polluted with the filler words).
def test_leading_details_about():
    assert clean_user_question("details about Amazon")["entity_candidate"] == "Amazon"

def test_leading_show_me():
    assert clean_user_question("show me Apple")["entity_candidate"] == "Apple"

def test_leading_info_about():
    assert clean_user_question("info about Apple")["entity_candidate"] == "Apple"

def test_leading_more_about():
    assert clean_user_question("more about Apple")["entity_candidate"] == "Apple"

def test_leading_search_bare():
    assert clean_user_question("search Apple")["entity_candidate"] == "Apple"


# Regression: stray punctuation that previously polluted the entity
def test_stray_quotes_inside_token_are_stripped():
    p = clean_user_question("apple'; DROP TABLE")
    assert p["entity_candidate"] is not None
    assert ";" not in p["entity_candidate"]
    assert "'" not in p["entity_candidate"]
