"""Officeholder / succession heuristics for executive lookup."""

from app.services.chat_engine import (
    _is_current_officeholder_question,
    _is_forward_looking_exec_question,
)


def test_minister_of_investment_is_officeholder():
    assert _is_current_officeholder_question(
        "Who is the Minister of Investment of Saudi Arabia?"
    )
    assert _is_current_officeholder_question(
        "Who is the current Saudi Minister of Investment?"
    )


def test_named_person_bio_is_not_officeholder_gate():
    # Named-person bios can still be wrong if DB is stale, but the
    # officeholder gate is specifically for "who holds role X" asks.
    assert not _is_current_officeholder_question("Tell me about Khalid Al-Falih")


def test_corporate_ceo_is_not_officeholder_gate():
    """Regression: CEO-of-company asks must use MISA execs, not live web."""
    assert not _is_current_officeholder_question("Who is the CEO of Apple?")
    assert not _is_current_officeholder_question("Who is the CFO of Saudi Aramco?")
    assert not _is_current_officeholder_question("Who chairs Saudi Aramco?")


def test_succession_still_forward_looking():
    assert _is_forward_looking_exec_question("Who will replace Tim Cook as CEO?")
    assert not _is_forward_looking_exec_question(
        "Who is the Minister of Investment of Saudi Arabia?"
    )
