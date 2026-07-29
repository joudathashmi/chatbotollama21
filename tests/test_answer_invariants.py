"""Durable invariant battery for chat-answer quality.

This suite exists so the recurring answer-quality defects can never
silently come back. Each test locks ONE invariant of the output, tested
against the REAL code paths (the enrichment gate and the routing
detectors), not hand-crafted scenarios. If a future change re-introduces
a branch-local skip or a placeholder leak, CI fails here before deploy.

Defect classes locked (each was a real production bug):
  1. "this account" placeholder leaking into person briefs
  2. A heading welded onto the previous line ("library.## From the web")
  3. Person answers losing their ## Role section
  4. Forward-looking / succession questions must route to web verification
  5. Current-officeholder questions must route to web verification
  6. Plain current-CEO questions must NOT trigger succession web noise
  7. finalize_answer must be idempotent (safe on any path, any number
     of times)
"""

from __future__ import annotations

import pytest

from app.services.answer_finalize import finalize_answer
from app.services.jul21_surface import enrich_entity_brief_depth
from app.services.chat_engine import (
    _is_forward_looking_exec_question,
    _is_current_officeholder_question,
)


# ── 1. Person enrichment never leaks the "this account" placeholder ──────
def test_person_brief_never_leaks_this_account_placeholder():
    thin = (
        "## Role\n\n**Tim Cook is CEO at Apple Inc.**\n"
        "- Current tenure.\n\n## Background\n\n* Joined 1998.\n\n"
        "## 🇸🇦 Strategic Read\n\n* Engage on RHQ.\n"
    )
    out, _ = enrich_entity_brief_depth(thin, intent="executive_lookup")
    assert "this account" not in out.lower(), (
        "person brief leaked the 'this account' template placeholder"
    )
    # The company the person leads must be named instead.
    assert "Apple" in out


def test_person_brief_no_leak_with_plain_nonbold_role_lead():
    """The real pipeline often emits the Role lead as PLAIN text, not bold
    ('Tim Cook is the current CEO of Apple Inc.'). That format used to make
    name resolution fall back to the 'this account' placeholder and leak it
    into the Strategic Context. Lock the non-bold case explicitly."""
    real = (
        "## Role\n\n"
        "Tim Cook is the current CEO of Apple Inc.\n"
        "Apple's regional headquarters for the Middle East is in Dubai, UAE.\n\n"
        "## Background\n\n- Tim Cook succeeded Steve Jobs.\n\n"
        "## 🇸🇦 Strategic Read\n\n- Engage.\n"
    )
    out, _ = enrich_entity_brief_depth(real, intent="executive_lookup")
    assert "this account" not in out.lower()
    # The person is named, and the employer is clean (not welded to the
    # next line's "regional headquarters").
    assert "**Tim Cook**" in out
    assert "regional headquarters** on" not in out


def test_person_brief_frames_sections_on_company_not_person():
    thin = (
        "## Role\n\n**Satya Nadella is CEO at Microsoft Corp.**\n\n"
        "## Background\n\n* Long tenure.\n\n## 🇸🇦 Strategic Read\n\n* Engage.\n"
    )
    out, _ = enrich_entity_brief_depth(thin, intent="executive_lookup")
    # Must never call the PERSON an investment account.
    assert "Nadella is a priority account" not in out
    assert "Nadella** is a priority account" not in out


# ── 2. Headings are never welded onto the previous line ──────────────────
@pytest.mark.parametrize("glued", [
    "_Sources: document library.## From the web\n\n- Bar.",
    "End of section.### Next Section\n\nBody.",
    "Some text.## Strategic Read\n\n- Point.",
])
def test_gate_unglues_headings(glued):
    out = finalize_answer(glued, user_question="x", pack={})
    assert ".##" not in out and ".###" not in out, (
        "a heading was left welded to the previous line"
    )


# ── 3. Person answers keep their ## Role section through the gate ────────
def test_gate_preserves_role_section():
    person = (
        "## Role\n\n**Tim Cook is CEO at Apple Inc.**\n\n"
        "## Background\n\n- Joined 1998.\n"
    )
    out = finalize_answer(
        person, user_question="who is apples CEO",
        pack={"_intent": "executive_lookup"},
    )
    assert "## Role" in out


# ── 4/5/6. Web-verification routing detectors (the root-cause class) ──────
@pytest.mark.parametrize("q", [
    "who is the upcoming new ceo for apple",
    "who is the next CEO of Apple",
    "who is Tim Cook's successor",
    "who will replace Tim Cook as Apple CEO",
    "Apple's incoming CEO",
])
def test_forward_looking_exec_detected(q):
    assert _is_forward_looking_exec_question(q), (
        f"forward-looking succession phrasing not detected: {q!r}"
    )


@pytest.mark.parametrize("q", [
    "who is apples CEO",
    "tell me about Tim Cook",
    "who is the CEO of Aramco",
])
def test_plain_current_exec_not_forward_looking(q):
    assert not _is_forward_looking_exec_question(q), (
        f"plain current-CEO question wrongly flagged as succession: {q!r}"
    )


@pytest.mark.parametrize("q", [
    "who is the current Minister of Investment of Saudi Arabia",
    "who is the Minister of Investment",
])
def test_current_officeholder_detected(q):
    assert _is_current_officeholder_question(q), (
        f"current-officeholder question not detected: {q!r}"
    )


def test_private_company_ceo_is_not_officeholder():
    # A company CEO must stay on the MISA table, not the cabinet web path.
    assert not _is_current_officeholder_question("who is the CEO of Apple")


# ── 7. The single gate is idempotent ─────────────────────────────────────
@pytest.mark.parametrize("ans", [
    "## Role\n\n**Tim Cook is CEO at Apple Inc.**\n\n## Background\n\n- x.\n",
    "## Company — Executive Briefing\n\nText.\n\n## 🇸🇦 Strategic Read\n\n- y.\n",
    "Plain answer with no headings at all.",
])
def test_gate_does_not_duplicate_sections(ans):
    """The meaningful idempotency invariant: a second pass through the gate
    must not DUPLICATE any section or grow the content. (Cosmetic
    whitespace may differ; content must not.)"""
    import re as _re
    from collections import Counter

    once = finalize_answer(ans, user_question="q", pack={})
    twice = finalize_answer(once, user_question="q", pack={})

    def headings(s: str):
        return Counter(
            h.strip() for h in _re.findall(r"(?m)^#{2,4}\s+.+$", s)
        )

    h_once, h_twice = headings(once), headings(twice)
    # No heading may appear more times after a second pass.
    for head, n in h_twice.items():
        assert n <= max(1, h_once.get(head, 1)), (
            f"section duplicated on second pass: {head!r}"
        )
    # Content must not grow on a second pass (allow tiny whitespace delta).
    assert len(twice) <= len(once) + 4, (
        "content grew on a second finalize pass"
    )
