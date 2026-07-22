"""Prompt masking — secrets/PII only; must not alter business facts."""

from __future__ import annotations

from app.services.prompt_masking import (
    mask_messages_for_llm,
    mask_text,
    scrub_system_prompt_leak,
)


def test_masks_email_and_api_key_not_company_facts():
    raw = (
        "Contact Jane at jane.doe@example.com about Apple Inc. "
        "There are 95,671 licensed companies. "
        "key=sk-abcdefghijklmnopqrstuvwxyz012345"
    )
    out = mask_text(raw)
    assert "[EMAIL_MASKED]" in out
    assert "Apple Inc" in out
    assert "95,671" in out
    assert "sk-abc" not in out
    assert "[API_KEY_MASKED]" in out


def test_does_not_mask_bare_revenue_digits():
    raw = "Annual revenue_usd 391000000000 and Vision 2030."
    out = mask_text(raw)
    assert "391000000000" in out
    assert "2030" in out


def test_masks_labeled_phone_only():
    raw = "Phone: +966 11 234 5678 and RHQ count is 727"
    out = mask_text(raw)
    assert "PHONE_MASKED" in out
    assert "727" in out


def test_system_messages_not_masked():
    msgs = [
        {"role": "system", "content": "You are a helper. password=secret123"},
        {"role": "user", "content": "email me at a@b.com"},
    ]
    out = mask_messages_for_llm(msgs)
    assert out[0]["content"] == msgs[0]["content"]
    assert "[EMAIL_MASKED]" in out[1]["content"]


def test_scrub_system_prompt_leak():
    ans = (
        "## Snapshot\n"
        "You are a senior investment-promotion strategist for MISA.\n"
        "India has many licensed firms.\n"
    )
    scrubbed = scrub_system_prompt_leak(ans)
    assert "senior investment-promotion" not in scrubbed
    assert "India has many licensed firms" in scrubbed


def test_password_assignment_masked():
    assert "[REDACTED]" in mask_text("password=hunter2extra")
