"""Systemic quality-hardening regression tests (failure classes)."""

from __future__ import annotations

from app.services.quality_gate import detect_quality_issues, run_quality_gate
from app.services.quality_eval import evaluate_answer
from app.services.retrieval_status import (
    RetrievalStatus,
    failure,
    success_counts,
    user_facing_retrieval_message,
)
from app.services.source_policy import source_policy_system_addon
from app.prompts.chat_system import advisory_system_prompt


def test_retrieval_failure_never_looks_like_verified_zero():
    rr = failure(
        RetrievalStatus.CONNECTION_ERROR,
        source_name="company_profiles",
        error="connection refused",
    )
    assert rr.is_failure
    assert rr.to_context_dict()["do_not_claim_zero"] is True
    msg = user_facing_retrieval_message(rr)
    assert "not" in msg.lower() and "zero" in msg.lower()
    assert "0 verified" not in msg.lower() or "not a verified zero" in msg.lower()


def test_verified_empty_is_labelled_with_source():
    rr = success_counts(
        source_name="company_profiles",
        count=0,
        filters={"origin_country": "Atlantis"},
    )
    assert rr.is_verified_empty
    msg = user_facing_retrieval_message(rr)
    assert "0" in msg and "Atlantis" in msg or "filters" in msg.lower()


def test_quality_gate_keeps_long_answer_on_truncation_alone():
    """Truncated ranking must soft-repair — never wipe into a withhold stub."""
    # Long enough that soft-trim / keep-long paths engage.
    preamble = (
        "## Priority Company Ranking\n\n"
        "Strategic targeting brief for India-origin companies already "
        "licensed in the Kingdom, with conversion and expansion angles.\n\n"
    ) + ("Context paragraph. " * 40) + "\n\n"
    rows = "\n".join(
        f"| {i} | Company {i} | Sector | High | Thesis {i} | Next step |"
        for i in range(1, 12)
    )
    body = (
        preamble
        + "| Rank | Company | Sector | Priority | Thesis | Action |\n"
        + "|---|---|---|---|---|---|\n"
        + rows
        + "\n| 12 | Company 12 | ICT"  # truncated mid-row
    )
    text, _issues, fixes = run_quality_gate(
        body,
        question="best Indian companies to target",
        db_context={"origin_country": "India", "licensed_count": 2437},
        hard_block=True,
    )
    assert "Response withheld" not in text
    assert "Priority Company Ranking" in text
    assert len(text) > 400
    joined = " ".join(fixes)
    assert any(
        f in joined
        for f in (
            "trimmed_truncated_table_row",
            "repaired_truncation_kept_answer",
            "kept_long_answer_despite_critical",
        )
    )


def test_market_fit_incomplete_pipe_not_flagged_as_ranking_truncation():
    """Market-fit sector tables must not trip ranking-truncation hard-block."""
    body = (
        "## Market Fit Briefing: Attracting Indian Companies\n\n"
        + ("Opportunity narrative. " * 50)
        + "\n\n| Rank | Sector | Why |\n|---|---|---|\n"
        "| 1 | ICT | Digital |\n| 2 | Health |"
    )
    issues = detect_quality_issues(
        body,
        question="make me a market for to atrract indian companies",
        db_context={"origin_country": "India"},
    )
    assert not any(i["code"] == "truncated_ranking_table" for i in issues)


def test_quality_gate_blocks_false_zero_when_unavailable():
    bad = (
        "# Plan\n\nThere are zero Indian companies licensed in Saudi Arabia.\n"
    )
    fixed, issues, fixes = run_quality_gate(
        bad,
        question="best companies from India",
        db_context={"footprint_data_unavailable": True, "retrieval_status": "error"},
    )
    assert any(i["code"] == "false_zero_on_retrieval_failure" for i in issues) or fixes
    assert "not a verified zero" in fixed.lower() or "unavailable" in fixed.lower()


def test_quality_gate_retitles_licensing_snapshot():
    bad = (
        "## Saudi RHQ & Licensing — Snapshot\n\n"
        "**95,671 companies hold an active MISA licence**.\n"
    )
    fixed, issues, fixes = run_quality_gate(
        bad, question="Tell me the active MISA licenses",
    )
    assert "retitled_licensing_snapshot" in fixes or fixed.startswith(
        "## Licensing Snapshot"
    )
    assert "Licensing Snapshot" in fixed


def test_quality_gate_flags_contradiction_with_db_counts():
    issues = detect_quality_issues(
        "There are zero licensed companies from India.",
        db_context={
            "companies_from_origin_licensed_in_saudi": 2437,
            "companies_from_origin_with_rhq": 14,
            "retrieval_status": "ok",
        },
        question="how many Indian licensed companies",
    )
    assert any(
        i["code"] == "contradicts_internal_licensed_count" for i in issues
    )


def test_source_policy_injected_into_advisory_prompt():
    p = advisory_system_prompt("en", "company_targeting")
    assert "SOURCE HIERARCHY" in p
    assert "SUCCESS_EMPTY" in p or "zero_records" in p or "ERROR" in p
    assert "Requires validation" in p or "REQUIRES_VALIDATION" in p
    addon = source_policy_system_addon()
    assert "Priority" in addon or "1." in addon


def test_evaluator_fails_critical_false_zero():
    result = evaluate_answer(
        "zero companies licensed",
        question="licenses",
        db_context={"footprint_data_unavailable": True},
    )
    # Either fails or scores low
    assert result["score"] < 100


def test_country_licensing_answer_never_zeros_on_db_error():
    from app.services import chat_engine as ce
    out = ce._format_country_licensing_answer(
        "India",
        {
            "total_licensed": 0,
            "total_rhq": 0,
            "_db_error": "column does not exist",
            "retrieval_status": "SCHEMA_MISMATCH",
            "retrieval": {"do_not_claim_zero": True},
        },
    )
    assert "0 hold an active" not in out.lower()
    assert "not" in out.lower() and "zero" in out.lower()
    assert "India" in out


def test_prompt_counts_prefer_company_profiles_over_rhq_licenses():
    from pathlib import Path
    from app.prompts import chat_system as cs
    src = Path(cs.__file__).read_text()
    assert "NEVER use `rhq_licenses`" in src
    assert "authoritative source is `company_profiles`" in src
    # Aggregate examples must count company_profiles, not rhq_licenses
    assert (
        'how many RHQ licenses do we have?":\n'
        '       query_table(table="company_profiles"'
        in src
        or 'table=\\"company_profiles\\"' in src
        or 'table="company_profiles", filters=\n'
           '{"is_rhq"' in src
        or '{"is_rhq": {"op": "=", "value": true}}' in src
    )
    # CODE enforcer — not prompt-only (tool path rewrite)
    from app.services.source_policy import rewrite_aggregate_licensing_query
    t, f, n = rewrite_aggregate_licensing_query(
        "rhq_licenses", count_only=True, question="total licenses",
    )
    assert t == "company_profiles" and f.get("licensed") is True and n


def test_live_india_retrieval_status_ok():
    from app.services.engagement_data import fetch_country_saudi_investors
    stats = fetch_country_saudi_investors("India")
    assert not stats.get("_db_error"), stats.get("_db_error")
    assert stats.get("retrieval_status") in (
        "SUCCESS_WITH_RESULTS", "ok",
    ) or int(stats.get("total_licensed") or 0) > 0
    assert int(stats.get("total_licensed") or 0) > 1000
    assert int(stats.get("total_rhq") or 0) >= 5
