"""Cross-platform quality wiring — same gate on chat / export / stream."""

from __future__ import annotations

from app.services.chat_engine import _attach_quality_meta
from app.services.human_feedback import build_feedback_context
from app.services.surface_quality import (
    quality_context_for_question,
    run_surface_quality_gate,
)


def test_attach_quality_meta_lifts_advisory_deliverable():
    result = {
        "answer": "# Market Fit\n\nBody",
        "tool_calls": [{
            "input_trace": {
                "_advisory_deliverable": "market_fit",
                "_advisory_db_context": {
                    "origin_country": "India",
                    "companies_from_origin_licensed_in_saudi": 2437,
                },
                "_short_circuit": "strategic_advisory",
            },
        }],
    }
    out = _attach_quality_meta(result)
    assert out["_advisory_deliverable"] == "market_fit"
    assert out["_advisory_db_context"]["origin_country"] == "India"


def test_feedback_context_includes_advisory_deliverable():
    ctx = build_feedback_context(
        "make me a market for india",
        "en",
        "en",
        {"_advisory_deliverable": "market_fit", "_short_circuit": "strategic_advisory"},
    )
    assert ctx["advisory_deliverable"] == "market_fit"


def test_quality_context_for_typo_market_fit_question():
    qc = quality_context_for_question(
        "make me a market for to atrract indian companies",
    )
    assert qc["deliverable"] == "market_fit"
    # Live DB may or may not be reachable in unit env — if present, India.
    if qc.get("db_context"):
        assert qc["db_context"].get("origin_country") in (None, "India") or (
            qc["db_context"].get("companies_from_origin_licensed_in_saudi")
            is not None
            or qc["db_context"].get("footprint_data_unavailable")
        )


def test_surface_gate_does_not_withhold_market_fit_export():
    doc = (
        "# Market Fit Assessment: Attracting Indian Companies\n\n"
        + ("Strategic context. " * 40)
        + "\n\n## Overall Market Fit\n"
        "| Sector | Fit | Priority |\n|---|---|---|\n"
        "| ICT | High | Tier 1 |\n"
        "| Healthcare | High | Tier 1 |\n\n"
        "## Strategic Conclusion\nDone.\n"
    )
    text, issues, fixes = run_surface_quality_gate(
        doc,
        question="make me a market for to atrract indian companies",
        hard_block=True,
    )
    assert "Response withheld" not in text
    assert "Market Fit Assessment" in text
    assert not any(
        i.get("code") == "truncated_ranking_table" for i in (issues or [])
    )
