"""Systemic quality-hardening regression suite (failure classes).

Covers intent, safe fetch, evidence context, schemas, recommendations,
licensing false-zero, quality gate hard-block, and evaluator dimensions.
"""

from __future__ import annotations

import pytest

from app.services.query_intent import build_query_intent
from app.services.safe_fetch import safe_dict_fetch, safe_list_fetch
from app.services.evidence_context import (
    EvidenceBlock,
    assemble_evidence_context,
    strip_unusable_from_db_context,
)
from app.services.retrieval_status import RetrievalStatus
from app.services.recommendation_quality import (
    filter_recommendations,
    is_generic_recommendation,
    score_recommendation,
)
from app.services.quality_gate import run_quality_gate
from app.services.quality_eval import evaluate_answer
from app.schemas.quality_response import (
    LicensingSnapshot,
    QualityResponse,
    RecommendationItem,
    licensing_fallback_message,
    render_licensing_snapshot,
    validate_quality_response,
)
from app.services import chat_engine as ce


def test_query_intent_licensing_count():
    qi = build_query_intent("Tell me the active MISA licenses")
    assert qi.task_type == "licensing_count"
    assert qi.current_data_required
    assert "company_profiles.licensed/is_rhq" in qi.required_sources
    assert qi.output_type == "licensing_snapshot"


def test_query_intent_company_targeting_geography():
    qi = build_query_intent(
        "Which Indian companies should we prioritise for investment attraction?"
    )
    assert qi.task_type == "company_targeting"
    assert "India" in qi.geographies
    assert qi.ranking_required


def test_safe_list_fetch_never_returns_empty_on_exception():
    def boom():
        raise ConnectionError("db down")

    rr = safe_list_fetch(boom, source_name="company_profiles")
    assert rr.is_failure
    assert rr.to_context_dict()["do_not_claim_zero"] is True
    assert rr.record_count == 0  # structural zero, not verified empty
    assert not rr.is_verified_empty


def test_safe_dict_fetch_marks_unavailable():
    def boom():
        raise RuntimeError("timeout")

    out = safe_dict_fetch(boom, source_name="company_profiles", count_keys=("total_licensed",))
    assert out["do_not_claim_zero"] is True
    assert out.get("_db_error")
    assert "total_licensed" not in out or out.get("counts_unavailable")


def test_evidence_assembly_keeps_failures_as_limitations():
    ok = EvidenceBlock(
        claim_or_payload={"n": 10},
        source_name="company_profiles",
        record_count=10,
    )
    bad = EvidenceBlock(
        claim_or_payload={"n": 0},
        source_name="rhq_licenses",
        retrieval_status=RetrievalStatus.CONNECTION_ERROR.value,
        record_count=0,
        notes="connection refused",
    )
    ctx = assemble_evidence_context([ok, bad])
    assert ctx["evidence_count"] == 1
    assert ctx["failed_retrieval_count"] == 1
    assert ctx["data_limitations"][0]["do_not_claim_zero"] is True
    assert ctx["evidence"][0].get("payload") == {"n": 10}


def test_strip_unusable_removes_zero_counts_on_failure():
    ctx = strip_unusable_from_db_context({
        "origin_country": "India",
        "companies_from_origin_licensed_in_saudi": 0,
        "companies_from_origin_with_rhq": 0,
        "footprint_data_unavailable": True,
        "retrieval_status": "error",
    })
    assert "companies_from_origin_licensed_in_saudi" not in ctx
    assert ctx["do_not_claim_zero"] is True


def test_generic_recommendations_rejected():
    assert is_generic_recommendation("Engage stakeholders")
    meta = score_recommendation("Engage stakeholders to drive growth")
    assert not meta["ok"]
    kept, rejected = filter_recommendations([
        "Engage stakeholders",
        "Schedule a 30-minute RHQ briefing with **Company X** country head "
        "mapped to **SDAIA / LEAP** within 2 weeks",
    ])
    assert len(kept) == 1
    assert len(rejected) == 1


def test_recommendation_schema_rejects_generic():
    with pytest.raises(Exception):
        RecommendationItem(action="Engage stakeholders")


def test_licensing_formatter_never_zeros_on_db_error():
    out = ce._format_saudi_licensing_briefing({
        "total_licensed": 0,
        "total_rhq": 0,
        "_db_error": "connection refused",
        "retrieval_status": "CONNECTION_ERROR",
        "retrieval": {"do_not_claim_zero": True},
    })
    assert "0 companies hold" not in out.lower()
    assert "not" in out.lower() and "zero" in out.lower()


def test_licensing_snapshot_schema_render():
    snap = LicensingSnapshot(
        total_licensed=100, total_rhq=5, focus="licensed",
        retrieval_status="SUCCESS_WITH_RESULTS",
        by_country=[{"country": "Germany", "n": 3}],
    )
    md = render_licensing_snapshot(snap)
    assert "Licensing Snapshot" in md
    assert "100" in md
    bad = LicensingSnapshot(counts_unavailable=True, retrieval_status="TIMEOUT")
    assert "not" in render_licensing_snapshot(bad).lower()


def test_quality_response_validation():
    model, errs = validate_quality_response({
        "title": "Test",
        "facts": [{
            "statement": "Licensed count is positive",
            "verification_status": "VERIFIED_INTERNAL",
            "confidence": "high",
        }],
        "data_limitations": [],
    })
    assert model is not None
    assert not errs


def test_quality_gate_hard_blocks_unrepaired_false_zero():
    # Repair may scrub some phrases; remaining critical → hard block
    text, issues, fixes = run_quality_gate(
        "There are zero Indian companies licensed in Saudi Arabia.",
        question="best companies from India",
        db_context={
            "footprint_data_unavailable": True,
            "origin_country": "India",
            "retrieval_status": "error",
        },
        hard_block=True,
    )
    assert "hard_block" in " ".join(fixes) or "not a verified zero" in text.lower() or (
        "unavailable" in text.lower()
    )
    assert "zero indian companies licensed" not in text.lower() or "not a verified" in text.lower()


def test_evaluator_dimensions_present():
    result = evaluate_answer(
        "Engage stakeholders to drive growth with zero companies licensed.",
        question="India targeting",
        db_context={"footprint_data_unavailable": True},
    )
    assert "dimensions" in result
    assert "actionability" in result["dimensions"]
    assert result["pass"] is False


def test_licensing_fallback_message():
    msg = licensing_fallback_message(status="TIMEOUT", error="deadline")
    assert "TIMEOUT" in msg
    assert "not" in msg.lower() and "zero" in msg.lower()
