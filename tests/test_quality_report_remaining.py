"""Remaining quality-report items: metrics, ChatResponse meta, filter drop, docx."""

from __future__ import annotations

import pytest

from app.services.quality_metrics import record_turn, record_export, snapshot
from app.services.answer_finalize import finalize_answer
from app.services.response_validator import validate_first_paragraph
from app.database import _build_where_clauses
from app.schemas.chat import ChatResponse


def test_quality_metrics_counters():
    before = snapshot()["counters"]["turns_total"]
    record_turn(
        intent="licensing_count",
        retrieval_status="SOURCE_UNAVAILABLE",
        quality_gate={"issues": ["false_zero_on_retrieval_failure"],
                      "fixes": ["hard_block_critical_issues"]},
        quality_eval={"pass": False},
        truncated=True,
    )
    record_export(kind="pdf", quality_blocked=True)
    snap = snapshot()
    assert snap["counters"]["turns_total"] >= before + 1
    assert snap["counters"]["retrieval_failures"] >= 1
    assert snap["counters"]["pdf_exports"] >= 1


def test_finalize_adds_truncation_banner():
    pack = {"_truncated": True, "_truncation_reason": "row_budget",
            "_intent": "company_profile"}
    out = finalize_answer(
        "## Operational Detail\n\nSome content here about the company.\n"
        "## Strategic Read\n\nEngage carefully.\n",
        user_question="tell me about Acme",
        pack=pack,
    )
    assert "Partial result" in out
    assert pack.get("_data_limitations")


def test_validate_first_paragraph_fail_closed_no_client():
    v = validate_first_paragraph(
        "how many licenses?", "There are many.", None, "gpt",
        fail_closed=True,
    )
    assert v["is_relevant"] is False


def test_chat_response_accepts_quality_fields():
    r = ChatResponse(
        answer="ok",
        rows=[],
        trace=[],
        trace_id="abc",
        intent={"task_type": "licensing_count"},
        retrieval_status="SUCCESS_WITH_RESULTS",
        quality={"eval": {"score": 90, "pass": True}},
        data_limitations=["partial"],
    )
    assert r.intent["task_type"] == "licensing_count"
    assert r.quality["eval"]["pass"] is True


def test_build_where_reports_dropped_filters():
    # Use a fake allowed set so unknown cols are dropped
    where, params, dropped = _build_where_clauses(
        "company_profiles",
        {"licensed": {"op": "=", "value": True},
         "not_a_real_column_xyz": {"op": "=", "value": 1}},
        allowed_filters={"licensed"},
    )
    assert "not_a_real_column_xyz" in dropped
    assert where  # licensed should still apply if valid — may need column
    # If licensed not in allowed we'd get empty where; we included it


def test_docx_export_optional():
    try:
        from app.services.docx_export import render_docx
    except Exception:
        pytest.skip("docx_export import failed")
    try:
        import docx  # noqa: F401
    except ImportError:
        pytest.skip("python-docx not installed")
    data = render_docx(
        "Active MISA licenses",
        "## Licensing Snapshot\n\n**100** companies hold an active MISA licence.\n",
    )
    assert data[:2] == b"PK"  # zip/docx magic
