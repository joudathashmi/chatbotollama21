"""Shared quality context for every surface (JSON, SSE, PDF, DOCX).

One helper so export / polish / finalize don't invent divergent
behaviour — if chat can gate on DB counts, export must too.
"""

from __future__ import annotations

from typing import Any


def quality_context_for_question(
    question: str,
    *,
    existing_db_context: dict | None = None,
    existing_deliverable: str | None = None,
) -> dict[str, Any]:
    """Resolve deliverable + optional advisory DB footprint for gating.

    Safe to call from chat, export, or finalize. Never raises.
    """
    out: dict[str, Any] = {
        "deliverable": existing_deliverable,
        "db_context": dict(existing_db_context or {}),
        "retrieval_meta": None,
    }
    q = (question or "").strip()
    if not q and not out["db_context"]:
        return out

    try:
        from app.services.chat_engine import (
            _detect_advisory_deliverable,
            _is_advisory_question,
            _advisory_country_context,
        )
        if not out["deliverable"] and q:
            if _is_advisory_question(q):
                out["deliverable"] = _detect_advisory_deliverable(q)
            else:
                # Still useful for export of targeting-shaped pastes
                out["deliverable"] = _detect_advisory_deliverable(q)

        need_ctx = not out["db_context"] and out["deliverable"] in {
            "market_fit", "engagement_plan", "company_targeting",
            "sector_priorities", "strategy_analysis",
        }
        if need_ctx and q:
            ctx = _advisory_country_context(q)
            if ctx:
                out["db_context"] = ctx
    except Exception:
        pass

    ctx = out["db_context"] or {}
    out["retrieval_meta"] = ctx.get("retrieval") or {
        "retrieval_status": ctx.get("retrieval_status"),
        "do_not_claim_zero": ctx.get("do_not_claim_zero"),
        "counts_unavailable": ctx.get("counts_unavailable")
        or ctx.get("footprint_data_unavailable"),
    }
    return out


def run_surface_quality_gate(
    answer: str,
    *,
    question: str = "",
    deliverable: str | None = None,
    db_context: dict | None = None,
    hard_block: bool = True,
) -> tuple[str, list, list[str]]:
    """Quality gate with auto-resolved context — use from PDF/DOCX/SSE."""
    from app.services.quality_gate import run_quality_gate

    qc = quality_context_for_question(
        question,
        existing_db_context=db_context,
        existing_deliverable=deliverable,
    )
    return run_quality_gate(
        answer or "",
        question=question or "",
        db_context=qc.get("db_context") or None,
        retrieval_meta=qc.get("retrieval_meta"),
        hard_block=hard_block,
        deliverable=qc.get("deliverable"),
    )
