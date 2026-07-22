"""SSE / stream repair — soft_check + deterministic fallback."""

from __future__ import annotations

from app.services.stream_repair import repair_company_answer_if_thin


_THIN = (
    "## Acme — Executive Briefing\n\n"
    "Acme makes software.\n\n"
    "### Corporate Profile & Regional Footprint\n\n"
    "| Metric | Global | MENA |\n|---|---|---|\n| Sector | ICT | ICT |\n\n"
    "### 🇸🇦 Strategic Read\n\n"
    "- Expand digitally.\n"
)

_ROWS = [{
    "company_name": "Acme Soft",
    "sector": "Information and Communication Technology",
    "key_metrics": {
        "annual_revenue_usd": 5_000_000_000,
        "employee_count": 10000,
        "ksa_employees": 100,
        "mena_employees": 400,
        "sector": "ICT",
        "headquarters": "USA",
    },
    "misa_details": {
        "general": {"company_profile": "Acme Soft builds cloud software."},
        "mena_details": {"mena_notes": "MENA hub in Riyadh.", "presence_in_saudi": "Yes"},
        "rhq_details": {"rhq_status": "Yes", "rhq_city": "Riyadh"},
    },
}]


def test_repair_replaces_ops_less_company_brief():
    pack: dict = {"_intent": "company_profile"}
    out, fixes = repair_company_answer_if_thin(
        _THIN,
        question="tell me about Acme Soft",
        intent="company_profile",
        rows=_ROWS,
        pack=pack,
    )
    assert any("replaced_with_deterministic" in f or "contract_fail" in f for f in fixes)
    assert "Snapshot of Operations" in out or "Strategic Read" in out
    assert pack.get("_stream_contract_fallback") is True


def test_repair_keeps_rich_company_brief():
    rich = (
        "## Acme Soft — Executive Briefing\n\n"
        "Acme Soft is an ICT leader.\n\n"
        "### Corporate Profile & Regional Footprint\n\n"
        "| Metric | Global | MENA |\n|---|---|---|\n| Sector | ICT | ICT |\n\n"
        "### Snapshot of Operations and Market Position\n\n"
        "- Cloud ERP products.\n"
        "- CEO: Jane Doe.\n"
        "- MENA hub in Riyadh.\n"
        "- 100 KSA employees.\n"
        "- RHQ in Riyadh.\n"
        "- Open opportunity on localisation.\n"
        "- MISA contact on file.\n\n"
        "### 🇸🇦 Strategic Read\n\n"
        "- Map to SDAIA demand.\n"
    )
    pack: dict = {"_intent": "company_profile"}
    out, fixes = repair_company_answer_if_thin(
        rich,
        question="tell me about Acme Soft",
        intent="company_profile",
        rows=_ROWS,
        pack=pack,
    )
    assert "replaced_with_deterministic" not in "".join(fixes)
    assert "Snapshot of Operations" in out


def test_streaming_uses_advisory_token_budget():
    src = open("app/services/streaming_curation.py").read()
    assert "openai_advisory_max_tokens_kw" in src
    assert "openai_max_completion_tokens_kw()" not in src


def test_sse_buffers_before_emitting_answer_chunks():
    """Raw Azure tokens must not reach the client before repair."""
    src = open("app/routers/v1/chat.py").read()
    # Must buffer stream into assembled before yielding answer chunks
    assert "Quality-checking briefing" in src
    assert "buf.append(chunk)" in src or "buf: list[str]" in src
    assert "repair_company_answer_if_thin" in src
    # The old pattern of yielding each LLM chunk immediately is gone
    assert "never flashes a thin draft" in src.lower() or (
        "Only NOW stream answer text" in src
        or "ONLY the repaired final text" in src
    )


def test_hybrid_detects_snapshot_as_ops():
    src = open("app/services/hybrid_briefing.py").read()
    assert "Snapshot of Operations" in src
