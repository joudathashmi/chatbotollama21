"""
Response validator.

Step 9 of the executive-intelligence spec: before returning an answer,
check that the FIRST paragraph directly answers the user's question.
If not, regenerate with a stricter intent directive.

Implementation:
  - One small LLM call (~200ms with gpt-4o-mini): takes the user's
    question + the answer's first ~600 chars; returns a JSON verdict
    {is_relevant: bool, reason: str}.
  - Caller decides what to do — typically: if NOT relevant AND we
    have remaining retry budget, regenerate the curated answer with
    an even more directive system prompt.

This module is deliberately small and stateless. It does not regenerate
on its own (regeneration is the chat_engine's job because it knows the
full curation context). It just renders the verdict.
"""

from __future__ import annotations

import json


_VALIDATOR_PROMPT = """You are a strict response-quality reviewer for an executive-intelligence chatbot.

USER QUESTION: {question}

ANSWER (first 600 characters):
{answer_head}

TASK: Decide whether the FIRST paragraph of the answer DIRECTLY answers what the user asked. The user is a busy executive — they want the answer at the top, not buried after generic context.

Examples of FAILURE (is_relevant=false):
  Q: "Who is the CEO of Apple?"
  A: "Apple Inc. is a leading technology company..."   <- buries the CEO
  -> is_relevant: false; reason: "Leads with company description, not CEO name."

  Q: "What is Apple's revenue?"
  A: "## Snapshot\\nApple Inc. is headquartered in..."  <- no number up top
  -> is_relevant: false; reason: "First paragraph doesn't state the revenue figure."

Examples of SUCCESS (is_relevant=true):
  Q: "Who is the CEO of Apple?"
  A: "## CEO\\n- Name: Tim Cook\\n- Role: CEO..."
  -> is_relevant: true; reason: "First section names the CEO directly."

  Q: "What is Apple's revenue?"
  A: "> Revenue: $391B (latest in DB)..."
  -> is_relevant: true; reason: "Lead line states the revenue number."

Return ONLY a JSON object:
{{
  "is_relevant": <true|false>,
  "reason": "<one short sentence>"
}}"""


def validate_first_paragraph(
    user_question: str, answer: str, client, model: str,
) -> dict:
    """Quick relevance check on the answer's opening. Returns
    {"is_relevant": bool, "reason": str}.

    Defaults to is_relevant=true on any failure (network, parse,
    empty inputs) - we never want the validator to break a turn.
    The downside risk is a missed regeneration, not a broken response.
    """
    q = (user_question or "").strip()
    a = (answer or "").strip()
    if not q or not a or client is None:
        return {"is_relevant": True, "reason": "validator skipped (empty input)"}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": _VALIDATOR_PROMPT.format(
                    question=q, answer_head=a[:600],
                ),
            }],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=80,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "is_relevant": bool(data.get("is_relevant", True)),
            "reason": str(data.get("reason") or "")[:200],
        }
    except Exception as e:
        # Fail-open: assume relevant so we never reject a real answer
        # over a validator network blip.
        return {"is_relevant": True, "reason": f"validator error: {e}"}


# ─── Advisory answer validation (deterministic, no LLM) ──────────────
# The advisory path composes long strategy documents. Two failure
# modes observed in production that prompt instructions alone do not
# reliably prevent:
#   1. FABRICATED FOOTPRINT — no DB context was supplied, yet the
#      answer contains a "Current MISA Footprint" section with
#      invented figures ("zero Indian companies licensed").
#   2. WRONG COUNTS — DB context was supplied, but the stated
#      licensed/RHQ counts do not match it.
# Both are enforced here in code: sections are stripped or rebuilt
# from the real numbers. Never trust the model with these claims.

import re as _re

_FOOTPRINT_HEADING_RE = _re.compile(
    r"^#{1,3}\s*(current\s+misa\s+footprint|the\s+evidence\s+base)\s*$",
    _re.I | _re.M,
)


def _split_sections(answer: str) -> list[tuple[str | None, str]]:
    """Split markdown into (heading_line, body) chunks. The first chunk
    may have heading None (preamble before any heading)."""
    lines = (answer or "").splitlines()
    chunks: list[tuple[str | None, list[str]]] = [(None, [])]
    for ln in lines:
        if _re.match(r"^#{1,3}\s+\S", ln):
            chunks.append((ln, []))
        else:
            chunks[-1][1].append(ln)
    return [(h, "\n".join(b)) for h, b in chunks]


def _is_footprint_heading(heading: str | None) -> bool:
    if not heading:
        return False
    return bool(_FOOTPRINT_HEADING_RE.match(heading.strip()))


def _build_footprint_section(db_context: dict) -> str:
    """Deterministically render the footprint paragraph from the real
    DB numbers — used to REPLACE a model-written section whose counts
    don't match the database."""
    country = db_context.get("origin_country") or "the origin country"
    licensed = db_context.get("companies_from_origin_licensed_in_saudi")
    rhq = db_context.get("companies_from_origin_with_rhq")
    lines = [
        f"According to MISA's database, **{licensed}** companies from "
        f"{country} are licensed in Saudi Arabia, of which **{rhq}** "
        "hold Regional Headquarters (RHQ) status."
    ]
    tops = db_context.get("top_rhq_companies") or []
    if tops:
        names = ", ".join(
            str(t.get("name")) for t in tops[:5] if t.get("name"))
        if names:
            lines.append(f"Top RHQ holders: {names}.")
    tops_l = db_context.get("top_licensed_companies") or []
    if tops_l:
        names = ", ".join(
            str(t.get("name")) for t in tops_l[:5] if t.get("name"))
        if names:
            lines.append(f"Top licensed companies: {names}.")
    return "\n\n".join(lines)


def validate_advisory_answer(
    answer: str, db_context: dict | None,
) -> tuple[str, list[str]]:
    """Deterministic post-generation guard for advisory documents.
    Returns (possibly-corrected answer, list of fixes applied)."""
    if not answer:
        return answer, []
    fixes: list[str] = []
    sections = _split_sections(answer)

    counts_available = bool(
        db_context
        and not db_context.get("footprint_data_unavailable")
        and db_context.get("companies_from_origin_licensed_in_saudi")
        is not None
    )

    out_chunks: list[str] = []
    for heading, body in sections:
        if _is_footprint_heading(heading):
            if not counts_available:
                # Fabricated: no data existed to write this section.
                fixes.append("stripped_fabricated_footprint_section")
                continue
            licensed = db_context["companies_from_origin_licensed_in_saudi"]
            rhq = db_context.get("companies_from_origin_with_rhq")
            nums = {int(n) for n in _re.findall(r"\b(\d{1,7})\b", body)}
            zero_claim = bool(_re.search(
                r"\b(zero|no)\s+(\w+\s+){0,3}(companies|firms|entities|"
                r"investors|rhq)", body, _re.I)) and int(licensed or 0) > 0
            counts_wrong = (
                int(licensed) not in nums
                or (rhq is not None and int(rhq) not in nums)
            )
            if zero_claim or counts_wrong:
                fixes.append("rebuilt_footprint_from_db_counts")
                out_chunks.append(
                    (heading or "## Current MISA Footprint")
                    + "\n\n" + _build_footprint_section(db_context)
                )
                continue
        out_chunks.append((heading + "\n" if heading else "") + body)

    # Drop empty chunks (e.g. the zero-length preamble before the
    # first heading) so re-joining doesn't add stray leading newlines.
    result = "\n".join(c for c in out_chunks if c.strip())
    if not counts_available:
        # Also scrub a stale source-line claim of DB figures.
        result = result.replace(
            "; MISA database figures cited where noted", "")
    return result, fixes
