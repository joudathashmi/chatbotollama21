"""
Response validator.

Step 9 of the executive-intelligence spec: before returning an answer,
check that the FIRST paragraph directly answers the user's question.
Also hosts deterministic advisory footprint guards (false-zero prevention).
"""

from __future__ import annotations

import json
import re as _re


_VALIDATOR_PROMPT = """You are a strict response-quality reviewer for an executive-intelligence chatbot.

USER QUESTION: {question}

ANSWER (first 600 characters):
{answer_head}

TASK: Decide whether the FIRST paragraph of the answer DIRECTLY answers what the user asked. The user is a busy executive — they want the answer at the top, not buried after generic context.

Return ONLY a JSON object:
{{
  "is_relevant": <true|false>,
  "reason": "<one short sentence>"
}}"""


def validate_first_paragraph(
    user_question: str, answer: str, client, model: str,
    *,
    fail_closed: bool = False,
) -> dict:
    """Quick relevance check on the answer's opening.

    When ``fail_closed`` is True, validator errors / empty client return
    ``is_relevant: False`` so callers regenerate or withhold — used for
    advisory / count paths where a false pass is worse than a retry.
    """
    q = (user_question or "").strip()
    a = (answer or "").strip()
    if not q or not a or client is None:
        if fail_closed and client is None:
            return {
                "is_relevant": False,
                "reason": "validator unavailable (no client)",
            }
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
        if fail_closed:
            return {
                "is_relevant": False,
                "reason": f"validator error (fail-closed): {e}",
            }
        return {"is_relevant": True, "reason": f"validator error: {e}"}


# ─── Advisory answer validation (deterministic, no LLM) ──────────────

# Footprint headings ONLY. "The Evidence Base" is the sector-priorities
# spine — never treat it as a footprint section to strip/rebuild.
_FOOTPRINT_HEADING_RE = _re.compile(
    r"^#{0,3}\s*current\s+(misa|saudi)\s+footprint\s*$",
    _re.I | _re.M,
)

_ZERO_CLAIM_RE = _re.compile(
    r"\b(zero|no|0)\s+(\w+\s+){0,4}(companies|firms|entities|"
    r"investors|rhq|licensed)",
    _re.I,
)


def _split_sections(answer: str) -> list[tuple[str | None, str]]:
    lines = (answer or "").splitlines()
    chunks: list[tuple[str | None, list[str]]] = [(None, [])]
    for ln in lines:
        if _re.match(r"^#{1,3}\s+\S", ln):
            chunks.append((ln, []))
        else:
            chunks[-1][1].append(ln)
    return [(h, "\n".join(b)) for h, b in chunks]


def _is_footprint_heading(heading: str | None) -> bool:
    from app.services.advisory_safety import is_footprint_heading
    return is_footprint_heading(heading)


def _build_footprint_section(db_context: dict) -> str:
    country = db_context.get("origin_country") or "the origin country"
    status = db_context.get("retrieval_status") or "SUCCESS_WITH_RESULTS"
    filters = db_context.get("retrieval_filters") or {}
    source = filters.get("source") or "company_profiles + nationality/origin join"

    _fail = {
        "error", "ERROR", "SOURCE_UNAVAILABLE", "TIMEOUT",
        "CONNECTION_ERROR", "AUTHENTICATION_ERROR", "PERMISSION_ERROR",
        "INVALID_QUERY", "MALFORMED_RESPONSE", "PARSING_ERROR",
        "UNKNOWN_ERROR", "NO_RELEVANT_CONTEXT",
    }
    if (
        db_context.get("footprint_data_unavailable")
        or status in _fail
        or db_context.get("do_not_claim_zero")
        or db_context.get("counts_unavailable")
    ):
        try:
            from app.services.retrieval_status import (
                RetrievalStatus, failure, user_facing_retrieval_message,
            )
            st = status if status in RetrievalStatus.__members__ else "SOURCE_UNAVAILABLE"
            rr = failure(
                RetrievalStatus[st] if st in RetrievalStatus.__members__
                else RetrievalStatus.SOURCE_UNAVAILABLE,
                source_name=source,
                error=str(db_context.get("_db_error") or ""),
                filters={"origin_country": country},
            )
            return user_facing_retrieval_message(rr)
        except Exception:
            return (
                f"Internal MISA footprint data for **{country}** could not be "
                f"retrieved (retrieval_status=`{status}`). This is **not** a "
                f"verified zero — do not treat licensed or RHQ counts as 0."
            )

    licensed = db_context.get("companies_from_origin_licensed_in_saudi")
    rhq = db_context.get("companies_from_origin_with_rhq")

    if status in ("SUCCESS_EMPTY", "zero_records") or (
        licensed == 0 and rhq == 0 and status in ("SUCCESS_EMPTY", "zero_records")
    ):
        return (
            f"The queried MISA source returned **0** verified licensed "
            f"records and **0** RHQ records for **{country}** "
            f"(source: {source}; filters: origin_country={country}). "
            f"This is a successful zero-result query, not a retrieval failure."
        )

    lines = [
        f"According to MISA's database ({source}), **{licensed}** companies "
        f"from {country} are licensed in Saudi Arabia, of which **{rhq}** "
        "hold Regional Headquarters (RHQ) status."
    ]
    tops = db_context.get("top_rhq_companies") or []
    if tops:
        names = ", ".join(
            str(t.get("name") or t.get("company"))
            for t in tops[:5]
            if (t.get("name") or t.get("company"))
        )
        if names:
            lines.append(f"Top RHQ holders: {names}.")
    tops_l = db_context.get("top_licensed_companies") or []
    if tops_l:
        names = ", ".join(
            str(t.get("name") or t.get("company"))
            for t in tops_l[:5]
            if (t.get("name") or t.get("company"))
        )
        if names:
            lines.append(f"Top licensed companies: {names}.")
    expansion = db_context.get("expansion_targets") or []
    if expansion:
        names = ", ".join(
            str(t.get("company")) for t in expansion[:6] if t.get("company")
        )
        if names:
            lines.append(
                f"Priority expansion candidates from the footprint: {names}."
            )
    return "\n\n".join(lines)


def _scrub_false_zeros_in_prose(body: str, licensed: int) -> str:
    """When DB has positive counts, rewrite zero-claims in executive text."""
    if int(licensed or 0) <= 0:
        return body
    # Replace common false-zero phrases while leaving other numbers alone.
    body = _re.sub(
        r"\b(zero|no)\s+(\w+\s+){0,3}(Indian\s+|from\s+\w+\s+)?"
        r"(companies|firms|entities|investors)\s+"
        r"(licensed|with\s+RHQ|holding\s+RHQ)",
        f"**{licensed}** companies licensed",
        body,
        flags=_re.I,
    )
    body = _re.sub(
        r"\blicensed\s+(?:in\s+Saudi\s+Arabia\s*)?(?:[:\-–]\s*)?(?:is\s+|are\s+)?0\b",
        f"licensed: **{licensed}**",
        body,
        flags=_re.I,
    )
    body = _re.sub(
        r"\bRHQ[s]?\s*(?:status\s*)?(?:[:\-–]\s*)?(?:is\s+|are\s+)?0\b",
        "RHQ: (see Current Saudi Footprint)",
        body,
        flags=_re.I,
    )
    return body


def _ensure_expansion_block(answer: str, db_context: dict) -> tuple[str, bool]:
    """If expansion_targets exist but none are named in the answer, append."""
    targets = db_context.get("expansion_targets") or []
    if not targets:
        return answer, False
    names = [
        str(t.get("company")).strip()
        for t in targets[:8]
        if t.get("company")
    ]
    if not names:
        return answer, False
    hit = sum(1 for n in names if n.casefold() in answer.casefold())
    if hit >= min(3, len(names)):
        return answer, False
    lines = [
        "",
        "## Priority Expansion Targets (MISA database)",
        "",
        "The following companies are already in the MISA Saudi footprint "
        "and should be treated as **expansion** targets:",
        "",
        "| Rank | Company | Sector | Saudi Presence | Target Type |",
        "|---|---|---|---|---|",
    ]
    for i, t in enumerate(targets[:8], 1):
        lines.append(
            f"| {i} | {t.get('company')} | {t.get('sector') or ''} | "
            f"{t.get('current_saudi_presence') or ''} | expansion |"
        )
    lines.append("")
    return answer.rstrip() + "\n" + "\n".join(lines), True


def validate_advisory_answer(
    answer: str,
    db_context: dict | None,
    *,
    deliverable: str | None = None,
) -> tuple[str, list[str]]:
    """Deterministic post-generation guard for advisory documents.

    ``deliverable`` gates destructive rebuilds via ``advisory_safety``.
    Without it, full company-targeting rebuilds are refused unless the
    answer already has an unmistakable targeting shape.
    """
    if not answer:
        return answer, []
    from app.services.advisory_safety import (
        may_rebuild_as_company_targeting,
        ranking_midrow_truncated,
    )

    fixes: list[str] = []
    sections = _split_sections(answer)

    unavailable = bool(
        db_context and db_context.get("footprint_data_unavailable")
    )
    counts_available = bool(
        db_context
        and not unavailable
        and db_context.get("companies_from_origin_licensed_in_saudi")
        is not None
    )

    out_chunks: list[str] = []
    saw_footprint = False
    for heading, body in sections:
        if _is_footprint_heading(heading):
            saw_footprint = True
            if unavailable:
                fixes.append("rebuilt_footprint_unavailable_notice")
                out_chunks.append(
                    (heading or "## Current Saudi Footprint")
                    + "\n\n" + _build_footprint_section(db_context)
                )
                continue
            if not counts_available:
                fixes.append("stripped_fabricated_footprint_section")
                continue
            licensed = db_context["companies_from_origin_licensed_in_saudi"]
            rhq = db_context.get("companies_from_origin_with_rhq")
            nums = {int(n) for n in _re.findall(r"\b(\d{1,7})\b", body)}
            zero_claim = (
                bool(_ZERO_CLAIM_RE.search(body)) and int(licensed or 0) > 0
            )
            # Require BOTH counts missing before rebuilding — narrative
            # footprint prose often cites licensed without repeating RHQ.
            counts_wrong = (
                int(licensed) not in nums
                and (rhq is None or int(rhq) not in nums)
            )
            if zero_claim or counts_wrong:
                fixes.append("rebuilt_footprint_from_db_counts")
                out_chunks.append(
                    (heading or "## Current Saudi Footprint")
                    + "\n\n" + _build_footprint_section(db_context)
                )
                continue
        elif counts_available and body:
            licensed = int(
                db_context["companies_from_origin_licensed_in_saudi"] or 0
            )
            scrubbed = _scrub_false_zeros_in_prose(body, licensed)
            if scrubbed != body:
                fixes.append("scrubbed_false_zero_in_section")
                body = scrubbed
        out_chunks.append((heading + "\n" if heading else "") + body)

    result = "\n".join(c for c in out_chunks if c.strip())

    if counts_available and not saw_footprint:
        from app.services.source_policy import may_inject_missing_footprint
        if may_inject_missing_footprint(deliverable):
            inject = (
                "## Current Saudi Footprint\n\n"
                + _build_footprint_section(db_context)
            )
            parts = result.split("\n", 1)
            if len(parts) == 2 and parts[0].startswith("#"):
                result = parts[0] + "\n\n" + inject + "\n\n" + parts[1]
            else:
                result = inject + "\n\n" + result
            fixes.append("injected_missing_footprint_section")

    if unavailable:
        if _ZERO_CLAIM_RE.search(result):
            result = _ZERO_CLAIM_RE.sub(
                "footprint data unavailable (not a verified zero)", result
            )
            fixes.append("scrubbed_zero_claim_while_unavailable")
        result = result.replace(
            "; MISA database figures cited where noted", "")

    if not counts_available and not unavailable:
        result = result.replace(
            "; MISA database figures cited where noted", "")

    if (
        db_context
        and counts_available
        and (deliverable == "company_targeting"
             or (deliverable is None
                 and "Priority Company Ranking" in result))
    ):
        result, added = _ensure_expansion_block(result, db_context)
        if added:
            fixes.append("injected_expansion_targets_from_db")

    # Targeting rebuild ONLY — never wipe other advisory deliverables.
    try:
        from app.services.advisory_structured import (
            render_company_targeting_markdown,
            seed_company_targeting_payload_from_db,
        )
        if (
            db_context
            and may_rebuild_as_company_targeting(
                deliverable=deliverable, answer=result,
            )
            and ranking_midrow_truncated(result)
            and (db_context.get("expansion_targets") or counts_available)
        ):
            seed = seed_company_targeting_payload_from_db(db_context)
            if seed and seed.get("targets"):
                result = render_company_targeting_markdown(seed)
                fixes.append("rebuilt_truncated_company_targeting_from_db")
    except Exception:
        pass

    return result, fixes
