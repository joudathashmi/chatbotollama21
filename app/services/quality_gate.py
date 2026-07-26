"""Deterministic quality gate — shared failure-class checks.

Runs after compose / advisory / licensing briefings. Catches patterns
that previously shipped as confident wrong answers:

- Failed retrieval presented as zero / none
- False-zero against known positive DB context
- Truncated ranking tables (mid-row only)
- RHQ-led title on a licensing-focused answer

Bulletproof contract (see ``advisory_safety``):
- Truncation never hard-withholds a long answer
- Company-targeting rebuild never runs for other deliverables
- Hard withhold only for unrepaired factual criticals
"""

from __future__ import annotations

import re


_ZERO_FACT_RE = re.compile(
    r"\b(zero|no|0)\s+(\w+\s+){0,4}"
    r"(companies|firms|entities|investors|licen[cs]e?s?|rhq|records|"
    r"results|matches)\b",
    re.I,
)

_UNAVAILABLE_OK_RE = re.compile(
    r"(could not be retrieved|unavailable|retrieval\s+failed|"
    r"data\s+unavailable|not\s+a\s+verified\s+zero|SUCCESS_EMPTY|"
    r"zero-result\s+query|verified\s+empty)",
    re.I,
)

_PIPE_TABLE_RE = re.compile(r"(?m)^\s*\|.+\|\s*$")


def detect_quality_issues(
    answer: str,
    *,
    db_context: dict | None = None,
    retrieval_meta: dict | None = None,
    question: str = "",
    deliverable: str | None = None,
) -> list[dict[str, str]]:
    """Return list of {code, severity, detail} issues (empty = clean)."""
    from app.services.advisory_safety import (
        may_rebuild_as_company_targeting,
        ranking_midrow_truncated,
    )

    issues: list[dict[str, str]] = []
    text = answer or ""
    q = (question or "").lower()
    ctx = db_context or {}
    meta = retrieval_meta or {}

    status_raw = str(
        meta.get("retrieval_status")
        or ctx.get("retrieval_status")
        or ""
    )
    _FAILURE_STATUSES = {
        "TIMEOUT", "AUTHENTICATION_ERROR", "PERMISSION_ERROR",
        "CONNECTION_ERROR", "INVALID_QUERY", "MALFORMED_RESPONSE",
        "PARSING_ERROR", "SOURCE_UNAVAILABLE", "UNKNOWN_ERROR",
        "NO_RELEVANT_CONTEXT",
        # legacy aliases
        "error", "ERROR", "ok_error",
    }
    unavailable = bool(
        ctx.get("footprint_data_unavailable")
        or ctx.get("counts_unavailable")
        or meta.get("do_not_claim_zero")
        or meta.get("counts_unavailable")
        or status_raw in _FAILURE_STATUSES
        or status_raw.startswith(
            ("TIMEOUT", "AUTH", "PERMISSION", "CONNECTION", "INVALID",
             "MALFORMED", "PARSING", "SOURCE_UNAVAILABLE", "UNKNOWN",
             "error", "ERROR")
        )
    )

    # 1) Failure masquerading as zero
    if unavailable and _ZERO_FACT_RE.search(text) and not _UNAVAILABLE_OK_RE.search(text):
        issues.append({
            "code": "false_zero_on_retrieval_failure",
            "severity": "critical",
            "detail": "Answer claims zero/none while retrieval was unavailable",
        })

    # 2) Positive DB counts contradicted by zero prose
    licensed = ctx.get("companies_from_origin_licensed_in_saudi")
    rhq = ctx.get("companies_from_origin_with_rhq")
    try:
        if licensed is not None and int(licensed) > 0 and _ZERO_FACT_RE.search(text):
            if re.search(r"\b(licen[cs]|compan)", text, re.I):
                if str(int(licensed)) not in re.sub(r"[,\s]", "", text):
                    issues.append({
                        "code": "contradicts_internal_licensed_count",
                        "severity": "critical",
                        "detail": f"DB licensed={licensed} but answer implies zero/absence",
                    })
        if rhq is not None and int(rhq) > 0:
            # Only absolute origin-level denials. Sector-scoped
            # "no RHQ in textiles yet" is legitimate and must NOT withhold
            # a full sector-priorities brief.
            for m in re.finditer(
                r"\b(zero|no|0)\s+(\w+\s+){0,3}rhq\w*\b",
                text,
                re.I,
            ):
                window = text[max(0, m.start() - 48): m.end() + 48].lower()
                if re.search(
                    r"\b(sector|segment|industry|vertical|in this|"
                    r"among|within|for (?:the )?\w+\s+sector)\b",
                    window,
                ):
                    continue
                issues.append({
                    "code": "contradicts_internal_rhq_count",
                    "severity": "critical",
                    "detail": f"DB rhq={rhq} but answer claims no RHQ",
                })
                break
    except (TypeError, ValueError):
        pass

    # 3) Licensing question with RHQ-led title
    if re.search(r"licen[cs]", q) and "rhq" not in q and "headquarter" not in q:
        if re.search(r"(?im)^#+\s*Saudi\s+RHQ\b", text):
            issues.append({
                "code": "wrong_licensing_title",
                "severity": "major",
                "detail": "License-focused question led with Saudi RHQ title",
            })

    # 4) Truncated ranking — mid-row corruption ONLY, and only when a
    # targeting rebuild would be legal for this deliverable/shape.
    if may_rebuild_as_company_targeting(
        deliverable=deliverable, answer=text,
    ) and ranking_midrow_truncated(text):
        issues.append({
            "code": "truncated_ranking_table",
            "severity": "critical",
            "detail": "Ranking table appears cut mid-row",
        })

    # 5) Wide markdown tables (informational for PDF layout)
    pipe_rows = len(_PIPE_TABLE_RE.findall(text))
    if pipe_rows >= 3:
        for ln in text.splitlines():
            if ln.strip().startswith("|") and ln.count("|") >= 10:
                issues.append({
                    "code": "very_wide_markdown_table",
                    "severity": "minor",
                    "detail": "Table has many columns; PDF should use cards/landscape",
                })
                break

    return issues


def apply_quality_repairs(
    answer: str,
    issues: list[dict[str, str]],
    *,
    db_context: dict | None = None,
    deliverable: str | None = None,
) -> tuple[str, list[str]]:
    """Best-effort deterministic repairs. Returns (answer, fix_codes)."""
    fixes: list[str] = []
    text = answer or ""
    codes = {i["code"] for i in issues}
    ctx = db_context or {}

    if "wrong_licensing_title" in codes:
        text2 = re.sub(
            r"(?im)^#+\s*Saudi\s+RHQ(?:\s*&\s*Licensing)?(?:\s*[—\-–]\s*)?Snapshot\s*$",
            "## Licensing Snapshot",
            text,
            count=1,
        )
        if text2 != text:
            text = text2
            fixes.append("retitled_licensing_snapshot")

    if "false_zero_on_retrieval_failure" in codes:
        text = _ZERO_FACT_RE.sub(
            "data unavailable (not a verified zero)", text, count=3
        )
        fixes.append("scrubbed_false_zero_on_failure")

    if (
        "contradicts_internal_licensed_count" in codes
        or "contradicts_internal_rhq_count" in codes
        or "truncated_ranking_table" in codes
    ):
        try:
            from app.services.response_validator import validate_advisory_answer
            text, adv_fixes = validate_advisory_answer(
                text, ctx, deliverable=deliverable,
            )
            fixes.extend(adv_fixes)
        except Exception:
            pass

    return text, fixes


def _trim_trailing_incomplete_pipe_row(text: str) -> str:
    lines = (text or "").rstrip().splitlines()
    while lines and lines[-1].strip().startswith("|") and (
        lines[-1].count("|") < 8 or not lines[-1].rstrip().endswith("|")
    ):
        lines.pop()
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def run_quality_gate(
    answer: str,
    *,
    question: str = "",
    db_context: dict | None = None,
    retrieval_meta: dict | None = None,
    hard_block: bool = True,
    deliverable: str | None = None,
) -> tuple[str, list[dict[str, str]], list[str]]:
    """Detect + repair. Returns (answer, issues, fixes_applied).

    ``hard_block`` only withholds for unrepaired factual criticals
    (false-zero / count contradiction). Truncation is always soft-
    repaired — never replaced with a withhold stub.
    """
    from app.services.advisory_safety import (
        is_truncation_only,
        may_hard_withhold,
    )

    issues = detect_quality_issues(
        answer,
        db_context=db_context,
        retrieval_meta=retrieval_meta,
        question=question,
        deliverable=deliverable,
    )
    if not issues:
        return answer, [], []
    repaired, fixes = apply_quality_repairs(
        answer, issues, db_context=db_context, deliverable=deliverable,
    )
    remaining = [
        i for i in detect_quality_issues(
            repaired,
            db_context=db_context,
            retrieval_meta=retrieval_meta,
            question=question,
            deliverable=deliverable,
        )
        if i["severity"] == "critical"
    ]
    if hard_block and remaining:
        codes = {i["code"] for i in remaining}
        ctx = db_context or {}

        # Truncation class → soft repair only (never withhold).
        if is_truncation_only(codes) or "truncated_ranking_table" in codes:
            try:
                from app.services.response_validator import (
                    validate_advisory_answer,
                )
                rebuilt, adv_fixes = validate_advisory_answer(
                    repaired, ctx, deliverable=deliverable,
                )
                if rebuilt and len(rebuilt) > 400:
                    return rebuilt, [], fixes + list(adv_fixes) + [
                        "repaired_truncation_kept_answer"
                    ]
            except Exception:
                pass
            soft = _trim_trailing_incomplete_pipe_row(repaired)
            if len(soft) > 400:
                return soft, [], fixes + ["trimmed_truncated_table_row"]
            if len(repaired) > 400:
                return repaired, remaining, fixes + [
                    "kept_long_answer_despite_truncation"
                ]

        # Factual integrity — rebuild if possible, else withhold.
        if may_hard_withhold(codes):
            codes_s = ", ".join(sorted(codes))
            # Long drafts: scrub + keep rather than destroy a full
            # brief over a single loose RHQ/zero phrase.
            if (
                len(repaired) > 1200
                and codes <= {
                    "contradicts_internal_rhq_count",
                    "contradicts_internal_licensed_count",
                }
            ):
                scrubbed = repaired
                if "contradicts_internal_rhq_count" in codes:
                    scrubbed = re.sub(
                        r"\b(zero|no|0)\s+(\w+\s+){0,3}rhq\w*\b",
                        f"{ctx.get('companies_from_origin_with_rhq') or 'verified'} RHQ",
                        scrubbed,
                        count=3,
                        flags=re.I,
                    )
                if "contradicts_internal_licensed_count" in codes and ctx.get(
                    "companies_from_origin_licensed_in_saudi"
                ) is not None:
                    scrubbed = re.sub(
                        r"\b(zero|no|0)\s+(\w+\s+){0,4}"
                        r"(companies|firms|licen[cs]e?s?)\b",
                        f"{ctx['companies_from_origin_licensed_in_saudi']} licensed companies",
                        scrubbed,
                        count=3,
                        flags=re.I,
                    )
                if len(scrubbed) > 800:
                    return scrubbed, remaining, fixes + [
                        "scrubbed_count_contradiction_kept_answer"
                    ]

            limitation = (
                "## Response withheld for quality review\n\n"
                "The drafted answer failed critical quality checks "
                f"({codes_s}). Shipping it would risk false or unsupported "
                "claims.\n\n"
                "**What we know:** authoritative internal retrieval must be "
                "re-checked; do not treat missing data as a verified zero.\n\n"
                "_Please retry the question, or ask for a narrower scope "
                "(e.g. a single country or count-only)._\n"
            )
            if ctx.get("origin_country") and not (
                ctx.get("footprint_data_unavailable")
                or ctx.get("counts_unavailable")
            ):
                try:
                    from app.services.response_validator import (
                        validate_advisory_answer,
                    )
                    rebuilt, adv_fixes = validate_advisory_answer(
                        repaired, ctx, deliverable=deliverable,
                    )
                    if adv_fixes and rebuilt and len(rebuilt) > 400:
                        return rebuilt, remaining, fixes + adv_fixes + [
                            "hard_block_rebuilt_from_db"
                        ]
                except Exception:
                    pass
            if (
                ctx.get("footprint_data_unavailable")
                or ctx.get("counts_unavailable")
                or (retrieval_meta or {}).get("do_not_claim_zero")
            ):
                country = ctx.get("origin_country") or "the requested scope"
                limitation = (
                    f"## Data unavailable\n\n"
                    f"Internal MISA data for **{country}** could not be "
                    f"verified for this answer. This is **not** a verified "
                    f"zero.\n\n"
                    f"_Quality gate: {codes_s}_\n"
                )
            return limitation, remaining, fixes + ["hard_block_critical_issues"]

        # Non-factual critical leftovers on a long draft → keep it.
        if len(repaired) > 800:
            return repaired, remaining, fixes + [
                "kept_long_answer_despite_critical"
            ]
        return repaired, remaining, fixes + ["kept_short_answer_non_factual"]

    return repaired, remaining or issues, fixes
