"""Lightweight deterministic quality evaluator for regression CI.

Scores answers on factual/source/format/strategic axes without an LLM.
Used in tests and pack telemetry. Critical issues ⇒ fail regardless of score.
"""

from __future__ import annotations

import re
from typing import Any


_GENERIC_RECO_RE = re.compile(
    r"\b(engage stakeholders|leverage synergies|explore opportunities|"
    r"continue monitoring|stay aligned with vision\s*2030|"
    r"foster collaboration|drive growth)\b",
    re.I,
)
_ACTIONABLE_RE = re.compile(
    r"\b(schedule|brief|invite|propose|pilot|open\s+an?\s+rhq|"
    r"localis|localiz|joint\s+venture|next\s+step|within\s+\d+\s+"
    r"(?:days|weeks|months)|assign|contact)\b",
    re.I,
)


def evaluate_answer(
    answer: str,
    *,
    question: str = "",
    db_context: dict | None = None,
    expect_licensing_snapshot: bool = False,
    expect_ranking: bool = False,
    min_pass_score: int = 70,
) -> dict[str, Any]:
    from app.services.quality_gate import detect_quality_issues

    text = answer or ""
    issues = detect_quality_issues(
        answer, db_context=db_context, question=question,
    )
    critical = [i for i in issues if i["severity"] == "critical"]
    major = [i for i in issues if i["severity"] == "major"]
    score = 100
    score -= 25 * len(critical)
    score -= 10 * len(major)
    score -= 3 * sum(1 for i in issues if i["severity"] == "minor")

    dimensions = {
        "factual_accuracy": 100,
        "source_grounding": 100,
        "completeness": 100,
        "relevance": 100,
        "strategic_depth": 100,
        "specificity": 100,
        "actionability": 100,
        "internal_consistency": 100,
        "uncertainty_handling": 100,
        "formatting_quality": 100,
        "readability": 100,
        "hallucination_absence": 100,
    }

    if critical:
        dimensions["factual_accuracy"] = 20
        dimensions["hallucination_absence"] = 10
    if "[object Object]" in text:
        dimensions["formatting_quality"] = 0
        score = min(score, 40)
    if not text.strip():
        for k in dimensions:
            dimensions[k] = 0
        score = 0

    ctx = db_context or {}
    if ctx.get("footprint_data_unavailable") or ctx.get("counts_unavailable"):
        if re.search(r"\b(zero|no companies|0\s+licen)", text, re.I) and not (
            re.search(r"not\s+a\s+verified\s+zero|unavailable", text, re.I)
        ):
            dimensions["uncertainty_handling"] = 15
            dimensions["factual_accuracy"] = min(
                dimensions["factual_accuracy"], 25
            )
            score -= 20
        elif re.search(r"unavailable|not\s+a\s+verified\s+zero", text, re.I):
            dimensions["uncertainty_handling"] = 95

    if _GENERIC_RECO_RE.search(text):
        dimensions["specificity"] -= 30
        dimensions["actionability"] -= 25
        score -= 15
        # Soft-rec answers fail the world-class bar — do not pass.
        critical.append({
            "code": "soft_recommendation_boilerplate",
            "severity": "critical",
            "detail": "Generic recommendation phrasing present",
        })

    # World-class rec bar: if a Recommended section exists, every bullet
    # must be named + dated/counterpart-grounded.
    try:
        from app.services.recommendation_quality import (
            is_world_class_recommendation,
        )
        rec_m = re.search(
            r"(?is)^#{1,3}\s*(?:Recommended|Strategic Targeting)[^\n]*\n(.*?)(?=^#{1,3}\s|\Z)",
            text,
            re.M,
        )
        if rec_m:
            bullets = re.findall(r"(?m)^\s*[-*•]\s+(.+)$", rec_m.group(1))
            soft_n = sum(
                1 for b in bullets
                if b.strip() and not is_world_class_recommendation(b)
            )
            if bullets and soft_n >= max(1, len(bullets) // 2):
                dimensions["actionability"] = min(dimensions["actionability"], 35)
                dimensions["specificity"] = min(dimensions["specificity"], 40)
                score -= 20
                critical.append({
                    "code": "recs_not_world_class",
                    "severity": "critical",
                    "detail": f"{soft_n}/{len(bullets)} recommendation bullets fail world-class bar",
                })
    except Exception:
        pass

    # Company contract thinness
    try:
        from app.services.answer_contracts import company_brief_violations
        if "Executive Briefing" in text or "Corporate Profile" in text:
            cv = company_brief_violations(text)
            if any("ops body" in v for v in cv):
                dimensions["completeness"] = min(dimensions["completeness"], 30)
                score -= 25
                critical.append({
                    "code": "missing_ops_body",
                    "severity": "critical",
                    "detail": "Company brief missing Snapshot of Operations",
                })
    except Exception:
        pass
    if expect_ranking or re.search(r"priority company ranking", text, re.I):
        if not re.search(r"(?m)^\|.+\|", text):
            dimensions["completeness"] -= 40
            score -= 15
        if not _ACTIONABLE_RE.search(text):
            dimensions["actionability"] -= 20

    if len(text) > 12000:
        dimensions["readability"] -= 15
    if len(text) < 40 and text.strip():
        dimensions["completeness"] -= 20

    # Source / limitation presence
    has_source = bool(re.search(r"(?i)source|limitation|_Sources", text))
    if not has_source:
        dimensions["source_grounding"] -= 20
        score -= 5

    for k, v in list(dimensions.items()):
        dimensions[k] = max(0, min(100, int(v)))

    score = max(0, min(100, score))

    checks = {
        "has_content": bool(text.strip()),
        "no_critical_issues": not critical,
        "no_raw_object_error": "[object Object]" not in text,
        "has_sources_or_limitations": has_source,
        "no_generic_reco_boilerplate": not bool(_GENERIC_RECO_RE.search(text)),
        "no_soft_recommendation_critical": not any(
            i.get("code") in (
                "soft_recommendation_boilerplate", "recs_not_world_class",
            )
            for i in critical
        ),
    }
    if expect_licensing_snapshot:
        checks["licensing_title"] = bool(
            re.search(r"(?im)^#+\s*Licensing Snapshot\b", text)
        )
        if not checks["licensing_title"]:
            score = min(score, 70)
            dimensions["relevance"] = min(dimensions["relevance"], 60)
    if expect_ranking:
        checks["has_ranking_table"] = bool(
            re.search(r"(?im)priority company ranking", text)
            and re.search(r"(?m)^\|.+\|", text)
        )
        if not checks["has_ranking_table"]:
            score = min(score, 65)

    passed = score >= min_pass_score and not critical and checks["has_content"]
    return {
        "score": score,
        "pass": passed,
        "issues": issues,
        "checks": checks,
        "dimensions": dimensions,
        "min_pass_score": min_pass_score,
    }
