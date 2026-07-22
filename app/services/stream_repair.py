"""Shared post-compose repair for streamed / curated company answers.

Closes the SSE hole: soft_check → deterministic template fallback → finalize.
"""

from __future__ import annotations

import re
from typing import Any


_COMPANY_ASK_RE = re.compile(
    r"(?i)\b(company\s+(?:profile|briefing)|briefing\s+on|profile\s+of|"
    r"tell\s+me\s+about|brief\s+me\s+on)\b"
)


def _looks_like_company_ask(question: str) -> bool:
    # "tell me about the CEO of X" is a person ask — never treat as company.
    try:
        from app.services.db_briefing import _question_looks_like_person
        if _question_looks_like_person(question):
            return False
    except Exception:
        pass
    return bool(_COMPANY_ASK_RE.search(question or ""))


def _entity_hint_from_question(question: str) -> str:
    q = (question or "").strip()
    m = re.search(
        r"(?i)(?:company\s+(?:profile|briefing)\s+for|"
        r"briefing\s+on|profile\s+of|tell\s+me\s+about|"
        r"brief\s+me\s+on)\s+(.+?)\s*$",
        q,
    )
    if m:
        return m.group(1).strip().rstrip("?.!,")
    return ""


def _pick_best_company_rows(rows: list[dict], hint: str) -> list[dict]:
    if not rows:
        return []
    if len(rows) == 1:
        return rows
    hint_l = (hint or "").casefold()
    scored: list[tuple[int, dict]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("company_name") or r.get("name") or "")
        score = 0
        if hint_l and hint_l in name.casefold():
            score += 5
        if r.get("is_rhq") is True or str(r.get("is_rhq")).lower() in (
            "true", "1", "yes",
        ):
            score += 3
        if r.get("licensed") is True or str(r.get("licensed")).lower() in (
            "true", "1", "yes",
        ):
            score += 2
        if "regional headquarter" in name.casefold() or "rhq" in name.casefold():
            score += 2
        scored.append((score, r))
    if not scored:
        return rows[:1]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    best = [r for s, r in scored if s == best_score and s > 0]
    return best[:1] or [scored[0][1]]


def _fetch_rows_for_hint(hint: str) -> list[dict]:
    if not hint or len(hint) < 2:
        return []
    try:
        from app.services.ambiguity import discover_candidates
        cands = discover_candidates(hint) or []
        rows = []
        for c in cands[:5]:
            if not isinstance(c, dict):
                continue
            row = dict(c)
            if row.get("name") and not row.get("company_name"):
                row["company_name"] = row["name"]
            rows.append(row)
        return rows
    except Exception:
        return []


def repair_company_answer_if_thin(
    answer: str,
    *,
    question: str = "",
    intent: str | None = None,
    rows: list[dict] | None = None,
    locale: str = "en",
    pack: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """If company shape fails contracts, replace with deterministic template.

    Returns (answer, fixes). Always safe — on any error returns original text.
    """
    pack = pack or {}
    fixes: list[str] = []
    text = answer or ""
    try:
        from app.services.answer_contracts import (
            company_brief_violations,
            soft_check_answer,
        )
        try:
            viol = soft_check_answer(
                text,
                intent=intent or pack.get("_intent"),
                user_question=question,
                db_context=pack.get("_advisory_db_context")
                or pack.get("_db_context"),
            )
        except Exception as exc:
            # Fail closed — treat gate errors as contract failure.
            viol = [f"soft_check_exception:{type(exc).__name__}"]
            fixes.append("soft_check_failed_open_blocked")

        # Also check company shape directly when intent is company-like
        # or the ask is a company briefing/profile phrasing.
        intent_l = (intent or pack.get("_intent") or "").strip().lower()
        company_ask = _looks_like_company_ask(question)
        if intent_l in (
            "company_profile", "saudi_presence", "financial_lookup",
            "opportunity_alignment", "relationship_intelligence",
            "entity_lookup",
        ) or "Executive Briefing" in text or "Corporate Profile" in text or company_ask:
            try:
                viol = list(viol or []) + company_brief_violations(text)
            except Exception:
                pass
        # Raw multi-hit listings are never Jul21 company briefs.
        if company_ask and re.search(
            r"(?i)Your search matched|Multiple possible matches|Retrieval trace",
            text,
        ):
            viol = list(viol or []) + [
                "missing 'Executive Briefing' headline",
                "missing ops body "
                "('Snapshot of Operations and Market Position' or 'Operational Detail')",
            ]

        # Deduplicate
        seen: set[str] = set()
        uniq: list[str] = []
        for v in viol or []:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        viol = uniq

        needs_ops = any(
            "ops body" in v
            or "Executive Briefing" in v
            or "Corporate Profile" in v
            or "Strategic Read" in v
            or "soft_check_exception" in v
            for v in viol
        )
        if not viol or not needs_ops:
            if viol:
                pack["_contract_violations"] = viol
            return text, fixes

        pack["_contract_violations"] = viol
        fixes.append(f"contract_fail:{','.join(viol[:4])}")

        hint = (
            str(pack.get("entity_candidate") or pack.get("_entity") or "").strip()
            or _entity_hint_from_question(question)
        )
        use_rows = list(rows or [])
        if not use_rows:
            use_rows = _fetch_rows_for_hint(hint)
            if use_rows:
                fixes.append("fetched_rows_for_repair")
        use_rows = _pick_best_company_rows(use_rows, hint)
        if not use_rows:
            return text, fixes

        from app.services.db_briefing import render_db_briefing
        db_brief = render_db_briefing(
            use_rows,
            intent=intent or pack.get("_intent") or "company_profile",
            table="company_profiles",
            user_question=question,
            locale=locale,
            force=True,
        )
        if not db_brief:
            return text, fixes

        try:
            from app.services.hybrid_briefing import enrich_db_briefing
            enriched = enrich_db_briefing(
                db_brief,
                question,
                entity_hint=hint,
            )
            if enriched.get("answer"):
                db_brief = enriched["answer"]
                if enriched.get("web_sources"):
                    pack["_web_sources"] = enriched["web_sources"]
        except Exception:
            pass

        fixes.append("replaced_with_deterministic_template")
        pack["_answer_source"] = "db"
        pack["_stream_contract_fallback"] = True
        return db_brief, fixes
    except Exception:
        return text, fixes
