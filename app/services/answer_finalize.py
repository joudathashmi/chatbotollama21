"""Single post-compose gate for every entity answer.

Why this exists: quality kept regressing case-by-case because answers
exited through ~10 composers (deterministic brief, hybrid, curation,
officeholder web, SSE stream, commentary, relationship stubs…) and only
some of them ran style scrub / enrichment. ONE finalize pass after every
composer is the only way to stop "CEO thin", "forbidden headers", and
"Strategic Read missing" from returning one ask at a time.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.style_guide import make_footer


_FORBIDDEN_SECTION_HEADERS = (
    "From the MISA Record",
    "Background (general knowledge)",
    "Verified facts",
    "Current officeholder",
    "Database draft",
)

_SECTION_RE = re.compile(r"(?m)^(#{1,3}\s+[^\n]+)\n?")


def _strip_forbidden_sections(answer: str) -> str:
    if not answer:
        return answer
    tokens = _SECTION_RE.split(answer)
    if len(tokens) < 2:
        return answer
    out = [tokens[0]]
    for i in range(1, len(tokens), 2):
        header = tokens[i]
        body = tokens[i + 1] if i + 1 < len(tokens) else ""
        h = re.sub(r"^#+\s*", "", header).strip().casefold()
        if any(bad.casefold() == h or bad.casefold() in h for bad in _FORBIDDEN_SECTION_HEADERS):
            continue
        out.append(header if header.endswith("\n") else header + "\n")
        if body and not body.startswith("\n"):
            out.append("\n")
        out.append(body)
    text = "".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# Phrases safe to delete inline everywhere. Do NOT put strategy-filler
# words like "leverage" here — those are validator-only, not scrubbers.
_INLINE_FORBIDDEN = (
    "(High)", "(Medium)", "(Low)", "(Unknown)",
    "[DB]", "[gk]", "[inferred]",
    "Source: DB", "**Source:** DB", "Source: web",
    "_(general knowledge)_",
    "Not available in the current database",
    "From the MISA Record",
    "Background (general knowledge)",
    "Internal records do not currently show",
    "Company: Multiple",
)


def _strip_forbidden_phrases(answer: str) -> str:
    if not answer:
        return answer
    text = answer
    for phrase in _INLINE_FORBIDDEN:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*\s*\*\*", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s*$", "", text)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def _ensure_sources_footer(answer: str, *, is_person: bool) -> str:
    if not answer:
        return answer
    if re.search(r"(?im)_Sources:", answer[-320:]):
        return answer
    sources = (
        ["executive records", "company_profiles"]
        if is_person
        else ["company_profiles", "company_executives", "opportunities"]
    )
    return answer.rstrip() + "\n\n" + make_footer(sources)


def finalize_answer(
    answer: str,
    *,
    user_question: str = "",
    pack: dict[str, Any] | None = None,
) -> str:
    """Normalize any composer output before it reaches the client.

    Idempotent. Safe to call on non-entity answers (returns cleaned text).
    """
    if not answer or not str(answer).strip():
        return answer or ""
    pack = pack or {}
    text = str(answer).strip()
    try:
        from app.services.prompt_masking import scrub_system_prompt_leak
        text = scrub_system_prompt_leak(text)
    except Exception:
        pass
    text = _strip_forbidden_sections(text)
    text = _strip_forbidden_phrases(text)
    try:
        from app.services.jul21_surface import apply_jul21_surface_polish
        text, jfixes = apply_jul21_surface_polish(
            text, question=user_question or "", pack=pack,
        )
        if jfixes:
            pack.setdefault("_jul21_surface_fixes", []).extend(jfixes)
    except Exception:
        pass
    # Company briefing asks that still lack Jul21 shape → deterministic
    # template. Never run this on corridor advisories — it can replace a
    # full engagement plan with a named-company Engagement Recommendation.
    _adv = (
        pack.get("_answer_source") == "strategic_advisory"
        or pack.get("_short_circuit") == "strategic_advisory"
        or pack.get("_advisory_deliverable")
    )
    if not _adv:
        try:
            from app.services.stream_repair import repair_company_answer_if_thin
            text, rfixes = repair_company_answer_if_thin(
                text,
                question=user_question or "",
                intent=pack.get("_intent"),
                rows=pack.get("_rows") or pack.get("rows") or [],
                locale=pack.get("_locale") or "en",
                pack=pack,
            )
            if rfixes:
                pack.setdefault("_finalize_repair_fixes", []).extend(rfixes)
        except Exception:
            pass
        # CEO / person asks that still lack ## Role → deterministic person brief.
        try:
            from app.services.db_briefing import (
                _question_looks_like_person,
                render_db_briefing,
            )
            from app.services.answer_contracts import person_brief_violations
            if _question_looks_like_person(user_question or ""):
                pviol = person_brief_violations(text)
                if pviol or "Executive Briefing" in text:
                    rows = pack.get("_rows") or pack.get("rows") or []
                    if not rows:
                        # Synthesize a minimal company row from the question.
                        import re as _re
                        m = _re.search(
                            r"(?i)(?:ceo|cfo|cto|chairman|president|founder)"
                            r"\s+of\s+(.+)$|"
                            r"tell\s+me\s+about\s+the\s+ceo\s+of\s+(.+)$|"
                            r"who\s+(?:runs|leads)\s+(.+)$",
                            user_question or "",
                        )
                        cname = ""
                        if m:
                            cname = next(
                                (g for g in m.groups() if g), ""
                            ).strip().rstrip("?.!,")
                        if cname:
                            rows = [{"company_name": cname}]
                    if rows:
                        person_brief = render_db_briefing(
                            rows,
                            intent="executive_lookup",
                            table="company_profiles",
                            user_question=user_question or "",
                            locale=pack.get("_locale") or "en",
                            force=True,
                        )
                        if person_brief and "## Role" in person_brief:
                            text = person_brief
                            pack.setdefault(
                                "_finalize_repair_fixes", [],
                            ).append("replaced_with_person_template")
                            pack["_answer_source"] = "db"
        except Exception:
            pass
    try:
        from app.services.curation import (
            collapse_repetitive_briefing,
            _strip_empty_source_lanes,
        )
        text = collapse_repetitive_briefing(
            text,
            deliverable=pack.get("_advisory_deliverable"),
            answer_source=pack.get("_answer_source"),
        )
        text = _strip_empty_source_lanes(text)
    except Exception:
        pass

    is_person = bool(re.search(r"(?m)^##\s+Role\b", text))
    is_company = bool(
        re.search(r"(?m)^##\s+.+\s+—\s+Executive Briefing\b", text)
        or "Operational Detail" in text
        or "Corporate Profile" in text
    )
    text = _ensure_sources_footer(text, is_person=is_person or (
        (pack.get("_intent") or "") in (
            "executive_lookup", "person_lookup", "executive_succession",
        )
    ))

    # World-class rec bar: drop soft bullets; rebuild from footprint when possible.
    try:
        from app.services.recommendation_quality import scrub_recommendation_section
        from app.services.advisory_enrichment import _named_actions_from_footprint
        db_ctx = (
            pack.get("_advisory_db_context")
            or pack.get("_db_context")
            or {}
        )
        replacements = None
        if isinstance(db_ctx, dict) and (
            db_ctx.get("expansion_targets")
            or db_ctx.get("origin_country")
            or db_ctx.get("licensed_sector_distribution")
        ):
            replacements = _named_actions_from_footprint(db_ctx)
        text, rfixes = scrub_recommendation_section(
            text, replacement_actions=replacements,
        )
        if rfixes:
            pack.setdefault("_rec_quality_fixes", []).extend(rfixes)
    except Exception:
        pass

    # Honest truncation / partial-result banners (never silent)
    if pack.get("_truncated") or pack.get("_partial_result"):
        reason = pack.get("_truncation_reason") or "context or row budget"
        banner = (
            "> **Partial result:** this answer was truncated "
            f"({reason}). Do not treat it as a complete census.\n\n"
        )
        if "Partial result" not in text[:200]:
            text = banner + text
        pack.setdefault("_data_limitations", []).append(
            f"truncated:{reason}"
        )

    try:
        from app.services.quality_gate import run_quality_gate
        db_ctx = (
            pack.get("_advisory_db_context")
            or pack.get("_db_context")
            or {}
        )
        try:
            from app.services.evidence_context import (
                strip_unusable_from_db_context,
            )
            db_ctx = strip_unusable_from_db_context(db_ctx)
        except Exception:
            pass
        text, issues, fixes = run_quality_gate(
            text,
            question=user_question or "",
            db_context=db_ctx,
            retrieval_meta=pack.get("_retrieval"),
            hard_block=True,
            deliverable=pack.get("_advisory_deliverable"),
        )
        if fixes or issues:
            pack["_quality_gate"] = {
                "fixes": fixes,
                "issues": [i.get("code") for i in (issues or [])],
            }
        try:
            from app.services.quality_eval import evaluate_answer
            qi = pack.get("_query_intent") or {}
            ev = evaluate_answer(
                text,
                question=user_question or "",
                db_context=db_ctx,
                expect_licensing_snapshot=(
                    qi.get("output_type") == "licensing_snapshot"
                ),
                expect_ranking=bool(qi.get("ranking_required")),
            )
            pack["_quality_eval"] = {
                "score": ev.get("score"),
                "pass": ev.get("pass"),
                "dimensions": ev.get("dimensions"),
            }
            if not ev.get("pass") and ev.get("score", 100) < 50:
                if "Response withheld" not in text and "Data unavailable" not in text:
                    pack["_quality_eval"]["forced_limitation"] = True
        except Exception:
            pass
        try:
            from app.services.quality_metrics import record_turn
            qi = pack.get("_query_intent") or {}
            ret = pack.get("_retrieval") or {}
            record_turn(
                intent=qi.get("task_type") or pack.get("_intent"),
                retrieval_status=(
                    ret.get("retrieval_status")
                    or pack.get("_retrieval_status")
                    or (db_ctx or {}).get("retrieval_status")
                ),
                quality_gate=pack.get("_quality_gate"),
                quality_eval=pack.get("_quality_eval"),
                truncated=bool(pack.get("_truncated")),
                filter_drop=bool(pack.get("_filter_drop")),
            )
        except Exception:
            pass
    except Exception:
        pass

    # Soft quality markers for telemetry — never raise.
    pack["_finalize"] = {
        "is_person": is_person,
        "is_company": is_company,
        "has_background": bool(re.search(r"(?m)^##\s+Background\b", text)),
        "has_ops": "Operational Detail" in text,
        "has_strategic": "Strategic Read" in text,
        "question": (user_question or "")[:120],
        "query_intent": (pack.get("_query_intent") or {}).get("task_type"),
    }
    # Forward-looking executive / succession safety net. The direct
    # exec-lookup branch web-augments succession questions, but some
    # phrasings ("who is the upcoming new CEO", "Tim Cook's successor")
    # reach the client through the general curation path, which skipped
    # it — so the answer named only the CURRENT CEO. Fire the web
    # augmentation here, on every path, when the question is
    # forward-looking and the answer has no reported-web section yet.
    try:
        intent = (pack.get("_intent") or "")
        already = pack.get("_exec_web_augmented")
        has_web = bool(re.search(
            r"(?im)^#{1,3}\s*(What'?s\s+Reported|From\s+the\s+web|Live\s+Web)",
            text,
        ))
        from app.services.chat_engine import _is_forward_looking_exec_question
        forward = (
            intent == "executive_succession"
            or _is_forward_looking_exec_question(user_question or "")
        )
        # Only for person / executive answers (a ## Role brief), never for
        # advisories or company corridor docs.
        if (forward and not already and not has_web and is_person):
            from app.database import get_openai_client
            from app.config import ADVISORY_MODEL, OPENAI_MODEL
            from app.services.chat_engine import _augment_exec_answer_with_web
            _c = get_openai_client()
            if _c is not None:
                srcs: list = []
                text = _augment_exec_answer_with_web(
                    text, user_question or "", _c,
                    ADVISORY_MODEL or OPENAI_MODEL,
                    lead_with_web=(intent == "executive_succession"),
                    capture_sources=srcs,
                    mode="succession",
                )
                pack["_exec_web_augmented"] = True
                if srcs:
                    pack.setdefault("web_sources", []).extend(srcs)
    except Exception:
        pass

    # Emit pipeline trace if present
    try:
        tr = pack.get("_pipeline_trace")
        if tr is not None and hasattr(tr, "emit"):
            tr.quality = pack.get("_quality_gate") or {}
            tr.meta["quality_eval"] = pack.get("_quality_eval") or {}
            tr.emit()
    except Exception:
        pass
    return text.strip() + ("\n" if not text.endswith("\n") else "")
