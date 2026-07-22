"""Hybrid enrichment for DB-first briefs.

Residency rules:
  - Postgres / document *row JSON* never goes to Azure/OpenAI.
  - Deterministic DB brief stays local.
  - Document library: cite excerpts locally (no cloud compose).
  - Live web: question text + public snippets may use public OpenAI /
    Azure for a short narrative section (no DB rows).

Goal: restore earlier multi-source narrative quality without the slow
Ollama full-row rewrite and without leaking MISA tables to the cloud.
"""

from __future__ import annotations

import re
from typing import Any

from app.logger import logger
from app.services.style_guide import HEADERS, make_footer


def _strip_section(answer: str, header_substr: str) -> str:
    """Remove a ## section whose header contains header_substr (casefold)."""
    if not answer:
        return answer
    section_re = re.compile(r"(?m)^(#{1,3}\s+[^\n]+)\n?")
    tokens = section_re.split(answer)
    if len(tokens) < 2:
        return answer
    out = [tokens[0]]
    needle = header_substr.casefold()
    for i in range(1, len(tokens), 2):
        header = tokens[i]
        body = tokens[i + 1] if i + 1 < len(tokens) else ""
        if needle in header.casefold():
            continue
        out.append(header if header.endswith("\n") else header + "\n")
        if body and not body.startswith("\n"):
            out.append("\n")
        out.append(body)
    text = "".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def _doc_section(question: str) -> tuple[str, list]:
    """Deterministic document cites — no LLM, residency-safe."""
    try:
        from app import config
        if not getattr(config, "DOCUMENTS_ENABLED", False):
            return "", []
        from app.services.audit_log import get_audit_user
        from app.services.document_store import get_document_store
        user = get_audit_user()
        if not user or user in ("unknown", "anonymous", "invalid-token"):
            return "", []
        hits = get_document_store().retrieve(question, user)
        if not hits:
            return "", []
        min_score = getattr(config, "DOCUMENTS_RETRIEVAL_MIN_SCORE", 0.12)
        strong = [h for h in hits if getattr(h, "score", 0) >= min_score] or list(hits)
        if not strong:
            return "", []
        lines = [HEADERS["from_documents"], ""]
        sources = []
        for h in strong[:6]:
            cite = f"[doc:{h.filename}#{h.chunk_index}]"
            snippet = (h.text or "").strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277].rsplit(" ", 1)[0] + "…"
            lines.append(f"- {cite} {snippet}")
            sources.append({
                "title": h.filename,
                "url": f"doc://{h.document_id}#{h.chunk_index}",
                "snippet": snippet,
                "type": "document",
            })
        lines.append("")
        lines.append("_Sources: document library._")
        return "\n".join(lines).strip(), sources
    except Exception as e:
        logger.info(f"hybrid doc section skipped: {e}")
        return "", []


def _web_section(question: str) -> tuple[str, list]:
    """Live web section — question + public snippets only (no DB rows)."""
    try:
        from app.services.llm_residency import public_web_allowed
        if not public_web_allowed():
            return "", []
        from app.services import web_search
        from app.services.document_ingest import _curate_web_section
        results = web_search.search(question, max_results=6)
        if not results:
            return "", []
        return _curate_web_section(question, results)
    except Exception as e:
        logger.info(f"hybrid web section skipped: {e}")
        return "", []


def _azure_public_narrative(
    question: str,
    *,
    web_section: str,
    entity_hint: str = "",
) -> str:
    """Short Azure narrative from question + web text only — never DB JSON.

    Produces Background + Strategic Read complementary prose so the brief
    feels like the earlier curated answers, while MISA numbers stay in the
    local deterministic block above.
    """
    if not web_section and not question:
        return ""
    try:
        from app.database import get_openai_client
        from app.config import OPENAI_MODEL
        from app.services.style_guide import STYLE_GUIDE_PROMPT
        client = get_openai_client()
        if client is None:
            return ""
        prompt = (
            STYLE_GUIDE_PROMPT
            + "\nYou are writing ONLY the public-context complement of a MISA brief.\n"
            "You do NOT have MISA database access. Do not invent MISA headcount, "
            "RHQ status, revenue, or internal contacts.\n\n"
            "Write exactly these sections (headers VERBATIM), then STOP:\n"
            f"{HEADERS['person_background']}\n"
            "3–5 bullets of stable public context (company positioning / market / "
            "products, or person career if the question is about a person).\n\n"
            f"{HEADERS['strategic_read']}\n"
            "2–4 concrete engagement angles for MISA grounded in the WEB evidence "
            "and the user question. No Vision filler. No 'leverage' / 'explore'.\n\n"
            "If web evidence is thin, keep Background short; still give Strategic Read.\n"
            "Do not write Corporate Profile tables. Do not write From the web again.\n\n"
            f"ENTITY HINT: {entity_hint or '(from question)'}\n"
            f"QUESTION:\n{question.strip()}\n\n"
            f"WEB SECTION (already shown to the reader — use as evidence, don't repeat):\n"
            f"{(web_section or '(none)')[:3500]}\n"
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=700,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return ""
        # Keep only Background + Strategic Read if model added junk.
        keep = []
        for block in re.split(r"(?m)(?=^## )", text):
            h = block.lstrip()
            if h.startswith(HEADERS["person_background"]) or h.startswith(
                HEADERS["strategic_read"]
            ) or h.lower().startswith("## background") or "strategic read" in h[:40].lower():
                keep.append(block.strip())
        return "\n\n".join(keep).strip() if keep else text
    except Exception as e:
        logger.info(f"hybrid azure narrative skipped: {e}")
        return ""


def _extract_role_identity(role_brief: str) -> tuple[str, str, str]:
    """Pull verified name / title / employer from the ## Role lead sentence."""
    m = re.search(
        r"\*\*(.+?)\s+is\s+(.+?)\s+at\s+(.+?)\.\*\*",
        role_brief or "",
        flags=re.I,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip(" .*")
    m2 = re.search(r"\*\*(.+?)\s+is\s+(.+?)\.\*\*", role_brief or "", flags=re.I)
    if m2:
        return m2.group(1).strip(), m2.group(2).strip(), ""
    return "", "", ""


def _person_public_background(
    question: str,
    *,
    name: str,
    title: str = "",
    employer: str = "",
) -> str:
    """Question-only Azure Background for a verified executive — never DB JSON.

    Locks the MISA-verified name/title/employer so the model cannot invent a
    different CEO (the John Ternus failure mode).
    """
    if not name:
        return ""
    try:
        from app.services.llm_residency import public_web_allowed
        web_sec = ""
        if public_web_allowed():
            from app.services import web_search
            from app.services.document_ingest import _curate_web_section
            q = f"{name} {title} {employer} career biography".strip()
            results = web_search.search(q or question, max_results=6)
            if results:
                web_sec, _ = _curate_web_section(q or question, results)
                if web_sec and len(web_sec) > 2200:
                    web_sec = "\n".join(web_sec.splitlines()[:40])
        from app.database import get_openai_client
        from app.config import OPENAI_MODEL
        from app.services.style_guide import STYLE_GUIDE_PROMPT
        client = get_openai_client()
        if client is None:
            return ""
        locked = f"{name}" + (f", {title}" if title else "") + (
            f" at {employer}" if employer else ""
        )
        prompt = (
            STYLE_GUIDE_PROMPT
            + "\nYou are writing ONLY public career Background for a MISA "
            "executive brief.\n"
            "VERIFIED IDENTITY (do not change, contradict, or replace): "
            f"{locked}.\n"
            "Do NOT invent a different officeholder. Do NOT write ## Role.\n"
            "Do NOT invent MISA headcount, RHQ, or revenue figures.\n\n"
            f"Write exactly this section header VERBATIM, then STOP:\n"
            f"{HEADERS['person_background']}\n"
            "4–7 stable public bullets: career path, prior roles, education, "
            "notable public moves. Cap ~180 words. No speculation. No "
            "training-cutoff talk. No 'From the web' header.\n\n"
            f"QUESTION:\n{question.strip()}\n\n"
            f"PUBLIC WEB EVIDENCE:\n{(web_sec or '(none)')[:3500]}\n"
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=550,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return ""
        # Keep only Background.
        keep = []
        for block in re.split(r"(?m)(?=^## )", text):
            h = block.lstrip()
            if h.startswith(HEADERS["person_background"]) or h.lower().startswith(
                "## background"
            ):
                keep.append(block.strip())
        out = "\n\n".join(keep).strip() if keep else text
        # Refuse if the model swapped the person name.
        if name.split()[0].casefold() not in out.casefold() and name.casefold() not in out.casefold():
            # Still accept generic career bullets that omit the surname.
            pass
        wrong = re.search(
            r"\*\*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+is\s+(?:the\s+)?(?:CEO|Chief)",
            out,
        )
        if wrong and name.casefold() not in wrong.group(1).casefold():
            logger.info(
                "person public background rejected — model swapped identity "
                f"to {wrong.group(1)!r}"
            )
            return ""
        return out
    except Exception as e:
        logger.info(f"person public background skipped: {e}")
        return ""


def _merge_background_sections(core: str, public_bg: str) -> str:
    """Append public Background bullets under an existing ## Background."""
    if not public_bg:
        return core
    pub_bullets = [
        ln for ln in public_bg.splitlines()
        if ln.strip().startswith(("*", "-", "•"))
    ]
    if not pub_bullets:
        return core

    def _sig(ln: str) -> str:
        s = re.sub(r"^[\*\-•]\s*", "", ln.strip()).casefold()
        s = re.sub(r"\s+", " ", s)
        # Near-dup: first ~10 tokens after dropping leading name noise.
        toks = s.split()
        if toks and toks[0] in {"he", "she", "they"}:
            toks = toks[1:]
        return " ".join(toks[:10])

    if re.search(r"(?m)^##\s+Background\b", core):
        # Insert before the next ## section or end.
        m = re.search(r"(?m)^##\s+Background\b.*?(?=^##\s|\Z)", core, flags=re.S)
        if not m:
            return core.rstrip() + "\n\n" + public_bg
        block = m.group(0).rstrip()
        existing = {
            _sig(ln)
            for ln in block.splitlines()
            if ln.strip().startswith(("*", "-", "•"))
        }
        extra = []
        for ln in pub_bullets:
            key = _sig(ln)
            if not key or key in existing:
                continue
            # Also skip if any existing bullet shares a long prefix.
            if any(key[:40] and key[:40] in ex for ex in existing):
                continue
            existing.add(key)
            extra.append(ln if ln.strip().startswith("*") else "* " + ln.strip().lstrip("-• "))
            if len(extra) >= 4:
                break
        if not extra:
            return core
        merged_block = block + "\n" + "\n".join(extra) + "\n"
        return core[: m.start()] + merged_block + core[m.end():]
    # No Background yet — insert after Role block.
    m = re.search(r"(?m)^##\s+Role\b.*?(?=^##\s|\Z)", core, flags=re.S)
    if m:
        insert = "\n" + HEADERS["person_background"] + "\n\n" + "\n".join(pub_bullets[:6]) + "\n\n"
        return core[: m.end()] + insert + core[m.end():]
    return core.rstrip() + "\n\n" + public_bg


def enrich_db_briefing(
    db_brief: str,
    question: str,
    *,
    entity_hint: str = "",
    include_web: bool = True,
    include_docs: bool = True,
    include_public_narrative: bool = False,
) -> dict[str, Any]:
    """Stitch DB core (incl. Operational Detail + Strategic Read) with docs/web.

    Public Azure narrative is OFF by default for companies — MISA Operational
    Detail is the insight layer. Person briefs get a question-only public
    Background layered under ## Background (verified Role stays authoritative).
    """
    if not db_brief:
        return {"answer": "", "doc_sources": [], "web_sources": []}

    core = re.sub(r"(?im)\n*_Sources:[^\n]*_?\s*$", "", db_brief).strip()
    is_person = bool(re.search(r"(?m)^##\s+Role\b", core))
    is_engagement = bool(re.search(r"(?m)^##\s+Engagement Recommendation\b", core))
    has_ops = (
        "Operational Detail" in core
        or "Snapshot of Operations" in core
    )

    doc_sec, doc_sources = ("", [])
    if include_docs:
        doc_sec, doc_sources = _doc_section(question)

    web_sec, web_sources = ("", [])
    # Engagement plans already have Snapshot/MENA/Strategic Read — skip web.
    if include_web and not is_person and not is_engagement:
        web_sec, web_sources = _web_section(question)
        cap = 1200 if has_ops else 1800
        max_bullets = 3 if has_ops else 4
        if web_sec and len(web_sec) > cap:
            lines = web_sec.splitlines()
            kept: list[str] = []
            n_bullets = 0
            for ln in lines:
                if ln.strip().startswith(("-", "*", "•")):
                    n_bullets += 1
                    if n_bullets > max_bullets:
                        continue
                kept.append(ln)
            web_sec = "\n".join(kept).strip()

    if is_person and include_web:
        name, title, employer = _extract_role_identity(core)
        public_bg = _person_public_background(
            question, name=name, title=title, employer=employer or entity_hint,
        )
        if public_bg:
            core = _merge_background_sections(core, public_bg)
            web_sources = [{"title": "public background", "type": "web"}]

    narrative = ""
    if include_public_narrative and not is_person:
        narrative = _azure_public_narrative(
            question, web_section=web_sec, entity_hint=entity_hint,
        )
        narrative = _strip_section(narrative, "Strategic Read")
        if has_ops:
            narrative = _strip_section(narrative, "Background")

    parts = [core]
    if doc_sec:
        parts.append(doc_sec)
    if web_sec and not is_person:
        parts.append(web_sec)
    if narrative:
        parts.append(narrative)

    answer = "\n\n".join(p for p in parts if p).strip()
    try:
        from app.services.curation import (
            collapse_repetitive_briefing,
            _strip_empty_source_lanes,
        )
        answer = collapse_repetitive_briefing(
            answer,
            answer_source="hybrid_briefing",
        )
        answer = _strip_empty_source_lanes(answer)
    except Exception:
        pass

    sources = ["company_profiles", "company_executives"]
    if "company_geographic_revenues" in core or "Geographic revenue" in core:
        sources.append("company_geographic_revenues")
    if "opportunities" in core.lower() or "Strategic Read" in core:
        sources.append("opportunities")
    if is_person:
        sources = ["executive records", "company_profiles"]
        if web_sources:
            sources.append("public background")
    if doc_sources:
        sources.append("document library")
    if web_sources and not is_person:
        sources.append("public reporting")
    footer = make_footer(sources)
    if footer and "_Sources:" not in answer[-240:]:
        answer = answer.rstrip() + "\n\n" + footer
    return {
        "answer": answer.strip() + "\n",
        "doc_sources": doc_sources,
        "web_sources": web_sources,
    }
