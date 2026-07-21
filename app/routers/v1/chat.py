"""
Chat endpoints:
  POST /api/v1/chat      — NL question → SSE stream or JSON
  GET  /api/v1/questions — preset example questions
  POST /api/v1/feedback  — thumbs up/down + optional comment on a turn
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.auth import require_role, verify_credentials
from app.config import (
    CHAT_RATE_LIMIT,
    FEEDBACK_RATE_LIMIT,
    META_RATE_LIMIT,
    PDF_EXPORT_RATE_LIMIT,
    ROLE_ANALYST,
)
from app import config
from app.logger import logger
from app.prompts.chat_system import PRESET_QUESTIONS
from app.rate_limit import rate_limit
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.audit_log import set_audit_user
from app.services.chat_engine import chat
from app.services.feedback_log import append_feedback
from app.utils.text_validation import reject_html_markup

router = APIRouter()

_feedback_rl = rate_limit("feedback", *FEEDBACK_RATE_LIMIT)
_chat_rl = rate_limit("chat", *CHAT_RATE_LIMIT)
_pdf_rl = rate_limit("pdf_export", *PDF_EXPORT_RATE_LIMIT)
_questions_rl = rate_limit("questions", *META_RATE_LIMIT)


def _persist_chat_turn(
    user: str,
    session_id: str | None,
    question: str,
    answer: str,
    *,
    answer_source: str | None = None,
    web_sources: list | None = None,
    state: dict | None = None,
    summary: str | None = None,
) -> str | None:
    """Append user+assistant messages to a session; create one if needed.

    Returns the session id used, or None when sessions are disabled / fail.
    Persistence must never fail the chat turn.
    """
    if not getattr(config, "SESSIONS_ENABLED", True):
        return None
    try:
        from app.services.session_store import get_session_store
        store = get_session_store()
        sid = (session_id or "").strip() or None
        if sid:
            if store.get(sid, user) is None:
                # Unknown / other-user session → start a fresh one.
                sid = store.create(user).id
        else:
            sid = store.create(user).id
        store.append_message(
            sid, user, "user", question, auto_title=True,
        )
        store.append_message(
            sid,
            user,
            "assistant",
            answer or "",
            answer_source=answer_source,
            web_sources=web_sources,
        )
        if state is not None or summary is not None:
            store.save_state(sid, user, state, summary=summary)
        return sid
    except Exception as e:
        logger.warning(f"Chat session persistence failed: {e}")
        return session_id


def _prepare_session_history(
    user: str,
    session_id: str | None,
    question: str,
    history: list[dict],
) -> tuple[list[dict], dict | None, str | None, dict]:
    """Load session state card and build a smart, trimmed prompt history."""
    meta: dict = {}
    state_dict = None
    sid = (session_id or "").strip() or None
    if not getattr(config, "SESSIONS_ENABLED", True):
        return history, None, sid, meta
    try:
        from app.services.chat_context import (
            StateCard,
            prepare_prompt_history,
        )
        from app.services.session_store import get_session_store
        store = get_session_store()
        card = StateCard()
        if sid:
            sess = store.get(sid, user)
            if sess and sess.state:
                card = StateCard.from_dict(sess.state)
        effective, card, meta = prepare_prompt_history(question, history, card)
        state_dict = card.to_dict()
        return effective, state_dict, sid, meta
    except Exception as e:
        logger.warning(f"Smart history prepare failed: {e}")
        return history, None, sid, meta


def _finalize_state(
    pre_state: dict | None,
    question: str,
    result: dict | None,
) -> tuple[dict | None, str | None]:
    if pre_state is None and not getattr(config, "SESSIONS_ENABLED", True):
        return None, None
    try:
        from app.services.chat_context import StateCard, update_state_after_turn
        card = StateCard.from_dict(pre_state)
        debug = (result or {}).get("debug") if isinstance(result, dict) else None
        entity = None
        intent = None
        if isinstance(debug, dict):
            entity = debug.get("entity") or debug.get("extracted_entity")
            intent = debug.get("intent") or debug.get("detected_intent")
        # Prefer pack-ish fields on the result when present
        if isinstance(result, dict):
            entity = entity or result.get("_entity") or result.get("extracted_entity")
            intent = intent or result.get("_intent")
        card = update_state_after_turn(
            card,
            question,
            answer_source=(result or {}).get("_answer_source") if result else None,
            intent=intent,
            entity=entity,
        )
        return card.to_dict(), card.summary
    except Exception:
        return pre_state, (pre_state or {}).get("summary") if pre_state else None


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """One thumbs-up or thumbs-down on a chat turn. The chat UI emits
    this from the buttons below each assistant message. Persisted to
    feedback.jsonl so the weekly review script can pick up patterns.

    All text fields are bounded server-side (not just in the UI) —
    empty/whitespace-only required fields are rejected with 422, and
    every free-text field has a hard max length so a client that
    bypasses the browser UI can't push unbounded or blank data into
    the log. `feedback_log.append_feedback` also clamps on write as a
    second, independent layer of defense. HTML/script-tag-shaped input
    (<script>, <img onerror=...>, etc.) is rejected outright — this data
    is logged verbatim and never sanitized on read, so refusing markup
    on write is the cheaper, safer guarantee.
    """
    verdict: str = Field(..., pattern="^(up|down)$")
    question: str = Field(..., min_length=1, max_length=2_000)
    answer: str = Field(..., min_length=1, max_length=20_000)
    comment: Optional[str] = Field(None, max_length=2_000)
    intent: Optional[str] = Field(None, max_length=100)
    intent_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    entity_resolved: Optional[str] = Field(None, max_length=500)
    tables_searched: Optional[list[str]] = Field(None, max_length=50)
    row_count_total: Optional[int] = Field(None, ge=0)
    ui_locale: Optional[str] = Field("en", max_length=10)

    @field_validator("question", "answer", "comment", "entity_resolved", "intent", mode="before")
    @classmethod
    def _reject_blank(cls, v):
        """Whitespace-only input passes a naive `min_length` check
        (e.g. "   " has length 3) — strip first so it's actually
        rejected, not just cosmetically non-empty."""
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
        return v

    @field_validator("question", "answer", "comment", "entity_resolved", "intent")
    @classmethod
    def _reject_markup(cls, v):
        return reject_html_markup(v)


class FeedbackResponse(BaseModel):
    persisted: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

class PdfExportRequest(BaseModel):
    """Render a briefing answer as a downloadable PDF. The chat UI
    sends question + the rendered answer markdown + the web_sources
    list so the offline document carries the cited URLs too."""
    question: str
    answer: str
    web_sources: Optional[list[dict]] = None


@router.post(
    "/export/pdf",
    summary="Render a chat briefing as a downloadable PDF",
    response_description=(
        "application/pdf with Content-Disposition: attachment so the "
        "browser fires a download. Designed for ministers who consume "
        "intelligence via PDF, not browser."
    ),
    dependencies=[Depends(_pdf_rl), Depends(require_role(ROLE_ANALYST))],
    responses={429: {"description": "Rate limit exceeded — see Retry-After header."}},
)
async def export_pdf(req: PdfExportRequest):
    """Render the answer markdown + sources as a 1-2 page PDF with
    MISA letterhead, the question as the title, and a numbered
    sources footer. Returns the raw bytes with PDF content type.
    """
    from fastapi import HTTPException
    from fastapi.responses import Response
    from app.services.pdf_export import render_pdf
    try:
        pdf_bytes = await asyncio.to_thread(
            render_pdf, req.question, req.answer, req.web_sources,
        )
    except Exception:
        # Log the real cause server-side, but never leak raw exception
        # text (paths, library internals) to the client.
        logger.exception("PDF render failed")
        raise HTTPException(status_code=500, detail="PDF rendering failed.")
    # Build a filename that's safe for browser downloads — strip
    # punctuation, cap at 60 chars.
    import re
    safe = re.sub(r"[^a-zA-Z0-9_\- ]+", "", req.question or "briefing")[:60].strip()
    safe = re.sub(r"\s+", "-", safe) or "briefing"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="MISA-{safe}.pdf"',
            "Cache-Control": "no-store",
        },
    )


@router.post(
    "/feedback",
    summary="Thumbs up/down + optional comment on a chat turn",
    response_description=(
        "Appends a feedback record to feedback.jsonl on disk for "
        "later quality review."
    ),
    dependencies=[Depends(_feedback_rl)],
    responses={
        422: {"description": "Malformed feedback payload (blank/oversized fields, bad verdict)."},
        429: {"description": "Rate limit exceeded — see Retry-After header."},
    },
)
async def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
    """Persist a thumbs up/down with optional comment. Echo back
    persistence status. Malformed input (blank required fields,
    oversized text, invalid verdict) is rejected with 422 by
    `FeedbackRequest`'s validators before this body ever runs — this
    endpoint only has to handle the disk-write step, which never
    raises: write failures are swallowed at the log layer so a
    filesystem hiccup can't turn a thumbs-click into a 500.
    """
    rec = append_feedback({
        "verdict": req.verdict,
        "question": req.question,
        "answer_head": req.answer,  # log layer clamps to 800 chars
        "answer_chars": len(req.answer or ""),
        "comment": req.comment,
        "intent": req.intent,
        "intent_confidence": req.intent_confidence,
        "entity_resolved": req.entity_resolved,
        "tables_searched": req.tables_searched,
        "row_count_total": req.row_count_total,
        "ui_locale": req.ui_locale,
    })
    return FeedbackResponse(
        persisted="_persist_error" not in rec,
        error=rec.get("_persist_error"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _clean_value(v):
    """Coerce a single value into something pydantic / JSON can serialise.

    Handles:
      - NaN/inf floats → None
      - pandas.NaT, pandas.NA → None
      - pandas.Timestamp / datetime → ISO string
      - numpy scalars → native Python ints/floats
      - Decimal → float
      - bytes → utf-8 str (replacement on bad bytes)
      - dict/list → recursive clean
    Anything else is returned unchanged.
    """
    if v is None:
        return None
    # Float NaN / inf
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    # pandas / numpy NaT / NA
    try:
        import pandas as _pd
        if v is _pd.NaT or v is getattr(_pd, "NA", object()):
            return None
        if isinstance(v, _pd.Timestamp):
            return v.isoformat() if not _pd.isna(v) else None
    except Exception:
        pass
    # numpy scalars
    try:
        import numpy as _np
        if isinstance(v, _np.integer):
            return int(v)
        if isinstance(v, _np.floating):
            return None if _np.isnan(v) or _np.isinf(v) else float(v)
        if isinstance(v, _np.bool_):
            return bool(v)
    except Exception:
        pass
    # Decimal → float
    try:
        from decimal import Decimal as _Decimal
        if isinstance(v, _Decimal):
            return float(v)
    except Exception:
        pass
    # bytes → utf-8 str
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return None
    # datetime stdlib
    try:
        import datetime as _dt
        if isinstance(v, (_dt.datetime, _dt.date)):
            return v.isoformat()
    except Exception:
        pass
    # Recurse into containers
    if isinstance(v, dict):
        return {k: _clean_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_value(x) for x in v]
    return v


def _clean_row(row: dict) -> dict:
    return {k: _clean_value(v) for k, v in (row or {}).items()}


def _extract_rows(tool_calls: list) -> list[dict]:
    merged: list[dict] = []
    seen_ids: set = set()
    for tc in tool_calls or []:
        df = tc.get("rows_df")
        if df is None or df.empty:
            continue
        for row in df.to_dict(orient="records"):
            rid = row.get("id")
            if rid is not None and rid in seen_ids:
                continue
            if rid is not None:
                seen_ids.add(rid)
            merged.append(_clean_row(row))

    from app.services.curation import redact_rows_for_response, cap_rows_for_turn
    merged = redact_rows_for_response(merged)
    merged, _ = cap_rows_for_turn(merged, context="chat_router._extract_rows")
    return merged


def _build_trace_meta(tool_calls: list) -> list[dict]:
    return [
        {
            "table": tc.get("table"),
            "row_count": tc.get("row_count"),
            "sql": tc.get("sql"),
            "params": tc.get("params"),
            "filters": tc.get("filters"),
            "sql_entity_check_passed": tc.get("sql_entity_check_passed"),
            "row_entity_sanity_passed": tc.get("row_entity_sanity_passed"),
            "closest_names": tc.get("closest_names"),
            "error": tc.get("error"),
        }
        for tc in (tool_calls or [])
    ]


def _pack_answer_sources(result: dict | None) -> tuple[list[dict], list[dict]]:
    """Build (unified sources, legacy web_sources) for the always-on panel.

    ``web_sources`` stays a flat list of clickable entries (documents +
    web) for older clients / PDF export. ``sources`` adds typed DB rows.
    """
    result = result or {}
    from app.services.answer_sources import build_unified_sources

    doc_sources = list(result.get("doc_sources") or [])
    web_sources = list(result.get("web_sources") or [])
    # Some paths stash docs inside web_sources (doc://…); keep both.
    unified = build_unified_sources(
        doc_sources=doc_sources,
        web_sources=web_sources,
        tool_calls=result.get("tool_calls") or [],
    )
    # Legacy field: everything the reader can open (not bare DB tables).
    clickable = [
        {
            "title": s.get("title") or "(untitled)",
            "url": s.get("url") or "",
            "snippet": s.get("snippet") or "",
            "type": s.get("type"),
        }
        for s in unified
        if s.get("type") in ("document", "web") and s.get("url")
    ]
    return unified, clickable


def _polish_answer(answer: str, *, keep_citations: bool = False) -> str:
    """Post-process streaming / legacy answers through the style scrubber."""
    if not answer:
        return answer
    try:
        from app.services.curation import _scrub_backend_noise
        return _scrub_backend_noise(answer, keep_citations=keep_citations)
    except Exception:
        return answer


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------

async def _prepare_fast_stream(req: ChatRequest):
    """Run the fast-path prep (intent classify + entity extract +
    correlator). Returns a dict of prep state if the turn is
    streamable, or None if the caller should fall back to the legacy
    non-streaming SSE path.

    A turn is streamable when:
      - The intent is a company-anchored briefing (profile / presence /
        financial / opportunity / relationship) — specialised flows
        (succession, advisory, docs) stay on the legacy path
      - The entity resolves to one or more company IDs in the DB
      - The correlator finds a primary row
    """
    from app.database import get_openai_client
    from app.config import OPENAI_MODEL
    from app.services.intent_router import classify_intent
    from app.services.depth_detector import detect_depth
    from app.services.chat_engine import _extract_exec_target
    from app.services.engagement_data import resolve_company_ids
    from app.services.correlator import (
        correlate_company, bundle_summary_for_prompt,
    )

    # Company-row briefings that share the streaming curation prompt.
    _STREAMABLE = frozenset({
        "company_profile",
        "saudi_presence",
        "financial_lookup",
        "opportunity_alignment",
        "relationship_intelligence",
    })

    client = get_openai_client()
    if client is None:
        return None

    history = [{"role": m.role, "content": m.content} for m in req.history]
    user_q = req.question

    def _classify():
        try:
            return classify_intent(user_q, history, client, OPENAI_MODEL)
        except Exception:
            return {"intent": "general_research", "confidence": 0.0}

    def _extract():
        try:
            return _extract_exec_target(user_q, client, OPENAI_MODEL)
        except Exception:
            return {}

    loop = asyncio.get_event_loop()
    intent_fut = loop.run_in_executor(None, _classify)
    entity_fut = loop.run_in_executor(None, _extract)
    intent_meta = await intent_fut
    target_dict = await entity_fut

    intent = intent_meta.get("intent")
    if intent not in _STREAMABLE:
        return None

    target = (target_dict or {}).get("company")
    if not target:
        return None

    def _resolve_and_correlate():
        ids, canon = resolve_company_ids(target)
        if not ids:
            return None, None, None
        bundle = correlate_company(ids[:5])
        if not bundle.get("primary"):
            return None, None, None
        return ids, canon, bundle

    ids, canon, bundle = await loop.run_in_executor(None, _resolve_and_correlate)
    if not ids:
        return None

    depth, _ = detect_depth(user_q)
    summary = bundle_summary_for_prompt(bundle)
    primary_row = summary.get("primary") or {}
    return {
        "client": client,
        "model": OPENAI_MODEL,
        "intent": intent,
        "target": target,
        "canon": canon,
        "depth": depth,
        "rows": [primary_row],
    }


async def _stream_fast_path(req: ChatRequest, prep: dict) -> AsyncGenerator[str, None]:
    """Stream the curation LLM call as SSE chunk events using the
    prep state from _prepare_fast_stream(). Yields status + chunk +
    done events. NEVER returns a value (it's an async generator)."""
    from app.services.streaming_curation import stream_company_insights_chunks

    yield _sse({"type": "status",
                "message": f"Composing briefing for {prep['canon'] or prep['target']}…"})

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def _producer():
        try:
            for chunk in stream_company_insights_chunks(
                prep["rows"], req.question,
                locale=req.locale or "en",
                entity_candidate=prep["target"],
                entity_matched=prep["canon"] or prep["target"],
                table="company_profiles",
                client=prep["client"], model=prep["model"],
                intent=prep["intent"], depth=prep["depth"],
            ):
                queue.put_nowait(chunk)
        except Exception:
            pass
        queue.put_nowait(SENTINEL)

    loop.run_in_executor(None, _producer)

    streamed_anything = False
    while True:
        chunk = await queue.get()
        if chunk is SENTINEL:
            break
        streamed_anything = True
        yield _sse({"type": "chunk", "text": chunk})

    if streamed_anything:
        yield _sse({"type": "done", "trace": []})


async def _chat_sse_generator(
    req: ChatRequest, user: str,
) -> AsyncGenerator[str, None]:
    """SSE generator. Tries the fast streaming path first (text appears
    in ~2s for company_profile queries); falls back to the legacy path
    (full chat() call, then paragraph-chunked SSE) for everything else."""
    # Try fast streaming path
    prep = await _prepare_fast_stream(req)
    if prep is not None:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        _hist, pre_state, sid, hist_meta = _prepare_session_history(
            user, req.session_id, req.question, history,
        )
        # Collect streamed text for session persistence; replace the
        # fast-path "done" with one that includes session_id.
        parts: list[str] = []
        async for evt in _stream_fast_path(req, prep):
            try:
                line = evt.strip()
                if line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                else:
                    payload = {}
                if payload.get("type") == "chunk" and payload.get("text"):
                    parts.append(payload["text"])
                    yield evt
                elif payload.get("type") == "done":
                    continue
                else:
                    yield evt
            except Exception:
                yield evt
        answer = "".join(parts)
        answer = _polish_answer(answer, keep_citations=False)
        # Synthetic result so DB table from the fast path shows in Sources.
        fake_result = {
            "_answer_source": "db",
            "tool_calls": [{
                "table": "company_profiles",
                "row_count": len(prep.get("rows") or []),
            }],
        }
        sources, clickable = _pack_answer_sources(fake_result)
        post_state, summary = _finalize_state(
            pre_state, req.question, fake_result,
        )
        out_sid = _persist_chat_turn(
            user, sid or req.session_id, req.question, answer,
            answer_source="db",
            web_sources=clickable or sources or None,
            state=post_state,
            summary=summary,
        )
        # Rewrite the bubble with polished text (citations / noise scrubbed).
        if answer:
            yield _sse({"type": "final", "text": answer})
        done = {
            "type": "done",
            "trace": [],
            "sources": sources or None,
            "web_sources": clickable or None,
        }
        if out_sid:
            done["session_id"] = out_sid
        if hist_meta.get("topic_shift"):
            done["topic_shift"] = True
        yield _sse(done)
        return

    # Legacy path
    yield _sse({"type": "status", "message": "Routing query and retrieving data…"})

    history = [{"role": m.role, "content": m.content} for m in req.history]
    history, pre_state, sid, hist_meta = _prepare_session_history(
        user, req.session_id, req.question, history,
    )
    if hist_meta.get("topic_shift"):
        yield _sse({
            "type": "status",
            "message": "New topic detected — prior chat context cleared for this answer.",
        })
    result = await asyncio.to_thread(chat, req.question, history, req.locale)

    if result.get("error"):
        yield _sse({
            "type": "error",
            "message": result["error"],
            "recovery": ["retry", "rephrase", "documents"],
        })
        return

    rows = _extract_rows(result.get("tool_calls") or [])
    if rows:
        yield _sse({"type": "rows", "data": rows, "row_count": len(rows)})

    sources, clickable = _pack_answer_sources(result)
    keep_cites = any(s.get("type") in ("web", "document") for s in sources)
    answer = _polish_answer(result.get("answer") or "", keep_citations=keep_cites)
    if not answer.strip():
        yield _sse({
            "type": "error",
            "message": "No answer was produced for this question.",
            "recovery": ["retry", "rephrase", "documents"],
        })
        return

    for para in answer.split("\n\n"):
        if para.strip():
            yield _sse({"type": "chunk", "text": para + "\n\n"})

    post_state, summary = _finalize_state(pre_state, req.question, result)
    out_sid = _persist_chat_turn(
        user,
        sid or req.session_id,
        req.question,
        answer,
        answer_source=result.get("_answer_source"),
        web_sources=clickable or sources or None,
        state=post_state,
        summary=summary,
    )
    done = {
        "type": "done",
        "trace": _build_trace_meta(result.get("tool_calls") or []),
        "sources": sources or None,
        "web_sources": clickable or None,
    }
    if out_sid:
        done["session_id"] = out_sid
    if hist_meta.get("topic_shift"):
        done["topic_shift"] = True
    yield _sse(done)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    summary="NL chat — streaming (SSE) or JSON",
    response_description=(
        "stream=true → text/event-stream (status | rows | chunk | error | done); "
        "stream=false → application/json with answer, rows, trace, error"
    ),
    dependencies=[Depends(_chat_rl)],
)
async def chat_endpoint(req: ChatRequest, user: str = Depends(verify_credentials)):
    """
    Send a natural-language question. Use **`stream`** to choose the response format.

    **`stream: true` (default)** — Server-Sent Events:

    | Event | Payload |
    |-------|---------|
    | `status` | processing update |
    | `rows` | filtered table data (list of company objects) |
    | `chunk` | one paragraph of commentary text |
    | `error` | fatal error |
    | `done` | SQL trace metadata |

    **`stream: false`** — single JSON:
    ```json
    {"answer": "...", "rows": [...], "trace": [...], "error": null}
    ```

    Answers are curated by OpenAI from retrieved rows that are **privacy-filtered**
    first (internal comments, reviewer notes and audit fields are stripped) and sent
    with no-retention. When the DB has no match, OpenAI answers from general knowledge,
    clearly labelled as not sourced from the MISA database. Curation/fallback can be
    disabled via `MISA_CHAT_CURATION` / `MISA_CHAT_FALLBACK`.
    """
    # Risk-20-1: attribute this request's security events (blocked table
    set_audit_user(user)
    if not req.stream:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        history, pre_state, sid, hist_meta = _prepare_session_history(
            user, req.session_id, req.question, history,
        )
        result = await asyncio.to_thread(chat, req.question, history, req.locale)
        if hist_meta.get("topic_shift") and isinstance(result, dict):
            result = dict(result)
            result["_topic_shift"] = True
        debug_payload = None
        if req.debug:
            # Step 10 of the spec: when debug=true, expose intent /
            # entity / aliases / tables searched / evidence used /
            # resolver reasoning. Helper lives in chat_engine so the
            # logic stays next to the pipeline that produces it.
            try:
                from app.services.chat_engine import build_debug_payload
                debug_payload = build_debug_payload(req.question, result)
            except Exception as e:
                debug_payload = {"error": f"debug builder failed: {e}"}
            if debug_payload is not None and hist_meta:
                debug_payload["history_meta"] = hist_meta
        # Prefer typed unified sources; keep web_sources for older clients.
        sources, clickable = _pack_answer_sources(result)
        keep_cites = any(s.get("type") in ("web", "document") for s in sources)
        polished = _polish_answer(
            result.get("answer") or "", keep_citations=keep_cites,
        )
        post_state, summary = _finalize_state(pre_state, req.question, {
            **(result or {}),
            "debug": debug_payload,
        })
        out_sid = None
        if not result.get("error"):
            out_sid = _persist_chat_turn(
                user,
                sid or req.session_id,
                req.question,
                polished,
                answer_source=result.get("_answer_source"),
                web_sources=clickable or sources or None,
                state=post_state,
                summary=summary,
            )
        return ChatResponse(
            answer=polished,
            rows=_extract_rows(result.get("tool_calls") or []),
            trace=_build_trace_meta(result.get("tool_calls") or []),
            error=result.get("error"),
            debug=debug_payload,
            web_sources=clickable or None,
            sources=sources or None,
            session_id=out_sid,
        )

    return StreamingResponse(
        _chat_sse_generator(req, user),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/questions", summary="Preset example questions", dependencies=[Depends(_questions_rl)])
async def get_questions():
    """Returns preset questions originally shown as quick-start cards in the Streamlit UI."""
    return {"questions": PRESET_QUESTIONS}
