from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.config import MAX_USER_MESSAGE_CHARS
from app.utils.text_validation import reject_html_markup, strip_html_to_plaintext


class HistoryMessage(BaseModel):
    role: str = Field(..., min_length=1, max_length=32)
    content: str = Field(..., max_length=MAX_USER_MESSAGE_CHARS)

    @field_validator("content")
    @classmethod
    def _strip_history_markup(cls, v):
        # Chat UI may re-send citation HTML (<sup class="cite">) from a
        # prior turn. Strip it instead of 422 — otherwise every follow-up
        # after a web-cited answer fails with a cryptic client error.
        if isinstance(v, str):
            return strip_html_to_plaintext(v)
        return v


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_USER_MESSAGE_CHARS)
    history: list[HistoryMessage] = []
    locale: str = "en"
    stream: bool = True
    # When true, ChatResponse.debug is populated with the full pipeline
    # trace (intent, entity, aliases, tables searched, evidence rows used,
    # resolver/intent reasoning). Diagnostic-only — no behaviour change.
    debug: bool = False
    # Optional persistent session id. When sessions are enabled and this is
    # omitted, the server creates a new session for the turn.
    session_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def _reject_markup(cls, v):
        return reject_html_markup(v)


class ChatResponse(BaseModel):
    answer: str
    rows: list[dict]
    trace: list[dict]
    error: Optional[str] = None
    # Populated only when the request had debug=true. Otherwise None so
    # production responses stay lean.
    debug: Optional[dict] = None
    # Live-web / document / DB provenance for the Sources panel.
    # Order matches citation numbering where applicable.
    web_sources: Optional[list[dict]] = None
    # Echo / assigned chat session id when persistence is enabled.
    session_id: Optional[str] = None
    # Unified sources (documents + web + DB tables) for the always-on panel.
    sources: Optional[list[dict]] = None
    # Quality / retrieval observability (safe for clients — no PII payloads).
    trace_id: Optional[str] = None
    intent: Optional[dict] = None
    retrieval_status: Optional[str] = None
    quality: Optional[dict] = None
    data_limitations: Optional[list[str]] = None
