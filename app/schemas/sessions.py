from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)


class SessionRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class SessionPatch(BaseModel):
    """Partial update: rename and/or pin and/or archive."""

    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    pinned: Optional[bool] = None
    archived: Optional[bool] = None


class SessionOut(BaseModel):
    id: str
    owner_username: str
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message_count: int = 0
    pinned: bool = False
    archived_at: Optional[str] = None
    state: Optional[dict[str, Any]] = None
    summary: str = ""
    active_entity: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: list[SessionOut]


class MessageOut(BaseModel):
    id: int | str
    session_id: str
    role: str
    content: str
    answer_source: Optional[str] = None
    web_sources: Optional[list[dict]] = None
    created_at: Optional[str] = None


class SessionDetailResponse(BaseModel):
    session: SessionOut
    messages: list[MessageOut]
