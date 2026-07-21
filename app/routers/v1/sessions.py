"""Chat sessions API — persistent conversation history per user.

  POST   /api/v1/sessions              — create session
  GET    /api/v1/sessions              — list (q=, include_archived=)
  GET    /api/v1/sessions/{id}         — session + messages
  PATCH  /api/v1/sessions/{id}         — rename / pin / archive
  DELETE /api/v1/sessions/{id}         — delete
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app import config
from app.auth import verify_credentials
from app.config import SESSIONS_RATE_LIMIT
from app.rate_limit import rate_limit
from app.schemas.sessions import (
    MessageOut,
    SessionCreate,
    SessionDetailResponse,
    SessionListResponse,
    SessionOut,
    SessionPatch,
)
from app.services.session_store import get_session_store
from app.utils.error_handler import create_error_response

router = APIRouter(prefix="/sessions", tags=["sessions"])
_rl = rate_limit("sessions", *SESSIONS_RATE_LIMIT)


def _err(code: str, message: str, status: int, path: str) -> JSONResponse:
    body = create_error_response(
        code=code, message=message, status=status, path=path
    ).model_dump()
    return JSONResponse(status_code=status, content=body)


def _session_out(s) -> SessionOut:
    return SessionOut(**s.to_dict())


@router.post(
    "",
    summary="Create a chat session",
    response_model=SessionOut,
    dependencies=[Depends(_rl)],
)
async def create_session(
    body: SessionCreate | None = None,
    user: str = Depends(verify_credentials),
):
    if not config.SESSIONS_ENABLED:
        return _err("DISABLED", "Chat sessions are disabled.", 503, "/api/v1/sessions")
    title = (body.title if body else None)
    sess = get_session_store().create(user, title=title)
    return _session_out(sess)


@router.get(
    "",
    summary="List chat sessions for the current user",
    response_model=SessionListResponse,
    dependencies=[Depends(_rl)],
)
async def list_sessions(
    user: str = Depends(verify_credentials),
    q: str | None = Query(default=None, max_length=100),
    include_archived: bool = Query(default=False),
):
    if not config.SESSIONS_ENABLED:
        return _err("DISABLED", "Chat sessions are disabled.", 503, "/api/v1/sessions")
    rows = get_session_store().list_for_user(
        user, include_archived=include_archived, q=q,
    )
    return SessionListResponse(sessions=[_session_out(s) for s in rows])


@router.get(
    "/{session_id}",
    summary="Get a session and its messages",
    response_model=SessionDetailResponse,
    dependencies=[Depends(_rl)],
)
async def get_session(session_id: str, user: str = Depends(verify_credentials)):
    if not config.SESSIONS_ENABLED:
        return _err(
            "DISABLED", "Chat sessions are disabled.", 503,
            f"/api/v1/sessions/{session_id}",
        )
    store = get_session_store()
    sess = store.get(session_id, user)
    if sess is None:
        return _err(
            "NOT_FOUND", "Session not found.", 404,
            f"/api/v1/sessions/{session_id}",
        )
    msgs = store.list_messages(session_id, user) or []
    return SessionDetailResponse(
        session=_session_out(sess),
        messages=[MessageOut(**m.to_dict()) for m in msgs],
    )


@router.patch(
    "/{session_id}",
    summary="Rename, pin, or archive a chat session",
    response_model=SessionOut,
    dependencies=[Depends(_rl)],
)
async def patch_session(
    session_id: str,
    body: SessionPatch,
    user: str = Depends(verify_credentials),
):
    if not config.SESSIONS_ENABLED:
        return _err(
            "DISABLED", "Chat sessions are disabled.", 503,
            f"/api/v1/sessions/{session_id}",
        )
    store = get_session_store()
    sess = store.get(session_id, user)
    if sess is None:
        return _err(
            "NOT_FOUND", "Session not found.", 404,
            f"/api/v1/sessions/{session_id}",
        )
    if body.title is not None:
        sess = store.rename(session_id, user, body.title)
    if body.pinned is not None:
        sess = store.set_pinned(session_id, user, body.pinned)
    if body.archived is not None:
        sess = store.set_archived(session_id, user, body.archived)
    if sess is None:
        return _err(
            "NOT_FOUND", "Session not found.", 404,
            f"/api/v1/sessions/{session_id}",
        )
    return _session_out(sess)


@router.delete(
    "/{session_id}",
    summary="Delete a chat session",
    dependencies=[Depends(_rl)],
)
async def delete_session(session_id: str, user: str = Depends(verify_credentials)):
    if not config.SESSIONS_ENABLED:
        return _err(
            "DISABLED", "Chat sessions are disabled.", 503,
            f"/api/v1/sessions/{session_id}",
        )
    ok = get_session_store().delete(session_id, user)
    if not ok:
        return _err(
            "NOT_FOUND", "Session not found.", 404,
            f"/api/v1/sessions/{session_id}",
        )
    return {"ok": True, "id": session_id}
