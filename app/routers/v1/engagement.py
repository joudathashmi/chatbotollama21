"""
Engagement dossier endpoints:
  POST /api/v1/engagement/generate — streaming SSE or JSON dossier
  GET  /api/v1/engagement/modes    — available dossier modes
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import ENGAGEMENT_RATE_LIMIT, META_RATE_LIMIT
from app.rate_limit import rate_limit
from app.schemas.engagement import EngagementRequest, EngagementResponse
from app.services.engagement_engine import engagement_generate, engagement_sse_stream

router = APIRouter(prefix="/engagement")

_engagement_rl = rate_limit("engagement", *ENGAGEMENT_RATE_LIMIT)
_modes_rl = rate_limit("engagement_modes", *META_RATE_LIMIT)

_MODES = [
    {
        "id": "quick",
        "label": "Quick brief",
        "description": "Four tiles: strategic relevance, signals, opening lines, deferral.",
    },
    {
        "id": "full",
        "label": "Full dossier",
        "description": "19-section institutional dossier: thesis, qualification, conversion pathway, risks.",
    },
]


@router.post(
    "/generate",
    summary="Generate engagement dossier — SSE or JSON",
    response_description=(
        "stream=true → text/event-stream; "
        "stream=false → application/json with text and error"
    ),
    dependencies=[Depends(_engagement_rl)],
)
async def generate_endpoint(req: EngagementRequest):
    """
    Generate a MISA-grade strategic investor engagement dossier.

    Uses OpenAI Responses API with **web_search** for live research.

    **`stream: true` (default)** — Server-Sent Events:

    | Event | Payload |
    |-------|---------|
    | `{"meta": {"phase": "opening"}}` | before API call |
    | `{"meta": {"phase": "research"}}` | web_search started |
    | `{"delta": "...text..."}` | markdown chunk |
    | `{"error": "...message..."}` | error (followed by [DONE]) |
    | `[DONE]` | stream finished |

    **`stream: false`** — single JSON:
    ```json
    {"text": "...full markdown dossier...", "error": null}
    ```
    """
    if not req.stream:
        result = await engagement_generate(req.entity, req.mode, req.context)
        return EngagementResponse(**result)

    return StreamingResponse(
        engagement_sse_stream(req.entity, req.mode, req.context),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/modes", summary="Available dossier modes", dependencies=[Depends(_modes_rl)])
async def get_modes():
    """Returns the two supported dossier modes: quick (4 tiles) and full (19 sections)."""
    return {"modes": _MODES}
