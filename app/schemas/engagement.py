from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class EngagementRequest(BaseModel):
    entity: str
    mode: Literal["quick", "full"]
    context: str = ""
    stream: bool = True


class EngagementResponse(BaseModel):
    text: str
    error: Optional[str] = None
