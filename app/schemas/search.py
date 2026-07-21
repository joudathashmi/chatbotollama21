from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    filters: dict[str, Any] = {}
    order_by: Optional[str] = None
    descending: bool = True
    limit: int = 25
