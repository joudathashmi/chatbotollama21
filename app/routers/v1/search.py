"""
Search endpoint:
  POST /api/v1/search — direct structured-filter company lookup, no OpenAI
"""

from __future__ import annotations

import asyncio
import json
import math

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import SEARCH_RATE_LIMIT
from app.database import COMPANY_TABLE, generate_query_and_run_query
from app.rate_limit import rate_limit
from app.schemas.search import SearchRequest

router = APIRouter()

_search_rl = rate_limit("search", *SEARCH_RATE_LIMIT)


def _clean_row(row: dict) -> dict:
    return {
        k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
        for k, v in row.items()
    }


async def _search_ndjson_generator(req: SearchRequest):
    def _run():
        return generate_query_and_run_query(
            table=COMPANY_TABLE,
            filters=req.filters,
            order_by=req.order_by,
            descending=req.descending,
            limit=req.limit,
        )

    df, _sql, _params = await asyncio.to_thread(_run)
    for row in df.to_dict(orient="records"):
        yield json.dumps(_clean_row(row), ensure_ascii=False, default=str) + "\n"


@router.post(
    "/search",
    summary="Direct company lookup — no OpenAI",
    response_description="application/x-ndjson — one JSON object per company row",
    dependencies=[Depends(_search_rl)],
)
async def search_endpoint(req: SearchRequest) -> StreamingResponse:
    """
    Query `company_profiles` directly with structured filters. OpenAI is not involved.

    Returns **NDJSON** (newline-delimited JSON): one company object per line.

    Filterable: company_name, sector, status, type_of_entity, global_headquarters,
    ultimate_parent_company, rhq_country, rhq_city, company_profile, history_in_mena,
    mena_notes, review_status, type_of_presence, rhq_entity_name, internal_code.

    Example:
    ```json
    {"filters": {"rhq_city": {"op": "=", "value": "Riyadh"}}, "order_by": "revenue_usd", "limit": 20}
    ```
    """
    return StreamingResponse(
        _search_ndjson_generator(req),
        media_type="application/x-ndjson",
    )
