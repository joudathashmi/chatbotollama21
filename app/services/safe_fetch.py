"""Safe DB/API fetch helpers — never convert exceptions into empty facts."""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from app.services.retrieval_status import (
    RetrievalResult,
    RetrievalStatus,
    classify_exception,
    failure,
    success,
    success_counts,
)

log = logging.getLogger(__name__)
T = TypeVar("T")


def safe_list_fetch(
    fn: Callable[[], list],
    *,
    source_name: str,
    filters: dict | None = None,
    query: str = "",
) -> RetrievalResult:
    """Run a list-returning fetch; exceptions → failure, not []."""
    try:
        rows = fn()
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            return failure(
                RetrievalStatus.MALFORMED_RESPONSE,
                source_name=source_name,
                error=f"expected list, got {type(rows).__name__}",
                filters=filters,
                query=query,
            )
        return success(
            rows,
            source_name=source_name,
            filters=filters,
            query=query,
        )
    except Exception as exc:
        log.warning(
            "safe_list_fetch failed source=%s err=%s",
            source_name, exc,
        )
        return failure(
            classify_exception(exc),
            source_name=source_name,
            error=str(exc),
            filters=filters,
            query=query,
        )


def safe_count_fetch(
    fn: Callable[[], int],
    *,
    source_name: str,
    filters: dict | None = None,
    query: str = "",
) -> RetrievalResult:
    try:
        n = fn()
        return success_counts(
            source_name=source_name,
            count=int(n),
            filters=filters,
            query=query,
        )
    except Exception as exc:
        log.warning(
            "safe_count_fetch failed source=%s err=%s",
            source_name, exc,
        )
        return failure(
            classify_exception(exc),
            source_name=source_name,
            error=str(exc),
            filters=filters,
            query=query,
        )


def safe_dict_fetch(
    fn: Callable[[], dict],
    *,
    source_name: str,
    filters: dict | None = None,
    count_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run a dict-returning aggregate fetch.

    On success: returns the dict plus retrieval metadata.
    On failure: returns a dict with zeros CLEARED and ``_db_error`` /
    ``do_not_claim_zero`` set — callers must not treat missing counts
    as verified zeros.
    """
    try:
        out = fn()
        if not isinstance(out, dict):
            return {
                "_db_error": f"expected dict, got {type(out).__name__}",
                "retrieval_status": RetrievalStatus.MALFORMED_RESPONSE.value,
                "counts_unavailable": True,
                "do_not_claim_zero": True,
                "footprint_data_unavailable": True,
            }
        # Attach success envelope if caller didn't
        if "retrieval_status" not in out:
            total = 0
            for k in count_keys:
                try:
                    total += int(out.get(k) or 0)
                except (TypeError, ValueError):
                    pass
            rr = success_counts(
                source_name=source_name,
                count=total,
                filters=filters,
            )
            out = dict(out)
            out["retrieval_status"] = rr.status.value
            out["retrieval"] = rr.to_context_dict()
        return out
    except Exception as exc:
        log.warning(
            "safe_dict_fetch failed source=%s err=%s",
            source_name, exc,
        )
        rr = failure(
            classify_exception(exc),
            source_name=source_name,
            error=str(exc),
            filters=filters,
        )
        return {
            "_db_error": str(exc),
            "retrieval_status": rr.status.value,
            "retrieval": rr.to_context_dict(),
            "counts_unavailable": True,
            "do_not_claim_zero": True,
            "footprint_data_unavailable": True,
        }


def merge_retrieval_into_pack(pack: dict, rr: RetrievalResult) -> None:
    """Attach retrieval envelope onto the turn pack for validators."""
    pack["_retrieval"] = rr.to_context_dict()
    pack["_retrieval_status"] = rr.status.value
    if rr.is_failure:
        pack["_degraded"] = f"retrieval_{rr.status.value}"
        pack["counts_unavailable"] = True
