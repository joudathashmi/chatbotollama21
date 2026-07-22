"""Standard retrieval outcome model — never confuse failure with empty.

Every DB / API / RAG / search call should surface a RetrievalResult so
callers and the model cannot treat TIMEOUT / AUTH errors as "0 records".

Statuses mirror the production quality contract:
  SUCCESS_WITH_RESULTS | SUCCESS_EMPTY | PARTIAL_RESULT | *ERROR*
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class RetrievalStatus(str, Enum):
    SUCCESS_WITH_RESULTS = "SUCCESS_WITH_RESULTS"
    SUCCESS_EMPTY = "SUCCESS_EMPTY"
    PARTIAL_RESULT = "PARTIAL_RESULT"
    TIMEOUT = "TIMEOUT"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    INVALID_QUERY = "INVALID_QUERY"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    PARSING_ERROR = "PARSING_ERROR"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    NO_RELEVANT_CONTEXT = "NO_RELEVANT_CONTEXT"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


_FAILURE = frozenset({
    RetrievalStatus.TIMEOUT,
    RetrievalStatus.AUTHENTICATION_ERROR,
    RetrievalStatus.PERMISSION_ERROR,
    RetrievalStatus.CONNECTION_ERROR,
    RetrievalStatus.INVALID_QUERY,
    RetrievalStatus.MALFORMED_RESPONSE,
    RetrievalStatus.PARSING_ERROR,
    RetrievalStatus.SOURCE_UNAVAILABLE,
    RetrievalStatus.UNKNOWN_ERROR,
})


@dataclass
class RetrievalResult:
    """Typed envelope for one retrieval attempt."""

    status: RetrievalStatus
    source_name: str
    source_type: str = "internal_db"  # internal_db|internal_api|document|web|official
    records: list[Any] = field(default_factory=list)
    record_count: int = 0
    filters: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    error: Optional[str] = None
    confidence: str = "high"  # high|medium|low
    as_of: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (
            RetrievalStatus.SUCCESS_WITH_RESULTS,
            RetrievalStatus.SUCCESS_EMPTY,
            RetrievalStatus.PARTIAL_RESULT,
        )

    @property
    def is_failure(self) -> bool:
        return self.status in _FAILURE

    @property
    def is_verified_empty(self) -> bool:
        return self.status == RetrievalStatus.SUCCESS_EMPTY

    def to_context_dict(self) -> dict[str, Any]:
        """Compact dict safe to inject into LLM context / advisory ctx."""
        d = {
            "retrieval_status": self.status.value,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "record_count": int(self.record_count),
            "filters": dict(self.filters or {}),
            "confidence": self.confidence,
        }
        if self.as_of:
            d["as_of"] = self.as_of
        if self.error:
            d["error"] = str(self.error)[:400]
        if self.query:
            d["query"] = self.query[:500]
        # Never pass raw zeros as "facts" on failure.
        if self.is_failure:
            d["counts_unavailable"] = True
            d["do_not_claim_zero"] = True
        elif self.is_verified_empty:
            d["verified_empty"] = True
        return d

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["status"] = self.status.value
        return raw


def success(records: list[Any] | None, *, source_name: str,
            source_type: str = "internal_db",
            filters: dict | None = None,
            query: str = "",
            as_of: str | None = None,
            metadata: dict | None = None) -> RetrievalResult:
    rows = list(records or [])
    return RetrievalResult(
        status=(
            RetrievalStatus.SUCCESS_WITH_RESULTS if rows
            else RetrievalStatus.SUCCESS_EMPTY
        ),
        source_name=source_name,
        source_type=source_type,
        records=rows,
        record_count=len(rows),
        filters=dict(filters or {}),
        query=query,
        as_of=as_of,
        metadata=dict(metadata or {}),
        confidence="high",
    )


def success_counts(*, source_name: str, count: int,
                   source_type: str = "internal_db",
                   filters: dict | None = None,
                   query: str = "",
                   metadata: dict | None = None) -> RetrievalResult:
    """For aggregate COUNT(*) style retrievals (no row payload)."""
    n = int(count or 0)
    return RetrievalResult(
        status=(
            RetrievalStatus.SUCCESS_WITH_RESULTS if n > 0
            else RetrievalStatus.SUCCESS_EMPTY
        ),
        source_name=source_name,
        source_type=source_type,
        records=[],
        record_count=n,
        filters=dict(filters or {}),
        query=query,
        metadata=dict(metadata or {}),
        confidence="high",
    )


def failure(status: RetrievalStatus, *, source_name: str,
            error: str,
            source_type: str = "internal_db",
            filters: dict | None = None,
            query: str = "") -> RetrievalResult:
    if status not in _FAILURE:
        status = RetrievalStatus.UNKNOWN_ERROR
    return RetrievalResult(
        status=status,
        source_name=source_name,
        source_type=source_type,
        records=[],
        record_count=0,
        filters=dict(filters or {}),
        query=query,
        error=error,
        confidence="low",
    )


def classify_exception(exc: BaseException) -> RetrievalStatus:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if "timeout" in msg or "timed out" in msg:
        return RetrievalStatus.TIMEOUT
    if "auth" in msg or "credential" in msg or "401" in msg:
        return RetrievalStatus.AUTHENTICATION_ERROR
    if "permission" in msg or "403" in msg or "denied" in msg:
        return RetrievalStatus.PERMISSION_ERROR
    if "connect" in msg or "connection" in msg or "could not translate" in msg:
        return RetrievalStatus.CONNECTION_ERROR
    if "syntax" in msg or "invalid" in msg or "undefinedcolumn" in name:
        return RetrievalStatus.INVALID_QUERY
    if "json" in msg or "parse" in msg or "decode" in msg:
        return RetrievalStatus.PARSING_ERROR
    return RetrievalStatus.UNKNOWN_ERROR


def user_facing_retrieval_message(result: RetrievalResult) -> str:
    """Honest fallback copy — never implies a verified zero."""
    if result.is_verified_empty:
        filt = result.filters or {}
        return (
            f"The queried source **{result.source_name}** returned "
            f"**0** verified records"
            + (f" for filters `{filt}`." if filt else ".")
            + " This is a successful empty result, not a retrieval failure."
        )
    if result.status == RetrievalStatus.PARTIAL_RESULT:
        return (
            f"Only a partial result set was retrieved from "
            f"**{result.source_name}** ({result.record_count} records). "
            "Do not treat this as a complete census."
        )
    if result.is_failure:
        return (
            f"Internal data from **{result.source_name}** could not be "
            f"retrieved ({result.status.value}"
            + (f": {result.error}" if result.error else "")
            + "). This is **not** a verified zero — do not conclude that "
            "no records exist."
        )
    return ""
