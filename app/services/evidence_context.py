"""Evidence-bearing context blocks with mandatory provenance.

Prevents empty / failed retrievals from being injected into the model
as if they were valid evidence. Every fact-bearing block must carry
source metadata and a retrieval status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.services.retrieval_status import RetrievalResult, RetrievalStatus
from app.services.source_policy import SourceTier, VerificationStatus


@dataclass
class EvidenceBlock:
    """One provenance-tagged evidence unit for context assembly."""

    claim_or_payload: Any
    source_name: str
    source_type: str = "internal_db"
    source_tier: str = SourceTier.INTERNAL.value
    retrieval_status: str = RetrievalStatus.SUCCESS_WITH_RESULTS.value
    verification_status: str = VerificationStatus.VERIFIED_INTERNAL.value
    record_count: int = 0
    filters: dict[str, Any] = field(default_factory=dict)
    confidence: str = "high"
    retrieved_at: str = ""
    data_as_of: Optional[str] = None
    is_inference: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.retrieved_at:
            self.retrieved_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_usable_evidence(self) -> bool:
        """False when failure / empty context must not drive factual claims."""
        st = self.retrieval_status
        if st in (
            RetrievalStatus.TIMEOUT.value,
            RetrievalStatus.AUTHENTICATION_ERROR.value,
            RetrievalStatus.PERMISSION_ERROR.value,
            RetrievalStatus.CONNECTION_ERROR.value,
            RetrievalStatus.INVALID_QUERY.value,
            RetrievalStatus.MALFORMED_RESPONSE.value,
            RetrievalStatus.PARSING_ERROR.value,
            RetrievalStatus.SOURCE_UNAVAILABLE.value,
            RetrievalStatus.UNKNOWN_ERROR.value,
            "error", "ERROR",
        ):
            return False
        return True

    @property
    def do_not_claim_zero(self) -> bool:
        return not self.is_usable_evidence

    def to_context_dict(self) -> dict[str, Any]:
        d = {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "source_tier": self.source_tier,
            "retrieval_status": self.retrieval_status,
            "verification_status": self.verification_status,
            "record_count": self.record_count,
            "filters": dict(self.filters or {}),
            "confidence": self.confidence,
            "retrieved_at": self.retrieved_at,
            "is_inference": self.is_inference,
        }
        if self.data_as_of:
            d["data_as_of"] = self.data_as_of
        if self.notes:
            d["notes"] = self.notes[:400]
        if self.do_not_claim_zero:
            d["counts_unavailable"] = True
            d["do_not_claim_zero"] = True
            d["payload_omitted"] = True
        else:
            d["payload"] = self.claim_or_payload
        return d


def from_retrieval_result(
    rr: RetrievalResult,
    *,
    payload: Any = None,
    verification: str | None = None,
    is_inference: bool = False,
) -> EvidenceBlock:
    if rr.is_failure:
        ver = VerificationStatus.UNAVAILABLE.value
    elif rr.is_verified_empty:
        ver = VerificationStatus.VERIFIED_INTERNAL.value
    elif is_inference:
        ver = VerificationStatus.ANALYTICAL_INFERENCE.value
    else:
        ver = verification or VerificationStatus.VERIFIED_INTERNAL.value
    return EvidenceBlock(
        claim_or_payload=payload if payload is not None else rr.records,
        source_name=rr.source_name,
        source_type=rr.source_type,
        retrieval_status=rr.status.value,
        verification_status=ver,
        record_count=rr.record_count,
        filters=dict(rr.filters or {}),
        confidence=rr.confidence,
        data_as_of=rr.as_of,
        is_inference=is_inference,
        notes=(rr.error or "")[:400],
    )


def assemble_evidence_context(
    blocks: list[EvidenceBlock],
    *,
    max_blocks: int = 24,
) -> dict[str, Any]:
    """Rank usable evidence first; keep failures as limitations, not facts."""
    usable = [b for b in blocks if b.is_usable_evidence]
    failed = [b for b in blocks if not b.is_usable_evidence]
    # Prefer internal tier, then higher record counts
    tier_rank = {
        SourceTier.INTERNAL.value: 0,
        SourceTier.OFFICIAL.value: 1,
        SourceTier.EXTERNAL.value: 2,
        SourceTier.MODEL_KNOWLEDGE.value: 3,
    }
    usable.sort(
        key=lambda b: (
            tier_rank.get(b.source_tier, 9),
            -int(b.record_count or 0),
        )
    )
    selected = usable[:max_blocks]
    limitations = [
        {
            "source_name": b.source_name,
            "retrieval_status": b.retrieval_status,
            "detail": b.notes or "retrieval failed",
            "do_not_claim_zero": True,
        }
        for b in failed
    ]
    conflicts: list[dict[str, Any]] = []
    # Simple numeric conflict: same metric key with different values
    # (callers may attach metric_key in filters)
    by_metric: dict[str, list[EvidenceBlock]] = {}
    for b in selected:
        mk = (b.filters or {}).get("metric_key")
        if mk:
            by_metric.setdefault(str(mk), []).append(b)
    for mk, group in by_metric.items():
        vals = {str(g.claim_or_payload) for g in group}
        if len(vals) > 1:
            conflicts.append({
                "metric_key": mk,
                "values": [
                    {
                        "source": g.source_name,
                        "value": g.claim_or_payload,
                        "tier": g.source_tier,
                    }
                    for g in group
                ],
            })

    return {
        "evidence": [b.to_context_dict() for b in selected],
        "data_limitations": limitations,
        "source_conflicts": conflicts,
        "evidence_count": len(selected),
        "failed_retrieval_count": len(failed),
        "any_counts_unavailable": bool(failed) or any(
            b.do_not_claim_zero for b in selected
        ),
    }


def strip_unusable_from_db_context(db_context: dict | None) -> dict:
    """If footprint retrieval failed, remove zero-looking count fields
    so the model cannot treat them as verified zeros."""
    ctx = dict(db_context or {})
    unavailable = bool(
        ctx.get("footprint_data_unavailable")
        or ctx.get("counts_unavailable")
        or ctx.get("do_not_claim_zero")
        or (
            isinstance(ctx.get("retrieval"), dict)
            and ctx["retrieval"].get("do_not_claim_zero")
        )
    )
    if not unavailable:
        return ctx
    for key in (
        "companies_from_origin_licensed_in_saudi",
        "companies_from_origin_with_rhq",
        "total_licensed",
        "total_rhq",
    ):
        # Keep key only if explicitly marked verified empty
        if ctx.get("retrieval_status") in (
            "SUCCESS_EMPTY", "zero_records",
            RetrievalStatus.SUCCESS_EMPTY.value,
        ):
            continue
        ctx.pop(key, None)
    ctx["counts_unavailable"] = True
    ctx["do_not_claim_zero"] = True
    return ctx
