"""Typed schemas for major chatbot response types.

Structured output is the internal contract; Markdown/HTML/PDF are
renderings of validated structures — never the primary interface.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class VerificationStatus(str, Enum):
    VERIFIED_INTERNAL = "VERIFIED_INTERNAL"
    VERIFIED_OFFICIAL = "VERIFIED_OFFICIAL"
    VERIFIED_EXTERNAL = "VERIFIED_EXTERNAL"
    ANALYTICAL_INFERENCE = "ANALYTICAL_INFERENCE"
    PROPOSAL = "PROPOSAL"
    REQUIRES_VALIDATION = "REQUIRES_VALIDATION"
    UNAVAILABLE = "UNAVAILABLE"


class FactClaim(BaseModel):
    statement: str
    value: Any = None
    source_id: str = ""
    source_type: str = "internal_db"
    verification_status: VerificationStatus = VerificationStatus.REQUIRES_VALIDATION
    confidence: str = "medium"


class RecommendationItem(BaseModel):
    action: str
    owner: str = ""
    target_stakeholder: str = ""
    justification: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    expected_outcome: str = ""
    next_step: str = ""
    risks: list[str] = Field(default_factory=list)
    requires_validation: bool = False

    @field_validator("action")
    @classmethod
    def action_not_generic(cls, v: str) -> str:
        bad = (
            "engage stakeholders", "leverage synergies", "explore opportunities",
            "continue monitoring", "stay aligned",
        )
        low = (v or "").strip().lower()
        if low in bad or len(low) < 12:
            raise ValueError("recommendation too generic or empty")
        return v


class RankingRow(BaseModel):
    rank: int
    entity: str
    scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    investment_motion: str = ""  # expansion|new_entry|rhq|localization|…


class SourceRef(BaseModel):
    source_id: str
    source_name: str
    source_type: str = "internal_db"
    retrieval_status: str = ""
    record_count: Optional[int] = None
    filters: dict[str, Any] = Field(default_factory=dict)


class QualityResponse(BaseModel):
    """Generic envelope for analytical / factual answers."""

    title: str = ""
    executive_summary: list[str] = Field(default_factory=list)
    facts: list[FactClaim] = Field(default_factory=list)
    analysis: list[str] = Field(default_factory=list)
    rankings: list[RankingRow] = Field(default_factory=list)
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    data_limitations: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    rendering_hints: dict[str, Any] = Field(default_factory=dict)

    def has_critical_limitation(self) -> bool:
        return any(
            "unavailable" in (d or "").lower()
            or "not a verified zero" in (d or "").lower()
            for d in self.data_limitations
        )


class LicensingSnapshot(BaseModel):
    title: str = "Licensing Snapshot"
    total_licensed: Optional[int] = None
    total_rhq: Optional[int] = None
    focus: str = "both"  # licensed|rhq|both
    by_country: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_status: str = ""
    source_name: str = "company_profiles.licensed/is_rhq"
    data_limitations: list[str] = Field(default_factory=list)
    counts_unavailable: bool = False


def validate_quality_response(data: dict | QualityResponse) -> tuple[QualityResponse | None, list[str]]:
    """Validate structured output. Returns (model|None, errors)."""
    try:
        if isinstance(data, QualityResponse):
            return data, []
        model = QualityResponse.model_validate(data)
        return model, []
    except Exception as exc:
        return None, [str(exc)]


def licensing_fallback_message(*, status: str, error: str = "") -> str:
    return (
        "## Licensing Snapshot\n\n"
        f"Internal MISA licensing aggregates could not be retrieved "
        f"(`{status}`"
        + (f": {error}" if error else "")
        + "). This is **not** a verified zero — do not conclude that "
        "no licensed or RHQ companies exist.\n\n"
        "_Source: `company_profiles.licensed` / `is_rhq` (unavailable)._"
    )


def render_licensing_snapshot(snap: LicensingSnapshot) -> str:
    """Deterministic Markdown from a validated licensing snapshot."""
    if snap.counts_unavailable or snap.total_licensed is None:
        return licensing_fallback_message(
            status=snap.retrieval_status or "SOURCE_UNAVAILABLE",
        )
    lines = [f"## {snap.title}", ""]
    if snap.focus == "rhq":
        lines.append(
            f"**{snap.total_rhq:,} companies hold an active Saudi Regional "
            f"Headquarters (RHQ) licence**, out of a broader pool of "
            f"**{snap.total_licensed:,} MISA-licensed companies**."
        )
    else:
        lines.append(
            f"**{snap.total_licensed:,} companies hold an active MISA licence**. "
            f"Of these, **{snap.total_rhq:,}** hold RHQ status."
        )
    if snap.by_country:
        lines += ["", "### Top Origin Countries", "",
                  "| Rank | Origin Country | Count |", "|---|---|---|"]
        for i, r in enumerate(snap.by_country[:10], start=1):
            lines.append(
                f"| {i} | {r.get('country', '')} | **{r.get('n', 0)}** |"
            )
    if snap.data_limitations:
        lines += ["", "### Data Limitations", ""]
        lines += [f"- {d}" for d in snap.data_limitations]
    lines += [
        "",
        f"_Source: `{snap.source_name}` "
        f"(retrieval_status=`{snap.retrieval_status}`)._",
    ]
    return "\n".join(lines)
