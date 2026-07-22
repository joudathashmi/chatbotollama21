"""Structured query-intent object (pre-retrieval / pre-generation).

Replaces ad-hoc string labels with a typed contract the pipeline can
log, route on, and pass into validators. Heuristic only — no LLM.
Does not hardcode country/company facts.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from app.services.intent import detect_intent


_RANK_RE = re.compile(
    r"\b(rank|top\s+\d+|priorit[sy]|best\s+(?:compan|target|investor)|"
    r"which\s+compan(?:y|ies)\s+(?:to|should)|target(?:ing)?)\b",
    re.I,
)
_RECOMMEND_RE = re.compile(
    r"\b(recommend|should\s+we|next\s+steps?|action\s+plan|"
    r"how\s+(?:do|can)\s+we|attract|engage|outreach)\b",
    re.I,
)
_COMPARE_RE = re.compile(r"\b(compare|vs\.?|versus|difference\s+between)\b", re.I)
_DOC_RE = re.compile(
    r"\b(pdf|export|report|briefing|document|powerpoint|pptx|docx)\b", re.I,
)
_CURRENT_RE = re.compile(
    r"\b(current|latest|today|as\s+of|how\s+many|total|count|"
    r"active\s+licen|licensed|rhq)\b",
    re.I,
)
_COMPLETE_RE = re.compile(
    r"\b(all|complete|full\s+list|every|exhaustive|census)\b", re.I,
)
_LICENSING_RE = re.compile(
    r"\b(licen[cs]e?s?|misa\s+licen|is_rhq|rhq)\b", re.I,
)
_TARGETING_RE = re.compile(
    r"\b(compan(?:y|ies)\s+target|target(?:ing)?\s+compan|"
    r"attract\s+investor|priority\s+compan|investment\s+thes)\b",
    re.I,
)
_SECTOR_RE = re.compile(
    r"\b(sector|industry|vertical|oil|gas|ict|fintech|health|"
    r"pharma|manufactur|logistics|tourism|mining|energy)\b",
    re.I,
)

# Common origin adjectives / country nouns — detection only, not facts.
_ORIGIN_HINTS = {
    "india": "India", "indian": "India", "china": "China", "chinese": "China",
    "japan": "Japan", "japanese": "Japan", "germany": "Germany",
    "german": "Germany", "france": "France", "french": "France",
    "uk": "United Kingdom", "britain": "United Kingdom", "british": "United Kingdom",
    "usa": "United States", "us": "United States", "american": "United States",
    "korea": "South Korea", "korean": "South Korea", "uae": "United Arab Emirates",
    "emirates": "United Arab Emirates", "singapore": "Singapore",
    "pakistan": "Pakistan", "pakistani": "Pakistan",
}


@dataclass
class QueryIntent:
    """Structured intent before retrieval and generation."""

    task_type: str = "unknown"
    decision_objective: str = ""
    entities: list[str] = field(default_factory=list)
    geographies: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    time_scope: str = "unspecified"
    required_sources: list[str] = field(default_factory=list)
    output_type: str = "prose"
    ranking_required: bool = False
    current_data_required: bool = False
    completeness_required: bool = False
    ambiguities: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    # Routing helpers
    legacy_intent_label: str = "unknown"
    needs_internal_db: bool = True
    needs_clarification: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_log_dict(self) -> dict[str, Any]:
        """Compact telemetry — no PII beyond extracted entities."""
        return {
            "task_type": self.task_type,
            "geographies": self.geographies[:5],
            "sectors": self.sectors[:5],
            "ranking_required": self.ranking_required,
            "current_data_required": self.current_data_required,
            "completeness_required": self.completeness_required,
            "output_type": self.output_type,
            "required_sources": self.required_sources[:8],
            "needs_clarification": self.needs_clarification,
            "legacy_intent_label": self.legacy_intent_label,
            "ambiguity_count": len(self.ambiguities),
        }


def _detect_geographies(q: str) -> list[str]:
    found: list[str] = []
    lower = f" {q.lower()} "
    for hint, canon in _ORIGIN_HINTS.items():
        if re.search(rf"\b{re.escape(hint)}\b", lower) and canon not in found:
            found.append(canon)
    return found


def _infer_task_type(q: str, legacy: str) -> str:
    if _TARGETING_RE.search(q) or (
        _RANK_RE.search(q) and re.search(r"\bcompan", q, re.I)
    ):
        return "company_targeting"
    if (
        re.search(r"\bcompan", q, re.I)
        and re.search(r"\b(attract|priorit|target|outreach|investor)", q, re.I)
    ):
        return "company_targeting"
    if _LICENSING_RE.search(q) and (
        re.search(r"\b(how\s+many|total|count|active)\b", q, re.I)
        or legacy == "count"
    ):
        return "licensing_count"
    if legacy == "count":
        return "factual_count"
    if _COMPARE_RE.search(q) or legacy == "comparison":
        return "comparative_analysis"
    if _RECOMMEND_RE.search(q):
        return "recommendation"
    if _RANK_RE.search(q):
        return "ranking"
    if legacy == "person_lookup":
        return "executive_lookup"
    if legacy == "browse":
        return "browse_list"
    if legacy == "entity_lookup":
        return "entity_profile"
    if legacy == "off_topic":
        return "off_topic"
    if _SECTOR_RE.search(q) and re.search(r"\b(attract|target|opportunit)", q, re.I):
        return "sector_analysis"
    if re.search(r"\b(country|sovereign|gdp|vision\s+2030)\b", q, re.I):
        return "country_analysis"
    return legacy if legacy not in ("unknown",) else "analytical"


def build_query_intent(
    user_question: str,
    history: list | None = None,
    *,
    entity_candidate: str | None = None,
) -> QueryIntent:
    """Build a structured intent from the user question."""
    q = (user_question or "").strip()
    legacy = detect_intent(q, history)
    geos = _detect_geographies(q)
    task = _infer_task_type(q, legacy)

    entities: list[str] = []
    if entity_candidate and entity_candidate.strip():
        entities.append(entity_candidate.strip())

    ranking = bool(_RANK_RE.search(q) or task == "company_targeting")
    current = bool(_CURRENT_RE.search(q) or task in (
        "licensing_count", "factual_count",
    ))
    complete = bool(_COMPLETE_RE.search(q))

    required: list[str] = []
    if task in ("licensing_count", "factual_count", "company_targeting",
                "country_analysis", "entity_profile"):
        required.append("company_profiles.licensed/is_rhq")
    if task == "company_targeting":
        required.append("target_ranking")
    if task == "executive_lookup":
        required.append("company_executives")
    if geos and task in ("company_targeting", "country_analysis",
                         "licensing_count", "sector_analysis"):
        required.append("origin_country_filter")

    output = "prose"
    if ranking or task == "company_targeting":
        output = "ranked_report"
    elif task == "licensing_count":
        output = "licensing_snapshot"
    elif _DOC_RE.search(q):
        output = "document"
    elif legacy == "browse":
        output = "table_list"

    ambiguities: list[str] = []
    assumptions: list[str] = []
    if "Saudi Arabia" in geos and len(geos) == 1 and task == "company_targeting":
        ambiguities.append(
            "origin_vs_destination: Saudi may be destination, not origin"
        )
    if complete and ranking:
        ambiguities.append(
            "complete_list_vs_prioritised_list: treating as prioritised unless "
            "user insists on census"
        )
        assumptions.append("ranking_not_exhaustive_census")
    if task == "licensing_count":
        assumptions.append(
            "canonical_counts_use_company_profiles.licensed_and_is_rhq"
        )

    needs_clarify = False
    # Only ask when material: e.g. targeting with no geography and vague "companies"
    if task == "company_targeting" and not geos and not entities:
        if not re.search(r"\b(global|worldwide|all\s+countries)\b", q, re.I):
            ambiguities.append("missing_origin_geography")
            # Inferable often from context — do not force clarify by default
            needs_clarify = False

    objective = ""
    if task == "company_targeting":
        objective = "prioritise companies for investment attraction / engagement"
    elif task == "licensing_count":
        objective = "report verified MISA licensing / RHQ counts"
    elif task == "recommendation":
        objective = "produce actionable next steps grounded in evidence"
    elif task == "comparative_analysis":
        objective = "compare entities or geographies on evidenced criteria"

    return QueryIntent(
        task_type=task,
        decision_objective=objective,
        entities=entities,
        geographies=geos,
        sectors=[],
        time_scope="current" if current else "unspecified",
        required_sources=required,
        output_type=output,
        ranking_required=ranking,
        current_data_required=current,
        completeness_required=complete,
        ambiguities=ambiguities,
        assumptions=assumptions,
        legacy_intent_label=legacy,
        needs_internal_db=legacy != "off_topic",
        needs_clarification=needs_clarify,
    )
