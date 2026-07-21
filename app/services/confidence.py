"""
Confidence scoring for chat answers.

Computes five scores from the pipeline state so the caller (logging
layer, UI footer, regression tests) can see how reliable the answer is:

  - db_retrieval_score    — did the DB return any rows? (0 / 1)
  - company_match_score   — how well the entity matched the rows
                            (exact / fuzzy / broad / none, mapped to
                             1.0 / 0.75 / 0.55 / 0.0)
  - table_relevance_score — proxy for "right table picked" (1.0 if
                            a known primary table, 0.7 generic, 0.4
                            if invalid-filter trace marker, 0 if no
                            tool call)
  - answer_grounding_score — DB-grounded (1.0), fallback (0.4),
                             clarification (0.6), error (0.0)
  - final_confidence_score — weighted product, clamped 0–1

Scores are deliberately heuristic, not precise — the goal is "good
enough that an operator can grep for low-confidence turns in the log
and triage them", not statistical calibration.
"""

from __future__ import annotations

# Primary tables we have purpose-built curation templates for. Picking
# one of these counts as high table-relevance.
_PRIMARY_TABLES = frozenset({
    "company_profiles", "country_profiles", "countries",
    "rhq_company", "rhq_licenses", "rhq_new_data", "rhq_topexecutives",
    "executives", "company_executives", "board_positions",
    "opportunities", "deals", "leads", "engagements", "meetings",
    "strategic_investors", "country_associated_companies", "fdi_data",
})


def _classification_score(classification: str | None) -> float:
    return {
        "exact": 1.0,
        "fuzzy": 0.75,
        "broad": 0.55,
        "none":  0.0,
    }.get(classification or "", 0.5)


def compute_confidence(
    *,
    tool_calls_executed: list,
    classification: str | None,
    answer_source: str,
    intent: str = "unknown",
) -> dict:
    """Returns dict of all five scores plus a one-line description."""
    tcs = tool_calls_executed or []

    # DB RETRIEVAL — did we get any rows?
    row_count_total = sum(int(tc.get("row_count") or 0) for tc in tcs)
    db_retrieval_score = 1.0 if row_count_total > 0 else 0.0

    # COMPANY MATCH — how well did the entity match the returned rows?
    company_match_score = _classification_score(classification)
    # If there was no entity at all (browse, count, off-topic), match
    # is N/A — score as 1.0 so it doesn't drag final down.
    if intent in ("browse", "count", "off_topic", "unknown") and not classification:
        company_match_score = 1.0

    # TABLE RELEVANCE — did the model pick a sensible table?
    if not tcs:
        table_relevance_score = 0.0 if intent != "off_topic" else 1.0
    else:
        any_primary = any(
            (tc.get("table") in _PRIMARY_TABLES) for tc in tcs
        )
        any_invalid = any(
            ("_dropped_unknown_filters" in (tc.get("filters") or {})
             or "_invalid_filter_for_count" in (tc.get("filters") or {}))
            for tc in tcs
        )
        if any_invalid:
            table_relevance_score = 0.4
        elif any_primary:
            table_relevance_score = 1.0
        else:
            table_relevance_score = 0.7

    # ANSWER GROUNDING
    grounding = {
        "db": 1.0,
        "count_only": 1.0,
        "fallback": 0.4,
        "off_topic_fallback": 0.5,
        "clarification": 0.6,
        "conversational": 0.8,
        "deterministic_no_match": 0.7,  # honest answer with no rows
        "error": 0.0,
    }.get(answer_source or "", 0.5)
    answer_grounding_score = grounding

    # FINAL — weighted average. Grounding matters most; retrieval and
    # match next; table relevance is a sanity-check, low weight.
    final = (
        0.40 * answer_grounding_score +
        0.25 * db_retrieval_score +
        0.20 * company_match_score +
        0.15 * table_relevance_score
    )
    final = max(0.0, min(1.0, final))

    return {
        "db_retrieval_score":    round(db_retrieval_score, 2),
        "company_match_score":   round(company_match_score, 2),
        "table_relevance_score": round(table_relevance_score, 2),
        "answer_grounding_score": round(answer_grounding_score, 2),
        "final_confidence_score": round(final, 2),
        # short human label for the UI footer
        "label": _final_label(final),
    }


def _final_label(score: float) -> str:
    if score >= 0.85:
        return "High"
    if score >= 0.65:
        return "Moderate"
    if score >= 0.40:
        return "Low"
    return "Very low"
