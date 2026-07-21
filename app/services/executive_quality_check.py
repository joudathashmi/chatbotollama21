"""
Executive quality check — Rule 10 from the Universal Executive
Intelligence Reasoning Rules.

After curation produces an answer, this module asks ONE additional LLM
call: "Would a MISA Minister find this useful for decision-making?
What's missing?". If the score is below threshold, we regenerate ONCE
with the feedback. If still low, we ship the answer anyway (never block
the user) but record the low score in the audit trail.

DESIGN PRINCIPLES — these protect the work shipped in earlier commits:

  1. SKIP on simple_fact depth. The depth_detector + DIRECT-ANSWER RULE
     work explicitly keeps single-line factual answers short. Running an
     "is this executive-grade?" check on "Apple HQ is in Cupertino, CA"
     would inflate it back into a 10-section briefing. Hard skip.

  2. SKIP on pure-lookup intents (executive_lookup, saudi_presence,
     financial_lookup). These already have DIRECT-ANSWER RULEs that
     lead with the answer. No need to second-guess.

  3. ONE REGEN MAX. Same discipline as response_validator — bounded
     work, never loops, never blocks the user on quality failure.

  4. Cheap by default. Single LLM call with a small structured prompt.
     ~1500 tok in + 200 tok out = ~$0.0006 per check on gpt-4.1-mini.

  5. Reversible via env flag. MISA_EXEC_QUALITY_CHECK=false bypasses
     the entire module — answer flows through unchanged.

Wiring: called from curation.py AFTER response_validator runs, BEFORE
returning the answer to the caller.
"""

from __future__ import annotations

import json
import os
from typing import Any

from app.config import openai_max_completion_tokens_kw


# ─── Configuration ───────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


EXEC_QC_ENABLED: bool = _env_bool("MISA_EXEC_QUALITY_CHECK", True)
EXEC_QC_THRESHOLD: int = max(1, min(10, int(
    os.getenv("MISA_EXEC_QUALITY_THRESHOLD", "7")
)))

# Depths at which the check is genuinely useful. The check adds
# ~2-3s of latency per turn (one extra LLM call). Originally fired
# on operational_detail too, but "tell me about X" briefings don't
# need the extra grading — the curator already produces solid output
# for that depth and the user feels the latency more than the quality
# bump. Narrowed to the two depths where decision-grade quality
# genuinely matters: executive briefings and strategic recommendations.
_CHECK_DEPTHS = frozenset({
    "executive_briefing",
    "strategic_recommendation",
})

# Intents that have DIRECT-ANSWER RULEs in intent_router and should
# NOT be second-guessed by the executive checker (their concise answers
# are by design). Add new pure-lookup intents here if you create them.
_SKIP_INTENTS = frozenset({
    "executive_lookup",        # "Who is the CEO of X" — lead with name
    "executive_succession",    # "Who replaces X" — lead with successor
    "saudi_presence",          # "Is X in Saudi" — lead with Yes/No
    "financial_lookup",        # "What is X revenue" — lead with number
    "off_topic",               # don't check off-topic refusals
})


# ─── Checker prompt ──────────────────────────────────────────────────

_CHECKER_SYSTEM = """You are a strict executive-intelligence quality reviewer
for the Saudi Ministry of Investment (MISA). You audit answers BEFORE
they reach a Minister, Deputy Minister, or CEO-level reader.

Score each answer 1-10 on whether it would help a senior decision-maker
take action:
  9-10: Decision-grade. Clear recommendation, evidence-backed, actionable.
  7-8:  Solid briefing. Useful but missing some strategic angle.
  5-6:  Descriptive only. Lists facts; lacks recommendation or "so what".
  1-4:  Low value. Generic, vague, or fails to answer the question.

Also identify what's MISSING that an executive would expect — pick from:
  market_implications, sector_implications, competitive_implications,
  policy_implications, investment_implications, strategic_recommendation,
  named_decision_makers, named_opportunities, evidence_for_recommendations,
  gap_analysis (what's absent from the data), prioritization,
  next_action.

Be strict. Generic recommendations like "build partnerships" or
"increase awareness" without evidence = score 5-6 max.

Output JSON:
{
  "score": <int 1-10>,
  "missing": [<list of strings from the categories above>],
  "should_regenerate": <bool: true ONLY if score < THRESHOLD AND the
                        gaps are addressable from the existing payload>,
  "feedback": "<one-sentence guidance for the regen if applicable>"
}
"""


_REGEN_DIRECTIVE_TEMPLATE = """EXECUTIVE QUALITY GATE — regeneration required:

The previous draft scored {score}/10 on executive usefulness.
Missing elements: {missing}

Reviewer guidance: {feedback}

Address these gaps in the new answer. Stay strictly within the
retrieved records (do not invent facts). If a gap cannot be filled
from the records, surface it explicitly in the missing-data line.
"""


# ─── Public API ──────────────────────────────────────────────────────

def should_run_check(intent: str | None, depth: str | None) -> bool:
    """Gate: when should the executive quality check fire?
    Skip when the module is disabled, the depth is simple_fact, or the
    intent is a pure-lookup with its own DIRECT-ANSWER RULE."""
    if not EXEC_QC_ENABLED:
        return False
    if depth not in _CHECK_DEPTHS:
        return False
    if intent in _SKIP_INTENTS:
        return False
    return True


def grade_answer(
    *,
    user_question: str,
    answer: str,
    intent: str | None,
    depth: str | None,
    client: Any,
    model: str,
) -> dict | None:
    """Run the checker. Returns a dict with score / missing / feedback,
    or None if the check failed (network, parse error). Never raises —
    the caller must handle None gracefully (= ship the answer as-is).
    """
    if not client or not answer or not user_question:
        return None
    user_content = (
        f"User question: {user_question}\n"
        f"Intent: {intent or 'unknown'}\n"
        f"Depth: {depth or 'unknown'}\n"
        f"Threshold for regeneration: {EXEC_QC_THRESHOLD}\n\n"
        f"Draft answer to grade:\n---\n{answer}\n---"
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CHECKER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            **openai_max_completion_tokens_kw(),
        )
        content = (r.choices[0].message.content or "").strip()
        if not content:
            return None
        data = json.loads(content)
        # Defensive shape checking — never trust the LLM's JSON to
        # match exactly what we asked for.
        score = int(data.get("score") or 0)
        missing = data.get("missing") or []
        if not isinstance(missing, list):
            missing = []
        should_regen = bool(data.get("should_regenerate"))
        feedback = str(data.get("feedback") or "")[:500]
        # Sanity: if score >= threshold we should NEVER regenerate even
        # if the LLM said true (model occasionally contradicts itself).
        if score >= EXEC_QC_THRESHOLD:
            should_regen = False
        return {
            "score": score,
            "missing": missing[:8],          # cap list length
            "should_regenerate": should_regen,
            "feedback": feedback,
            "threshold": EXEC_QC_THRESHOLD,
        }
    except Exception:
        return None


def build_regen_directive(verdict: dict) -> str:
    """Format the regen directive that gets prepended to the curator's
    user content on the second attempt. Returns a small block ready
    to drop into the existing extra_directive slot in curation.py."""
    missing = ", ".join(verdict.get("missing") or []) or "(none specified)"
    return _REGEN_DIRECTIVE_TEMPLATE.format(
        score=verdict.get("score", 0),
        missing=missing,
        feedback=verdict.get("feedback") or "Improve actionability and evidence-citation.",
    )
