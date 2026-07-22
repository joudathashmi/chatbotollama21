"""Recommendation quality — reject soft recs; require named, dated, actionable asks.

World-class bar (platform-wide, any origin / any account):
  every recommendation states WHO (named company or IPA), WHAT (concrete verb),
  WITH WHOM (Saudi counterpart / programme), and WHEN (dated next step).
"""

from __future__ import annotations

import re
from typing import Any


_GENERIC_PHRASES = (
    "engage stakeholders",
    "leverage synergies",
    "leverage networks",
    "explore opportunities",
    "continue monitoring",
    "stay aligned",
    "foster collaboration",
    "drive growth",
    "unlock value",
    "deepen relationship",
    "raise awareness",
    "develop a framework",
    "identify and map",
    "showcase opportunities",
    "strengthen bilateral",
    "focus on ict",
    "build partnerships",
    "enhance collaboration",
    "pursue synergies",
    "create awareness",
    "potential opportunities",
    "mutually beneficial",
)

_MOTION_LABELS = (
    "existing_investor_expansion",
    "new_market_entry",
    "rhq_conversion",
    "manufacturing_localization",
    "regional_export_platform",
    "rd_innovation_presence",
    "joint_venture",
    "shared_services",
    "procurement_opportunity",
    "pure_sales_opportunity",
)

_ACTION_RE = re.compile(
    r"\b(schedule|brief|invite|propose|pilot|open|establish|locali[sz]e|"
    r"convert|assign|contact|negotiate|launch|map|qualify|run|table|"
    r"publish|advance|align|stand\s+up)\b",
    re.I,
)

_NAMED_BOLD_RE = re.compile(r"\*\*[^*]{2,60}\*\*")
_DATED_RE = re.compile(
    r"\b(within\s+\d+\s*(?:day|days|week|weeks|month|months)|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|Q[1-4]\s*20\d{2}|"
    r"90[\s-]?day|12[\s-]?month)\b",
    re.I,
)
_COUNTERPART_RE = re.compile(
    r"\b(NEOM|SDAIA|NUPCO|LEAP|FII|NIDLP|RHQ\s+Program|"
    r"MCIT|MoH|Ministry|PIF|giga[\s-]?project)\b",
    re.I,
)

_RECS_SECTION_RE = re.compile(
    r"(?is)(^#{1,3}\s*(?:Strategic\s+Targeting\s+Recommendations|"
    r"Recommended\s+Next\s+(?:Moves|Actions)|"
    r"Recommendations\s+for\s+MISA|"
    r"Closing\s+Recommendations(?:\s+to\s+MISA)?)\b[^\n]*\n)"
    r"(.*?)(?=^#{1,3}\s|\Z)",
    re.M,
)


def saudi_counterpart_for_sector(sector: str | None) -> str:
    """Named Saudi demand counterpart from sector cues — not a universal paste."""
    s = (sector or "").casefold()
    if any(k in s for k in ("health", "pharma", "life", "bio", "hospital")):
        return "NUPCO / MoH localisation"
    if any(k in s for k in (
        "ict", "tech", "digital", "software", "telecom", "ai", "cloud",
        "semiconductor", "gaming", "esport",
    )):
        return "SDAIA / LEAP"
    if any(k in s for k in (
        "energy", "oil", "gas", "power", "renew", "water", "petro",
    )):
        return "NEOM / energy-transition programmes"
    if any(k in s for k in (
        "industrial", "manufactur", "mining", "metal", "logistic",
        "construct", "infra", "engineer",
    )):
        return "NIDLP / PIF industrial zones"
    if any(k in s for k in ("financ", "bank", "insur", "fintech")):
        return "Financial sector development programmes"
    if any(k in s for k in ("tourism", "hospital", "entertain", "real estate")):
        return "Tourism / giga-project programmes (NEOM, Red Sea, Qiddiya)"
    return "a named Vision 2030 programme desk (SDAIA, NEOM, or NIDLP)"


def is_generic_recommendation(text: str) -> bool:
    low = (text or "").strip().lower()
    if len(low) < 20:
        return True
    return any(p in low for p in _GENERIC_PHRASES)


def is_world_class_recommendation(text: str) -> bool:
    """Named actor + concrete verb + (counterpart OR dated step)."""
    t = (text or "").strip()
    if not t or is_generic_recommendation(t):
        return False
    if not _ACTION_RE.search(t):
        return False
    named = bool(_NAMED_BOLD_RE.search(t)) or bool(
        re.search(r"\b(Invest |GTAI|JETRO|KOTRA|SelectUSA|BOI|GIPC|ApexBrasil)\b", t)
    )
    if not named:
        return False
    return bool(_COUNTERPART_RE.search(t) or _DATED_RE.search(t))


def score_recommendation(item: dict[str, Any] | str) -> dict[str, Any]:
    """Return {ok, score, reasons} for a recommendation blob."""
    if isinstance(item, str):
        action = item
        owner = next_step = justification = ""
        motion = ""
    else:
        action = str(item.get("action") or item.get("recommendation") or "")
        owner = str(item.get("owner") or "")
        next_step = str(item.get("next_step") or "")
        justification = str(item.get("justification") or item.get("why") or "")
        motion = str(item.get("investment_motion") or "")

    reasons: list[str] = []
    score = 100
    if is_generic_recommendation(action):
        reasons.append("generic_phrasing")
        score -= 50
    if not _ACTION_RE.search(action):
        reasons.append("no_concrete_verb")
        score -= 20
    if not next_step and not _DATED_RE.search(action):
        reasons.append("missing_next_step")
        score -= 15
    if not _NAMED_BOLD_RE.search(action) and not re.search(
        r"\b(Invest |GTAI|JETRO|KOTRA|SelectUSA)\b", action
    ):
        reasons.append("missing_named_actor")
        score -= 15
    if not _COUNTERPART_RE.search(action):
        reasons.append("missing_saudi_counterpart")
        score -= 10
    if not justification and len(action) < 80:
        reasons.append("weak_justification")
        score -= 10
    if motion and motion not in _MOTION_LABELS:
        reasons.append("unknown_investment_motion")
        score -= 5
    if motion == "pure_sales_opportunity":
        reasons.append("sales_not_investment_unless_local_capex")
    ok = (
        score >= 60
        and "generic_phrasing" not in reasons
        and "missing_named_actor" not in reasons
    )
    return {
        "ok": ok,
        "score": max(0, score),
        "reasons": reasons,
        "investment_motions": list(_MOTION_LABELS),
        "world_class": is_world_class_recommendation(action),
    }


def filter_recommendations(
    items: list[Any],
    *,
    min_score: int = 60,
) -> tuple[list[Any], list[dict]]:
    """Keep only non-generic recommendations; return (kept, rejected_meta)."""
    kept: list[Any] = []
    rejected: list[dict] = []
    for it in items or []:
        meta = score_recommendation(it)
        if meta["score"] >= min_score and meta["ok"]:
            kept.append(it)
        else:
            rejected.append({"item": it, **meta})
    return kept, rejected


def scrub_recommendation_section(
    answer: str,
    *,
    replacement_actions: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Drop soft bullets in rec sections; optionally rebuild from grounded actions."""
    if not answer:
        return answer or "", []
    fixes: list[str] = []
    text = answer

    def _clean_body(body: str) -> str:
        lines = body.splitlines()
        kept: list[str] = []
        dropped = 0
        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                kept.append(ln)
                continue
            if stripped.startswith(("-", "*", "•")):
                bullet = re.sub(r"^[-*•]\s*", "", stripped)
                # World-class bar: named + verb + (counterpart or dated).
                if is_world_class_recommendation(bullet):
                    kept.append(ln)
                else:
                    dropped += 1
                continue
            kept.append(ln)
        if dropped:
            fixes.append(f"dropped_soft_rec_bullets:{dropped}")
        return "\n".join(kept)

    m = _RECS_SECTION_RE.search(text)
    if not m:
        return text, fixes

    header, body = m.group(1), m.group(2)
    cleaned = _clean_body(body)
    # If too few actionable bullets remain, rebuild from grounded replacements.
    bullet_count = len(re.findall(r"(?m)^\s*[-*•]\s+\S", cleaned))
    if bullet_count < 2 and replacement_actions:
        rebuilt = "\n".join(f"- {a}" for a in replacement_actions[:6]) + "\n\n"
        text = text[: m.start()] + header + rebuilt + text[m.end():]
        fixes.append("rebuilt_recs_after_soft_scrub")
        return text, fixes

    text = text[: m.start()] + header + cleaned + text[m.end():]
    return text, fixes


RECOMMENDATION_PROMPT_ADDON = """
RECOMMENDATION RULES (mandatory — world-class bar):
Every recommendation MUST state:
  1) WHAT to do (concrete verb: brief / schedule / qualify / convert / table)
  2) WITH WHOM — a **named company** or **named IPA / chamber**
  3) Saudi counterpart or programme (SDAIA, NEOM, NUPCO, NIDLP, RHQ Program…)
  4) WHEN — a dated next step (within 90 days / 12-month plan / Q-date)
Reject generic phrases (engage stakeholders, leverage synergies, explore
opportunities, develop a framework, strengthen bilateral, etc.).
Label the investment motion when relevant:
  existing_investor_expansion | new_market_entry | rhq_conversion |
  manufacturing_localization | regional_export_platform |
  rd_innovation_presence | joint_venture | shared_services |
  procurement_opportunity | pure_sales_opportunity.
Do NOT present pure sales/procurement as investment unless a local
investment component is evidenced.
""".strip()
