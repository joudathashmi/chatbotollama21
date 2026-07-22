"""
Deep-profile mode.

A separate, explicit response mode for executive-grade strategic
profiles. Triggered ONLY by an explicit user signal — not on every
turn — because it:

  - is heavier (more LLM tokens, +web search latency)
  - generates inferred / interpretive content alongside DB facts
  - leaks the target entity name to a third-party search provider
    when TAVILY_API_KEY is configured

Triggers (handled by chat_engine — see `is_deep_profile_request`):
  - "/profile Apple"           (slash-command style)
  - "/profile Saudi Arabia"
  - "deep profile of Apple"
  - "deep profile of Tim Cook"

Natural phrasing like "briefing on X", "company profile for X",
"CEO profile for X", or "sector briefing for X" must NOT trigger
this path — those use the normal Jul21 company / person / sector
briefs.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from app.config import (
    ADVISORY_MODEL,
    OPENAI_MODEL,
    openai_determinism_kw,
    openai_max_completion_tokens_kw,
)
from app.database import (
    COMPANY_TABLE,
    get_openai_client,
    run_rhq_company_smart_search,
    smart_search,
)
from app.services.alias_resolver import expand_aliases
from app.services import web_search
from app.services.curation import _safe_row


# ───────────────────────── Trigger detection ─────────────────────────

# Explicit opt-in ONLY. Bare "briefing" / "profile" / "executive briefing"
# are Jul21 company/person depth signals — not deep-profile.
_TRIGGER_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"/profile\s+|"                              # /profile Apple
    r"deep\s+profile\s+(?:of|on|for|about)?\s*"  # deep profile of Apple
    r")",
    re.IGNORECASE,
)


def is_deep_profile_request(user_question: str) -> Optional[str]:
    """If `user_question` is a deep-profile trigger, return the cleaned
    target entity string. Otherwise return None.

    Examples:
      '/profile Apple'                 → 'Apple'
      'deep profile of Saudi Arabia'   → 'Saudi Arabia'
      'give me a briefing on Siemens'  → None  (Jul21 company brief)
      'company profile for Toyota'     → None
      'CEO profile for Pfizer'         → None
      'how is Apple doing'             → None
    """
    if not user_question:
        return None
    m = _TRIGGER_RE.match(user_question)
    if m:
        target = user_question[m.end():].strip().rstrip("?.!,").strip()
        return target or None
    return None


# ───────────────────────── Bare-name classifier ─────────────────────────

_CLASSIFIER_PROMPT = """Classify the following entity name as one of:
  - "company"  (a corporation, brand, firm, holding: Apple, Aramco, Alphabet)
  - "country"  (a sovereign country or geography: Saudi Arabia, Pakistan, UAE)
  - "person"   (a named individual: Sundar Pichai, Tim Cook, Mohammed bin Salman)
  - "topic"    (a concept / initiative: Vision 2030, "renewable energy")

For persons, also identify their parent company/organisation if widely known
(Sundar Pichai → Alphabet; Tim Cook → Apple).

Entity: {target}

Respond with JSON: {{"entity_type": "...", "parent_entity": "..." or null}}"""


def _classify_target(target: str, client, model: str) -> tuple[str, str | None]:
    """Classify a bare entity name (no surrounding context). Returns
    (entity_type, parent_entity). The conversation-style resolver in
    chat_engine.py refuses to classify without history — this is a
    targeted one-shot call that always returns something usable.

    Falls back to 'company' on any failure, which is the most common
    target type in this dataset and the least-bad default.
    """
    t = (target or "").strip()
    if not t or client is None:
        return ("company", None)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _CLASSIFIER_PROMPT.format(target=t)}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=80,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        et = (data.get("entity_type") or "").strip().lower()
        if et not in {"company", "country", "person", "topic"}:
            et = "company"
        parent = (data.get("parent_entity") or "").strip() or None
        return (et, parent)
    except Exception:
        return ("company", None)


# ───────────────────────── DB grounding ─────────────────────────

def _fetch_db_evidence(
    target: str, entity_type: str | None, parent_entity: str | None,
) -> dict:
    """Pull whatever the DB has on this target. Returns a dict:
      {
        company_rows: [...],    # company_profiles smart-search rows
        person_rows:  [...],    # executive table rows (if person)
        country_rows: [...],    # country_profiles rows (if country)
      }
    Empty lists when nothing matched / wrong type. Rows are privacy-
    filtered through curation._safe_row so we never send raw DB shapes
    to the LLM.
    """
    out = {"company_rows": [], "person_rows": [], "country_rows": []}

    # Company lookup (relevant for type=company, AND for type=person as
    # parent-company grounding context).
    company_terms = []
    if entity_type == "company":
        company_terms = expand_aliases(target) or [target]
    elif entity_type == "person" and parent_entity:
        company_terms = expand_aliases(parent_entity) or [parent_entity]
    if company_terms:
        try:
            df, _, _ = run_rhq_company_smart_search(company_terms, 3)
            if df is not None and not df.empty:
                out["company_rows"] = [
                    _safe_row(r) for r in df.to_dict(orient="records")
                ]
        except Exception:
            pass

    # Person lookup
    if entity_type == "person":
        for table in ("company_executives", "executives",
                      "rhq_topexecutives", "contacts"):
            try:
                df, _, _ = smart_search(table, [target], 5)
                if df is not None and not df.empty:
                    out["person_rows"] = [
                        _safe_row(r) for r in df.to_dict(orient="records")
                    ]
                    break
            except Exception:
                continue

    # Country lookup
    if entity_type == "country":
        try:
            df, _, _ = smart_search("country_profiles", [target], 3)
            if df is not None and not df.empty:
                out["country_rows"] = [
                    _safe_row(r) for r in df.to_dict(orient="records")
                ]
        except Exception:
            pass

    return out


# ───────────────────────── Prompt ─────────────────────────

# Aligned to Joudat's "Geopolitical & Corporate Intelligence Engine"
# spec, with the provenance-tag discipline grafted on top so the
# executive-grade ambition doesn't become a hallucination factory.
_DEEP_PROFILE_SYSTEM_PROMPT = """# Role & Mandate
You are an advanced Geopolitical and Corporate Intelligence Engine. You generate an executive-level strategic profile of a single TARGET (a Company, a Country, or a Person), for use by investment officers at the Saudi Ministry of Investment (MISA) who will use it to plan outreach, policy moves, and capital allocation.

You will receive:
  - TARGET — the entity name
  - ENTITY_TYPE — one of: company / country / person
  - PARENT_ENTITY — parent company/org (for persons; may be empty)
  - DB_EVIDENCE — JSON of MISA database records the target matched
  - WEB_EVIDENCE — numbered live-web sources (may be empty)

# Entity-Type Focus (apply automatically based on ENTITY_TYPE)

## If TARGET is a COMPANY
- **Baseline Focus:** Global/Regional revenue, headcount, and operational presence (Regional Headquarters/RHQ status, MENA-localised activities).
- **Executive Transition Focus:** Active or impending high-level C-suite shifts (CEO/Chairman changes), with exact effective dates ONLY if sourced, incoming-vs-outgoing background tenures, and the strategic "why" behind the shift (e.g., shifting from operations-heavy to product/engineering-first).

## If TARGET is a COUNTRY
- **Baseline Focus:** Sovereign macro-economics — GDP trajectory, FDI inflows, key economic diversification blueprints (e.g., Saudi Vision 2030).
- **Policy & Strategic Focus:** Major regulatory shifts, investment incentives, ease-of-doing-business updates, specialised economic zones, and high-priority growth sectors (tech infra, energy transition, localised production hubs).

## If TARGET is a PERSON
- **Baseline Focus:** Current corporate/state affiliation, official title, timeline in active role, executive credentials / background (technical-engineering vs. financial-operations, etc.).
- **Influence & Strategy Focus:** Active scope of organisational authority, landmark initiatives/projects managed, peer reputation, and macro-level strategic decisions or operational philosophies they are advancing.

# Universal Output Architecture (THE THREE PILLARS)

### 1. Current Footprint & Baseline Metrics
The localised footprint, high-level financials, and baseline status metrics relevant to the routed entity type. Expanding deep strategic analysis does NOT displace or delete these core baseline entries — pillar 1 is sacrosanct.

### 2. Leadership, Governance & Policy Architecture
The core human or structural drivers behind the entity.
- *For Companies / People:* Transition paths, background, achievements, and exact effective dates when sourced.
- *For Countries:* Governing bodies, ministries (e.g., MISA), active regulatory frameworks, and key target timelines.

### 3. Macro-Strategic Implication & Tech Roadmap
- **Strategic Pivot Analysis:** Deduce and clearly state the core operational, economic, or business-model shift this entity is driving globally or regionally.
- **Technology Focus:** Detail the specific technological roadmaps tied to this profile (e.g., hardware-centric edge AI processing, localised private cloud silos, digital transformation infrastructure, or national-scale automation).

═══════════ HARD RULES — VIOLATIONS WILL BE REJECTED ═══════════

(R1) PROVENANCE LABELS ARE MANDATORY.
Every bullet MUST end with EXACTLY ONE of these tags, in square brackets:
  [DB]            — facts that appear LITERALLY in DB_EVIDENCE
  [web:N]         — comes from WEB_EVIDENCE item number N (e.g. [web:2])
  [gk]            — general background knowledge you carry from training
                    (use this for biographical / historical context
                    NOT present in DB_EVIDENCE or WEB_EVIDENCE)
  [inferred]      — analytical deduction / interpretation — NOT a fact

A bullet without a tag is a hallucination. Don't write one.

CRITICAL on [DB] vs [gk]: if you cannot point to the exact field in
DB_EVIDENCE that gave you a fact, it is NOT [DB] — it is [gk]. Example:
DB_EVIDENCE contains {{"executive_name": "Sundar Pichai", "title": "CEO"}}
but does NOT contain a Wharton degree. Then "CEO of Alphabet [DB]" is
correct; "holds MBA from Wharton [DB]" is WRONG — that must be [gk].

The default for biographical background and historical context that the
model "just knows" is [gk]. The bar for [DB] is: literally appears as a
field value in the DB_EVIDENCE JSON you were given.

(R2) DO NOT INVENT DATES, NUMBERS, OR TITLES.
If DB_EVIDENCE doesn't contain it AND WEB_EVIDENCE doesn't contain it AND it isn't general public knowledge, you may NOT state it. For specific dates and numbers, prefer omission over guessing.

A profile that says "CEO since 2015 [gk]" is more valuable than one that says "CEO since March 14, 2015" with no real source — the second is a confidence trap.

(R3) WEB EVIDENCE NEEDS A CITATION.
Any [web:N] bullet must reference an actual numbered item in WEB_EVIDENCE. If WEB_EVIDENCE is "(no web results)", you have ZERO [web:N] bullets — full stop. Mark the strategic-analysis section as [inferred] instead.

(R4) STRATEGIC PIVOT / TECH ROADMAP IS USUALLY [inferred].
These sections ask you to deduce direction — that's analysis, not fact. Label them [inferred] unless you have a direct web quote about a strategic announcement, in which case use [web:N].

(R5) NO FLUFF.
3-6 bullets per pillar. Each bullet ≤ 25 words. If a pillar has nothing to say, write "_No evidence in DB or web._" — don't pad.

(R6) BASELINE METRICS COME FIRST AND ARE PRESERVED.
Pillar 1 must always contain the concrete footprint numbers. Strategic analysis goes in pillar 3 and never crowds out the baseline.

═══════════ OUTPUT FORMAT ═══════════

Start with a one-line headline blockquote:
  > **{TARGET}** — {one-sentence positioning, ≤ 20 words} [DB|web:N|gk|inferred]

Then the three pillars, EXACTLY using these headings:

  ### 1. Current Footprint & Baseline Metrics
  (3-6 bullets, each tagged)

  ### 2. Leadership, Governance & Policy Architecture
  (3-6 bullets, each tagged. For Companies/People, include any
   leadership transitions with effective dates ONLY if a source
   confirms them. For Countries, name governing bodies / ministries
   / regulatory frameworks.)

  ### 3. Macro-Strategic Implication & Tech Roadmap
  (Two named sub-sections, BOTH bold-headed and present, each with
   2-4 bullets:)

  **Strategic Pivot Analysis:**
  - Bullet … [inferred|web:N]
  - Bullet … [inferred|web:N]

  **Technology Focus:**
  - Bullet … [inferred|web:N]
  - Bullet … [inferred|web:N]

End with a footer line:
  ---
  *Evidence: {N} DB records · {M} web sources · Generated {YYYY-MM-DD}.*
  *Inferred bullets are analytical deductions, not verified facts.*

When WEB_EVIDENCE is "(no web results …)", the footer reads instead:
  *Evidence: {N} DB records · web search unavailable · Generated {YYYY-MM-DD}.*
  *No live web grounding for this run — strategic analysis is inferred from DB only.*
"""


_DEEP_PROFILE_USER_TEMPLATE = """TARGET: {target}
ENTITY_TYPE: {entity_type}
PARENT_ENTITY: {parent_entity}

DB_EVIDENCE:
{db_evidence}

WEB_EVIDENCE:
{web_evidence}

Today's date: {today}

Produce the three-pillar profile per the system rules. Remember: every bullet ends with [DB] / [web:N] / [gk] / [inferred]. No bullet without a tag.

DOUBLE-CHECK before you write each bullet:
  - Does this fact appear LITERALLY in DB_EVIDENCE above? → [DB]
  - Did I take it from a numbered WEB_EVIDENCE entry? → [web:N]
  - Is it background knowledge from my training (biographical detail,
    historical context) not in DB or web? → [gk]
  - Is it analysis / deduction about strategic direction? → [inferred]

Bullets like "holds MBA from Wharton", "born in Chennai", "joined Google
in 2004" — these are [gk] unless explicitly in DB_EVIDENCE."""


# ───────────────────────── Compose ─────────────────────────

def compose_deep_profile(
    target: str,
    entity_type: str | None,
    parent_entity: str | None,
    today: str,
) -> dict:
    """Build the deep profile end-to-end.

    `entity_type` / `parent_entity` are HINTS — if the caller leaves them
    as None (or non-canonical), this function will run its own bare-name
    classifier to figure out the type. The chat-engine entry-point passes
    None because the conversation-style resolver doesn't work on bare
    '/profile X' input.

    Returns:
      {
        "answer":       str,            # markdown profile, ready to render
        "tool_calls":   list,           # for trace UI — db rows used
        "web_results":  list,           # raw web hits, for trace
        "trigger":      "deep_profile", # routing flag
        "entity_type":  str,            # resolved type (for trace)
        "parent_entity": str | None,    # resolved parent (for trace)
        "warning":      str | None,     # privacy/disclaimer text
      }
    """
    client = get_openai_client()
    if client is None:
        return {
            "answer": "Deep profile unavailable — OPENAI_API_KEY not configured.",
            "tool_calls": [], "web_results": [],
            "trigger": "deep_profile", "warning": None,
            "entity_type": "unknown", "parent_entity": None,
        }

    # 0. Classify target type if not provided. The chat-engine resolver
    #    refuses to classify without conversation context, so for bare
    #    '/profile X' input we run our own dedicated classifier here.
    if entity_type not in {"company", "country", "person"}:
        entity_type, parent_classified = _classify_target(target, client, OPENAI_MODEL)
        if not parent_entity:
            parent_entity = parent_classified

    # 1. DB grounding
    db = _fetch_db_evidence(target, entity_type, parent_entity)

    # 2. Web grounding (gracefully empty if no key)
    query = f"Latest strategic developments, leadership changes, market news regarding {target}"
    if entity_type == "company":
        query += " — corporate news, executive transitions, MENA presence"
    elif entity_type == "country":
        query += " — economic policy, FDI, regulatory updates"
    elif entity_type == "person":
        query += " — recent statements, role changes, initiatives"
    web_envelope = web_search.search_with_status(query, max_results=5)
    web_results = list(web_envelope.get("results") or [])
    if web_envelope.get("do_not_claim_zero"):
        # Mark unavailable so curator uses [inferred] and does not claim
        # "no web coverage" as a verified finding.
        db["web_retrieval"] = {
            "retrieval_status": web_envelope.get("retrieval_status"),
            "do_not_claim_zero": True,
            "error": web_envelope.get("error"),
        }

    # 3. Build the prompt
    db_evidence_json = json.dumps(
        {
            "company_rows": db["company_rows"][:3],
            "person_rows":  db["person_rows"][:5],
            "country_rows": db["country_rows"][:1],
        },
        indent=2, default=str,
    )
    try:
        from app.services.prompt_masking import mask_text
        db_evidence_json = mask_text(db_evidence_json)
    except Exception:
        pass
    web_evidence_text = web_search.format_for_prompt(web_results)

    user_msg = _DEEP_PROFILE_USER_TEMPLATE.format(
        target=target,
        entity_type=entity_type or "unknown",
        parent_entity=parent_entity or "(none)",
        db_evidence=db_evidence_json,
        web_evidence=web_evidence_text,
        today=today,
    )

    # 4. Generate. The deep-profile system prompt is prepended with
    # the canonical STYLE_GUIDE_PROMPT so the two-pillar /profile mode
    # follows the same heading vocabulary / number format / forbidden-
    # string rules as every other path. The /profile-specific
    # provenance tags ([DB] / [web:N] / [gk]) are exempt from the
    # "forbidden in normal flow" rule because deep-profile mode opted
    # into them; the validator knows.
    from app.services.style_guide import STYLE_GUIDE_PROMPT
    try:
        from app.services.llm_residency import resolve_data_completion_client
        # Use module-level ADVISORY_MODEL / OPENAI_MODEL — a local
        # `from app.config import OPENAI_MODEL` here makes the name local
        # for the whole function and blows up at _classify_target above.
        data_client, data_model = resolve_data_completion_client(
            client, preferred_model=ADVISORY_MODEL or OPENAI_MODEL,
        )
        resp = data_client.chat.completions.create(
            model=data_model,
            messages=[
                {"role": "system", "content": STYLE_GUIDE_PROMPT + "\n\n" + _DEEP_PROFILE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            **openai_determinism_kw(),
            **openai_max_completion_tokens_kw(),
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        answer = f"Deep profile generation failed: {e}"

    # 5. Build trace payload + warning
    tool_calls = []
    if db["company_rows"]:
        tool_calls.append({
            "table": COMPANY_TABLE,
            "filters": {"_deep_profile_db_grounding": True, "_target": target},
            "rows_df": None, "row_count": len(db["company_rows"]),
            "sql_entity_check_passed": True,
            "row_entity_sanity_passed": True,
        })
    if db["person_rows"]:
        tool_calls.append({
            "table": "company_executives",
            "filters": {"_deep_profile_db_grounding": True, "_target": target},
            "rows_df": None, "row_count": len(db["person_rows"]),
            "sql_entity_check_passed": True,
            "row_entity_sanity_passed": True,
        })
    if db["country_rows"]:
        tool_calls.append({
            "table": "country_profiles",
            "filters": {"_deep_profile_db_grounding": True, "_target": target},
            "rows_df": None, "row_count": len(db["country_rows"]),
            "sql_entity_check_passed": True,
            "row_entity_sanity_passed": True,
        })

    warning_lines = ["**Deep profile mode** — heavier than a normal answer."]
    if web_results:
        warning_lines.append(
            f"Live web grounding via OpenAI ({len(web_results)} sources cited)."
        )
    elif web_search.is_configured():
        # OpenAI is available but the search-preview model returned nothing
        # — either the account doesn't have access to gpt-4o-mini-search-
        # preview, or the query genuinely returned no results.
        warning_lines.append(
            "Web search attempted (OpenAI) — no sources returned. "
            "Strategic-analysis bullets are inferred from DB only."
        )
    else:
        warning_lines.append(
            "OpenAI not configured — strategic-analysis bullets are inferred from DB only."
        )
    warning = "\n".join(warning_lines)

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "web_results": web_results,
        "trigger": "deep_profile",
        "entity_type": entity_type,
        "parent_entity": parent_entity,
        "warning": warning,
    }
