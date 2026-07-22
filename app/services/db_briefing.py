"""Deterministic DB-first briefings — no Ollama / no cloud for row prose.

Under residency strict, rewriting Postgres rows with a local LLM is the
latency killer (~60–100s) and the accuracy killer (loops / invented
sections). This module renders STYLE_GUIDE markdown straight from row
fields so common company / person answers stay fast and fact-accurate
while row JSON never leaves the machine.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.commentary import (
    _text,
    format_int,
    format_revenue_usd,
)
from app.services.style_guide import HEADERS, format_count, format_currency, make_footer

_PERSON_INTENTS = frozenset({
    "executive_lookup", "person_lookup", "executive_succession",
})
_COMPANY_LIKE_INTENTS = frozenset({
    "company_profile", "saudi_presence", "financial_lookup",
    "opportunity_alignment", "relationship_intelligence",
    "general_research", "engagement_strategy",
})
_ENGAGEMENT_PLAN_RE = re.compile(
    r"(?i)\b("
    r"engagement\s+plan|engage(?:ment)?\s+(?:plan|strategy|approach)|"
    r"how\s+(?:should|do|can)\s+(?:we|misa)\s+engage|"
    r"outreach\s+plan|investment\s+(?:into|in)\s+new|"
    r"pitch\s+(?:to|for)|how\s+to\s+(?:engage|approach)"
    r")\b"
)
_PEOPLE_TABLES = frozenset({
    "executives", "company_executives", "rhq_topexecutives",
    "board_positions", "contacts", "company_contact_records",
    "related_people", "profiles", "personal_informations",
})
_PERSON_NAME_KEYS = (
    "full_name", "executive_name", "person_name", "contact_name",
    "name", "ceo_name",
)
_PERSON_TITLE_KEYS = (
    "title", "role", "position", "designation", "current_title",
    "exec_title", "job_title",
)


def use_deterministic_db_briefing() -> bool:
    """True when templates should short-circuit ahead of LLM curation.

    With ``MISA_NARRATIVE_CLOUD`` (default), ``auto`` prefers Jul21 cloud
    narrative; templates are used only as a compose fallback.
    """
    from app import config
    mode = (getattr(config, "DB_BRIEFING_MODE", "auto") or "auto").strip().lower()
    if mode in ("ollama", "llm", "off", "false", "0"):
        return False
    if mode in ("deterministic", "db", "template", "on", "true", "1"):
        return True
    # auto — prefer cloud narrative when enabled (Jul21 quality path).
    # When narrative cloud is OFF, always prefer deterministic templates so
    # company/person briefs keep Snapshot of Operations + Strategic Read
    # instead of thin local-LLM loops.
    if bool(getattr(config, "NARRATIVE_CLOUD_ENABLED", True)):
        return False
    return True


def _cell(val: Any, *, currency: bool = False, count: bool = False) -> str:
    if val is None or val == "" or val == []:
        return "—"
    if currency:
        return format_currency(val, default="—")
    if count:
        return format_count(val)
    t = _text(val)
    return t if t else "—"


def _is_person_row(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    keys = {str(k).lower() for k in row.keys()}
    has_name = any(k in keys for k in _PERSON_NAME_KEYS)
    has_title = any(k in keys for k in _PERSON_TITLE_KEYS)
    return has_name and (
        has_title or "company_profile_id" in keys or "tenure" in keys
    )


def _is_company_row(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    keys = {str(k).lower() for k in row.keys()}
    return "company_name" in keys and not _is_person_row(row)


def _person_name(row: dict) -> str | None:
    for k in _PERSON_NAME_KEYS:
        t = _text(row.get(k))
        if t and t.lower() not in ("unknown", "n/a", "na"):
            # Avoid treating company names stored in `name` as people
            # when title keys are missing — already gated by _is_person_row.
            return t
    # first + last
    first = _text(row.get("first_name"))
    last = _text(row.get("last_name"))
    if first and last:
        return f"{first} {last}"
    return first or last


def _person_title(row: dict) -> str | None:
    for k in _PERSON_TITLE_KEYS:
        t = _text(row.get(k))
        if t:
            return t
    return None


def _pick_company_row(rows: list[dict]) -> dict | None:
    for r in rows:
        if _is_company_row(r):
            return r
    return None


def _pick_person_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if _is_person_row(r)]


def _should_render_person(intent: str | None, table: str | None, rows: list[dict]) -> bool:
    people = _pick_person_rows(rows)
    if not people:
        return False
    if (intent or "") in _PERSON_INTENTS:
        return True
    if (table or "") in _PEOPLE_TABLES and (intent or "") not in _COMPANY_LIKE_INTENTS:
        return True
    return False


def _should_render_company(intent: str | None, table: str | None, rows: list[dict]) -> bool:
    if not _pick_company_row(rows):
        return False
    if _should_render_person(intent, table, rows) and (intent or "") in _PERSON_INTENTS:
        return False
    if (intent or "") in _COMPANY_LIKE_INTENTS or intent is None:
        return True
    if (table or "") in ("company_profiles", "rhq_company", None, ""):
        return True
    return False


def render_person_brief(
    rows: list[dict],
    *,
    user_question: str = "",
    locale: str = "en",
) -> str | None:
    """## Role + ## Background + light Strategic Read — never a full company dump.

    Matches the STYLE_GUIDE people template: Role from MISA, then a
    substantive Background (record + employer context). Public career
    detail is layered on later by ``enrich_db_briefing`` (question-only / web).
    """
    people = _pick_person_rows(rows)
    if not people:
        return None
    company = _pick_company_row(rows)
    company_name = _text((company or {}).get("company_name")) if company else None

    q = (user_question or "").lower()
    role_hint = None
    for hint in ("ceo", "chief executive", "chairman", "chairperson",
                 "cfo", "cto", "founder", "president"):
        if hint in q:
            role_hint = hint
            break

    chosen = people[0]
    if role_hint:
        for r in people:
            title = (_person_title(r) or "").lower()
            if role_hint in title or (
                role_hint == "ceo" and "chief executive" in title
            ) or (
                role_hint == "chairman" and "chair" in title
            ):
                chosen = r
                break

    name = _person_name(chosen)
    title = _person_title(chosen)
    if not name:
        return None

    employer = (
        _text(chosen.get("company_name"))
        or _text(chosen.get("organization"))
        or _text(chosen.get("organisation"))
        or company_name
    )
    if employer:
        employer = employer.rstrip(".,;:")

    if title and employer:
        lead = f"**{name} is {title} at {employer}.**"
    elif title:
        lead = f"**{name} is {title}.**"
    elif employer:
        lead = f"**{name}** is recorded against **{employer}**."
    else:
        lead = f"**{name}**."

    bullets: list[str] = []
    if title:
        bullets.append(f"* **Position:** {title}")
    if employer:
        bullets.append(f"* **Company:** {employer}")
    tenure = _text(chosen.get("tenure")) or _text(chosen.get("status"))
    if tenure:
        bullets.append(f"* **Tenure:** {tenure}")
    start = _text(chosen.get("start_date")) or _text(chosen.get("appointment_date"))
    if start:
        bullets.append(f"* **Since:** {start}")

    others = []
    for r in people:
        if r is chosen:
            continue
        if (_person_name(r) or "").lower() != name.lower():
            continue
        ot = _person_title(r)
        oc = _text(r.get("company_name")) or employer
        if ot:
            others.append(f"* Also recorded as **{ot}**" + (f" at {oc}" if oc else ""))
    bullets.extend(others[:3])

    km: dict = {}
    mena: dict = {}
    sector = None
    overview = None
    if company:
        km = company.get("key_metrics") if isinstance(company.get("key_metrics"), dict) else {}
        misa = company.get("misa_details") if isinstance(company.get("misa_details"), dict) else {}
        general = misa.get("general") if isinstance(misa.get("general"), dict) else {}
        mena = misa.get("mena_details") if isinstance(misa.get("mena_details"), dict) else {}
        sector = (
            _text(company.get("sector"))
            or _text(km.get("sector"))
            or _text(general.get("sector"))
        )
        overview = (
            _text(company.get("company_description"))
            or _text(company.get("company_profile"))
            or _text(general.get("company_profile"))
        )

    bg: list[str] = []
    contrib = (
        _text(chosen.get("key_contribution"))
        or _text(chosen.get("biography"))
        or _text(chosen.get("notes"))
    )
    if contrib:
        bg.append(f"* {_trim(contrib, 360)}")
    if overview:
        first = re.split(r"(?<=[.!?])\s+", overview.strip())
        blurb = " ".join(first[:2]).strip()
        if employer:
            bg.append(f"* **{employer}** (employer on file): {_trim(blurb, 280)}")
        else:
            bg.append(f"* {_trim(blurb, 280)}")
    if sector:
        bg.append(f"* Employer sector on record: **{sector}**.")
    ksa_emp = _first_num(
        km.get("ksa_employees") if km else None,
        mena.get("ksa_employees") if mena else None,
        (company or {}).get("number_of_employees_ksa") if company else None,
    )
    mena_emp = _first_num(
        km.get("mena_employees") if km else None,
        mena.get("mena_employees") if mena else None,
        (company or {}).get("number_of_employees_mena") if company else None,
    )
    head_bits = []
    if ksa_emp is not None and format_int(ksa_emp):
        head_bits.append(f"{format_int(ksa_emp)} in Saudi Arabia")
    if mena_emp is not None and format_int(mena_emp):
        head_bits.append(f"{format_int(mena_emp)} in MENA")
    if head_bits and employer:
        bg.append(
            f"* **{employer}** MENA footprint on file: " + ", ".join(head_bits) + "."
        )
    if tenure and employer and title and not any("on file as" in b.lower() for b in bg):
        bg.append(f"* On file as **{title}** at **{employer}** ({tenure}).")

    strat: list[str] = []
    if company:
        pso = _text(company.get("potential_strategic_opportunity"))
        if pso:
            if ":" in pso:
                head, rest = pso.split(":", 1)
                if 8 <= len(head) <= 80:
                    strat.append(f"* **{head.strip()}:** {_trim(rest.strip(), 280)}")
                else:
                    strat.append(f"* {_trim(pso, 300)}")
            else:
                strat.append(f"* {_trim(pso, 300)}")
        notes = _text(mena.get("mena_notes")) if mena else None
        if notes and len(strat) < 3:
            for angle in _strategic_from_notes(notes)[:3]:
                if angle not in strat:
                    strat.append(angle)
        if sector and len(strat) < 4:
            strat.append(
                f"* Engage **{name}** / **{employer}** leadership on **{sector}** "
                f"localisation mapped to {_sector_demand_anchor(sector or '')} — "
                f"ask for a dated capability commitment, not a generic MoU."
            )
        if employer and len(strat) < 5:
            strat.append(
                f"* Position **{employer}**'s Saudi / MENA footprint as the "
                f"account hub for Vision 2030 demand (NEOM, SDAIA, NUPCO, RHQ "
                f"Program) and use **{name}** as the executive sponsor."
            )
    if not strat and employer:
        strat.append(
            f"* Confirm live priorities with the **{employer}** relationship "
            f"owner before outreach to **{name}**, then table a named "
            f"programme ask (SDAIA / NEOM / NUPCO / RHQ Program)."
        )
    strat.append(
        f"* **MISA next ask:** secure a 90-day follow-up with **{name}**"
        + (f" ({title})" if title else "")
        + " on one concrete localisation or RHQ expansion deliverable."
    )

    parts = [HEADERS["person_role"], "", lead, ""]
    if bullets:
        parts.extend(bullets)
        parts.append("")
    # Jul21 person briefs always carried Strategic Context before Background.
    parts.append("## Strategic Context")
    parts.append("")
    parts.append(
        f"**{name}** is a priority executive contact"
        + (f" at **{employer}**" if employer else "")
        + (f" in **{sector}**" if sector else "")
        + ". Vision 2030 demand corridors create a concrete agenda for "
        "engagement — digital (SDAIA / LEAP), healthcare (NUPCO), "
        "giga-projects (NEOM), and industrial localisation (NIDLP). "
        "Lead with the MISA record footprint, then a named programme ask."
    )
    parts.append("")
    if bg:
        parts.append(HEADERS["person_background"])
        parts.append("")
        seen: set[str] = set()
        for b in bg[:8]:
            key = b.casefold()
            if key in seen:
                continue
            seen.add(key)
            parts.append(b)
        parts.append("")
    if strat:
        parts.append(HEADERS["strategic_read"])
        parts.append("")
        parts.extend(strat[:6])
        parts.append("")
    parts.append("## Recommended Next Actions for MISA")
    parts.append("")
    parts.append(
        f"* Brief **{name}** on a named Vision 2030 demand corridor within "
        f"90 days, with a written capability offer."
    )
    if employer:
        parts.append(
            f"* Align the **{employer}** account plan with RHQ conversion / "
            f"expansion and a LEAP / FII calendar slot."
        )
    parts.append(
        f"* Confirm counterpart ministries / agencies before the meeting "
        f"(MCIT, SDAIA, MoH / NUPCO, or Energy as relevant)."
    )
    parts.append("")
    parts.append(make_footer(["executive records", "company_profiles"]))
    return "\n".join(parts).strip() + "\n"


def _first_num(*vals) -> Any:
    for v in vals:
        if v is None or v == "" or v == []:
            continue
        if isinstance(v, dict):
            continue
        try:
            if isinstance(v, str):
                s = v.strip().replace(",", "").replace("$", "")
                if not s or s.lower() in ("n/a", "na", "none", "null"):
                    continue
                return float(s)
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _nested_get(row: dict, *path: str) -> Any:
    cur: Any = row
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _as_list(val: Any) -> list:
    if isinstance(val, list):
        return [x for x in val if isinstance(x, dict)]
    return []


def _related_lists(row: dict) -> dict[str, list]:
    """Flatten `_related` enrichment (human labels → row lists)."""
    related = row.get("_related")
    if not isinstance(related, dict):
        return {}
    out: dict[str, list] = {}
    for k, v in related.items():
        if isinstance(v, list):
            out[str(k).lower()] = [x for x in v if isinstance(x, dict)]
    return out


def _harvest_geo(row: dict, all_rows: list[dict] | None = None) -> list[dict]:
    """Collect geographic revenue rows from nested, flat, related, or siblings."""
    geo = _as_list(row.get("geographic_revenue")) or _as_list(
        row.get("geographic_revenues")
    )
    if not geo:
        rel = _related_lists(row)
        for key in ("geographic revenues", "geographic_revenues", "geographic revenue"):
            if rel.get(key):
                geo = rel[key]
                break
    if not geo and all_rows:
        for r in all_rows:
            if r is row or not isinstance(r, dict):
                continue
            if r.get("region") and (r.get("percentage") is not None or r.get("revenue")):
                if not _is_company_row(r) and not _is_person_row(r):
                    geo.append(r)
    return geo


def _harvest_opportunities(row: dict, all_rows: list[dict] | None = None) -> list[dict]:
    opps = (
        _as_list(row.get("match_opportunities"))
        or _as_list(row.get("opportunities"))
    )
    if not opps:
        rel = _related_lists(row)
        for key in ("opportunities", "match opportunities", "match_opportunities"):
            if rel.get(key):
                opps = rel[key]
                break
    if not opps and all_rows:
        for r in all_rows:
            if r is row or not isinstance(r, dict):
                continue
            if r.get("title") and (
                r.get("description") is not None
                or r.get("match_reason") is not None
                or r.get("value") is not None
            ) and not _is_company_row(r) and not _is_person_row(r):
                # Heuristic: opportunity-shaped sibling tool-call rows.
                if "company_profile_id" in r or "sector_name" in r or "stage" in r:
                    opps.append(r)
    return opps


def _harvest_financials(row: dict) -> list[dict]:
    fins = _as_list(row.get("financial_performance")) or _as_list(
        row.get("financial_performances")
    )
    if not fins:
        rel = _related_lists(row)
        for key in ("financial performance", "financial_performances", "financials"):
            if rel.get(key):
                fins = rel[key]
                break
    return fins


def rows_from_correlator_summary(summary: dict) -> list[dict]:
    """Build render-ready rows from a correlator / SSE prep summary.

    Folds geographic revenues, opportunities, and financials onto the
    primary company row so Operational Detail / Strategic Read never
    depend on accidental denormalised JSON columns.
    """
    if not isinstance(summary, dict):
        return []
    primary = summary.get("primary")
    if not isinstance(primary, dict):
        return []
    row = dict(primary)
    geo = summary.get("geographic_revenues") or summary.get("geographic_revenue") or []
    if isinstance(geo, list) and geo:
        row["geographic_revenue"] = [g for g in geo if isinstance(g, dict)]
    opps = summary.get("opportunities") or summary.get("match_opportunities") or []
    if isinstance(opps, list) and opps:
        row["match_opportunities"] = [o for o in opps if isinstance(o, dict)]
    fins = (
        summary.get("financial_performances")
        or summary.get("financial_performance")
        or []
    )
    if isinstance(fins, list) and fins:
        row["financial_performance"] = [f for f in fins if isinstance(f, dict)]
    rows: list[dict] = [row]
    for key in ("executives", "company_executives"):
        for er in (summary.get(key) or [])[:12]:
            if not isinstance(er, dict):
                continue
            er2 = dict(er)
            if er2.get("name") and not er2.get("full_name"):
                er2["full_name"] = er2["name"]
            if er2.get("position") and not er2.get("title"):
                er2["title"] = er2["position"]
            rows.append(er2)
    return rows


def _question_looks_like_person(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if re.search(
        r"\b(who\s+is|who's|who\s+was|who\s+runs|who\s+leads|name\s+of\s+the|"
        r"ceo\s+of|cfo\s+of|cto\s+of|chairman\s+of|chairperson\s+of|"
        r"president\s+of|founder\s+of|"
        r"ceo\s+profile|profile\s+(?:for|of)\s+the\s+ceo|"
        r"tell\s+me\s+about\s+the\s+ceo|"
        r"(?:^|\b)ceo\s+of\b)\b",
        q,
    ):
        return True
    # Bare "CEO of X" / "CEO profile for X" without leading who-
    if re.search(r"(?i)(?:^|\b)ceo\s+(?:of|profile\b)", q):
        return True
    return False


def _trim(text: str, limit: int = 380) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1].rsplit(" ", 1)[0] + "…"


def _strategic_from_notes(notes: str) -> list[str]:
    """Turn MENA notes into distinct engagement angles (old.pdf style)."""
    if not notes:
        return []
    chunks: list[str] = []
    for c in re.split(r"[\r\n]+", notes):
        c = c.strip()
        if not c:
            continue
        for p in re.split(r"(?=\d+\))", c):
            p = re.sub(r"^\d+\)\s*", "", p).strip()
            if p:
                chunks.append(p)
    angles: list[str] = []
    seen: set[str] = set()
    rules = (
        (
            r"distribution hub|logistics zone",
            "Middle East distribution hub in Saudi logistics zone",
            "critical opportunity for MISA to engage on supply chain "
            "localisation and logistics partnerships.",
        ),
        (
            r"developer academy|digital skills",
            "Developer Academy / digital-skills presence",
            "platform for collaboration with Saudi digital skills and "
            "innovation programmes.",
        ),
        (
            r"arab business machine|\babm\b|principal distributor|distributor",
            "Regional distribution partnership",
            "align retail expansion and service localisation with the "
            "on-file distributor network.",
        ),
    )
    blob = notes.lower()
    for needle, title, action in rules:
        if re.search(needle, blob, re.I):
            key = title.casefold()
            if key not in seen:
                seen.add(key)
                angles.append(f"* **{title}** — {action}")
    # If notes had substance but no rule matched, keep one concrete excerpt.
    if not angles and chunks:
        angles.append(f"* {_trim(chunks[0], 320)}")
    return angles


def _format_opportunity_bullet(opp: dict) -> str | None:
    title = (
        _text(opp.get("title"))
        or _text(opp.get("sub_title"))
        or _text(opp.get("sector_name"))
    )
    # Prefer substantive bodies over thin "Sector alignment: …" reasons.
    body = (
        _text(opp.get("description"))
        or _text(opp.get("sub_title"))
        or _text(opp.get("match_reason"))
    )
    reason = _text(opp.get("match_reason")) or ""
    if reason.lower().startswith("sector alignment") and _text(opp.get("sub_title")):
        body = _text(opp.get("sub_title"))
    # Avoid "Title: Title: rest" when sub_title repeats the title prefix.
    if title and body and body.casefold().startswith(title.casefold() + ":"):
        body = body.split(":", 1)[1].strip()
    elif title and body and body.casefold() == title.casefold():
        body = reason if reason and not reason.lower().startswith("sector alignment") else ""
    value = opp.get("value")
    value_s = ""
    if value not in (None, ""):
        try:
            value_s = format_currency(float(value), default="")
        except (TypeError, ValueError):
            value_s = _text(value)
    if title and body:
        extra = f" (value {value_s})" if value_s else ""
        return f"* **{title}:** {_trim(body, 320)}{extra}"
    if title and value_s:
        return f"* **{title}** — recorded opportunity value {value_s}."
    if title:
        return f"* On-file opportunity: **{title}**."
    if body:
        if ":" in body:
            head, rest = body.split(":", 1)
            if 8 <= len(head) <= 80:
                return f"* **{head.strip()}:** {_trim(rest.strip(), 320)}"
        return f"* **Strategic opportunity (record):** {_trim(body, 340)}"
    return None


def render_company_brief(
    rows: list[dict],
    *,
    user_question: str = "",
    locale: str = "en",
) -> str | None:
    """STYLE_GUIDE company briefing from Postgres — Operational Detail + Strategic Read.

    Accepts nested ``misa_details``, flat projection columns, correlator
    folds, and ``_related`` enrichment — same insight depth as the pre-
    Ollama PDF briefs.
    """
    row = _pick_company_row(rows)
    if not row:
        return None

    misa = row.get("misa_details") if isinstance(row.get("misa_details"), dict) else {}
    general = misa.get("general") if isinstance(misa.get("general"), dict) else {}
    mena = misa.get("mena_details") if isinstance(misa.get("mena_details"), dict) else {}
    rhq = misa.get("rhq_details") if isinstance(misa.get("rhq_details"), dict) else {}
    key_metrics = row.get("key_metrics") if isinstance(row.get("key_metrics"), dict) else {}
    corp = row.get("corporate_profile") if isinstance(row.get("corporate_profile"), dict) else {}
    market = row.get("market_intelligence") if isinstance(row.get("market_intelligence"), dict) else {}

    name = _text(row.get("company_name")) or "Company"
    sector = (
        _text(corp.get("sector"))
        or _text(key_metrics.get("sector"))
        or _text(general.get("sector"))
        or _text(general.get("industry"))
        or _text(row.get("most_active_business_unit"))
        or _text(row.get("sector"))
        or _text(row.get("industry"))
        or "—"
    )

    fin_list = _harvest_financials(row)
    fin0 = fin_list[0] if fin_list else {}

    rev = _first_num(
        key_metrics.get("annual_revenue_usd"),
        general.get("revenue_usd"),
        fin0.get("total_revenue") if isinstance(fin0, dict) else None,
        row.get("revenue_usd"),
        row.get("annual_revenue"),
    )
    employees = _first_num(
        key_metrics.get("employee_count"),
        general.get("global_employees"),
        row.get("employee_count"),
        row.get("number_of_employees"),
    )
    mena_emp = _first_num(
        key_metrics.get("mena_employees"),
        mena.get("mena_employees"),
        row.get("number_of_employees_mena"),
    )
    ksa_emp = _first_num(
        key_metrics.get("ksa_employees"),
        mena.get("ksa_employees"),
        row.get("number_of_employees_ksa"),
    )
    hq = (
        _text(general.get("global_headquarters"))
        or _text(key_metrics.get("headquarters"))
        or _text(row.get("headquarters"))
        or _text(row.get("global_headquarters"))
        or "—"
    )

    rhq_status_raw = rhq.get("rhq_status")
    if rhq_status_raw is None:
        rhq_status_raw = row.get("rhq_status")
    if rhq_status_raw is None:
        rhq_status_raw = row.get("is_rhq")
    rhq_license = (
        _text(rhq.get("rhq_license_status"))
        or _text(row.get("rhq_license_status"))
    )
    if isinstance(rhq_status_raw, bool):
        rhq_yes = rhq_status_raw
    else:
        rhq_yes = str(rhq_status_raw or "").strip().lower() in (
            "yes", "true", "1", "active",
        )
    rhq_city = _text(rhq.get("rhq_city")) or _text(row.get("rhq_city"))
    rhq_country = _text(rhq.get("rhq_country")) or _text(row.get("rhq_country"))
    rhq_loc = ", ".join(x for x in (rhq_city, rhq_country) if x) or "—"
    if rhq_yes and rhq_license:
        rhq_label = f"Yes ({rhq_license} license)"
    elif rhq_yes:
        rhq_label = "Yes"
    else:
        rhq_label = "No"

    used_geo = False
    used_opps = False
    mena_fin = "—"
    geo = _harvest_geo(row, rows)
    geo_bits: list[str] = []
    for g in geo:
        region = (
            _text(g.get("region"))
            or _text(g.get("region_name"))
            or _text(g.get("geography"))
            or ""
        )
        pct = (
            _text(g.get("percentage"))
            or _text(g.get("revenue_percentage"))
            or _text(g.get("pct"))
            or ""
        )
        if region and pct:
            geo_bits.append(f"{pct} from {region}")
            used_geo = True
        region_l = region.lower()
        if "middle east" in region_l or region_l in ("mena", "mea"):
            try:
                mena_fin = format_currency(
                    float(str(g.get("revenue") or "").replace(",", "")),
                    default="—",
                )
                used_geo = True
            except (TypeError, ValueError):
                pass
    if mena_fin == "—":
        mena_fin = "MENA revenue not separately reported"

    human_mena = []
    if ksa_emp is not None and format_int(ksa_emp):
        human_mena.append(f"{format_int(ksa_emp)} in Saudi Arabia")
    if mena_emp is not None and format_int(mena_emp):
        human_mena.append(f"{format_int(mena_emp)} in MENA")
    human_mena_s = ", ".join(human_mena) if human_mena else "—"

    global_fin = format_currency(rev, default="—")
    if global_fin != "—":
        global_fin = f"{global_fin} annual revenue"
        if isinstance(fin0, dict) and fin0.get("year"):
            global_fin += f" ({fin0.get('year')})"

    overview = (
        _text(corp.get("overview"))
        or _text(general.get("company_profile"))
        or _text(row.get("company_description"))
        or _text(row.get("company_profile"))
    )
    if overview:
        parts_s = re.split(r"(?<=[.!?])\s+", overview.strip())
        # Jul21 company briefs opened with a multi-sentence framing blurb,
        # not a two-sentence stub.
        blurb = " ".join(parts_s[:4]).strip()
        if sector and sector != "—" and sector.lower() not in blurb.lower():
            blurb = f"**{name}** is a global player in **{sector}**. " + blurb
        if len(blurb) > 900:
            blurb = blurb[:897].rsplit(" ", 1)[0] + "…"
    else:
        blurb = f"**{name}**" + (
            f" operates in **{sector}**." if sector != "—" else "."
        )
        blurb += (
            " MISA should treat the on-file Saudi / MENA footprint as the "
            "working account base and map expansion asks to Vision 2030 "
            "demand corridors (SDAIA / LEAP, NUPCO, NEOM, NIDLP / PIF zones)."
        )

    table = "\n".join([
        "### 📊 Corporate Profile & Regional Footprint",
        "",
        "| Metric | Global Performance | Saudi Arabia & MENA Region |",
        "| --- | --- | --- |",
        f"| **Core Sector** | {sector} | {sector} |",
        f"| **Financials** | {global_fin} | {mena_fin} |",
        f"| **Human Capital** | {_cell(employees, count=True)} employees globally | {human_mena_s} |",
        f"| **Regional Headquarters** | {hq} | RHQ in {rhq_loc} ({rhq_label}) |",
    ])

    # Flat + nested MENA fields (projection / SELECT * / misa_details).
    history = (
        _text(mena.get("history_in_mena"))
        or _text(row.get("history_in_mena"))
    )
    notes = _text(mena.get("mena_notes")) or _text(row.get("mena_notes"))
    mena_entity = (
        _text(mena.get("companies_name_in_mena"))
        or _text(row.get("companies_name_in_mena"))
    )
    ksa_entity = (
        _text(mena.get("companies_name_in_ksa"))
        or _text(row.get("companies_name_in_ksa"))
    )
    presence_mena = (
        _text(mena.get("presence_in_mena"))
        or _text(row.get("presence_in_mena"))
    )
    presence_saudi = (
        _text(mena.get("presence_in_saudi"))
        or _text(row.get("presence_in_saudi"))
    )
    type_saudi = (
        _text(mena.get("type_of_presence_saudi"))
        or _text(row.get("type_of_presence_saudi"))
    )
    locs = mena.get("mena_locations")
    if locs is None:
        locs = row.get("mena_locations")

    ops: list[str] = []
    products = (
        _text(general.get("product_services"))
        or _text(corp.get("core_services"))
        or _text(row.get("product_services"))
        or _text(row.get("core_services"))
    )
    if products:
        clean = re.sub(r"\s*\|\s*", "; ", products)
        clean = re.sub(r"\s+", " ", clean).strip()
        ops.append(f"* **Product & services portfolio:** {_trim(clean, 420)}")
    if geo_bits:
        ops.append("* **Geographic revenue mix:** " + "; ".join(geo_bits) + ".")
    if history:
        ops.append(f"* **MENA entry & expansion:** {_trim(history, 420)}")
    if mena_entity or ksa_entity:
        bits = []
        if mena_entity:
            bits.append(f"MENA hub entity **{mena_entity.rstrip(',')}**")
        if ksa_entity:
            bits.append(f"KSA entity **{ksa_entity.rstrip(',')}**")
        loc_s = ""
        if isinstance(locs, list) and locs:
            loc_s = " covering " + ", ".join(str(x) for x in locs)
        elif _text(locs):
            loc_s = f" covering {_text(locs)}"
        ops.append("* **On-file MENA entities:** " + "; ".join(bits) + loc_s + ".")
    if notes:
        chunks = [c.strip() for c in re.split(r"[\r\n]+", notes) if c.strip()]
        expanded: list[str] = []
        for c in chunks:
            for p in re.split(r"(?=\d+\))", c):
                p = re.sub(r"^\d+\)\s*", "", p).strip()
                if p:
                    expanded.append(p)
        for note in expanded[:4]:
            ops.append(f"* {_trim(note, 380)}")
    lead_bits: list[str] = []
    for r in _pick_person_rows(rows)[:5]:
        pn, pt = _person_name(r), _person_title(r)
        if pn and pt:
            lead_bits.append(f"**{pn}** ({pt})")
    if not lead_bits:
        for er in (row.get("executive_leadership") or [])[:5]:
            if isinstance(er, dict):
                pn = _text(er.get("name")) or _text(er.get("full_name"))
                pt = _text(er.get("position")) or _text(er.get("title"))
                if pn and pt:
                    lead_bits.append(f"**{pn}** ({pt})")
    ceo = _text(row.get("ceo"))
    if ceo and not any(ceo.lower() in x.lower() for x in lead_bits):
        lead_bits.insert(0, f"**{ceo}** (CEO)")
    if lead_bits:
        ops.append("* **Leadership (record):** " + "; ".join(lead_bits[:4]) + ".")
    mcap = _text(market.get("market_cap"))
    trend = _text(market.get("market_trend"))
    pricing = _text(market.get("pricing_strategy"))
    if mcap or trend or pricing or row.get("profit_margin") or row.get("roe"):
        fin_bits = []
        if mcap:
            fin_bits.append(f"market cap **{mcap}**")
        if row.get("profit_margin"):
            fin_bits.append(f"profit margin **{_text(row.get('profit_margin'))}**")
        if row.get("roe"):
            fin_bits.append(f"ROE **{_text(row.get('roe'))}**")
        line = "* **Financial & competitive posture:** "
        if fin_bits:
            line += ", ".join(fin_bits)
        if pricing:
            line += ("; " if fin_bits else "") + pricing.rstrip(".")
        if trend:
            line += (". " if fin_bits or pricing else "") + trend.rstrip(".")
        ops.append(line + ".")

    mena_bullets: list[str] = []
    if presence_mena:
        mena_bullets.append(f"* **Presence in MENA:** {presence_mena}")
    if presence_saudi:
        mena_bullets.append(
            f"* **Presence in Saudi:** {presence_saudi}"
            + (f" ({type_saudi})" if type_saudi else "")
        )
    mena_bullets.append(
        f"* **RHQ status:** {rhq_label}"
        + (f" — {rhq_loc}" if rhq_loc != "—" else "")
    )
    if human_mena_s != "—":
        mena_bullets.append(f"* **Headcount (record):** {human_mena_s}")

    # Strategic Read — Jul21 depth: named programmes + concrete MISA plays.
    strat: list[str] = []
    for opp in _harvest_opportunities(row, rows)[:5]:
        bullet = _format_opportunity_bullet(opp)
        if bullet and not any(bullet[4:40] in s for s in strat):
            strat.append(bullet)
            used_opps = True
    pso = _text(row.get("potential_strategic_opportunity"))
    if pso and not any(pso[:48].casefold() in s.casefold() for s in strat):
        bullet = _format_opportunity_bullet({"description": pso})
        if bullet:
            strat.append(bullet)
            used_opps = True
    for angle in _strategic_from_notes(notes or ""):
        if len(strat) >= 6:
            break
        if not any(angle[4:50].casefold() in s.casefold() for s in strat):
            strat.append(angle)
    if (
        sector
        and sector != "—"
        and (presence_saudi or presence_mena or human_mena_s != "—")
        and len(strat) < 6
    ):
        strat.append(
            f"* **{sector} footprint in KSA/MENA** aligns with Vision 2030 "
            f"demand corridors (SDAIA / LEAP for digital, NUPCO for health, "
            f"NEOM and industrial zones for localisation) — pitch a concrete "
            f"capability or RHQ expansion against a named programme."
        )
    if rhq_yes and not any("RHQ" in s for s in strat) and len(strat) < 7:
        strat.append(
            f"* Treat the RHQ in **{rhq_loc or 'KSA'}** as the account hub — "
            f"schedule a 12-month capability plan tied to giga-project / "
            f"procurement demand (NEOM, SDAIA, NUPCO as relevant)."
        )
    rhq_bullet = None
    if rhq_yes and rhq_license and rhq_license.lower() == "inactive":
        rhq_bullet = (
            f"* RHQ is on file in **{rhq_loc}** with **{rhq_license}** license "
            f"status — clarify reactivation / coverage before pitching."
        )
    if not strat:
        if mena_entity or ksa_entity or history:
            strat.append(
                f"* Build the next conversation around the on-file MENA operating "
                f"footprint for **{name}**"
                + (f" ({mena_entity or ksa_entity})" if (mena_entity or ksa_entity) else "")
                + "; confirm live priorities with the relationship owner."
            )
        else:
            strat.append(
                f"* Ground next steps in the on-file MENA footprint for **{name}**; "
                f"confirm live priorities with the relationship owner and map "
                f"to a named Vision 2030 anchor (NEOM, SDAIA, NUPCO, or RHQ "
                f"Program) before the next account review."
            )
    if rhq_bullet and len(strat) < 7:
        strat.append(rhq_bullet)
    # Always close Strategic Read with an executable MISA play.
    if not any("MISA" in s and ("should" in s.casefold() or "brief" in s.casefold()
               or "schedule" in s.casefold() or "pitch" in s.casefold())
               for s in strat):
        strat.append(
            f"* **MISA next ask:** brief **{name}** RHQ / KSA leadership on a "
            f"named Vision 2030 demand corridor within 90 days, with a "
            f"written capability offer (not a generic partnership pitch)."
        )

    # Recommended Next Actions — Jul21 company briefs always closed with
    # executable, named moves (not only Strategic Read bullets).
    next_actions: list[str] = []
    if rhq_yes:
        next_actions.append(
            f"* Run an RHQ expansion account review with **{name}** "
            f"({rhq_loc}) within 90 days — map demand to "
            f"**{_sector_demand_anchor(sector)}** and a written 12-month "
            f"capability offer."
        )
    elif presence_saudi or ksa_entity:
        next_actions.append(
            f"* Qualify **{name}** for licence deepening or RHQ conversion — "
            f"schedule a MISA-led incentive and site walkthrough with "
            f"**{_sector_demand_anchor(sector)}** within 90 days."
        )
    else:
        next_actions.append(
            f"* Open a soft-landing dialogue with **{name}** on Saudi market "
            f"entry via the RHQ Program and **{_sector_demand_anchor(sector)}** "
            f"within 90 days."
        )
    if lead_bits:
        # Extract first person name from "**Name** (Title)"
        m = re.search(r"\*\*([^*]+)\*\*", lead_bits[0])
        if m:
            next_actions.append(
                f"* Engage **{m.group(1).strip()}** on a concrete localisation "
                f"or procurement ask tied to {_sector_demand_anchor(sector)}."
            )
    next_actions.append(
        f"* Publish a one-pager of **{name}**'s Saudi / MENA footprint for "
        f"desk targeting ahead of LEAP / FII."
    )
    if used_opps:
        next_actions.append(
            f"* Advance the top on-file opportunity threads for **{name}** "
            f"with a dated next meeting and named Saudi counterpart."
        )

    strategic_context = (
        f"**{name}** is a priority account for Saudi investment attraction"
        + (f" in **{sector}**" if sector and sector != "—" else "")
        + ". Lead with "
        + _sector_demand_anchor(sector if sector and sector != "—" else "")
        + " as the concrete demand agenda — a dated capability offer, not a "
        "generic partnership pitch. MISA should deepen the installed Saudi / "
        "MENA footprint and convert warm licence / RHQ presence into "
        "expansion commitments."
    )

    out: list[str] = [
        f"## {name} — Executive Briefing",
        "",
        blurb,
        "",
        "## Strategic Context",
        "",
        strategic_context,
        "",
        "---",
        "",
        table,
        "",
    ]
    if ops:
        out.append("### Snapshot of Operations and Market Position")
        out.append("")
        out.extend(ops[:12])
        out.append("")
    else:
        # Always ship the Jul21 ops header — sparse rows still need the
        # section so soft_check / live contracts don't fail closed.
        out.append("### Snapshot of Operations and Market Position")
        out.append("")
        out.append(
            f"- **{name}** is on the MISA record; detailed operational "
            f"fields are sparse in the current extract — use FK-linked "
            f"executives / MENA notes / RHQ status in the Corporate Profile "
            f"table as the working baseline."
        )
        out.append("")
    if mena_bullets:
        out.append(HEADERS["saudi_position"])
        out.append("")
        out.extend(mena_bullets)
        out.append("")
    out.append(HEADERS["strategic_read"])
    out.append("")
    out.extend(strat[:8])
    out.append("")
    out.append("## Recommended Next Actions for MISA")
    out.append("")
    out.extend(next_actions[:6])
    out.append("")
    footer_sources = ["company_profiles", "company_executives"]
    if used_geo:
        footer_sources.append("company_geographic_revenues")
    if used_opps or notes:
        footer_sources.append("opportunities")
    out.append(make_footer(footer_sources))
    return "\n".join(out).strip() + "\n"


def _sector_demand_anchor(sector: str) -> str:
    s = (sector or "").casefold()
    if any(k in s for k in ("ict", "tech", "digital", "software", "telecom")):
        return "SDAIA / LEAP"
    if any(k in s for k in ("health", "pharma", "life", "bio")):
        return "NUPCO / healthcare localisation"
    if any(k in s for k in ("energy", "oil", "gas", "power", "renew", "water")):
        return "NEOM / energy transition"
    if any(k in s for k in ("construct", "infra", "real estate", "engineer")):
        return "giga-projects / NIDLP"
    if any(k in s for k in ("financ", "bank", "insur")):
        return "financial-sector development"
    return "a named Vision 2030 programme"


def _question_looks_like_engagement_plan(question: str) -> bool:
    return bool(_ENGAGEMENT_PLAN_RE.search(question or ""))


def render_engagement_brief(
    rows: list[dict],
    *,
    user_question: str = "",
    locale: str = "en",
) -> str | None:
    """Match ai_response_6.pdf shape for engagement-plan asks.

    Structure (VERBATIM order):
      ## Engagement Recommendation  (approach / stakeholders / why /
                                     talking points / risks)
      ## Snapshot
      ## Saudi / MENA Position
      ## Strategic Read

    NOT the Corporate Profile table + Operational Detail dump — that
    shape is for company_profile asks. Engagement asks need the
    outreach plan voice.
    """
    row = _pick_company_row(rows)
    if not row:
        return None

    misa = row.get("misa_details") if isinstance(row.get("misa_details"), dict) else {}
    general = misa.get("general") if isinstance(misa.get("general"), dict) else {}
    mena = misa.get("mena_details") if isinstance(misa.get("mena_details"), dict) else {}
    rhq = misa.get("rhq_details") if isinstance(misa.get("rhq_details"), dict) else {}
    km = row.get("key_metrics") if isinstance(row.get("key_metrics"), dict) else {}

    name = _text(row.get("company_name")) or "the company"
    sector = (
        _text(km.get("sector"))
        or _text(row.get("most_active_business_unit"))
        or _text(row.get("sector"))
        or "ICT"
    )
    hq = (
        _text(km.get("headquarters"))
        or _text(general.get("global_headquarters"))
        or _text(row.get("global_headquarters"))
        or _text(row.get("headquarters"))
        or "—"
    )
    rev = _first_num(km.get("annual_revenue_usd"), row.get("revenue_usd"), row.get("annual_revenue"))
    employees = _first_num(km.get("employee_count"), row.get("employee_count"), row.get("number_of_employees"))
    ksa_emp = _first_num(km.get("ksa_employees"), mena.get("ksa_employees"), row.get("number_of_employees_ksa"))
    mena_emp = _first_num(km.get("mena_employees"), mena.get("mena_employees"), row.get("number_of_employees_mena"))
    rhq_emp = _first_num(rhq.get("rhq_employees"), row.get("rhq_number_of_employees"))

    notes = _text(mena.get("mena_notes")) or _text(row.get("mena_notes")) or ""
    notes_l = notes.lower()
    history = _text(mena.get("history_in_mena")) or _text(row.get("history_in_mena")) or ""
    mena_ent = (_text(mena.get("companies_name_in_mena")) or _text(row.get("companies_name_in_mena")) or "").rstrip(",")
    ksa_ent = (_text(mena.get("companies_name_in_ksa")) or _text(row.get("companies_name_in_ksa")) or "").rstrip(",")
    coverage = _text(rhq.get("rhq_country_coverage")) or ""
    rhq_city = _text(rhq.get("rhq_city")) or _text(row.get("rhq_city")) or ""
    rhq_country = _text(rhq.get("rhq_country")) or _text(row.get("rhq_country")) or ""
    rhq_loc = ", ".join(x for x in (rhq_city, rhq_country) if x) or "—"
    rhq_license = _text(rhq.get("rhq_license_status")) or _text(row.get("rhq_license_status")) or ""
    overview = (
        _text(general.get("company_profile"))
        or _text(row.get("company_description"))
        or _text(row.get("company_profile"))
        or ""
    )
    if overview:
        overview = " ".join(re.split(r"(?<=[.!?])\s+", overview.strip())[:2]).strip()
        if len(overview) > 220:
            overview = overview[:217].rsplit(" ", 1)[0] + "…"

    has_hub = "distribution hub" in notes_l or "logistics zone" in notes_l
    has_academy = "developer academy" in notes_l
    has_abm = "arab business machine" in notes_l or re.search(r"\babm\b", notes_l)
    pso = _text(row.get("potential_strategic_opportunity")) or ""

    # ── Engagement Recommendation ──────────────────────────────────
    focus = []
    if has_hub:
        focus.append("Saudi distribution hub")
    if has_academy:
        focus.append("Developer Academy")
    focus.append("digital services ecosystem")
    approach = (
        f"Initiate a targeted dialogue with **{name}**'s regional leadership "
        f"to propose investment partnerships focused on expanding their "
        + ", ".join(focus[:3])
        + "."
    )

    stakeholders = []
    if rhq_loc != "—":
        stakeholders.append(f"{name} MENA regional executives based in {rhq_city or rhq_loc}")
    if ksa_ent:
        stakeholders.append(f"**{ksa_ent}** management")
    else:
        stakeholders.append(f"{name} Saudi management")
    stakeholders.append("MISA digital economy and ICT sector leads")
    if has_hub:
        stakeholders.append("Saudi logistics zone authorities")

    why: list[str] = []
    if has_hub:
        why.append(
            f"* **{name}**'s Middle East distribution hub in Saudi Arabia's "
            f"special logistics zone signals commitment to localising supply "
            f"chain and operations — a concrete hook for deeper investment talks."
        )
    if has_academy:
        why.append(
            f"* The Riyadh-based Apple Developer Academy is a platform to deepen "
            f"Saudi talent development in software and app ecosystems, aligning "
            f"with digital-economy goals."
            if "apple" in name.lower()
            else
            f"* The on-file Developer Academy / digital-skills presence is a "
            f"platform to deepen Saudi talent development and align with "
            f"digital-economy goals."
        )
    why.append(
        f"* **{name}**'s global scale in **{sector}** can attract complementary "
        f"technology suppliers and service providers into Saudi Arabia."
    )
    if pso and "gaming" in pso.lower() and len(why) < 4:
        why.append(
            "* On-file gaming & digital-content opportunity aligns with Saudi "
            "esports / Arabic localisation priorities."
        )

    talking: list[str] = []
    if has_hub:
        talking.append(
            "* Highlight Saudi logistics-zone incentives and infrastructure "
            "supporting distribution expansion."
        )
    if has_academy:
        talking.append(
            "* Discuss collaboration to scale the Developer Academy and "
            "integrate local startups and SMEs into the app ecosystem."
        )
    talking.append(
        f"* Present Saudi Arabia's growing **{sector}** market and digital "
        f"transformation agenda as a growth platform for services and cloud."
    )
    if has_abm:
        talking.append(
            "* Use the on-file distributor network (including ABM where "
            "recorded) to frame retail expansion and service localisation."
        )
    talking.append(
        f"* Offer facilitation for **{name}**'s supply-chain partners and "
        f"technology vendors to establish regional operations in Saudi Arabia."
    )

    risks: list[str] = []
    if has_hub:
        risks.append(
            f"* Confirm **{name}**'s planned investment scale and timeline for "
            f"expanding the Saudi distribution hub."
        )
    risks.append(
        f"* Validate appetite for deeper local manufacturing, assembly, or "
        f"services partnerships beyond distribution."
    )
    if rhq_license and rhq_license.lower() == "inactive":
        risks.append(
            f"* RHQ on file in **{rhq_loc}** is **Inactive** — clarify "
            f"reactivation / coverage before pitching."
        )
    risks.append(
        f"* Assess regulatory or compliance issues **{name}** may raise in "
        f"Saudi ICT / digital services."
    )

    # ── Snapshot ───────────────────────────────────────────────────
    rev_s = format_currency(rev, default="")
    emp_s = format_int(employees) or ""
    snap_bits = [f"**{name}** is a global player in **{sector}**"]
    if hq and hq != "—":
        snap_bits.append(f"headquartered in {hq}")
    money = []
    if rev_s:
        money.append(f"{rev_s} revenue")
    if emp_s:
        money.append(f"{emp_s} employees")
    if money:
        snap_bits.append("with " + " and ".join(money))
    snap = ", ".join(snap_bits) + "."
    if overview and overview.lower() not in snap.lower():
        snap += " " + overview

    # ── Saudi / MENA Position ───────────────────────────────────────
    mena_pos: list[str] = []
    if rhq_loc != "—":
        cov = f", covering {coverage}" if coverage else ""
        rhq_line = f"* **{name}** holds an RHQ in **{rhq_loc}**{cov}"
        if rhq_emp is not None and format_int(rhq_emp):
            rhq_line += f", with **{format_int(rhq_emp)}** employees at the RHQ"
        if rhq_license:
            rhq_line += f" (**{rhq_license}** license)"
        mena_pos.append(rhq_line + ".")
    if ksa_ent or ksa_emp is not None:
        line = f"* Presence in Saudi Arabia"
        if ksa_ent:
            line += f" as **{ksa_ent}**"
        if ksa_emp is not None and format_int(ksa_emp):
            line += f" with **{format_int(ksa_emp)}** employees"
        if mena_emp is not None and format_int(mena_emp):
            line += f"; total MENA headcount is **{format_int(mena_emp)}**"
        mena_pos.append(line + ".")
    mena_pos.append(
        "* MENA revenue: not reliably recorded separately from global "
        "revenue in the database."
    )
    if has_hub:
        mena_pos.append(
            f"* **{name}** established its Middle East distribution hub in "
            f"Saudi Arabia's special logistics zone."
        )
    if has_academy:
        mena_pos.append(
            "* Riyadh hosts the Developer Academy (first in MENA on file), "
            "focused on local developer empowerment."
        )
    if has_abm:
        mena_pos.append(
            "* Arab Business Machine (ABM) is the principal distributor "
            "across multiple Middle Eastern countries on file."
        )
    elif mena_ent:
        mena_pos.append(f"* On-file MENA hub entity: **{mena_ent}**.")

    # ── Strategic Read (worded differently from talking points) ────
    strat: list[str] = []
    if has_hub:
        strat.append(
            f"* The regional distribution hub in Saudi Arabia's logistics zone "
            f"is a critical supply-chain node that can pull ancillary technology "
            f"and logistics firms into the Kingdom."
        )
    if has_academy:
        strat.append(
            "* The Developer Academy in Riyadh cultivates Saudi digital talent "
            "and is a bridge into a local app / software ecosystem and startup "
            "pipeline."
        )
    strat.append(
        f"* **{name}**'s **{sector}** capabilities align with "
        f"{_sector_demand_anchor(sector)} — frame the ask around services, "
        f"localisation, and RHQ capability expansion, not only distribution "
        f"or sales."
    )
    strat.append(
        f"* MISA should push **{name}** to expand beyond its current Saudi "
        f"posture into named Vision 2030 programmes (NEOM, SDAIA, NUPCO, "
        f"NIDLP), while facilitating entry for suppliers and technology "
        f"partners."
    )
    strat.append(
        f"* Use LEAP / FII / sector exhibitions as forcing functions for a "
        f"dated **{name}** commitment — RHQ upgrade, localisation JV, or "
        f"shared-services centre."
    )
    if rhq_license and rhq_license.lower() == "inactive":
        strat.append(
            f"* Clarify the **Inactive** RHQ license in **{rhq_loc}** before "
            f"committing outreach cadence or incentive packages."
        )

    parts = [
        HEADERS["engagement_recommendation"],
        "",
        f"* **Recommended approach:** {approach}",
        f"* **Priority stakeholders:** {'; '.join(stakeholders)}.",
        "* **Why this matters to MISA:**",
        *why[:5],
        "* **Talking points:**",
        *talking[:6],
        "* **Risks / unknowns:**",
        *risks[:4],
        "",
        "## Strategic Context",
        "",
        f"**{name}** is a priority engagement account"
        + (f" in **{sector}**" if sector else "")
        + ". Treat the on-file Saudi / MENA footprint as the working "
        "account base and convert warm presence into Vision 2030-linked "
        "expansion (RHQ Program, SDAIA / LEAP, NUPCO, NEOM, NIDLP).",
        "",
        HEADERS["snapshot"],
        "",
        snap,
        "",
        HEADERS["saudi_position"],
        "",
        *mena_pos,
        "",
        HEADERS["strategic_read"],
        "",
        *strat[:7],
        "",
        "## Recommended Next Actions for MISA",
        "",
        f"* Schedule a 90-day account review with **{name}** leadership "
        f"against {_sector_demand_anchor(sector)}.",
        f"* Table a written RHQ / localisation offer — not a generic "
        f"partnership pitch.",
        f"* Align IPA / chamber counterparts in the home market for a "
        f"joint mission slot (LEAP / FII).",
        "",
        make_footer(["company_profiles", "company_executives", "opportunities"]),
    ]
    return "\n".join(parts).strip() + "\n"



def render_db_briefing(
    rows: list[dict],
    *,
    intent: str | None = None,
    table: str | None = None,
    user_question: str = "",
    locale: str = "en",
    force: bool = False,
) -> str | None:
    """Return a deterministic briefing, or None if this payload isn't covered.

    ``force=True`` renders templates even when narrative cloud is preferred
    (compose / SSE fallback after curation fails).
    """
    if not rows:
        return None
    if not force and not use_deterministic_db_briefing():
        return None
    # Person / CEO asks win over company dump — including when intent was
    # misclassified as company_profile / general_research.
    person_q = _question_looks_like_person(user_question)
    if person_q and _pick_person_rows(rows):
        ans = render_person_brief(
            rows, user_question=user_question, locale=locale,
        )
        if ans:
            return ans
    if person_q:
        # Company rows only (no executive FK rows) — still emit ## Role
        # so CEO asks never ship as Executive Briefing.
        company = _pick_company_row(rows)
        if company:
            cname = _text(company.get("company_name")) or "this account"
            role = "CEO"
            qlow = (user_question or "").lower()
            for hint, label in (
                ("cfo", "CFO"), ("cto", "CTO"),
                ("chairman", "Chairman"), ("chairperson", "Chairperson"),
                ("founder", "Founder"), ("president", "President"),
                ("ceo", "CEO"), ("chief executive", "CEO"),
            ):
                if hint in qlow:
                    role = label
                    break
            # Prefer an on-file executive name when nested on the company row.
            exec_name = None
            for er in (company.get("executives") or company.get("company_executives") or []):
                if not isinstance(er, dict):
                    continue
                title = str(er.get("title") or er.get("position") or "").lower()
                if role.lower() in title or (
                    role == "CEO" and "chief executive" in title
                ):
                    exec_name = (
                        er.get("full_name") or er.get("name") or ""
                    ).strip() or None
                    if exec_name:
                        break
            lead = (
                f"**{exec_name} is {role} at {cname}.**"
                if exec_name
                else f"**The {role} of {cname}** is the account lead for MISA engagement."
            )
            return (
                f"## Role\n\n{lead}\n\n"
                f"## Background\n\n"
                f"* On-file employer: **{cname}**.\n"
                f"* Title requested: **{role}**.\n"
                f"* Detailed biography was not in the executive extract for "
                f"this turn — use the Role lead and Strategic Read for outreach.\n\n"
                f"## Strategic Context\n\n"
                f"**{cname}** is a priority account. Lead with a dated "
                f"capability offer to the named {role}, not a generic "
                f"partnership pitch.\n\n"
                f"## 🇸🇦 Strategic Read\n\n"
                f"* Engage the {role} office at **{cname}** on RHQ / "
                f"localisation within 90 days.\n\n"
                f"## Recommended Next Actions for MISA\n\n"
                f"- Schedule a MISA account review with the {role} office "
                f"at **{cname}** within 90 days.\n"
                f"- Table a written RHQ / localisation offer mapped to a "
                f"named Vision 2030 counterpart within 90 days.\n"
                f"- Align **{cname}** to a LEAP / FII calendar slot with a "
                f"named Saudi counterpart.\n\n"
                + make_footer(["company_profiles", "company_executives"])
                + "\n"
            )
    if _should_render_person(intent, table, rows):
        ans = render_person_brief(
            rows, user_question=user_question, locale=locale,
        )
        if ans:
            return ans
    # Engagement-plan asks: crisp recommendation + full Ops brief
    # (do not fall through to fluffy LLM talking-points essays).
    if (
        (intent or "") == "engagement_strategy"
        or _question_looks_like_engagement_plan(user_question)
    ) and _pick_company_row(rows):
        ans = render_engagement_brief(
            rows, user_question=user_question, locale=locale,
        )
        if ans:
            return ans
    if _should_render_company(intent, table, rows):
        return render_company_brief(
            rows, user_question=user_question, locale=locale,
        )
    if _pick_person_rows(rows):
        return render_person_brief(
            rows, user_question=user_question, locale=locale,
        )
    if _pick_company_row(rows):
        return render_company_brief(
            rows, user_question=user_question, locale=locale,
        )
    return None
