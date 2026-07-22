"""Structured company-targeting payloads for advisory answers.

Validates model JSON against a fixed schema, then renders readable
markdown (chat) and table-safe HTML fragments (PDF). Keeps MISA
footprint figures out of the model's hands when DB context exists.
"""

from __future__ import annotations

import json
import re
from typing import Any


_TARGET_TYPES = {"expansion", "new_entry"}
_EVIDENCE = {"high", "medium", "low"}


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v).strip()


def validate_company_targeting_payload(
    data: dict | None,
    *,
    db_context: dict | None = None,
) -> tuple[dict | None, list[str]]:
    """Return (normalized_payload, errors). None payload if unusable."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return None, ["payload_not_object"]

    footprint_in = data.get("current_footprint") or {}
    if not isinstance(footprint_in, dict):
        footprint_in = {}
        errors.append("current_footprint_not_object")

    # Prefer live DB context over model-claimed footprint figures.
    licensed = None
    rhq = None
    retrieval_status = _str(footprint_in.get("retrieval_status"), "unknown")
    source = _str(footprint_in.get("source"), "MISA database")
    limitations = [_str(x) for x in _as_list(footprint_in.get("limitations")) if _str(x)]

    if db_context and db_context.get("footprint_data_unavailable"):
        licensed = None
        rhq = None
        retrieval_status = "error"
        limitations.append("Internal footprint retrieval failed; counts omitted.")
    elif db_context and db_context.get("companies_from_origin_licensed_in_saudi") is not None:
        licensed = int(db_context["companies_from_origin_licensed_in_saudi"])
        rhq = int(db_context.get("companies_from_origin_with_rhq") or 0)
        retrieval_status = _str(db_context.get("retrieval_status"), "ok")
        source = (db_context.get("retrieval_filters") or {}).get(
            "source") or source
        if retrieval_status == "zero_records":
            limitations.append(
                "Query returned zero verified records for the applied filters."
            )
    else:
        try:
            if footprint_in.get("licensed_companies") is not None:
                licensed = int(footprint_in["licensed_companies"])
            if footprint_in.get("rhq_companies") is not None:
                rhq = int(footprint_in["rhq_companies"])
        except (TypeError, ValueError):
            errors.append("invalid_footprint_counts")
            licensed, rhq = None, None

    targets_raw = data.get("targets") or []
    if not isinstance(targets_raw, list) or not targets_raw:
        errors.append("targets_missing")
        return None, errors

    targets: list[dict] = []
    for i, t in enumerate(targets_raw[:12]):
        if not isinstance(t, dict):
            errors.append(f"target_{i}_not_object")
            continue
        company = _str(t.get("company"))
        if not company:
            errors.append(f"target_{i}_no_company")
            continue
        ttype = _str(t.get("target_type"), "new_entry").lower().replace("-", "_")
        if ttype not in _TARGET_TYPES:
            ttype = "new_entry"
            errors.append(f"target_{i}_bad_type")
        evidence = _str(t.get("evidence_strength"), "medium").lower()
        if evidence not in _EVIDENCE:
            evidence = "medium"
        proposed = _str(t.get("proposed_investment"))
        if not proposed:
            errors.append(f"target_{i}_no_investment")
            proposed = "Requires validation — proposed investment not specified"
        targets.append({
            "rank": int(t.get("rank") or (len(targets) + 1)),
            "company": company,
            "sector": _str(t.get("sector"), "Unclassified"),
            "current_saudi_presence": _str(
                t.get("current_saudi_presence"), "Unknown"),
            "target_type": ttype,
            "proposed_investment": proposed,
            "why_company": _str(t.get("why_company") or t.get("investment_thesis")),
            "why_saudi": _str(t.get("why_saudi")),
            "why_now": _str(t.get("why_now")),
            "saudi_strategic_fit": [
                _str(x) for x in _as_list(t.get("saudi_strategic_fit")) if _str(x)
            ],
            "misa_action": _str(t.get("misa_action")),
            "evidence": [_str(x) for x in _as_list(t.get("evidence")) if _str(x)],
            "evidence_strength": evidence,
            "validation_required": [
                _str(x) for x in _as_list(t.get("validation_required")) if _str(x)
            ],
        })

    if len(targets) < 3:
        errors.append("too_few_targets")
        return None, errors

    # Ensure expansion targets from DB appear when available.
    if db_context and db_context.get("expansion_targets"):
        known = {t["company"].casefold() for t in targets}
        injected = 0
        for et in db_context["expansion_targets"]:
            name = _str(et.get("company"))
            if not name or name.casefold() in known:
                continue
            targets.insert(injected, {
                "rank": 0,
                "company": name,
                "sector": _str(et.get("sector"), "Unclassified"),
                "current_saudi_presence": _str(
                    et.get("current_saudi_presence"), "Licensed"),
                "target_type": "expansion",
                "proposed_investment": (
                    "Deepen Saudi presence via RHQ / capability expansion"
                    if et.get("current_saudi_presence") == "RHQ"
                    else "Expand licensed operations into a substantive "
                         "local investment (facility, SSC, or JV)"
                ),
                "why_company": "Already present in MISA footprint.",
                "why_saudi": "Existing licence / RHQ is an expansion base.",
                "why_now": "Vision 2030 localisation and regional scale-up.",
                "saudi_strategic_fit": [],
                "misa_action": f"Account review with {name} RHQ/licence lead.",
                "evidence": ["MISA database footprint"],
                "evidence_strength": _str(et.get("evidence_strength"), "high"),
                "validation_required": [],
            })
            known.add(name.casefold())
            injected += 1
            if injected >= 4:
                break
        for i, t in enumerate(targets, 1):
            t["rank"] = i

    exec_sum = data.get("executive_summary") or {}
    if isinstance(exec_sum, str):
        exec_sum = {"key_findings": [exec_sum]}
    if not isinstance(exec_sum, dict):
        exec_sum = {}

    payload = {
        "title": _str(data.get("title")) or (
            f"Targeting companies from "
            f"{(db_context or {}).get('origin_country') or 'priority markets'}"
        ),
        "executive_summary": {
            "key_findings": [
                _str(x) for x in _as_list(
                    exec_sum.get("key_findings") or exec_sum.get("bullets")
                ) if _str(x)
            ][:8],
            "top_recommendation": _str(exec_sum.get("top_recommendation")),
        },
        "current_footprint": {
            "licensed_companies": licensed,
            "rhq_companies": rhq,
            "source": source,
            "retrieval_status": retrieval_status,
            "as_of_date": _str(footprint_in.get("as_of_date")),
            "limitations": limitations,
            "leading_companies": [
                _str(t.get("name") or t.get("company"))
                for t in (
                    (db_context or {}).get("top_rhq_companies")
                    or (db_context or {}).get("expansion_targets")
                    or []
                )[:6]
                if _str(t.get("name") or t.get("company"))
            ],
        },
        "targets": targets,
        "trade_bodies": [
            x if isinstance(x, dict) else {"organisation": _str(x)}
            for x in _as_list(data.get("trade_bodies"))
        ][:12],
        "recommendations": [
            _str(x) if not isinstance(x, dict) else _str(
                x.get("action") or x.get("text"))
            for x in _as_list(data.get("recommendations"))
        ][:12],
        "sources": [_str(x) if not isinstance(x, dict) else _str(
            x.get("label") or x.get("name") or x.get("url"))
            for x in _as_list(data.get("sources"))][:12],
        "data_limitations": [
            _str(x) for x in _as_list(data.get("data_limitations")) if _str(x)
        ][:12],
    }
    return payload, errors


def extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object from a model reply."""
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def render_company_targeting_markdown(payload: dict) -> str:
    """Chat-facing markdown from a validated payload."""
    fp = payload["current_footprint"]
    lines: list[str] = [f"# {payload['title']}", ""]
    if payload.get("_truncated"):
        lines += [
            "> **Partial result:** thesis enrichment was truncated; "
            "ranking below is DB-seeded and complete, but some thesis "
            "text may be thinner than usual.",
            "",
        ]

    lines.append("## Executive Summary")
    findings = payload["executive_summary"].get("key_findings") or []
    if fp.get("licensed_companies") is not None:
        findings = [
            f"MISA database: **{fp['licensed_companies']}** licensed companies "
            f"from this origin; **{fp.get('rhq_companies') or 0}** with RHQ "
            f"status (retrieval: {fp.get('retrieval_status')})."
        ] + [f for f in findings if "licensed" not in f.lower()]
    for f in findings[:6]:
        lines.append(f"- {f}")
    if payload["executive_summary"].get("top_recommendation"):
        lines.append(
            f"- **Top recommendation:** "
            f"{payload['executive_summary']['top_recommendation']}"
        )
    exp_n = sum(1 for t in payload["targets"] if t["target_type"] == "expansion")
    new_n = sum(1 for t in payload["targets"] if t["target_type"] == "new_entry")
    lines.append(
        f"- Priority list split: **{exp_n}** expansion targets and "
        f"**{new_n}** new-entry targets."
    )
    lines.append("")

    # Strategic Context — old Jul21 briefs always led with this framing.
    strat = (payload.get("strategic_context") or "").strip()
    if not strat:
        country = (
            (payload.get("current_footprint") or {}).get("origin_country")
            or "the origin market"
        )
        strat = (
            f"{country} is a priority source market for Saudi investment "
            f"attraction. Vision 2030 diversification — digital economy "
            f"(SDAIA, LEAP), giga-projects (NEOM, Red Sea Global, Qiddiya), "
            f"healthcare localisation (NUPCO), and industrial capability "
            f"(NIDLP / PIF zones) — creates concrete demand corridors for "
            f"{country} corporates seeking a **regional growth platform**. "
            f"MISA should deepen engagement with the installed licensed/"
            f"RHQ footprint as expansion anchors, then pursue high-fit "
            f"new entrants in Tier-1 sectors via the national IPA and "
            f"sector bodies of {country}."
        )
    lines.append("## Strategic Context")
    lines.append(strat)
    lines.append("")

    lines.append("## Current Saudi Footprint")
    status = fp.get("retrieval_status")
    if status in ("error", "SOURCE_UNAVAILABLE", "TIMEOUT", "CONNECTION_ERROR",
                  "UNKNOWN_ERROR") or (
        fp.get("licensed_companies") is None
        and status not in ("zero_records", "SUCCESS_EMPTY")
    ):
        lines.append(
            "Internal MISA footprint data could not be retrieved for this "
            "query. Do not interpret this as zero licensed companies or "
            "zero RHQs."
        )
    elif status in ("zero_records", "SUCCESS_EMPTY"):
        lines.append(
            f"The queried MISA source returned **0** verified licensed "
            f"records and **0** RHQ records for the applied filters "
            f"(source: {fp.get('source') or 'MISA database'})."
        )
    else:
        lines.append(
            f"According to MISA's database (**{fp.get('source')}**; "
            f"retrieval `{status}`), **{fp['licensed_companies']}** "
            f"companies from this origin are licensed in Saudi Arabia, "
            f"of which **{fp.get('rhq_companies') or 0}** hold RHQ status."
        )
        leads = fp.get("leading_companies") or []
        if leads:
            lines.append("Leading companies in the footprint: " + ", ".join(leads) + ".")
    for lim in fp.get("limitations") or []:
        lines.append(f"- Limitation: {lim}")
    lines.append("")

    lines.append("## Priority Company Ranking")
    lines.append("")
    # Lean 5-column table so PDF export keeps a real table (not profile
    # cards). Detail (proposed investment, MISA action) lives in theses.
    lines.append(
        "| Rank | Company | Sector | Saudi Presence | Investment Thesis |"
    )
    lines.append("|---|---|---|---|---|")
    for t in payload["targets"]:
        thesis = (t.get("why_company") or "")[:110]
        if t.get("why_saudi"):
            thesis = (thesis + " " + t["why_saudi"])[:140]
        thesis = thesis.replace("|", "/").replace("\n", " ")
        row = [
            str(t["rank"]),
            t["company"].replace("|", "/"),
            t["sector"].replace("|", "/"),
            t["current_saudi_presence"].replace("|", "/"),
            thesis,
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Detailed Investment Theses")
    lines.append("")
    for t in payload["targets"][:8]:
        lines.append(f"### {t['company']}")
        lines.append("")
        why_c = t.get("why_company") or "Requires validation"
        why_s = t.get("why_saudi") or "Requires validation"
        why_n = t.get("why_now") or "Requires validation"
        prop = t.get("proposed_investment") or "Requires validation"
        action = t.get("misa_action") or "Requires validation"
        # Jul21 narrative: short paragraphs, not one-line stubs.
        lines.append(
            f"**Target type:** {t['target_type']} · "
            f"**Evidence strength:** {t.get('evidence_strength') or 'medium'}"
        )
        lines.append("")
        lines.append(
            f"**Why this company.** {why_c}"
        )
        lines.append("")
        lines.append(
            f"**Why Saudi Arabia.** {why_s}"
        )
        lines.append("")
        lines.append(
            f"**Why now.** {why_n}"
        )
        lines.append("")
        lines.append(
            f"**Proposed investment.** {prop}"
        )
        if t.get("saudi_strategic_fit"):
            lines.append("")
            lines.append(
                "**Saudi strategic fit.** "
                + "; ".join(t["saudi_strategic_fit"])
            )
        lines.append("")
        lines.append(
            f"**Recommended MISA action.** {action}"
        )
        if t.get("evidence"):
            lines.append("")
            lines.append("**Evidence.** " + "; ".join(t["evidence"]))
        if t.get("validation_required"):
            lines.append("")
            lines.append(
                "**Requires validation.** "
                + "; ".join(t["validation_required"])
            )
        lines.append("")

    if payload.get("trade_bodies"):
        lines.append("## Investment and Trade Bodies")
        lines.append("")
        lines.append("| Organisation | Type | Role |")
        lines.append("|---|---|---|")
        for b in payload["trade_bodies"]:
            if not isinstance(b, dict):
                continue
            org = _str(b.get("organisation") or b.get("name"))
            if not org:
                continue
            lines.append(
                f"| {org} | {_str(b.get('type'))} | {_str(b.get('role'))} |"
            )
        lines.append("")

    lines.append("## Recommended Next Actions for MISA")
    # Prefer company-named actions (old Jul21 quality). Generic enrichment
    # bullets without a company name are demoted.
    recs = [r for r in (payload.get("recommendations") or []) if r]
    named = [r for r in recs if any(
        (t.get("company") or "")[:12].lower() in r.lower()
        for t in (payload.get("targets") or [])
        if t.get("company")
    )]
    if not named:
        named = [
            f"Engage **{t['company']}** — {t['misa_action']}"
            for t in (payload.get("targets") or [])[:8]
            if t.get("misa_action")
        ]
    for r in named[:10]:
        lines.append(f"- {r}")
    extras = [r for r in recs if r not in named][:4]
    for r in extras:
        lines.append(f"- {r}")
    if not named and not extras:
        lines.append(
            "- Prioritise account reviews with the top RHQ holders above."
        )
    lines.append("")

    lines.append("## Sources and Data Limitations")
    lines.append("- **Internal MISA data:** footprint counts and expansion targets where retrieval_status is `ok` or `zero_records`.")
    lines.append("- **Official / public data:** company sector and market context used in theses.")
    lines.append("- **Analytical inference:** ranking judgements and 'why now' timing arguments.")
    for s in payload.get("sources") or []:
        if s:
            lines.append(f"- {s}")
    for lim in payload.get("data_limitations") or []:
        lines.append(f"- Limitation: {lim}")
    for lim in fp.get("limitations") or []:
        lines.append(f"- Limitation: {lim}")
    lines.append("")
    lines.append(
        "*Strategic analysis synthesised from market knowledge; "
        "MISA database figures cited where noted.*"
    )
    return "\n".join(lines)


def company_targeting_json_system_addon() -> str:
    return (
        "Return ONLY a single compact JSON object (no markdown fences):\n"
        "{\n"
        '  "strategic_context": string,\n'
        '  "executive_summary": {"key_findings": [string], '
        '"top_recommendation": string},\n'
        '  "theses": {\n'
        '    "<ExactCompanyName>": {\n'
        '      "why_company": string, "why_saudi": string, "why_now": string,\n'
        '      "proposed_investment": string, "misa_action": string,\n'
        '      "saudi_strategic_fit": [string],\n'
        '      "validation_required": [string],\n'
        '      "evidence_strength": "high"|"medium"|"low"\n'
        "    }\n"
        "  },\n"
        '  "new_entry_targets": [{"company": string, "sector": string, '
        '"proposed_investment": string, "why_company": string, '
        '"why_saudi": string, "why_now": string, "misa_action": string, '
        '"validation_required": [string]}],\n'
        '  "trade_bodies": [{"organisation": string, "type": string, '
        '"role": string}],\n'
        '  "recommendations": [string],\n'
        '  "data_limitations": [string]\n'
        "}\n"
        "RULES:\n"
        "- strategic_context: 3–5 sentences framing THIS origin market vs "
        "Vision 2030 (giga-projects, digital, healthcare, industry). Name "
        "real source-market programmes where known; never borrow another "
        "country's IPA or industrial policy labels.\n"
        "- Thesis fields (why_company / why_saudi / why_now / "
        "proposed_investment / misa_action): EACH 2–4 sentences "
        "(≈40–80 words), SPECIFIC — name Saudi programmes "
        "(NEOM, SDAIA, NUPCO, LEAP, Qiddiya, Red Sea, PIF zones, "
        "Monsha'at) and a concrete investment modality (RHQ expansion, "
        "manufacturing, R&D, SSC, logistics hub, JV). No one-liners.\n"
        "- Provide theses ONLY for the company names listed in the user "
        "message. Add at most 3 new_entry_targets not already listed.\n"
        "- recommendations: EACH bullet MUST name a specific company from "
        "the list AND a concrete MISA next step (not generic 'engage "
        "stakeholders').\n"
        "- trade_bodies: ONLY bodies of the origin country in the "
        "request (national IPA + chambers + export finance). Never "
        "include another country's IPA (e.g. Invest India in a German "
        "brief, or GTAI in an Indian brief).\n"
        "- Do NOT emit the ranking table or full report — the system "
        "assembles those.\n"
    )


def seed_company_targeting_payload_from_db(
    db_context: dict | None,
    *,
    max_expansion: int = 10,
) -> dict | None:
    """Deterministic complete payload from MISA footprint — never truncated."""
    if not db_context or db_context.get("footprint_data_unavailable"):
        return None
    country = _str(db_context.get("origin_country"), "priority markets")
    expansion = list(db_context.get("expansion_targets") or [])[:max_expansion]
    if not expansion and db_context.get(
            "companies_from_origin_licensed_in_saudi") is None:
        return None

    targets: list[dict] = []
    for i, et in enumerate(expansion, 1):
        name = _str(et.get("company"))
        if not name:
            continue
        presence = _str(et.get("current_saudi_presence"), "Licensed")
        proposed = (
            "RHQ capability expansion (shared-services / delivery / R&D centre)"
            if presence == "RHQ"
            else "Expand licensed operations into a substantive local "
                 "investment (facility, SSC, JV, or logistics hub)"
        )
        targets.append({
            "rank": i,
            "company": name,
            "sector": _str(et.get("sector"), "Unclassified"),
            "current_saudi_presence": presence,
            "target_type": "expansion",
            "proposed_investment": proposed,
            "why_company": (
                f"Already present in the MISA Saudi footprint ({presence})."
            ),
            "why_saudi": (
                "Existing licence/RHQ is an expansion base for Vision 2030 "
                "localisation and regional scale-up."
            ),
            "why_now": (
                "Deepen installed-base relationships before competitors "
                "capture localisation mandates."
            ),
            "saudi_strategic_fit": [],
            "misa_action": f"Account review with {name} licence/RHQ lead.",
            "evidence": ["MISA database footprint"],
            "evidence_strength": _str(et.get("evidence_strength"), "high"),
            "validation_required": [],
        })

    filters = db_context.get("retrieval_filters") or {}
    licensed = db_context.get("companies_from_origin_licensed_in_saudi")
    rhq = db_context.get("companies_from_origin_with_rhq")
    status = _str(db_context.get("retrieval_status"), "ok")
    leads = [
        _str(t.get("name") or t.get("company"))
        for t in (
            db_context.get("top_rhq_companies")
            or db_context.get("expansion_targets")
            or []
        )[:6]
        if _str(t.get("name") or t.get("company"))
    ]
    return {
        "title": (
            f"Targeting {country} Companies for Investment Attraction: "
            "Strategic List and Investment Thesis"
        ),
        "executive_summary": {
            "key_findings": [
                f"Prioritise expansion of already-licensed / RHQ companies "
                f"from {country} before cold outreach to new entrants.",
            ],
            "top_recommendation": (
                f"Run an account-based expansion campaign on the top "
                f"{min(8, len(targets))} MISA footprint companies."
            ),
        },
        "current_footprint": {
            "origin_country": country,
            "licensed_companies": licensed,
            "rhq_companies": rhq,
            "source": filters.get("source") or "MISA database",
            "retrieval_status": status,
            "as_of_date": "",
            "limitations": [
                "Counts reflect MISA licensed/RHQ records for the applied "
                "origin filters; informal presence is out of scope.",
            ],
            "leading_companies": leads,
        },
        "targets": targets,
        "trade_bodies": _default_trade_bodies(country),
        "recommendations": [
            f"{t['company']}: {t['misa_action']}" for t in targets[:6]
        ],
        "sources": ["MISA database (licensed / RHQ footprint)"],
        "data_limitations": [
            "Investment theses beyond footprint facts require validation "
            "with the company and sector teams.",
        ],
    }


# Country-keyed IPA / chamber catalogs. Keys are matched as substrings
# against a normalised origin label (see ``_trade_body_country_key``).
_TRADE_BODY_CATALOG: dict[str, list[dict]] = {
    "india": [
        {"organisation": "Invest India", "type": "IPA",
         "role": "Pipeline and soft-landing for Indian corporates"},
        {"organisation": "CII", "type": "Industry body",
         "role": "CEO-level access across manufacturing and services"},
        {"organisation": "FICCI", "type": "Industry body",
         "role": "Policy dialogue and mission facilitation"},
        {"organisation": "NASSCOM", "type": "Sector body",
         "role": "IT / digital services targeting"},
        {"organisation": "EXIM Bank of India", "type": "Finance",
         "role": "Trade and project finance introductions"},
    ],
    "germany": [
        {"organisation": "Germany Trade & Invest (GTAI)", "type": "IPA",
         "role": "Outbound/inbound facilitation and market intelligence"},
        {"organisation": "BDI (Federation of German Industries)",
         "type": "Industry body",
         "role": "CEO-level industrial and manufacturing access"},
        {"organisation": "DIHK / AHK network", "type": "Chamber",
         "role": "Bilateral chamber outreach and SME soft-landing"},
        {"organisation": "GESALO (German-Saudi Liaison Office)",
         "type": "Bilateral office",
         "role": "On-ground German corporate support in the Kingdom"},
        {"organisation": "KfW / German development finance channels",
         "type": "Finance",
         "role": "Project and export-finance introductions"},
    ],
    "united states": [
        {"organisation": "SelectUSA", "type": "IPA",
         "role": "Federal investment-promotion channel and soft-landing"},
        {"organisation": "U.S. Chamber of Commerce", "type": "Chamber",
         "role": "CEO / policy access across sectors"},
        {"organisation": "AmCham Saudi Arabia", "type": "Bilateral chamber",
         "role": "On-ground US corporate network in the Kingdom"},
        {"organisation": "US Commercial Service / Embassy Riyadh",
         "type": "Trade promotion",
         "role": "Qualified company introductions and missions"},
        {"organisation": "EXIM Bank of the United States", "type": "Finance",
         "role": "Export and project finance introductions"},
    ],
    "united kingdom": [
        {"organisation": "Department for Business and Trade (DBT)",
         "type": "IPA / trade",
         "role": "UK outbound investment facilitation"},
        {"organisation": "UK Export Finance (UKEF)", "type": "Finance",
         "role": "Export and project finance introductions"},
        {"organisation": "CBI", "type": "Industry body",
         "role": "CEO-level access across UK industry"},
        {"organisation": "British Chambers of Commerce / BritCham KSA",
         "type": "Chamber",
         "role": "Bilateral networking and soft-landing"},
        {"organisation": "City of London / UK financial services bodies",
         "type": "Sector body",
         "role": "FS / fintech corridor engagement"},
    ],
    "china": [
        {"organisation": "MOFCOM / Invest in China outbound desks",
         "type": "IPA",
         "role": "State-guided outbound pipeline access"},
        {"organisation": "CCPIT", "type": "Trade promotion",
         "role": "Trade missions and corporate introductions"},
        {"organisation": "CICC / policy bank channels (CDB, Exim China)",
         "type": "Finance",
         "role": "Project finance and Belt-and-Road corridor capital"},
        {"organisation": "Provincial IPAs (e.g. Shanghai, Guangdong, Shenzhen)",
         "type": "Regional IPA",
         "role": "Champion enterprise targeting by province"},
        {"organisation": "Saudi-Chinese Business Council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "japan": [
        {"organisation": "JETRO", "type": "IPA / trade",
         "role": "Japanese corporate soft-landing and intelligence"},
        {"organisation": "JBIC", "type": "Finance",
         "role": "Project and outbound investment finance"},
        {"organisation": "Keidanren", "type": "Industry body",
         "role": "CEO-level industrial access"},
        {"organisation": "JCCI / Japanese Chamber in Saudi Arabia",
         "type": "Chamber",
         "role": "On-ground Japanese corporate network"},
        {"organisation": "METI outbound desks", "type": "Government",
         "role": "Policy-aligned sector targeting"},
    ],
    "south korea": [
        {"organisation": "KOTRA", "type": "IPA / trade",
         "role": "Korean corporate pipeline and soft-landing"},
        {"organisation": "KEXIM / Korea Development Bank channels",
         "type": "Finance",
         "role": "Export and project finance introductions"},
        {"organisation": "Korea Chamber of Commerce & Industry (KCCI)",
         "type": "Chamber",
         "role": "Industry access and missions"},
        {"organisation": "Korean Chamber / KOCHAM in Saudi Arabia",
         "type": "Bilateral chamber",
         "role": "On-ground Korean corporate network"},
        {"organisation": "MOTIE outbound desks", "type": "Government",
         "role": "Policy-aligned industrial targeting"},
    ],
    "france": [
        {"organisation": "Business France", "type": "IPA",
         "role": "French corporate outbound facilitation"},
        {"organisation": "MEDEF", "type": "Industry body",
         "role": "CEO-level industrial and services access"},
        {"organisation": "Bpifrance", "type": "Finance",
         "role": "SME / mid-cap outbound finance introductions"},
        {"organisation": "French Chamber of Commerce in Saudi Arabia",
         "type": "Chamber",
         "role": "On-ground French corporate network"},
        {"organisation": "French Treasury / DG Trésor desks",
         "type": "Government",
         "role": "Strategic sector and giga-project alignment"},
    ],
    "italy": [
        {"organisation": "ICE / Italian Trade Agency", "type": "IPA / trade",
         "role": "Italian corporate pipeline and soft-landing"},
        {"organisation": "Confindustria", "type": "Industry body",
         "role": "CEO-level manufacturing access"},
        {"organisation": "CDP / SACE", "type": "Finance",
         "role": "Export and project finance introductions"},
        {"organisation": "Italian Chamber of Commerce in Saudi Arabia",
         "type": "Chamber",
         "role": "On-ground Italian corporate network"},
    ],
    "netherlands": [
        {"organisation": "Netherlands Foreign Investment Agency (NFIA) / RVO",
         "type": "IPA / trade",
         "role": "Dutch corporate outbound facilitation"},
        {"organisation": "VNO-NCW", "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "Dutch Chamber / NBG in Saudi Arabia",
         "type": "Chamber",
         "role": "On-ground Dutch corporate network"},
        {"organisation": "Atradius / Dutch finance channels", "type": "Finance",
         "role": "Trade and project finance introductions"},
    ],
    "switzerland": [
        {"organisation": "Switzerland Global Enterprise (S-GE)",
         "type": "IPA / trade",
         "role": "Swiss corporate soft-landing and intelligence"},
        {"organisation": "economiesuisse", "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "Swiss Business Council / Chamber in KSA",
         "type": "Chamber",
         "role": "On-ground Swiss corporate network"},
        {"organisation": "SERV", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "spain": [
        {"organisation": "ICEX / Invest in Spain outbound desks",
         "type": "IPA / trade",
         "role": "Spanish corporate pipeline"},
        {"organisation": "CEOE", "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "CESCE", "type": "Finance",
         "role": "Export and project finance"},
        {"organisation": "Spanish Chamber in Saudi Arabia",
         "type": "Chamber",
         "role": "On-ground Spanish corporate network"},
    ],
    "canada": [
        {"organisation": "Invest in Canada / Trade Commissioner Service",
         "type": "IPA / trade",
         "role": "Canadian corporate outbound facilitation"},
        {"organisation": "Canadian Chamber of Commerce / CanCham KSA",
         "type": "Chamber",
         "role": "Bilateral networking"},
        {"organisation": "EDC", "type": "Finance",
         "role": "Export and project finance introductions"},
        {"organisation": "Provincial IPAs (Ontario, Quebec, Alberta)",
         "type": "Regional IPA",
         "role": "Champion enterprise targeting by province"},
    ],
    "australia": [
        {"organisation": "Austrade", "type": "IPA / trade",
         "role": "Australian corporate pipeline and soft-landing"},
        {"organisation": "Export Finance Australia (EFA)", "type": "Finance",
         "role": "Export and project finance"},
        {"organisation": "Australian Chamber / AustCham channels",
         "type": "Chamber",
         "role": "Bilateral networking"},
        {"organisation": "State IPAs (NSW, Victoria, Queensland)",
         "type": "Regional IPA",
         "role": "State champion targeting"},
    ],
    "singapore": [
        {"organisation": "Enterprise Singapore", "type": "IPA / trade",
         "role": "Singapore corporate outbound facilitation"},
        {"organisation": "Singapore Business Federation",
         "type": "Industry body",
         "role": "CEO / mid-cap access"},
        {"organisation": "IE Singapore legacy networks / bilateral councils",
         "type": "Bilateral",
         "role": "Gulf corridor relationship management"},
    ],
    "united arab emirates": [
        {"organisation": "UAE Ministry of Economy / Invest UAE desks",
         "type": "IPA",
         "role": "Federal investment-promotion channel"},
        {"organisation": "Dubai FDI / Abu Dhabi Investment Office outbound",
         "type": "Emirate IPA",
         "role": "Champion enterprise targeting by emirate"},
        {"organisation": "UAE–Saudi bilateral business councils",
         "type": "Bilateral",
         "role": "Cross-border corporate introductions"},
        {"organisation": "ADPorts / logistics and industrial free-zone desks",
         "type": "Sector body",
         "role": "Logistics and industrial corridor plays"},
    ],
    "turkey": [
        {"organisation": "Investment Office of the Presidency of Türkiye",
         "type": "IPA",
         "role": "Turkish corporate outbound facilitation"},
        {"organisation": "DEİK", "type": "Business council",
         "role": "Bilateral business council access"},
        {"organisation": "TOBB", "type": "Chamber",
         "role": "Industry and SME access"},
        {"organisation": "Türk Eximbank", "type": "Finance",
         "role": "Export and project finance"},
    ],
    "brazil": [
        {"organisation": "ApexBrasil", "type": "IPA / trade",
         "role": "Brazilian corporate outbound facilitation"},
        {"organisation": "CNI / FIESP", "type": "Industry body",
         "role": "Industrial and manufacturing access"},
        {"organisation": "BNDES / Brazilian finance channels", "type": "Finance",
         "role": "Project finance introductions"},
        {"organisation": "Brazilian–Saudi business council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "pakistan": [
        {"organisation": "Board of Investment (BOI) Pakistan", "type": "IPA",
         "role": "Pipeline and soft-landing for Pakistani corporates"},
        {"organisation": "FPCCI", "type": "Industry body",
         "role": "CEO-level access across trade and industry"},
        {"organisation": "TDAP", "type": "Trade promotion",
         "role": "Export and outbound mission facilitation"},
        {"organisation": "Pakistani–Saudi business council / Jeddah channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "mexico": [
        {"organisation": "Secretariat of Economy / Invest in Mexico desks",
         "type": "IPA",
         "role": "Mexican corporate outbound facilitation"},
        {"organisation": "COMCE", "type": "Business council",
         "role": "Bilateral business council access"},
        {"organisation": "Bancomext", "type": "Finance",
         "role": "Export and project finance introductions"},
        {"organisation": "Mexican Chamber / bilateral KSA channels",
         "type": "Chamber",
         "role": "On-ground Mexican corporate network"},
    ],
    "indonesia": [
        {"organisation": "BKPM / Ministry of Investment Indonesia",
         "type": "IPA",
         "role": "Indonesian corporate outbound facilitation"},
        {"organisation": "Kadin Indonesia", "type": "Chamber",
         "role": "CEO-level industrial and services access"},
        {"organisation": "LPEI / Indonesian EXIM", "type": "Finance",
         "role": "Export and project finance"},
        {"organisation": "Indonesian–Saudi business council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "sweden": [
        {"organisation": "Business Sweden", "type": "IPA / trade",
         "role": "Swedish corporate soft-landing and intelligence"},
        {"organisation": "Swedish Export Credit Agency (EKN) / SEK",
         "type": "Finance",
         "role": "Export and project finance introductions"},
        {"organisation": "Confederation of Swedish Enterprise",
         "type": "Industry body",
         "role": "CEO-level industrial access"},
        {"organisation": "Swedish Chamber in Saudi Arabia",
         "type": "Chamber",
         "role": "On-ground Swedish corporate network"},
    ],
    "nigeria": [
        {"organisation": "Nigerian Investment Promotion Commission (NIPC)",
         "type": "IPA",
         "role": "Nigerian corporate outbound facilitation"},
        {"organisation": "NECA / MAN", "type": "Industry body",
         "role": "Industrial and manufacturing access"},
        {"organisation": "Nigerian Export-Import Bank (NEXIM)",
         "type": "Finance",
         "role": "Trade and project finance"},
        {"organisation": "Nigerian–Saudi business council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "egypt": [
        {"organisation": "General Authority for Investment (GAFI)",
         "type": "IPA",
         "role": "Egyptian corporate outbound facilitation"},
        {"organisation": "Federation of Egyptian Industries / FEI",
         "type": "Industry body",
         "role": "CEO-level industrial access"},
        {"organisation": "Export Development Bank of Egypt channels",
         "type": "Finance",
         "role": "Trade and project finance"},
        {"organisation": "Egyptian–Saudi business council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "vietnam": [
        {"organisation": "Foreign Investment Agency (FIA) Vietnam",
         "type": "IPA",
         "role": "Vietnamese corporate outbound facilitation"},
        {"organisation": "VCCI", "type": "Chamber",
         "role": "Industry and SME access"},
        {"organisation": "Vietnam EXIM / development finance channels",
         "type": "Finance",
         "role": "Trade and project finance"},
        {"organisation": "Vietnamese–Saudi business council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "poland": [
        {"organisation": "Polish Investment and Trade Agency (PAIH)",
         "type": "IPA",
         "role": "Polish corporate outbound facilitation"},
        {"organisation": "KIG / Polish Chamber of Commerce",
         "type": "Chamber",
         "role": "Industry access and missions"},
        {"organisation": "KUKE", "type": "Finance",
         "role": "Export credit introductions"},
        {"organisation": "Polish–Saudi business council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "malaysia": [
        {"organisation": "MIDA / MATRADE outbound desks", "type": "IPA / trade",
         "role": "Malaysian corporate pipeline and soft-landing"},
        {"organisation": "EXIM Bank of Malaysia", "type": "Finance",
         "role": "Trade and project finance"},
        {"organisation": "MICCI / bilateral KSA channels", "type": "Chamber",
         "role": "On-ground Malaysian corporate network"},
    ],
    "thailand": [
        {"organisation": "BOI Thailand / Department of International Trade Promotion",
         "type": "IPA / trade",
         "role": "Thai corporate outbound facilitation"},
        {"organisation": "EXIM Thailand", "type": "Finance",
         "role": "Trade and project finance"},
        {"organisation": "Thai–Saudi business council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "bangladesh": [
        {"organisation": "BIDA (Bangladesh Investment Development Authority)",
         "type": "IPA",
         "role": "Bangladeshi corporate outbound facilitation"},
        {"organisation": "FBCCI", "type": "Chamber",
         "role": "Industry and SME access"},
        {"organisation": "Bangladeshi–Saudi business council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "south africa": [
        {"organisation": "InvestSA / DTIC outbound desks", "type": "IPA",
         "role": "South African corporate outbound facilitation"},
        {"organisation": "Business Unity South Africa (BUSA)",
         "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "ECIC / development finance channels", "type": "Finance",
         "role": "Export and project finance"},
        {"organisation": "South African–Saudi business council channels",
         "type": "Bilateral",
         "role": "On-ground relationship management"},
    ],
    "russia": [
        {"organisation": "Russian Export Center / Invest in Russia outbound",
         "type": "IPA / trade",
         "role": "Russian corporate outbound facilitation"},
        {"organisation": "RSPP", "type": "Industry body",
         "role": "CEO-level industrial access"},
        {"organisation": "EXIAR / Russian finance channels", "type": "Finance",
         "role": "Export and project finance"},
    ],
    "belgium": [
        {"organisation": "Flanders Investment & Trade / AWEX / hub.brussels",
         "type": "Regional IPA",
         "role": "Belgian regional outbound facilitation"},
        {"organisation": "FEB / VBO", "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "Credendo", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "austria": [
        {"organisation": "Advantage Austria / ABA", "type": "IPA / trade",
         "role": "Austrian corporate soft-landing"},
        {"organisation": "WKÖ", "type": "Chamber",
         "role": "Industry access"},
        {"organisation": "OeKB", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "denmark": [
        {"organisation": "Invest in Denmark / Trade Council",
         "type": "IPA / trade",
         "role": "Danish corporate outbound facilitation"},
        {"organisation": "DI — Confederation of Danish Industry",
         "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "EKF Denmark", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "norway": [
        {"organisation": "Innovation Norway", "type": "IPA / trade",
         "role": "Norwegian corporate outbound facilitation"},
        {"organisation": "NHO", "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "GIEK / Export Finance Norway", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "finland": [
        {"organisation": "Business Finland", "type": "IPA / trade",
         "role": "Finnish corporate outbound facilitation"},
        {"organisation": "EK — Confederation of Finnish Industries",
         "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "Finnvera", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "philippines": [
        {"organisation": "Board of Investments (BOI) Philippines / DTI",
         "type": "IPA",
         "role": "Philippine corporate outbound facilitation"},
        {"organisation": "PCC / bilateral KSA channels", "type": "Chamber",
         "role": "On-ground relationship management"},
        {"organisation": "Philippine EXIM / development finance",
         "type": "Finance",
         "role": "Trade and project finance"},
    ],
    # ── Long-tail origins (adjective-map coverage; same contract) ──
    "ghana": [
        {"organisation": "Ghana Investment Promotion Centre (GIPC)",
         "type": "IPA", "role": "Ghanaian corporate outbound facilitation"},
        {"organisation": "AGI / Ghana Chamber of Commerce",
         "type": "Industry body", "role": "Industry and SME access"},
        {"organisation": "Ghana EXIM / development finance channels",
         "type": "Finance", "role": "Trade and project finance"},
    ],
    "kenya": [
        {"organisation": "Kenya Investment Authority (KenInvest)",
         "type": "IPA", "role": "Kenyan corporate outbound facilitation"},
        {"organisation": "KEPSA / KNCCI", "type": "Industry body",
         "role": "Private-sector and chamber access"},
        {"organisation": "Kenya EXIM / development finance channels",
         "type": "Finance", "role": "Trade and project finance"},
    ],
    "ethiopia": [
        {"organisation": "Ethiopian Investment Commission (EIC)",
         "type": "IPA", "role": "Ethiopian corporate outbound facilitation"},
        {"organisation": "ECCSA / Ethiopian Chamber", "type": "Chamber",
         "role": "Industry access"},
    ],
    "czechia": [
        {"organisation": "CzechInvest", "type": "IPA",
         "role": "Czech corporate outbound facilitation"},
        {"organisation": "Czech Chamber of Commerce", "type": "Chamber",
         "role": "Industry access"},
        {"organisation": "EGAP / Czech export credit", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "greece": [
        {"organisation": "Enterprise Greece", "type": "IPA",
         "role": "Greek corporate outbound facilitation"},
        {"organisation": "SEV Hellenic Federation of Enterprises",
         "type": "Industry body", "role": "CEO-level access"},
        {"organisation": "Export Credit Greece / development finance",
         "type": "Finance", "role": "Trade and project finance"},
    ],
    "portugal": [
        {"organisation": "AICEP Portugal Global", "type": "IPA / trade",
         "role": "Portuguese corporate outbound facilitation"},
        {"organisation": "CIP", "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "COSEC / Portuguese export credit", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "hungary": [
        {"organisation": "HIPA (Hungarian Investment Promotion Agency)",
         "type": "IPA", "role": "Hungarian corporate outbound facilitation"},
        {"organisation": "MKIK / Hungarian Chamber", "type": "Chamber",
         "role": "Industry access"},
        {"organisation": "EXIM Hungary", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "romania": [
        {"organisation": "InvestRomania / Ministry of Economy desks",
         "type": "IPA", "role": "Romanian corporate outbound facilitation"},
        {"organisation": "CCIR / Romanian Chamber", "type": "Chamber",
         "role": "Industry access"},
        {"organisation": "EximBank Romania", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "ukraine": [
        {"organisation": "UkraineInvest", "type": "IPA",
         "role": "Ukrainian corporate outbound facilitation"},
        {"organisation": "Ukrainian Chamber of Commerce and Industry",
         "type": "Chamber", "role": "Industry access"},
        {"organisation": "Export Credit Agency of Ukraine", "type": "Finance",
         "role": "Export credit introductions"},
    ],
    "morocco": [
        {"organisation": "AMDIE (Moroccan Investment and Export Agency)",
         "type": "IPA", "role": "Moroccan corporate outbound facilitation"},
        {"organisation": "CGEM", "type": "Industry body",
         "role": "CEO-level access"},
        {"organisation": "ASMEX / Moroccan exporters", "type": "Trade body",
         "role": "Exporter pipeline"},
    ],
    "algeria": [
        {"organisation": "AAPI / Algerian investment promotion desks",
         "type": "IPA", "role": "Algerian corporate outbound facilitation"},
        {"organisation": "FCE / Algerian business forums",
         "type": "Industry body", "role": "Private-sector access"},
    ],
    "tunisia": [
        {"organisation": "FIPA Tunisia", "type": "IPA",
         "role": "Tunisian corporate outbound facilitation"},
        {"organisation": "UTICA", "type": "Industry body",
         "role": "Industry access"},
        {"organisation": "CEPEX", "type": "Trade promotion",
         "role": "Export mission facilitation"},
    ],
    "libya": [
        {"organisation": "Libyan Investment promotion / economy desks",
         "type": "IPA", "role": "Libyan corporate outbound facilitation"},
        {"organisation": "Libyan Chamber of Commerce channels",
         "type": "Chamber", "role": "Industry access"},
    ],
    "sudan": [
        {"organisation": "Sudanese Investment Authority desks",
         "type": "IPA", "role": "Sudanese corporate outbound facilitation"},
        {"organisation": "Sudanese Chamber of Commerce channels",
         "type": "Chamber", "role": "Industry access"},
    ],
    "qatar": [
        {"organisation": "Investment Promotion Agency Qatar / Invest Qatar",
         "type": "IPA", "role": "Qatari corporate outbound facilitation"},
        {"organisation": "Qatar Chamber", "type": "Chamber",
         "role": "Industry access"},
        {"organisation": "Qatar–Saudi bilateral business councils",
         "type": "Bilateral", "role": "Cross-border introductions"},
    ],
    "kuwait": [
        {"organisation": "Kuwait Direct Investment Promotion Authority (KDIPA)",
         "type": "IPA", "role": "Kuwaiti corporate outbound facilitation"},
        {"organisation": "Kuwait Chamber of Commerce and Industry",
         "type": "Chamber", "role": "Industry access"},
        {"organisation": "Kuwait–Saudi bilateral business councils",
         "type": "Bilateral", "role": "Cross-border introductions"},
    ],
    "bahrain": [
        {"organisation": "Bahrain EDB / Invest in Bahrain",
         "type": "IPA", "role": "Bahraini corporate outbound facilitation"},
        {"organisation": "Bahrain Chamber of Commerce and Industry",
         "type": "Chamber", "role": "Industry access"},
        {"organisation": "Bahrain–Saudi bilateral business councils",
         "type": "Bilateral", "role": "Cross-border introductions"},
    ],
    "oman": [
        {"organisation": "Invest Oman / Ministry of Commerce desks",
         "type": "IPA", "role": "Omani corporate outbound facilitation"},
        {"organisation": "Oman Chamber of Commerce and Industry",
         "type": "Chamber", "role": "Industry access"},
        {"organisation": "Oman–Saudi bilateral business councils",
         "type": "Bilateral", "role": "Cross-border introductions"},
    ],
    "jordan": [
        {"organisation": "Jordan Investment Commission (JIC)",
         "type": "IPA", "role": "Jordanian corporate outbound facilitation"},
        {"organisation": "Jordan Chamber of Industry / Commerce",
         "type": "Chamber", "role": "Industry access"},
        {"organisation": "Jordan–Saudi bilateral business councils",
         "type": "Bilateral", "role": "Cross-border introductions"},
    ],
    "lebanon": [
        {"organisation": "IDAL / Lebanese investment promotion desks",
         "type": "IPA", "role": "Lebanese corporate outbound facilitation"},
        {"organisation": "Association of Lebanese Industrialists / CCI",
         "type": "Industry body", "role": "Industry access"},
    ],
    "iraq": [
        {"organisation": "National Investment Commission (Iraq)",
         "type": "IPA", "role": "Iraqi corporate outbound facilitation"},
        {"organisation": "Iraqi Chambers of Commerce channels",
         "type": "Chamber", "role": "Industry access"},
    ],
    "iran": [
        {"organisation": "Organization for Investment (OIETAI) Iran",
         "type": "IPA", "role": "Iranian corporate outbound facilitation"},
        {"organisation": "Iran Chamber of Commerce channels",
         "type": "Chamber", "role": "Industry access"},
    ],
    "syria": [
        {"organisation": "Syrian investment / economy promotion desks",
         "type": "IPA", "role": "Syrian corporate outbound facilitation"},
        {"organisation": "Syrian Chamber of Commerce channels",
         "type": "Chamber", "role": "Industry access"},
    ],
    "yemen": [
        {"organisation": "Yemeni General Investment Authority desks",
         "type": "IPA", "role": "Yemeni corporate outbound facilitation"},
        {"organisation": "Yemeni Chamber of Commerce channels",
         "type": "Chamber", "role": "Industry access"},
    ],
    "palestine": [
        {"organisation": "PIPA (Palestinian Investment Promotion Agency)",
         "type": "IPA", "role": "Palestinian corporate outbound facilitation"},
        {"organisation": "Palestinian Chambers of Commerce",
         "type": "Chamber", "role": "Industry access"},
    ],
    "afghanistan": [
        {"organisation": "Afghanistan Investment Promotion desks",
         "type": "IPA", "role": "Afghan corporate outbound facilitation"},
        {"organisation": "Afghanistan Chamber of Commerce and Investment",
         "type": "Chamber", "role": "Industry access"},
    ],
    "sri lanka": [
        {"organisation": "Board of Investment of Sri Lanka (BOI)",
         "type": "IPA", "role": "Sri Lankan corporate outbound facilitation"},
        {"organisation": "CCC / Sri Lanka Chamber", "type": "Chamber",
         "role": "Industry access"},
        {"organisation": "Sri Lanka EXIM / development finance",
         "type": "Finance", "role": "Trade and project finance"},
    ],
}


# Markers used to detect wrong-country IPA bleed in generated prose/tables.
_FOREIGN_IPA_MARKERS: dict[str, tuple[str, ...]] = {
    "india": ("invest india", "nasscom", "exim bank of india", "pharmexcil"),
    "germany": ("germany trade", "gtai", "gesalo", "federation of german"),
    "united states": ("selectusa", "amcham saudi", "exim bank of the united"),
    "united kingdom": ("uk export finance", "department for business and trade",
                       "britcham"),
    "china": ("mofcom", "ccpit", "invest in china"),
    "japan": ("jetro", "jbic", "keidanren"),
    "south korea": ("kotra", "kexim", "kocham"),
    "france": ("business france", "medef", "bpifrance"),
    "italy": ("italian trade agency", "confindustria", "sace"),
    "netherlands": ("nfia", "vno-ncw", "atradius"),
    "switzerland": ("switzerland global enterprise", "economiesuisse", "serv"),
    "spain": ("icex", "ceoe", "cesce"),
    "canada": ("invest in canada", "trade commissioner service",
               "export development canada"),
    "australia": ("austrade", "export finance australia"),
    "singapore": ("enterprise singapore", "singapore business federation"),
    "united arab emirates": ("invest uae", "dubai fdi", "abu dhabi investment office"),
    "turkey": ("investment office of the presidency", "deik", "türk eximbank",
               "turk eximbank"),
    "brazil": ("apexbrasil", "apex-brasil", "bndes"),
    "pakistan": ("board of investment (boi) pakistan", "fpcci", "tdap"),
    "mexico": ("invest in mexico", "bancomext", "comce"),
    "indonesia": ("bkpm", "kadin indonesia", "lpei"),
    "sweden": ("business sweden", "confederation of swedish"),
    "nigeria": ("nigerian investment promotion", "nipc", "nexim"),
    "egypt": ("general authority for investment", "gafi"),
    "vietnam": ("foreign investment agency (fia) vietnam", "vcci"),
    "poland": ("polish investment and trade agency", "paih", "kuke"),
    "malaysia": ("mida", "matrade", "exim bank of malaysia"),
    "thailand": ("boi thailand", "exim thailand"),
    "bangladesh": ("bida", "fbcci"),
    "south africa": ("investsa", "busa", "ecic"),
    "russia": ("russian export center", "rspp", "exiar"),
    "belgium": ("flanders investment", "credendo"),
    "austria": ("advantage austria", "wkö", "oekb"),
    "denmark": ("invest in denmark", "ekf denmark"),
    "norway": ("innovation norway", "giek"),
    "finland": ("business finland", "finnvera"),
    "philippines": ("board of investments (boi) philippines",),
    "ghana": ("ghana investment promotion", "gipc"),
    "kenya": ("keninvest", "kenya investment authority"),
    "ethiopia": ("ethiopian investment commission",),
    "czechia": ("czechinvest", "egap"),
    "greece": ("enterprise greece",),
    "portugal": ("aicep",),
    "hungary": ("hipa", "exim hungary"),
    "romania": ("investromania", "eximbank romania"),
    "ukraine": ("ukraineinvest",),
    "morocco": ("amdie", "cgem"),
    "algeria": ("aapi",),
    "tunisia": ("fipa tunisia", "utica", "cepex"),
    "libya": ("libyan investment",),
    "sudan": ("sudanese investment",),
    "qatar": ("invest qatar", "qatar chamber"),
    "kuwait": ("kdipa",),
    "bahrain": ("bahrain edb", "invest in bahrain"),
    "oman": ("invest oman",),
    "jordan": ("jordan investment commission",),
    "lebanon": ("idal",),
    "iraq": ("national investment commission (iraq)",),
    "iran": ("oietai",),
    "syria": ("syrian investment",),
    "yemen": ("yemeni general investment",),
    "palestine": ("pipa",),
    "afghanistan": ("afghanistan investment",),
    "sri lanka": ("board of investment of sri lanka",),
}


def _trade_body_country_key(country: str) -> str:
    """Map free-text origin labels onto catalog keys."""
    c = (country or "").casefold().strip()
    if not c:
        return ""
    aliases = (
        ("united states", ("united states", "usa", "u.s.", "u.s.a", "america")),
        ("united kingdom", ("united kingdom", "uk", "u.k.", "britain",
                            "great britain", "england")),
        ("united arab emirates", ("united arab emirates", "uae", "u.a.e",
                                  "emirates")),
        # Bare "korea" → South Korea (DPRK is never an FDI source ask here).
        ("south korea", ("south korea", "korea, republic", "republic of korea",
                         "r.o.k", "korea")),
        ("germany", ("germany", "deutschland")),
        ("india", ("india",)),
        ("china", ("china", "prc", "people's republic of china")),
        ("japan", ("japan",)),
        ("france", ("france",)),
        ("italy", ("italy",)),
        ("netherlands", ("netherlands", "holland")),
        ("switzerland", ("switzerland",)),
        ("spain", ("spain",)),
        ("canada", ("canada",)),
        ("australia", ("australia",)),
        ("singapore", ("singapore",)),
        ("turkey", ("turkey", "türkiye", "turkiye")),
        ("brazil", ("brazil", "brasil")),
        ("pakistan", ("pakistan",)),
        ("mexico", ("mexico",)),
        ("indonesia", ("indonesia",)),
        ("sweden", ("sweden",)),
        ("nigeria", ("nigeria",)),
        ("egypt", ("egypt",)),
        ("vietnam", ("vietnam",)),
        ("poland", ("poland",)),
        ("malaysia", ("malaysia",)),
        ("thailand", ("thailand",)),
        ("bangladesh", ("bangladesh",)),
        ("south africa", ("south africa",)),
        ("russia", ("russia", "russian federation")),
        ("belgium", ("belgium",)),
        ("austria", ("austria",)),
        ("denmark", ("denmark",)),
        ("norway", ("norway",)),
        ("finland", ("finland",)),
        ("philippines", ("philippines",)),
        ("ghana", ("ghana",)),
        ("kenya", ("kenya",)),
        ("ethiopia", ("ethiopia",)),
        ("czechia", ("czechia", "czech republic", "czech")),
        ("greece", ("greece",)),
        ("portugal", ("portugal",)),
        ("hungary", ("hungary",)),
        ("romania", ("romania",)),
        ("ukraine", ("ukraine",)),
        ("morocco", ("morocco",)),
        ("algeria", ("algeria",)),
        ("tunisia", ("tunisia",)),
        ("libya", ("libya",)),
        ("sudan", ("sudan",)),
        ("qatar", ("qatar",)),
        ("kuwait", ("kuwait",)),
        ("bahrain", ("bahrain",)),
        ("oman", ("oman",)),
        ("jordan", ("jordan",)),
        ("lebanon", ("lebanon",)),
        ("iraq", ("iraq",)),
        ("iran", ("iran",)),
        ("syria", ("syria",)),
        ("yemen", ("yemen",)),
        ("palestine", ("palestine",)),
        ("afghanistan", ("afghanistan",)),
        ("sri lanka", ("sri lanka",)),
    )
    for key, needles in aliases:
        if any(n == c or c.startswith(n + " ") or c.endswith(" " + n) or n in c
               for n in needles):
            # Avoid "india" matching inside unrelated tokens via crude `in`
            # for short keys — require boundary-ish match for len<=5.
            matched = False
            for n in needles:
                if len(n) <= 5:
                    if c == n or c.startswith(n + " ") or c.endswith(" " + n):
                        matched = True
                        break
                elif n in c:
                    matched = True
                    break
            if matched:
                return key
    if c in _TRADE_BODY_CATALOG:
        return c
    return ""


def primary_trade_body_name(country: str) -> str:
    """Canonical IPA / lead organisation for an origin market."""
    bodies = _default_trade_bodies(country)
    if bodies:
        return str(bodies[0].get("organisation") or "")
    return ""


def foreign_ipa_markers_for_scrub(country: str) -> list[str]:
    """Markers belonging to OTHER countries — used to scrub bleed."""
    own = _trade_body_country_key(country)
    out: list[str] = []
    for key, markers in _FOREIGN_IPA_MARKERS.items():
        if key == own:
            continue
        out.extend(markers)
    # Generic placeholder row from older thin drafts (not country fallbacks).
    out.append("requires validation for this origin")
    return out


def _pad_sector_trade_bodies(country: str, bodies: list[dict]) -> list[dict]:
    """Ensure every origin has IPA + chamber depth AND sector associations.

    Jul21 Azure often named tech / health / industrial bodies. Catalog
    entries with only IPA+chamber get sector pads so outreach desks are
    never left with a 2-row table.
    """
    out = [dict(b) for b in (bodies or [])]
    if len(out) >= 5:
        return out
    label = (country or "the source market").strip() or "the source market"
    pads = [
        {
            "organisation": f"{label} technology / digital industry association",
            "type": "Sector body",
            "role": "IT / digital / software targeting into SDAIA–LEAP corridors",
        },
        {
            "organisation": f"{label} healthcare / life-sciences industry body",
            "type": "Sector body",
            "role": "Health localisation targeting into NUPCO corridors",
        },
        {
            "organisation": f"{label} industrial / manufacturing federation",
            "type": "Sector body",
            "role": "Industrial capability targeting into NIDLP / PIF zones",
        },
        {
            "organisation": f"{label} energy / utilities industry association",
            "type": "Sector body",
            "role": "Energy-transition targeting into NEOM / water corridors",
        },
    ]
    seen = {(b.get("organisation") or "").casefold() for b in out}
    for pad in pads:
        if len(out) >= 5:
            break
        org = pad["organisation"]
        if org.casefold() in seen:
            continue
        out.append(dict(pad))
        seen.add(org.casefold())
    return out


def _default_trade_bodies(country: str) -> list[dict]:
    key = _trade_body_country_key(country)
    if key and key in _TRADE_BODY_CATALOG:
        return _pad_sector_trade_bodies(
            country, [dict(b) for b in _TRADE_BODY_CATALOG[key]],
        )
    label = (country or "the source market").strip() or "the source market"
    return _pad_sector_trade_bodies(
        country,
        [
            {"organisation": f"{label} national investment promotion agency",
             "type": "IPA",
             "role": f"Pipeline and soft-landing for {label} corporates "
                     f"(confirm exact IPA name with desk)"},
            {"organisation": f"{label} chamber of commerce / industry federation",
             "type": "Industry body",
             "role": "CEO-level access and mission facilitation"},
            {"organisation": f"Bilateral {label}–Saudi business council / chamber",
             "type": "Bilateral chamber",
             "role": "On-ground networking and account introductions"},
            {"organisation": f"{label} export credit / development finance agency",
             "type": "Finance",
             "role": "Trade and project finance introductions"},
        ],
    )


def merge_thesis_enrichment(payload: dict, enrichment: dict | None) -> dict:
    """Overlay compact LLM theses / new-entry rows onto a DB-seeded payload."""
    if not enrichment or not isinstance(enrichment, dict):
        return payload
    out = dict(payload)
    # Preserve truncation markers from the seed path
    if payload.get("_truncated"):
        out["_truncated"] = True
        out["_truncation_reason"] = payload.get("_truncation_reason")
    exec_sum = enrichment.get("executive_summary") or {}
    if isinstance(exec_sum, dict):
        findings = [
            _str(x) for x in _as_list(exec_sum.get("key_findings")) if _str(x)
        ]
        if findings:
            out["executive_summary"] = {
                "key_findings": findings[:6],
                "top_recommendation": _str(
                    exec_sum.get("top_recommendation")
                    or (payload.get("executive_summary") or {}).get(
                        "top_recommendation")
                ),
            }
    theses = enrichment.get("theses") or {}
    if isinstance(theses, dict):
        by_name = {k.casefold(): v for k, v in theses.items() if isinstance(v, dict)}
        merged_targets = []
        for t in out.get("targets") or []:
            nt = dict(t)
            tip = by_name.get(nt["company"].casefold())
            if tip:
                for k in (
                    "why_company", "why_saudi", "why_now",
                    "proposed_investment", "misa_action", "evidence_strength",
                ):
                    if _str(tip.get(k)):
                        nt[k] = _str(tip.get(k))
                fits = [
                    _str(x) for x in _as_list(tip.get("saudi_strategic_fit"))
                    if _str(x)
                ]
                if fits:
                    nt["saudi_strategic_fit"] = fits
                vals = [
                    _str(x) for x in _as_list(tip.get("validation_required"))
                    if _str(x)
                ]
                if vals:
                    nt["validation_required"] = vals
            merged_targets.append(nt)
        out["targets"] = merged_targets

    # Append up to 3 new-entry targets after expansion rows.
    new_entries = enrichment.get("new_entry_targets") or []
    known = {t["company"].casefold() for t in out.get("targets") or []}
    added = 0
    for ne in new_entries:
        if not isinstance(ne, dict) or added >= 3:
            break
        name = _str(ne.get("company"))
        if not name or name.casefold() in known:
            continue
        known.add(name.casefold())
        out.setdefault("targets", []).append({
            "rank": 0,
            "company": name,
            "sector": _str(ne.get("sector"), "Unclassified"),
            "current_saudi_presence": "Not in MISA footprint",
            "target_type": "new_entry",
            "proposed_investment": _str(
                ne.get("proposed_investment"),
                "Requires validation — proposed investment not specified",
            ),
            "why_company": _str(ne.get("why_company"), "Requires validation"),
            "why_saudi": _str(ne.get("why_saudi"), "Requires validation"),
            "why_now": _str(ne.get("why_now"), "Requires validation"),
            "saudi_strategic_fit": [
                _str(x) for x in _as_list(ne.get("saudi_strategic_fit"))
                if _str(x)
            ],
            "misa_action": _str(ne.get("misa_action"), "Requires validation"),
            "evidence": [],
            "evidence_strength": "low",
            "validation_required": [
                _str(x) for x in _as_list(ne.get("validation_required"))
                if _str(x)
            ] or ["Saudi presence and demand thesis"],
        })
        added += 1
    for i, t in enumerate(out.get("targets") or [], 1):
        t["rank"] = i

    bodies = enrichment.get("trade_bodies")
    # Always keep canonical defaults (Invest India etc.) and merge LLM
    # extras by organisation name — never let a thin enrichment wipe them.
    country = ""
    try:
        country = str(
            (out.get("current_footprint") or {}).get("origin_country")
            or out.get("origin_country")
            or ""
        )
    except Exception:
        country = ""
    defaults = list(out.get("trade_bodies") or _default_trade_bodies(country))
    if isinstance(bodies, list) and bodies:
        by_org: dict[str, dict] = {}
        for b in defaults + [
            x if isinstance(x, dict) else {"organisation": _str(x)}
            for x in bodies
        ]:
            if not isinstance(b, dict):
                continue
            org = _str(b.get("organisation") or b.get("name"))
            if not org:
                continue
            key = org.casefold()
            prev = by_org.get(key) or {}
            by_org[key] = {
                "organisation": org,
                "type": _str(b.get("type") or prev.get("type")),
                "role": _str(b.get("role") or prev.get("role")),
            }
        # Hard-require the canonical primary IPA for THIS origin.
        primary = primary_trade_body_name(country)
        if primary and primary.casefold() not in by_org:
            defaults_by = {
                str(b.get("organisation") or "").casefold(): b
                for b in _default_trade_bodies(country)
            }
            if primary.casefold() in defaults_by:
                by_org[primary.casefold()] = defaults_by[primary.casefold()]
        # Drop foreign-country IPA rows that leaked from the LLM.
        foreign = foreign_ipa_markers_for_scrub(country)
        cleaned: dict[str, dict] = {}
        for k, b in by_org.items():
            blob = " ".join(
                str(b.get(x) or "") for x in ("organisation", "type", "role")
            ).casefold()
            if any(m in blob for m in foreign if len(m) >= 5):
                continue
            cleaned[k] = b
        # Prefer defaults first so catalog order wins when LLM is noisy.
        ordered: list[dict] = []
        seen: set[str] = set()
        for b in _default_trade_bodies(country) + list(cleaned.values()):
            org = str(b.get("organisation") or "")
            key = org.casefold()
            if not org or key in seen:
                continue
            seen.add(key)
            ordered.append(b)
        out["trade_bodies"] = ordered[:12]
    elif defaults:
        out["trade_bodies"] = defaults[:12]

    # Recommendations: keep company-named actions from seed; only adopt
    # enrichment bullets that name a target company.
    seed_recs = [
        _str(r) for r in (payload.get("recommendations") or []) if _str(r)
    ]
    enrich_recs = [
        _str(x) if not isinstance(x, dict)
        else _str(x.get("action") or x.get("text"))
        for x in _as_list(enrichment.get("recommendations"))
    ]
    enrich_recs = [r for r in enrich_recs if r]
    target_names = [
        (t.get("company") or "") for t in (out.get("targets") or [])
        if t.get("company")
    ]

    def _names_company(text: str) -> bool:
        tl = text.casefold()
        return any(n[:10].casefold() in tl for n in target_names if len(n) >= 4)

    merged_recs = []
    for r in seed_recs + [r for r in enrich_recs if _names_company(r)]:
        if r and r not in merged_recs:
            merged_recs.append(r)
    if not merged_recs:
        merged_recs = [
            f"{t['company']}: {t['misa_action']}"
            for t in (out.get("targets") or [])[:8]
            if t.get("misa_action")
        ]
    out["recommendations"] = merged_recs[:12]

    strat = _str(enrichment.get("strategic_context"))
    if strat:
        out["strategic_context"] = strat

    lims = [_str(x) for x in _as_list(enrichment.get("data_limitations")) if _str(x)]
    if lims:
        out["data_limitations"] = lims
    return out


def ranking_table_is_truncated(answer: str) -> bool:
    """True when an existing Priority Company Ranking table is cut mid-row.

    Delegates to ``advisory_safety.ranking_midrow_truncated`` — the single
    contract for this failure class. Absence of the ranking section, or
    absence of ``## Detailed…``, is NOT truncation.
    """
    from app.services.advisory_safety import ranking_midrow_truncated
    return ranking_midrow_truncated(answer)

