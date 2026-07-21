"""
Deterministic local narration from SQL result rows (no LLM).
Written for clarity, executive tone, and disciplined use of source fields only.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Localisation helpers
# ---------------------------------------------------------------------------

def _loc(locale: str | None) -> str:
    return "ar" if (locale or "").lower().startswith("ar") else "en"


def _tx(locale: str | None, key: str, **fmt) -> str:
    L = _loc(locale)
    bundles = {
        "sql_bad": {
            "en": (
                "The database query could not be constrained to the company you named. "
                "Try a shorter distinctive token in `company_name` or rephrase."
            ),
            "ar": (
                "تعذر ضبط استعلام قاعدة البيانات على الشركة التي ذكرتَها. "
                "جرّب مقطعاً أقصر وأكثر تميزاً في `company_name` أو أعد صياغة السؤال."
            ),
        },
        "no_rows_entity": {
            "en": (
                'No companies matching "{ec}" were found in `company_profiles`. '
                "I did not substitute unrelated rows."
            ),
            "ar": (
                'لم يُعثر على شركات مطابقة لـ «{ec}» في `company_profiles`. '
                "لم أستبدل بصفوف غير ذات صلة."
            ),
        },
        "no_rows_generic": {
            "en": "No matching companies were returned from `company_profiles` for this question.",
            "ar": "لم تُرجع `company_profiles` أي شركات مطابقة لهذا السؤال.",
        },
        "entity_mismatch": {
            "en": (
                'I couldn\'t find **"{ec}"** in `company_profiles` among the rows that matched the query. '
                "The closest matches by name on file are: {c}. "
                "Would you like details on any of these?"
            ),
            "ar": (
                'لم أتمكن من العثور على **«{ec}»** في `company_profiles` ضمن الصفوف المطابقة للاستعلام. '
                "أقرب الأسماء في الملف هي: {c}. "
                "هل تريد تفاصيل عن أي منها؟"
            ),
        },
        "sparse_row": {
            "en": "The row is sparse in the fields we summarise here—see the trace for detail.",
            "ar": "الصف ناقص في الحقول التي نلخّصها هنا—راجع التتبع للتفاصيل.",
        },
        "one_row_sparse": {
            "en": "One row was returned, but it has no populated narrative fields.",
            "ar": "أُرجع صف واحد، لكنه لا يحتوي حقولاً نصية مفعّلة للسرد.",
        },
        "this_company": {"en": "This company", "ar": "هذه الشركة"},
        "record": {"en": "record", "ar": "سجل"},
        "code": {"en": "code", "ar": "رمز"},
        "founded": {"en": "founded", "ar": "تأسست"},
        "regional_hdr": {
            "en": "**Regional picture (sourced from the record)**",
            "ar": "**الصورة الإقليمية (من السجل)**",
        },
        "scale_hdr": {"en": "**Scale**", "ar": "**الحجم**"},
        "notes_hdr": {"en": "**Internal notes**", "ar": "**ملاحظات داخلية**"},
        "team": {"en": "Team", "ar": "الفريق"},
        "company_lbl": {"en": "Company", "ar": "الشركة"},
        "status_lbl": {"en": "Status", "ar": "الحالة"},
        "reported_rev": {"en": "Reported revenue", "ar": "الإيرادات المذكورة"},
        "headcount": {"en": "global headcount", "ar": "عدد الموظفين عالمياً"},
        "multi_lead": {
            "en": (
                "Your search matched **{n}** companies{qpart}. "
                "Each numbered line is a short summary in plain English. "
                "Open **Retrieval trace** for the full row exactly as returned from Postgres.\n\n"
                "_Privacy:_ retrieved row data is **not** sent to OpenAI—only your questions "
                "(and fixed schema hints in the system prompt) are used to build SQL."
            ),
            "ar": (
                "طابق بحثك **{n}** شركة{qpart}. "
                "كل سطر مرقم ملخص قصير. "
                "افتح **تتبع الاسترجاع** لرؤية الصف كما عاد من Postgres.\n\n"
                "_الخصوصية:_ بيانات الصفوف المسترجعة **لا** تُرسل إلى OpenAI—يُرسل أسئلتك فقط "
                "(وملاحظات المخطط الثابتة في موجه النظام) لبناء SQL."
            ),
        },
        "multi_trunc": {
            "en": "_Showing {cap} of {n} matches; the rest appear only in the retrieval trace._",
            "ar": "_عرض {cap} من أصل {n} مطابقة؛ الباقي يظهر فقط في تتبع الاسترجاع._",
        },
        "co_default": {"en": "Company {i}", "ar": "شركة {i}"},
        "generic_no_rows": {
            "en": "No rows returned from `{table}`.",
            "ar": "لم تُرجع `{table}` أي صفوف.",
        },
        "none_names": {"en": "_none available_", "ar": "_لا توجد أسماء مقترحة_"},
        "fb_one_row": {"en": "**One row** from `{table}`. {body}", "ar": "**صف واحد** من `{table}`. {body}"},
        "fb_no_pop": {"en": "_No populated fields in this row._", "ar": "_لا توجد حقول مملوءة في هذا الصف._"},
        "fb_multi": {
            "en": "**{n} rows** from `{table}` (first **{cap}** in compact form):\n\n{grid}",
            "ar": "**{n} صفاً** من `{table}` (أول **{cap}** بصيغة مدمجة):\n\n{grid}",
        },
        "fb_omit": {
            "en": "_{n} additional rows omitted._",
            "ar": "_تم حذف {n} صف إضافي._",
        },
    }
    b = bundles[key]
    s = b.get(L) or b["en"]
    return s.format(**fmt) if fmt else s


# Columns suppressed from prose (paths, flags, audit noise).
_KEY_SKIP_SUBSTR = (
    "screenshot", "_id", "factiva", "capital_iq", "linkedin", "sec_filings",
    "annual_reports", "company_website", "uae_gcc", "logo", "ner",
    "confidence_level", "creation_date", "update_date", "review_date",
    "created_by", "reviewed_by", "updated_by",
    "reviewer_comments", "misa_comments", "misa_review_status", "review_status",
)


def _is_nonempty(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def _text(val: Any) -> str | None:
    if not _is_nonempty(val):
        return None
    if isinstance(val, (dict, list)):
        return None
    s = str(val).strip()
    return s if s else None


def _skip_key(key: str) -> bool:
    kl = key.lower()
    return any(s in kl for s in _KEY_SKIP_SUBSTR)


def format_revenue_usd(val: Any) -> str | None:
    if not _is_nonempty(val):
        return None
    try:
        n = float(val) if not isinstance(val, Decimal) else float(val)
    except (TypeError, ValueError):
        return None
    an = abs(n)
    if an >= 1e9:
        return f"USD {n / 1e9:.2f} billion"
    if an >= 1e6:
        return f"USD {n / 1e6:.1f} million"
    return f"USD {n:,.0f}"


def format_int(val: Any) -> str | None:
    if not _is_nonempty(val):
        return None
    try:
        i = int(float(val))
    except (TypeError, ValueError):
        return None
    return f"{i:,}"


def _display_places(val: Any) -> str | None:
    """Hide placeholder zeros (e.g. rhq_city '0') from prose."""
    t = _text(val)
    if not t:
        return None
    tl = t.lower().strip()
    if tl in ("0", "0.0", "00", "-", "—", "none", "n/a", "na", "null", "unknown"):
        return None
    try:
        if float(t) == 0:
            return None
    except (TypeError, ValueError):
        pass
    return t


def _prose_clauses_for_row(row: dict) -> list[str]:
    c: list[str] = []
    sec = _text(row.get("sector"))
    if sec:
        c.append(f"**{sec}** sector")
    ghq = _text(row.get("global_headquarters"))
    if ghq:
        c.append(f"global headquarters **{ghq}**")
    rcity = _display_places(row.get("rhq_city"))
    rctry = _display_places(row.get("rhq_country"))
    if rcity and rctry:
        c.append(f"regional HQ in **{rcity}**, **{rctry}**")
    elif rcity:
        c.append(f"regional HQ city **{rcity}**")
    elif rctry:
        c.append(f"regional HQ country **{rctry}**")
    rent = _text(row.get("rhq_entity_name"))
    if rent and rent not in {ghq or "", rcity or "", rctry or ""}:
        c.append(f"RHQ entity **{rent}**")
    rev = format_revenue_usd(row.get("revenue_usd"))
    if rev:
        c.append(f"revenue **{rev}**")
    ne = format_int(row.get("number_of_employees"))
    if ne:
        c.append(f"about **{ne}** employees globally (per file)")
    return c


def _ref_tags(row: dict, *, locale: str | None = None) -> str:
    bits = []
    ic = _text(row.get("internal_code"))
    rid = _text(row.get("id"))
    if ic:
        bits.append(f"internal code **{ic}**" if _loc(locale) == "en" else f"الرمز الداخلي **{ic}**")
    if rid:
        bits.append(f"record id **{rid}**" if _loc(locale) == "en" else f"معرّف السجل **{rid}**")
    return (" (" + ", ".join(bits) + ")") if bits else ""


def _format_local_revenue(val: Any, currency: str | None) -> str | None:
    if not _is_nonempty(val):
        return None
    try:
        n = float(val)
    except (TypeError, ValueError):
        return None
    cur = (currency or "").strip().upper()
    amt = f"{n:,.0f}"
    return f"{amt} {cur}" if cur else amt


def _first_sentences(text: str, *, max_sentences: int = 3, max_chars: int = 900) -> str:
    text = text.strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunk: list[str] = []
    for p in parts:
        if not p:
            continue
        chunk.append(p.strip())
        if len(chunk) >= max_sentences:
            break
    out = " ".join(chunk).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return out


def _bool_phrase(label: str, val: Any) -> str | None:
    if isinstance(val, bool) and val:
        mapping = {
            "rhq_status": "holds regional headquarters (RHQ) status in the record",
            "rhq_license_status": "shows an active RHQ licence flag in the record",
            "rhq_in_mena": "is flagged as operating an RHQ in MENA",
            "presence_in_saudi": "is recorded as present in Saudi Arabia",
            "presence_of_company_in_mena": "is recorded as present in MENA",
            "presence_of_parent_company_in_mena": "has the parent recorded as present in MENA",
        }
        return mapping.get(label)
    return None


def _compose_regional_narrative(row: dict) -> list[str]:
    sentences: list[str] = []

    hist = _text(row.get("history_in_mena"))
    if hist:
        sentences.append(_first_sentences(hist, max_sentences=2, max_chars=520))

    notes = _text(row.get("mena_notes"))
    if notes and notes != hist:
        sentences.append(_first_sentences(notes, max_sentences=2, max_chars=520))

    city = _text(row.get("rhq_city"))
    country = _text(row.get("rhq_country"))
    entity = _text(row.get("rhq_entity_name"))
    coverage = _text(row.get("rhq_country_coverage"))
    if city or country:
        loc = ", ".join(x for x in [city, country] if x)
        ent = f" under **{entity}**" if entity else ""
        cov = f", covering **{coverage}**" if coverage else ""
        sentences.append(f"The record places regional headquarters in **{loc}**{ent}{cov}.")

    tp = _text(row.get("type_of_presence_saudi"))
    if tp:
        sentences.append(f"In Saudi Arabia the recorded presence type is **{tp}**.")

    ml = _text(row.get("mena_locations"))
    if ml and len(ml) < 400:
        sentences.append(f"Recorded MENA locations include: {ml}.")

    c_ksa = _text(row.get("companies_name_in_ksa"))
    c_mena = _text(row.get("companies_name_in_mena"))
    if c_ksa and c_mena and c_ksa == c_mena:
        sentences.append(f"Associated operating names in KSA and MENA: **{c_ksa}**.")
    else:
        if c_ksa:
            sentences.append(f"Associated name in KSA: **{c_ksa}**.")
        if c_mena and c_mena != c_ksa:
            sentences.append(f"Associated name in MENA: **{c_mena}**.")

    mrev = _format_local_revenue(row.get("mena_revenue_local_currency"), row.get("currency"))
    if mrev:
        sentences.append(f"MENA revenue in local currency is recorded at **{mrev}**.")

    n_mena = format_int(row.get("number_of_employees_mena"))
    n_ksa = format_int(row.get("number_of_employees_ksa"))
    n_rhq = format_int(row.get("rhq_number_of_employees"))
    head_bits = []
    if n_rhq:
        head_bits.append(f"**{n_rhq}** at the RHQ entity")
    if n_mena:
        head_bits.append(f"**{n_mena}** across MENA")
    if n_ksa:
        head_bits.append(f"**{n_ksa}** in Saudi Arabia")
    if head_bits:
        sentences.append("Headcount in the file: " + ", ".join(head_bits) + ".")

    for bl in ("rhq_mandatory_activities", "rhq_optional_activities"):
        t = _text(row.get(bl))
        if t:
            lab = "Mandatory RHQ activities" if "mandatory" in bl else "Optional RHQ activities"
            sentences.append(f"{lab}: {_first_sentences(t, max_sentences=1, max_chars=280)}")

    bool_bits = []
    for k in sorted(row.keys()):
        if _skip_key(k):
            continue
        ph = _bool_phrase(k, row.get(k))
        if ph:
            bool_bits.append(ph)
    if bool_bits:
        sentences.append(
            "Flags on file: "
            + "; ".join(f"the company {b}" for b in bool_bits[:4])
            + "."
        )

    # Deduplicate near-identical sentences
    out: list[str] = []
    seen_lo: set[str] = set()
    for s in sentences:
        s = s.strip()
        if len(s) < 8:
            continue
        lo = s.lower()[:120]
        if lo in seen_lo:
            continue
        seen_lo.add(lo)
        out.append(s)
    return out


def _compose_identity_fallback(row: dict) -> str:
    bits = [
        _text(row.get(f))
        for f in ("sector", "legal_structure", "type_of_entity", "control_structure",
                  "ultimate_parent_company", "global_headquarters")
        if _text(row.get(f))
    ]
    return "; ".join(bits)  # type: ignore[arg-type]


def _paragraph_rhq_single(row: dict, *, locale: str | None = None) -> list[str]:
    sections: list[str] = []
    name = _text(row.get("company_name")) or _tx(locale, "this_company")
    id_t = _text(row.get("id"))
    code = _text(row.get("internal_code"))
    yf = _text(row.get("year_founded"))

    meta: list[str] = []
    if id_t:
        meta.append(f"{_tx(locale, 'record')} **{id_t}**")
    if code:
        meta.append(f"{_tx(locale, 'code')} **{code}**")
    meta_s = f" ({', '.join(meta)})" if meta else ""

    founded = ""
    if yf:
        sep = "، " if _loc(locale) == "ar" else ", "
        founded = f"{sep}{_tx(locale, 'founded')} **{yf}**"
    sections.append(f"**{name}**{meta_s}{founded}.")

    profile = _text(row.get("company_profile"))
    if profile:
        body = _first_sentences(profile, max_sentences=4, max_chars=1100)
        if body:
            sections.append(body)
    else:
        fb = _compose_identity_fallback(row)
        if fb:
            sections.append(fb + ".")

    regional = _compose_regional_narrative(row)
    if regional:
        sections.append(_tx(locale, "regional_hdr") + "\n\n" + "\n\n".join(regional))

    rev = format_revenue_usd(row.get("revenue_usd"))
    neg = format_int(row.get("number_of_employees"))
    scale_parts: list[str] = []
    if rev:
        scale_parts.append(f"{_tx(locale, 'reported_rev')} **{rev}**")
    if neg:
        scale_parts.append(f"{_tx(locale, 'headcount')} **{neg}**")
    if scale_parts:
        sep = "؛ " if _loc(locale) == "ar" else "; "
        sections.append(_tx(locale, "scale_hdr") + "\n\n" + sep.join(scale_parts) + ".")

    notes_out: list[str] = []
    for fld, title_key in (("team_comments", "team"), ("company_notes", "company_lbl"), ("status", "status_lbl")):
        t = _text(row.get(fld))
        if t:
            title = _tx(locale, title_key)
            notes_out.append(f"**{title}:** {_first_sentences(t, max_sentences=2, max_chars=400)}")
    if notes_out:
        sections.append(_tx(locale, "notes_hdr") + "\n\n" + "\n\n".join(notes_out))

    return [s for s in sections if s.strip()]


def _paragraph_rhq_multi(rows: list[dict], user_question: str, *, locale: str | None = None) -> str:
    n = len(rows)
    cap = min(n, 10)
    display = rows[:cap]
    qshort = (user_question or "").strip()
    if len(qshort) > 80:
        qshort = qshort[:77] + "…"
    qpart = (
        f" (من: «{qshort}»)" if _loc(locale) == "ar" else f' (from: "{qshort}")'
    ) if qshort else ""

    lead = _tx(locale, "multi_lead", n=n, qpart=qpart)
    lines = [lead, ""]
    for i, r in enumerate(display, start=1):
        nm = _text(r.get("company_name")) or _tx(locale, "co_default", i=i)
        ref = _ref_tags(r, locale=locale)
        clauses = _prose_clauses_for_row(r)
        tail = "; ".join(clauses) + "." if clauses else _tx(locale, "sparse_row")
        lines.append(f"{i}. **{nm}**{ref}. {tail}")

    if n > cap:
        lines.append("\n" + _tx(locale, "multi_trunc", cap=cap, n=n))
    return "\n\n".join(lines)


def _fallback_generic(rows: list[dict], table: str, *, locale: str | None = None) -> str:
    if not rows:
        return _tx(locale, "generic_no_rows", table=table)
    n = len(rows)
    if n == 1:
        r = rows[0]
        pairs = [
            f"**{k}:** {_text(v)}"
            for k, v in sorted(r.items())
            if _is_nonempty(v) and not _skip_key(k)
        ]
        body = "; ".join(pairs[:35]) if pairs else _tx(locale, "fb_no_pop")
        return _tx(locale, "fb_one_row", table=table, body=body)
    cap = min(n, 10)
    keys = [k for k in rows[0].keys() if not _skip_key(k)]
    scores = sorted(
        [(sum(1 for r in rows[:cap] if _is_nonempty(r.get(k))), k) for k in keys],
        reverse=True,
    )
    top_k = [k for _, k in scores[: min(8, len(scores))]]
    header = "| " + " | ".join(top_k) + " |"
    sep = "| " + " | ".join(["---"] * len(top_k)) + " |"
    body_lines = [header, sep]
    for r in rows[:cap]:
        cells = [_text(r.get(k)) or "" for k in top_k]
        body_lines.append("| " + " | ".join(cells) + " |")
    grid = "\n".join(body_lines)
    msg = _tx(locale, "fb_multi", n=n, table=table, cap=cap, grid=grid)
    if n > cap:
        msg += "\n\n" + _tx(locale, "fb_omit", n=n - cap)
    return msg


def generate_commentary(
    rows: list[dict],
    table: str,
    user_question: str,
    *,
    entity_candidate: str | None = None,
    entity_lookup_required: bool = False,
    sql_entity_check_passed: bool = True,
    row_entity_sanity_passed: bool = True,
    closest_names: list[str] | None = None,
    locale: str = "en",
) -> str:
    """
    Build markdown narration from retrieved rows. Never invents values;
    skips empty fields; never emits 'N/A', 'null', or 'unknown'.
    """
    rows = list(rows or [])
    q = user_question or ""
    ec = (entity_candidate or "").strip() or None

    if table in ("company_profiles", "rhq_company"):
        if not sql_entity_check_passed:
            return _tx(locale, "sql_bad")
        if not rows:
            if entity_lookup_required and ec:
                return _tx(locale, "no_rows_entity", ec=ec)
            return _tx(locale, "no_rows_generic")
        if entity_lookup_required and ec and not row_entity_sanity_passed:
            cns = [x for x in (closest_names or []) if x]
            c = ", ".join(f"**{x}**" for x in cns[:3]) or _tx(locale, "none_names")
            return _tx(locale, "entity_mismatch", ec=ec, c=c)
        if len(rows) == 1:
            paras = _paragraph_rhq_single(rows[0], locale=locale)
            return "\n\n".join(paras) if paras else _tx(locale, "one_row_sparse")
        return _paragraph_rhq_multi(rows, q, locale=locale)

    return _fallback_generic(rows, table, locale=locale)
