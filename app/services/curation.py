"""
Privacy-preserving OpenAI curation of chat answers.

Two paths, both degrade to None on any failure so the caller can fall back to
deterministic local commentary:

  * curate_company_insights — turn retrieved DB rows into an insight-rich answer.
    Rows are redacted (internal/audit fields stripped) and truncated before they
    leave the server, and requests are sent with store=False by default.
  * general_knowledge_answer — when the DB has nothing, answer from model
    knowledge with a mandatory "not from the MISA database" disclaimer.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from openai import OpenAI

from app.config import (
    CHAT_CURATION_MAX_ROWS,
    CHAT_OPENAI_STORE,
    openai_advisory_max_tokens_kw,
    openai_determinism_kw,
    openai_max_completion_tokens_kw,
)
from app.db_introspect import is_denied_column_name
from app.prompts.chat_system import (
    advisory_system_prompt,
    curation_system_prompt,
    fallback_system_prompt,
)

# Field-name substrings that must NEVER be sent to OpenAI (internal notes,
# reviewer/audit metadata, external source paths). Mirrors the suppression list
# used by deterministic commentary, plus internal free-text comment fields.
# Credential-style fields (password/token/secret/hash/etc.) are filtered
# separately via `is_denied_column_name` for defense in depth.
_SENSITIVE_KEY_SUBSTR = (
    "screenshot", "factiva", "capital_iq", "linkedin", "sec_filings",
    "annual_reports", "logo", "ner", "confidence_level",
    "creation_date", "update_date", "review_date",
    "review_status", "misa_review_status",
    "created_by", "reviewed_by", "updated_by",
    "reviewer_comments", "misa_comments",
    "team_comments", "company_notes",
)

_MAX_TEXT_CHARS = 1200


def _is_sensitive_key(key: str) -> bool:
    kl = (key or "").lower()
    if is_denied_column_name(kl):
        return True
    return any(s in kl for s in _SENSITIVE_KEY_SUBSTR)


def _safe_value(v: Any) -> Any:
    """Recursively clean a value: strip sensitive keys, drop empties,
    truncate long strings. Used by _safe_row for the top level AND for
    nested dicts/lists (e.g. enrichment under `_related`)."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, str):
        s = v.strip()
        return s[:_MAX_TEXT_CHARS] if s else None
    if isinstance(v, dict):
        out: dict[str, Any] = {}
        for k, vv in v.items():
            if _is_sensitive_key(k):
                continue
            cleaned = _safe_value(vv)
            if cleaned is None:
                continue
            out[k] = cleaned
        return out or None
    if isinstance(v, list):
        cleaned_list = [_safe_value(x) for x in v]
        cleaned_list = [x for x in cleaned_list if x is not None]
        return cleaned_list or None
    return v


def _safe_row(row: dict) -> dict:
    """Drop sensitive fields and empty values; truncate long text.
    Recurses into nested dicts/lists (e.g. `_related` enrichment)."""
    out: dict[str, Any] = {}
    for k, v in (row or {}).items():
        if _is_sensitive_key(k):
            continue
        cleaned = _safe_value(v)
        if cleaned is None:
            continue
        out[k] = cleaned
    return out


def _annotate_country_fks(row: dict) -> dict:
    """Resolve integer country FKs to names INSIDE the row the model
    sees. Without this, rows keyed only by `country_profile_id` are
    country-anonymous, and the curator attributes them to whatever
    country the question mentions (e.g. Saudi Arabia's aggregate
    fdi_data series presented as South Korea's FDI). Best-effort —
    lookup failures leave the row unchanged."""
    for col in ("country_id", "country_profile_id"):
        val = row.get(col)
        if val is None or isinstance(val, bool):
            continue
        try:
            from app.services.country_resolver import country_name_for_fk
            name = country_name_for_fk(col, int(val))
        except Exception:
            name = None
        if name:
            row[f"{col}_resolved_name"] = name
    return row


def safe_rows_for_curation(rows: list[dict]) -> list[dict]:
    """Public helper (also handy for tests): redact + cap a list of rows."""
    capped = list(rows or [])[:CHAT_CURATION_MAX_ROWS]
    return [_annotate_country_fks(r) for r in (_safe_row(r) for r in capped) if r]


def redact_rows_for_response(rows: list[dict]) -> list[dict]:
    """Risk-20-5: strip internal/audit/credential fields from rows that go
    back to the END USER (ChatResponse.rows / the SSE `rows` event). Reuses
    the same `_safe_row` privacy filter that already guards the OpenAI
    egress, so both boundaries redact identically. Does NOT cap length —
    per-turn volume is handled separately by `cap_rows_for_turn`."""
    return [r for r in (_safe_row(row) for row in (rows or [])) if r]


def cap_rows_for_turn(rows: list[dict], *, context: str = "") -> "tuple[list[dict], bool]":
    """Risk-20-5 aggregate row budget: cap total rows across ALL query_table
    calls in one chat turn at MISA_CHAT_MAX_ROWS_PER_TURN. Returns
    (capped_rows, was_truncated). On truncation, emits a security audit
    event; the turn still answers with the capped set (no refusal), so
    normal responses are unaffected (the default budget is well above any
    legitimate turn)."""
    from app.config import CHAT_MAX_ROWS_PER_TURN
    rows = list(rows or [])
    if len(rows) <= CHAT_MAX_ROWS_PER_TURN:
        return rows, False
    capped = rows[:CHAT_MAX_ROWS_PER_TURN]
    try:
        from app.services.audit_log import emit_security_event
        emit_security_event({
            "event": "row_budget_truncated",
            "context": context,
            "rows_returned": len(rows),
            "cap": CHAT_MAX_ROWS_PER_TURN,
        })
    except Exception:
        pass
    return capped, True


def _chat_text(resp) -> str | None:
    try:
        text = (resp.choices[0].message.content or "").strip()
    except (AttributeError, IndexError):
        return None
    return text or None


# Phrases the model is repeatedly observed to add from training knowledge
# even when they do not appear in any record field — most common Pakistan/
# China/Belt-and-Road auto-completes, plus generic version-numbering. Each
# entry is a phrase to look for (case-insensitive substring match).
_SUSPICIOUS_ADD_PHRASES: tuple[str, ...] = (
    "Phase II", "Phase III", "Phase IV", "Phase 2", "Phase 3",
    "Belt and Road",
    "Industrial Cooperation Zone",  # matches plural too
    "Free Economic Zone",
)


def repair_markdown_formatting(answer: str) -> str:
    """Deterministic fix for a recurring rendering defect: a bold
    lead-in run straight into the body text with no separator, e.g.
    '**Private Funds**Organize dedicated sessions...' which renders as
    the glued word 'FundsOrganize'. Insert ' — ' between the bold close
    and the following capitalised clause. Prompt rules ask for this
    format; this guarantees it regardless of model adherence."""
    if not answer:
        return answer
    # The glued defect is a closing '**' that ends a word and is
    # butted straight against a following uppercase letter, e.g.
    # 'Funds**Organize'. Target exactly that: a '**' preceded by a
    # word character (the end of the bold phrase) and followed by an
    # uppercase letter. The lookbehind means '** — **' emdash
    # sequences and legitimate '**Bold** Word' / '**Bold:** Word' are
    # never touched.
    return re.sub(r"(?<=[A-Za-z0-9)])\*\*(?=[A-Z])", "** — ", answer)


def _scrub_backend_noise(answer: str, *, keep_citations: bool = False) -> str:
    """Defence in depth for the executive-briefing data hygiene rules.
    The curation prompt forbids confidence tags / web-citation handles /
    "Source: DB" / "Not available in the current database." in output,
    but the LLM occasionally emits them anyway. Strip them here so the
    executive-facing answer is clean regardless of model adherence.

    Rules (in order):
      1. Remove inline confidence tags: '(High)', '(Medium)', '(Low)',
         '(Unknown)' — and the single space that often precedes them.
      2. Remove inline web-citation handles: '[web:1]', '[web:12]'
         (skipped when keep_citations=True — hybrid / deep-profile
         answers need them for the Sources panel chips).
      3. Remove inline provenance tags: '[DB]', '[gk]', '[inferred]'.
      4. Remove the '_(general knowledge)_' italic marker.
      5. Drop ENTIRE lines whose substantive content is "Not available
         in the current database." or "Unknown" — graceful omission
         beats a placeholder, per the user-facing spec.
      6. Drop 'Source: DB' / 'Source: web' style trailers.
      7. Collapse triple-blank lines that the strips may produce.

    Returns the cleaned answer.
    """
    if not answer:
        return answer
    import re as _re
    text = answer
    # 1. Confidence tags — and the space before them
    text = _re.sub(r"\s?\((High|Medium|Low|Unknown)\)", "", text)
    # 2. Web citation handles (preserve when the UI will hyperlink them)
    if not keep_citations:
        text = _re.sub(r"\s?\[web:\d+\]", "", text)
    # 3. Provenance tags
    text = _re.sub(r"\s?\[(DB|gk|inferred)\]", "", text, flags=_re.IGNORECASE)
    # 4. General-knowledge italic marker (whole standalone line OR inline)
    text = _re.sub(r"^\s*\*?_?\(?general knowledge[^_*\n]*\)?_?\*?\s*$",
                   "", text, flags=_re.IGNORECASE | _re.MULTILINE)
    text = _re.sub(r"_\(general knowledge\)_\s*", "", text, flags=_re.IGNORECASE)
    # 6. Source labels (before line-drop in step 5, so the line-drop can
    #    catch lines whose ONLY content was "Source: ...").
    #    Handles "Source: DB", "**Source:** DB", "- **Source:** DB".
    #    The colon sits BETWEEN the closing stars in markdown:
    #    `**Source:** DB` not `**Source**: DB`. Account for both.
    text = _re.sub(
        r"^\s*[-*]?\s*\*{0,2}Source:?\*{0,2}\s*:?\s*(DB|web|external|general\s+knowledge)\.?\s*$",
        "", text, flags=_re.IGNORECASE | _re.MULTILINE,
    )
    text = _re.sub(
        r"\s?[—\-]?\s?\*{0,2}Source:?\*{0,2}\s*:?\s*(DB|web|external|general\s+knowledge)\.?",
        "", text, flags=_re.IGNORECASE,
    )
    # 5a. Drop ENTIRE lines that are just a placeholder.
    cleaned_lines: list[str] = []
    placeholder_re = _re.compile(
        r"^\s*[-*]?\s*(?:\*\*[^*]+:\*\*\s*)?"
        r"(not available in (the )?current database\.?|unknown|n/?a)"
        r"\s*\.?\s*$",
        _re.IGNORECASE,
    )
    for line in text.splitlines():
        if placeholder_re.match(line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    # 5b. Strip INLINE occurrences. Catches sentences like "Specific
    #     FDI data is not available in the current database." that
    #     the per-line strip in 5a missed. We drop the entire
    #     sentence containing the phrase plus any leading "- "/"."
    #     so we don't leave a half-stub bullet behind.
    text = _re.sub(
        r"(?:^|(?<=\s))[^.!?\n]*\bnot available in (?:the )?current database\b[^.!?\n]*[.!?]?",
        "",
        text, flags=_re.IGNORECASE,
    )
    # And drop the whole "## Sources & Gaps" header when the section
    # body has been emptied by 5a+5b above. We collapse a header line
    # that's followed by only whitespace before the next heading.
    text = _re.sub(
        r"^##+\s*Sources\s*&\s*Gaps[^\n]*\n(?:\s*\n)+(?=##|\Z)",
        "", text, flags=_re.IGNORECASE | _re.MULTILINE,
    )
    # Also drop an empty "Sources & Gaps" at the end of the doc.
    text = _re.sub(
        r"^##+\s*Sources\s*&\s*Gaps[^\n]*\n\s*\Z",
        "", text, flags=_re.IGNORECASE | _re.MULTILINE,
    )
    # 7. Collapse triple-blank lines created by strips
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + ("\n" if text.endswith("\n") else "")


def _strip_unsourced_bullets(answer: str, records_blob: str) -> str:
    """Defence in depth against prompt-resistant LLM embellishment.

    Some named extensions (e.g. 'CPEC Phase II', 'Belt and Road',
    'Industrial Cooperation Zones') survive even strict verbatim-only
    curation prompts because the underlying model has very strong cached
    associations. If the curated answer contains one of these phrases but
    the source records do not, drop the *entire bullet/line* that holds
    the unsourced phrase rather than try to surgically rewrite it.

    `records_blob` should be a single string with ALL the row JSON the
    model was given, so we can verify presence by case-insensitive
    substring search.
    """
    if not answer:
        return answer
    blob_lc = records_blob.lower()
    suspicious_present = [
        p for p in _SUSPICIOUS_ADD_PHRASES
        if p.lower() in answer.lower() and p.lower() not in blob_lc
    ]
    if not suspicious_present:
        return answer

    out_lines: list[str] = []
    for line in answer.splitlines():
        ll = line.lower()
        if any(p.lower() in ll for p in suspicious_present):
            # Skip the line entirely (it's a bullet/sentence with an
            # unsourced named extension). Leaving the line in but redacting
            # the phrase tends to read worse than just dropping the bullet.
            continue
        out_lines.append(line)
    # Collapse any double-blank-lines we may have created.
    cleaned = "\n".join(out_lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


_REGIONAL_REVENUE_LINE_RE = re.compile(
    r"^.*\b(saudi|ksa|mena)\b.*\brevenue\b.*\$\s?[\d,.]+.*$",
    re.IGNORECASE | re.MULTILINE,
)
_REVENUE_FIELD_RE = re.compile(
    r'"([a-z0-9_]*revenue[a-z0-9_]*)"\s*:\s*"?(-?[\d][\d,]*\.?\d*)', re.IGNORECASE,
)
_REGION_TAG_RE = re.compile(r"ksa|saudi|mena", re.IGNORECASE)
# "ksa" (field names, e.g. ksa_revenue_local_currency) and "saudi" (answer
# prose, e.g. "Saudi revenue") name the same region — canonicalise both to
# "ksa" so a tag detected from one side matches a mention on the other.
_REGION_CANON = {"ksa": "ksa", "saudi": "ksa", "mena": "mena"}


def _suspect_regional_revenue_tags(records_blob: str) -> set[str]:
    """Which region tags ("ksa", "mena") have a revenue figure in the
    retrieved rows that cannot be trusted: the field is 0/blank, or its value
    is suspiciously close to the company's GLOBAL revenue for the same row —
    a real data artifact seen in practice (a country-specific revenue field
    populated with the same number as global revenue, not a genuine regional
    figure). Detected from the row JSON itself, not the model's prose, so it
    catches both "derived a figure that isn't there" and "restated a bad
    recorded value" the same way."""
    global_vals: list[float] = []
    regional: dict[str, list[float]] = {"ksa": [], "mena": []}
    for key, raw in _REVENUE_FIELD_RE.findall(records_blob or ""):
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        tag = _REGION_TAG_RE.search(key)
        if tag:
            regional[_REGION_CANON[tag.group(0).lower()]].append(val)
        else:
            global_vals.append(val)

    suspect: set[str] = set()
    for tag, vals in regional.items():
        for v in vals:
            if v == 0:
                suspect.add(tag)
                continue
            for g in global_vals:
                if g and abs(v - g) / g < 0.02:  # within 2% of global — a
                    suspect.add(tag)                # duplicate, not a real figure
                    break
    return suspect


def _neutralise_unreliable_regional_revenue(answer: str, records_blob: str) -> str:
    """Risk-20-4-adjacent output filter: replace a specific-dollar regional
    revenue line with a neutral statement when the underlying record can't
    actually support it (see `_suspect_regional_revenue_tags`), rather than
    let a fabricated-looking or duplicate-of-global figure reach the user
    with just an inline hedge. Global and other regions are untouched — this
    only fires on the specific tag(s) flagged as unreliable."""
    if not answer:
        return answer
    suspect = _suspect_regional_revenue_tags(records_blob)
    if not suspect:
        return answer
    out_lines = []
    for line in answer.splitlines():
        m = _REGIONAL_REVENUE_LINE_RE.match(line)
        if m and _REGION_CANON[m.group(1).lower()] in suspect:
            region = "Saudi" if m.group(1).lower() in ("saudi", "ksa") else "MENA"
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(
                f"{indent}{region} revenue: not reliably recorded separately "
                f"from global revenue in the database."
            )
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


_ATTRIBUTION_LINE = (
    "_All figures per the MISA record unless labelled general knowledge._"
)


def _ensure_figure_attribution(answer: str) -> str:
    """Guarantee the figure-provenance line is present whenever the
    answer carries numeric figures. The curation prompt asks for it,
    but model adherence varies — and an unattributed figure reads as
    a live current statistic, which mis-frames point-in-time record
    data. Inserted right under the first markdown header so it's
    visible before any number."""
    if not answer:
        return answer
    if "per the misa record" in answer.lower():
        return answer
    # Figures = digits next to $ / % / B / M or bolded numbers.
    if not re.search(r"(\$\s?\d|\d+(\.\d+)?\s?%|\*\*\s?\$?\d)", answer):
        return answer
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            lines.insert(i + 1, _ATTRIBUTION_LINE)
            return "\n".join(lines)
    return _ATTRIBUTION_LINE + "\n\n" + answer


def _best_fuzzy_ratio(entity: str, rows: list[dict]) -> tuple[float, str | None]:
    """Return (best_ratio, matched_name) between the entity and any row's
    name-like field. Used to decide whether smart-search produced a real
    fuzzy hit ("microsft" → Microsoft Corporation) vs a noise hit
    ("Acme Foo Bar Holdings" → Hyundai Steel because "Holdings" matched
    a common word).

    We compare the (single-token) entity against:
      - the full field value
      - its head (everything before the first comma)
      - each individual word of the field value
    and take the max. Without per-word comparison, "microsft" vs
    "Microsoft Corporation" gets diluted to ~0.5 by the long string;
    per-word it correctly hits ~0.94 against "microsoft".
    """
    import difflib
    import re as _re
    if not entity or not rows:
        return 0.0, None
    e = entity.strip().lower()
    best = 0.0
    best_name = None
    for r in rows:
        for col in ("company_name", "name", "country_name", "executive_name",
                    "title", "rhq_entity_name", "ultimate_parent_company"):
            v = r.get(col)
            if not v or not isinstance(v, str):
                continue
            vl = v.lower()
            head = vl.split(",", 1)[0].strip()
            r1 = difflib.SequenceMatcher(None, e, vl).ratio()
            r2 = difflib.SequenceMatcher(None, e, head).ratio()
            cand = max(r1, r2)
            # Per-word comparison: handles typos against names like
            # "Microsoft Corporation" / "Apple Inc." where the noise of the
            # suffix dilutes the full-string ratio.
            for word in _re.findall(r"[A-Za-z0-9]+", vl):
                if len(word) < 3:
                    continue
                wr = difflib.SequenceMatcher(None, e, word).ratio()
                if wr > cand:
                    cand = wr
            # Reward substring containment ONLY when the entity is
            # substantial enough that a substring match is meaningful.
            # Short entities like 'elon' are substrings of unrelated
            # words like 'sentinelone' purely by coincidence — boosting
            # those would mis-classify noise as fuzzy match.
            if len(e) >= 5 and (e in vl or vl in e or e in head or head in e):
                cand = max(cand, 0.75)
            if cand > best:
                best = cand
                best_name = v
    return best, best_name


_PRONOUNS_AND_NON_ENTITIES = frozenset({
    "it", "this", "that", "these", "those", "they", "them", "their", "theirs",
    "him", "her", "his", "hers", "its",
    "the", "a", "an",
    "one", "ones", "thing", "stuff",
    "yes", "no", "ok", "okay",
})


def _looks_like_proper_entity(entity: str) -> bool:
    """Heuristic: does the entity look like a specific named entity (e.g.
    'Acme Foo Bar Holdings', 'Apple', 'Pakistan Ordnance Factories')?
    vs a topic/concept phrase like 'renewable energy investments' or
    'tech giants in Saudi', or a pronoun like 'it' / 'this'.

    Used to scope the D2 NO-MATCH suppression and the history-
    inheritance decision: only treat as a proper entity when it's
    genuinely entity-shaped (not a pronoun, not a topic phrase).
    """
    if not entity:
        return False
    e = entity.strip()
    if not e:
        return False
    # Pronouns and obviously-not-entity tokens
    if e.lower() in _PRONOUNS_AND_NON_ENTITIES:
        return False
    # Quoted entity → definitely a proper-name lookup
    if e[0] in ('"', "'") and e[-1] in ('"', "'"):
        return True
    # Single token → proper entity ONLY if it has at least one alpha char
    # AND is not in the pronoun list above.
    words = e.split()
    if len(words) == 1:
        return any(c.isalpha() for c in e)
    # Multi-word: proper if ALL non-stopword tokens are Capitalized
    # ("Acme Foo Bar Holdings" = Yes, "renewable energy investments" = No,
    # "tech giants in Saudi" = No, "Apple Inc" = Yes).
    stopwords = {"and", "of", "the", "for", "in", "on", "with", "&", "or", "to"}
    sig = [w for w in words if w.lower() not in stopwords]
    if not sig:
        return False
    return all(w[:1].isupper() for w in sig if w[:1].isalpha())


_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z]{3,})\b")


def _extract_proper_nouns(entity: str) -> list[str]:
    """Return Capitalized words (4+ chars) from the entity — proxy for
    proper nouns even when the entity has lowercase filler around them.
    Pronouns and obvious non-entity tokens are excluded."""
    if not entity:
        return []
    found = _PROPER_NOUN_RE.findall(entity)
    return [w for w in found if w.lower() not in _PRONOUNS_AND_NON_ENTITIES]


def classify_match(
    entity_candidate: str | None,
    entity_matched: bool,
    rows: list[dict],
    *,
    fuzzy_threshold: float = 0.62,
    junk_threshold: float = 0.45,
) -> tuple[str, str | None]:
    """Classify the row set against the entity:
      'exact'  — entity_matched is True (whole-word hit in the row blob)
      'fuzzy'  — entity_matched is False BUT best difflib ratio ≥ threshold
                 (typo / spelling variation, the rows ARE the right ones)
      'broad'  — the entity isn't a proper named entity (it's a topic /
                 concept phrase like 'renewable energy investments').
                 The rows ARE the answer; curate normally without the
                 no-match preamble.
      'none'   — proper-entity lookup with no good match; rows are noise
                 and curation should NOT show them.
    Returns (classification, best_matching_name_if_any).
    """
    if not entity_candidate or not str(entity_candidate).strip():
        return "exact", None
    if entity_matched:
        return "exact", None
    ratio, name = _best_fuzzy_ratio(entity_candidate, rows)
    if ratio >= fuzzy_threshold:
        return "fuzzy", name
    # Even when the entity isn't strictly "proper-entity-shaped" by the
    # all-Capitalized heuristic, if it CONTAINS a proper noun (e.g.
    # "Apple" inside "will be leading Apple after Tim") AND no row
    # contains that proper noun, the user is asking about a specific
    # entity that wasn't found — classify as 'none', not 'broad'.
    proper_nouns_in_q = _extract_proper_nouns(entity_candidate)
    if proper_nouns_in_q and rows:
        import re as _re
        for pn in proper_nouns_in_q:
            pn_re = _re.compile(r"\b" + _re.escape(pn) + r"\b", _re.I)
            found_in_any_row = False
            for r in rows:
                for col in ("company_name", "ultimate_parent_company",
                            "company_profile", "name", "country_name",
                            "executive_name", "title"):
                    v = r.get(col)
                    if isinstance(v, str) and pn_re.search(v):
                        found_in_any_row = True
                        break
                if found_in_any_row:
                    break
            if not found_in_any_row:
                # A proper noun the user asked about doesn't appear in any
                # returned row — these aren't the answer, classify 'none'.
                return "none", name if ratio >= junk_threshold else None
    # Not a named-entity miss — it's a topic query. Curate the rows normally.
    if not _looks_like_proper_entity(entity_candidate):
        return "broad", name
    return "none", name if ratio >= junk_threshold else None


def curate_company_insights(
    rows: list[dict],
    user_question: str,
    *,
    locale: str = "en",
    entity_candidate: str | None = None,
    entity_matched: bool = True,
    table: str | None = None,
    client: OpenAI,
    model: str,
    intent: str | None = None,
    depth: str | None = None,
) -> str | None:
    """Compose an insight-rich answer from privacy-filtered DB rows. None on failure.

    Three-way classification on entity match quality:
      * exact: row contains a whole-word match → normal curation.
      * fuzzy: row's name is high-similarity to the entity (typo / spelling
        variation, e.g. microsft → Microsoft Corporation) → curate normally
        but tell the model to open with "Found via similar spelling: …".
      * none: rows are noise (Acme Foo Bar → Hyundai Steel because of a
        coincidental "Holdings" match) → DO NOT call OpenAI on those rows;
        return a short deterministic honest message instead. Profiling noise
        rows would let the model invent a confusing-but-confident answer.
    """
    safe = safe_rows_for_curation(rows)
    if not safe:
        return None

    # MODEL TIERING: quick facts stay on the cheap chat model; any
    # depth that requires analysis gets the advisory-tier model. The
    # mini tier collapses definitional questions ("what is the China
    # National IC Fund?") to two sentences and drops the mandatory
    # Strategic Read section regardless of prompt pressure.
    from app.config import curation_model_for_depth
    # PERSON-QUESTION EXCEPTION: person briefings are template-critical
    # (verbatim '## From the MISA Record' / '## Background (general
    # knowledge)' headers carry the data-provenance guarantee). Two
    # rules, keyed on the QUESTION being about a person — via people
    # table OR executive_lookup intent (multi-table merges often make
    # company_profiles the primary table for a person question):
    #   1. keep the default chat model (the advisory tier restructures
    #      the mandatory headers despite prompt pressure);
    #   2. force the people TEMPLATE (below) so the company template
    #      never overrides the provenance sections.
    _PEOPLE_TABLES = (
        "executives", "company_executives", "rhq_topexecutives",
        "board_positions", "contacts", "company_contact_records",
        "related_people", "profiles", "personal_informations",
        "misa_contact_details",
    )
    _person_question = (
        table in _PEOPLE_TABLES or (intent or "") == "executive_lookup"
    )
    if _person_question:
        if table not in _PEOPLE_TABLES:
            table = "company_executives"  # selects the people template
    else:
        model = curation_model_for_depth(depth, model)

    classification, matched_name = classify_match(entity_candidate, entity_matched, rows)

    # STRATEGIC-QUESTION OVERRIDE: when the user asked a policy / strategy
    # question (e.g. "How should MISA attract Chinese investment from
    # Europe?"), the input cleaner often grabs the whole sentence as a
    # phantom "entity_candidate". classify_match then returns "none"
    # because no row's name matches the sentence — and we used to dead-end
    # with "No record matching 'How should MISA attract...' was found",
    # which is absurd. Override to "broad" (topic) classification so the
    # curator composes an answer from whatever country / sector / company
    # data we DID retrieve. The intent_note (engagement_strategy) will
    # frame it correctly.
    try:
        from app.services.chat_engine import _is_strategic_policy_question
        _is_strategic = _is_strategic_policy_question(user_question)
    except Exception:
        _is_strategic = False
    if classification == "none" and _is_strategic and rows:
        classification = "broad"
        # Re-frame entity_candidate so the TOPIC-QUERY note that follows
        # doesn't echo the whole sentence. Use the intent if available,
        # otherwise leave a generic label.
        entity_candidate = (intent or "strategic policy question").replace(
            "_", " "
        )

    # TRUE NO-MATCH path — bypass OpenAI entirely so we never present
    # unrelated row data as if it were the user's entity.
    if classification == "none" and entity_candidate:
        ent = entity_candidate.strip()
        # SECOND-CHANCE ADVISORY ROUTE (question shape, NOT entity
        # length): if the question matches the advisory patterns
        # ("develop the dynamic between X and Y", "how will X be
        # reflected in Y"), it is a synthesis ask that slipped past
        # the gate — compose the advisory answer instead of the
        # absurd 'No record matching "<whole question>"' dead-end.
        # Deliberately reuses the SAME deterministic patterns as the
        # gate so entity lookups (even ones with very long names)
        # keep their honest no-match behaviour.
        try:
            from app.services.chat_engine import _is_advisory_question
            _advisory_shaped = _is_advisory_question(user_question)
        except Exception:
            _advisory_shaped = False
        if _advisory_shaped:
            advisory = strategic_advisory_answer(
                user_question,
                db_context=None,
                deliverable="strategy_analysis",
                locale=locale,
                client=client,
                model=model,
            )
            if advisory:
                return advisory
        # Honest no-match. Don't echo a question-length "entity" back
        # verbatim — truncate for readability; the guarantee (no
        # unrelated rows presented as the entity) is unchanged.
        ent_shown = ent if len(ent) <= 80 else ent[:77].rstrip() + "…"
        return (
            f'## Snapshot\n'
            f'**No record matching "{ent_shown}" was found in the MISA database.**\n\n'
            f'The closest hits returned by the database were not similar enough '
            f'to be presented as candidates — showing them would '
            f'misrepresent the data.\n\n'
            f'## Suggestion\n'
            f'- Try a shorter or differently-spelled name.\n'
            f'- If you meant a sector, country, or programme, ask for that '
            f'directly (e.g. "energy companies in MENA", '
            f'"opportunities in Egypt").'
        )

    # Build the per-classification note for the curation prompt.
    note = ""
    if classification == "broad" and entity_candidate:
        # Topic / concept query — the rows ARE the legitimate answer.
        # Tell the model to summarise across rows instead of profiling
        # one entity, and DO NOT prepend "No record found" — that
        # framing is wrong for topic queries.
        note = (
            f'TOPIC QUERY: The user asked about the broader topic '
            f'"{entity_candidate.strip()}", not a specific named entity. '
            f'The records below are the rows the database surfaced for '
            f'that topic. Summarise across them — pick out shared sectors, '
            f'common stages, notable counterparts — rather than profiling '
            f'one row. Do NOT prepend "No record found"; the rows ARE the '
            f'answer.\n\n'
        )
    elif classification == "fuzzy" and entity_candidate:
        ent = entity_candidate.strip()
        actual = matched_name or "the record below"
        note = (
            f'FUZZY MATCH FOUND:\n'
            f'The user typed "{ent}", which is a typo / spelling variation '
            f'of **{actual}** — the records below ARE the right ones.\n'
            f'You MUST open the Snapshot with: '
            f'"Found via similar spelling: **{actual}** (you typed '
            f'\\"{ent}\\"). Profile follows." Then proceed with the normal '
            f'company-snapshot structure, attributing every fact to '
            f'**{actual}**.\n\n'
        )
    elif not entity_matched and entity_candidate:
        # Reserved for completeness; with the classification above this branch
        # is effectively only reached if we relax junk_threshold in future.
        ent = entity_candidate.strip()
        note = (
            f'HARD CONSTRAINT — NO MATCH:\n'
            f'The user asked about "{ent}", but NONE of the records below '
            f'contain a whole-word match. Open the Snapshot with: "No record '
            f'found for \\"{ent}\\" in the MISA database." Attribute every '
            f'fact to the ACTUAL row company_name; never to "{ent}". Do NOT '
            f'write "Engage with {ent}" in the Strategic Read.\n\n'
        )

    # INTENT-FIRST LAYER: if the caller classified the user's question
    # into a specific intent (executive_lookup, saudi_presence, etc.),
    # prepend a direction block telling the model to LEAD with the
    # answer to that intent. Without this, the curator defaults to the
    # entity-type template (always company snapshot for company rows),
    # which buries direct answers like "Who is the CEO?" three sections
    # deep.
    intent_note = ""
    market_intel_note = ""
    missing_data_note = ""
    if intent:
        from app.services.intent_router import (
            intent_note_for_curation, ANTI_HALLUCINATION_NOTE,
            market_intel_note_for, missing_data_note_for,
        )
        intent_directive = intent_note_for_curation(intent)
        if intent_directive:
            intent_note = intent_directive + "\n" + ANTI_HALLUCINATION_NOTE + "\n"

        # MARKET INTEL + MISSING-DATA TRANSPARENCY (Tier 3 commit 2)
        # Both gated on depth — they would be noise at simple_fact
        # depth, where the answer is a single line.
        market_intel_note = market_intel_note_for(intent, depth or "")
        if market_intel_note:
            market_intel_note += "\n"
        missing_data_note = missing_data_note_for(depth or "", intent)
        if missing_data_note:
            missing_data_note += "\n"

    # DEPTH NOTE (Tier 3 commit 1) — same intent + different depth →
    # different answer breadth. "Where is Apple's RHQ?" gets 3 lines;
    # "Give me an executive briefing on Apple" gets the 10-section
    # format. The depth note prepends to the intent note so the
    # curator reads BREADTH-INTENT in that order.
    depth_note = ""
    if depth:
        try:
            from app.services.depth_detector import depth_note_for_curation
            depth_note = depth_note_for_curation(depth)
            if depth_note:
                depth_note += "\n"
        except Exception:
            pass

    payload = json.dumps(safe, ensure_ascii=False, default=str)
    table_label = f"`{table}`" if table else "the database"

    def _run_once(extra_directive: str = "") -> str | None:
        """One full curation call. Wrapped so the response-validator
        regeneration path can re-invoke with stricter direction."""
        uc = (
            f"User question:\n{user_question}\n\n"
            f"{extra_directive}"
            f"{depth_note}"
            f"{intent_note}"
            f"{market_intel_note}"
            f"{missing_data_note}"
            f"{note}"
            f"Retrieved records from {table_label} (privacy-filtered JSON):\n{payload}"
        )
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": curation_system_prompt(locale, table)},
                    {"role": "user", "content": uc},
                ],
                store=CHAT_OPENAI_STORE,
                **openai_determinism_kw(),
                **openai_max_completion_tokens_kw(),
            )
        except Exception:
            return None
        t = _chat_text(r)
        if not t:
            return None
        t = _strip_unsourced_bullets(t, payload)
        # Neutralise a Saudi/KSA/MENA revenue figure the record can't
        # actually support (0-valued field, or a field duplicating global
        # revenue — a real data artifact, not model invention). See
        # _neutralise_unreliable_regional_revenue.
        t = _neutralise_unreliable_regional_revenue(t, payload)
        # Defence-in-depth scrub for executive-briefing hygiene: even
        # though the prompt forbids confidence tags / [web:N] / "Not
        # available" placeholders, the model occasionally emits them.
        # Strip them here so the executive-facing answer is clean
        # regardless of model adherence. See _scrub_backend_noise.
        t = _scrub_backend_noise(t)
        t = repair_markdown_formatting(t)
        return t

    text = _run_once()
    if not text:
        return None

    # RESPONSE VALIDATION (Step 9 of the spec): does the first
    # paragraph directly answer the user's question? If not, and we
    # have a retry budget, regenerate with stricter direction.
    #
    # In practice the per-intent DIRECTIVE PROMPTS in intent_router
    # are already extremely directive ("Lead with ## CEO / Name:..."),
    # so the model produces the right shape on the first try ~all the
    # time. The validator was making ~1.5s of round-trip + occasional
    # ~15s regeneration calls for no quality gain — measured a clean
    # "Tell me about Apple" answer ballooning a turn from ~10s to ~48s
    # purely from a false-positive regen.
    #
    # Strategy: keep the validator wired in but DISABLED BY DEFAULT
    # in production. Toggle via MISA_CHAT_VALIDATE=1 in env for
    # debugging only — most useful when adding a new intent without
    # a directive, where the validator's safety net is highest value.
    # Golden tests pin the lead-with-the-answer behaviour without
    # paying per-turn validator latency.
    import os as _os
    _validate_on = (_os.getenv("MISA_CHAT_VALIDATE") or "").strip().lower() in (
        "1", "true", "yes",
    )
    if _validate_on and intent and intent == "general_research":
        try:
            from app.services.response_validator import validate_first_paragraph
            verdict = validate_first_paragraph(user_question, text, client, model)
        except Exception:
            verdict = {"is_relevant": True}
        if not verdict.get("is_relevant", True):
            stricter = (
                "REGENERATION (PRIOR ANSWER REJECTED): A previous draft of "
                "this answer did NOT directly answer the user's question in "
                "the first paragraph. Reviewer note: "
                f"{verdict.get('reason') or '(no reason given)'}. "
                "This regeneration MUST open with the exact answer to the "
                "user's question — no preamble, no generic snapshot, no "
                "company-history setup. The very first line/bullet must be "
                "the direct answer.\n\n"
            )
            retry = _run_once(stricter)
            if retry:
                text = retry

    # STYLE VALIDATOR (Tier 1 commit 3/3) — runs on EVERY curated
    # answer regardless of intent. Checks find_style_violations() from
    # the style_guide module; if the answer contains any forbidden
    # string, banned emoji, or wrong number format, regenerate ONCE
    # with a FORMAT_RECHECK directive that names the specific
    # violations. The scrubber (_scrub_backend_noise) is still the
    # defence-in-depth fallback — but the validator catches issues
    # earlier and tells the model what to fix instead of silently
    # stripping. Hard-capped at one regeneration to bound latency.
    try:
        from app.services.style_guide import (
            find_style_violations, STYLE_GUIDE_PROMPT,
        )
        violations = find_style_violations(text)
    except Exception:
        violations = []
    if violations:
        violation_lines = "\n".join(f"  - {v}" for v in violations[:10])
        format_recheck = (
            "FORMAT RECHECK: the previous draft violated the MISA Style "
            "Guide in the following ways:\n"
            f"{violation_lines}\n\n"
            "Re-generate the answer fixing each violation. Reminder of "
            "the rules:\n\n"
            f"{STYLE_GUIDE_PROMPT}\n\n"
        )
        retry = _run_once(format_recheck)
        if retry:
            text = retry
        # _scrub_backend_noise + _run_once already ran post-strip; the
        # retry will go through the same path so anything still present
        # gets stripped by the scrubber as a last line of defence.

    # EXECUTIVE QUALITY CHECK — Rule 10 from the Universal Executive
    # Intelligence Reasoning Rules. Asks one extra LLM call: "would a
    # MISA Minister find this useful for decision-making? what's
    # missing?". If the score is below threshold AND the gaps are
    # addressable from the existing payload, regenerate ONCE with the
    # feedback. Otherwise ship the answer as-is.
    #
    # GATING preserves prior work:
    #   - Skip on simple_fact depth (depth_detector keeps these short)
    #   - Skip on pure-lookup intents that already have DIRECT-ANSWER
    #     RULEs (executive_lookup, saudi_presence, financial_lookup, ...)
    #   - Hard-capped at ONE regen — same discipline as style validator
    #
    # Toggle: MISA_EXEC_QUALITY_CHECK=false bypasses entirely.
    try:
        from app.services.executive_quality_check import (
            should_run_check, grade_answer, build_regen_directive,
        )
    except Exception:
        should_run_check = lambda *a, **k: False  # noqa: E731
        grade_answer = None
        build_regen_directive = None
    if should_run_check(intent, depth) and grade_answer is not None:
        verdict = grade_answer(
            user_question=user_question,
            answer=text,
            intent=intent,
            depth=depth,
            client=client,
            model=model,
        )
        if verdict and verdict.get("should_regenerate"):
            regen_directive = build_regen_directive(verdict)
            retry = _run_once(regen_directive)
            if retry:
                text = retry
        # If the second draft also scores low, we ship it anyway —
        # the user is never blocked on quality. The low score is
        # captured in the audit trail via the trace below (caller can
        # surface it for offline review).

    # Deterministic figure-provenance guarantee (prompt adherence
    # varies; this makes the attribution line a code-level invariant).
    text = _ensure_figure_attribution(text)

    return text


def general_knowledge_answer(
    user_question: str,
    *,
    locale: str = "en",
    client: OpenAI,
    model: str,
) -> str | None:
    """Answer from model knowledge when the DB has no rows. None on failure."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": fallback_system_prompt(locale)},
                {"role": "user", "content": user_question},
            ],
            store=CHAT_OPENAI_STORE,
            **openai_determinism_kw(),
            **openai_max_completion_tokens_kw(),
        )
    except Exception:
        return None
    return _chat_text(resp)


def strategic_advisory_answer(
    user_question: str,
    *,
    db_context: dict | None = None,
    deliverable: str = "strategy_analysis",
    locale: str = "en",
    client: OpenAI,
    model: str,
) -> str | None:
    """Compose a full consultant-grade strategy document for advisory
    questions (market fit, engagement plans, investment-attraction
    strategy, sector opportunity analysis). None on failure so the
    caller can fall back to the normal pipeline.

    `deliverable` picks the document structure ('market_fit',
    'engagement_plan', or 'strategy_analysis' for adaptive) so the
    answer matches the artefact the user asked for — an engagement-plan
    request must yield phases/stakeholders/KPIs, not a market-fit
    assessment.

    `db_context` is an optional dict of MISA-database facts (e.g. how
    many companies from the origin country are already licensed / hold
    RHQs, plus the top names). When present it is appended to the user
    message so the report can ground its 'Current MISA Footprint'
    section in real figures instead of generic prose.
    """
    user_content = user_question
    if db_context:
        user_content += (
            "\n\n---\nMISA DATABASE CONTEXT (system of record — cite "
            "these figures in the 'Current MISA Footprint' section; do "
            "not alter them):\n"
            + json.dumps(db_context, default=str, ensure_ascii=False)
        )
    else:
        # Deterministic guard: without this line the model sometimes
        # writes a 'Current MISA Footprint' section anyway — from
        # nothing — fabricating database figures.
        user_content += (
            "\n\n---\nNOTE: No MISA database context is available for "
            "this question. Do NOT include a 'Current MISA Footprint' "
            "or 'Evidence Base' section, and do not cite any MISA "
            "database figures — this analysis is general market "
            "knowledge only."
        )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": advisory_system_prompt(locale, deliverable)},
                {"role": "user", "content": user_content},
            ],
            store=CHAT_OPENAI_STORE,
            **openai_determinism_kw(),
            **openai_advisory_max_tokens_kw(),
        )
    except Exception:
        return None
    return repair_markdown_formatting(_chat_text(resp))
