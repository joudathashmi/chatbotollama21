"""
Chat pipeline: Arabic helpers, entity matching, OpenAI retry, and the main
chat() entry point that drives the SQL-routing → DB → local commentary flow.

Row data is NEVER included in OpenAI messages — only the user's question and
fixed schema hints (in the system prompt) reach the model.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import random
import re
import sys
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock

import pandas as pd
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

import app.services.human_feedback as hf
from app.config import (
    COUNT_ONLY_RATE_LIMIT,
    MAX_USER_MESSAGE_CHARS,
    OPENAI_MAX_RETRIES,
    OPENAI_RETRY_DELAY_SEC,
    OPENAI_MODEL,
    CHAT_CURATION_ENABLED,
    CHAT_FALLBACK_ENABLED,
    openai_determinism_kw as _det_kw,
    openai_max_completion_tokens_kw,
    max_history_user_turns,
)
from app.database import (
    COMPANY_TABLE,
    _ARABIC_SCRIPT_RE,
    _AR_ENTITY_LATIN_ALIASES,
    _coerce_filters_mapping,
    _normalize_filter_condition,
    _rhq_filters_include_name_search,
    _table_source_sql,
    get_db,
    get_openai_client,
    generate_query_and_run_query,
    run_rhq_company_smart_search,
    smart_search,
)
from app.db_introspect import get_table_info, is_allowed_table, name_columns
from app.services.country_resolver import resolve_country_id_for_fk
from app.services.rate_limiter import RateLimiter

# Map common country-adjective forms to the noun form used in the data's
# `global_headquarters` field. Used to rewrite misrouted filters where the
# model picked `company_name=<adjective>` (which never matches anything)
# instead of `global_headquarters=<country>`. Extend as needed.
_COUNTRY_ADJECTIVE_TO_NOUN: dict[str, str] = {
    "pakistani": "Pakistan", "indian": "India", "egyptian": "Egypt",
    "chinese": "China", "japanese": "Japan", "korean": "South Korea",
    "south korean": "South Korea",
    "french": "France", "german": "Germany", "british": "United Kingdom",
    "spanish": "Spain", "italian": "Italy", "russian": "Russia",
    "brazilian": "Brazil", "mexican": "Mexico", "canadian": "Canada",
    "australian": "Australia", "american": "United States",
    "dutch": "Netherlands", "greek": "Greece", "swedish": "Sweden",
    "swiss": "Switzerland", "polish": "Poland", "portuguese": "Portugal",
    "belgian": "Belgium", "danish": "Denmark", "norwegian": "Norway",
    "finnish": "Finland", "austrian": "Austria", "romanian": "Romania",
    "hungarian": "Hungary", "ukrainian": "Ukraine", "czech": "Czechia",
    "singaporean": "Singapore", "afghan": "Afghanistan",
    "saudi": "Saudi Arabia", "emirati": "United Arab Emirates",
    "qatari": "Qatar", "kuwaiti": "Kuwait", "bahraini": "Bahrain",
    "omani": "Oman", "iraqi": "Iraq", "iranian": "Iran",
    "syrian": "Syria", "lebanese": "Lebanon", "jordanian": "Jordan",
    "yemeni": "Yemen", "palestinian": "Palestine",
    "turkish": "Turkey", "moroccan": "Morocco", "algerian": "Algeria",
    "tunisian": "Tunisia", "libyan": "Libya", "sudanese": "Sudan",
    "nigerian": "Nigeria", "south african": "South Africa",
    "kenyan": "Kenya", "ethiopian": "Ethiopia", "ghanaian": "Ghana",
    "vietnamese": "Vietnam", "thai": "Thailand", "indonesian": "Indonesia",
    "malaysian": "Malaysia", "filipino": "Philippines",
    "bangladeshi": "Bangladesh", "sri lankan": "Sri Lanka",
}

# Short-form aliases not in CANONICAL_COUNTRIES (≤3 chars or alternate nouns).
_COUNTRY_SHORT_ALIASES: dict[str, str] = {
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "ksa": "Saudi Arabia",
}


def _normalise_country_id_filters(table: str, filters: dict) -> dict:
    """If the model passed a string country name to an integer country-FK
    column (`country_id`, `country_profile_id`, …), resolve it server-side
    to the integer id. Without this the filter would fail or be silently
    coerced to garbage.
    """
    if not filters or not table:
        return filters
    info = get_table_info(table)
    if info is None and table != COMPANY_TABLE:
        return filters
    col_types = (info or {}).get("columns", {}) if info else {}
    out = dict(filters)
    for col, raw in list(filters.items()):
        cl = col.lower()
        if "country" not in cl or "_id" not in cl:
            continue
        # Bigint columns only; if column type is unknown (company_profiles
        # virtual view), still try since the names country_id /
        # country_profile_id are conventional FKs.
        dtype = col_types.get(col, "")
        if dtype and dtype not in ("bigint", "integer", "smallint"):
            continue
        spec = _normalize_filter_condition(raw)
        val = spec.get("value")
        # Already an int → leave it.
        if isinstance(val, int) or (isinstance(val, str) and val.strip().isdigit()):
            continue
        if not isinstance(val, str):
            continue
        rid = resolve_country_id_for_fk(val, col)
        if rid is None:
            continue
        out[col] = {"op": "=", "value": int(rid)}
        out.setdefault(
            "_resolved_country_fk", {}
        )[col] = {"from": val, "to": int(rid)}
    return out


_SELF_REFERENCE_FILTER_NOISE = frozenset({
    "misa", "the misa", "ministry of investment",
    "ministry of investment of saudi arabia", "miosa",
    "iisd", "iisd-analyst", "ministry",
})


def _strip_self_reference_filters(filters: dict) -> dict:
    """Drop filters like `company_name = 'MISA'` — MISA is the audience
    (the IPA reading the briefing), not a company in the data. Without
    this, the model frequently treats organisation keywords from the
    question as company-name filter values and gets back nothing."""
    if not filters:
        return filters
    out = dict(filters)
    # Strip MISA-as-value only from the explicit-name columns. Do NOT strip
    # from `company_profile` (a long free-text field where "MISA" can
    # legitimately appear in real company prose) — over-stripping leaves
    # the model with no filters and silently degrades the answer.
    for col in (
        "company_name", "ultimate_parent_company",
        "name", "title", "rhq_entity_name", "executive_name",
    ):
        if col not in out:
            continue
        spec = _normalize_filter_condition(out[col])
        val = spec.get("value")
        if isinstance(val, str) and val.strip().lower() in _SELF_REFERENCE_FILTER_NOISE:
            out.pop(col)
            out.setdefault("_dropped_self_reference_filter", []).append(
                {col: val}
            )
    return out


def _normalise_country_adjective_filters(filters: dict) -> dict:
    """If the model issued `company_name=<country-adjective>` (e.g.
    'Pakistani', 'Indian'), rewrite to `global_headquarters=<country>`
    server-side. Smaller models occasionally lock onto company_name even
    when explicit examples in the system prompt say otherwise, so this is
    a deterministic backstop."""
    if not filters:
        return filters
    raw = filters.get("company_name")
    if raw is None:
        return filters
    spec = _coerce_filters_mapping({"company_name": raw}).get("company_name")
    val = None
    if isinstance(spec, dict):
        val = spec.get("value")
    elif isinstance(spec, str):
        val = spec
    if not isinstance(val, str):
        return filters
    key = re.sub(r"\s+", " ", val.strip().lower())
    noun = _COUNTRY_ADJECTIVE_TO_NOUN.get(key)
    if not noun:
        return filters
    out = dict(filters)
    out.pop("company_name", None)
    # Preserve a structured-filter shape so the query builder routes via
    # the substring-on-= path for global_headquarters.
    out["global_headquarters"] = {"op": "=", "value": noun}
    out["_rewritten_company_name_adjective"] = {"from": val, "to": noun}
    return out
from app.prompts.chat_system import system_prompt, tools as _build_tools
from app.services.commentary import generate_commentary
from app.services.confidence import compute_confidence
from app.services.curation import (
    curate_company_insights, general_knowledge_answer, strategic_advisory_answer,
)
from app.services.input_cleaner import (
    clean_user_question, detect_pure_browse,
    looks_like_general_knowledge_question,
    looks_like_schema_browse_question,
)
from app.services.alias_resolver import expand_aliases, matches_any_alias
from app.services.ambiguity import (
    detect_ambiguity, discover_candidates,
    format_clarification, is_short_entity_query,
)
from app.services.intent import detect_intent

# ---------------------------------------------------------------------------
# OpenAI retry types
# ---------------------------------------------------------------------------

_OPENAI_RETRYABLE_TYPES = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

# ---------------------------------------------------------------------------
# Arabic helpers
# ---------------------------------------------------------------------------

def _text_has_arabic(s: str | None) -> bool:
    return bool(s and _ARABIC_SCRIPT_RE.search(s))


def _latin_company_aliases_for_arabic_entity(entity: str | None) -> list[str]:
    if not entity or not str(entity).strip():
        return []
    key = re.sub(r"\s+", " ", str(entity).strip())
    if not _text_has_arabic(key):
        return []
    al = _AR_ENTITY_LATIN_ALIASES.get(key)
    return list(al) if al else []


def _effective_response_locale(ui_locale: str, user_question: str) -> str:
    if ui_locale == "ar":
        return "ar"
    if _text_has_arabic(user_question):
        return "ar"
    return "en"

# ---------------------------------------------------------------------------
# Search-term extraction
# ---------------------------------------------------------------------------

_RHQ_Q_STOPWORDS = frozenset({
    "what", "whats", "who", "whos", "where", "when", "why", "how", "tell", "give",
    "show", "list", "find", "search", "lookup", "about", "please", "kindly",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "for", "from", "with", "without", "and", "or", "but", "not", "this", "that",
    "these", "those", "into", "onto", "per", "via", "company", "companies",
    "firm", "firms", "organization", "business", "details", "information", "info",
    "describe", "definition", "meaning", "name", "called", "known",
    "mentioning", "mentioned", "holding", "holdings", "matches", "match",
    "null", "cmpany", "campany", "comapny",
})


def _token_is_schema_noise_keyword(w: str) -> bool:
    wl = (w or "").lower().strip()
    if not wl:
        return True
    if wl.startswith("rhq_"):
        return True
    if "ultimate_parent" in wl or wl in (
        "mentioning", "mentioned", "company_name", "company_profile",
        "revenue_usd", "non-null", "nonnull",
    ):
        return True
    return wl == "ilike"


def _search_terms_from_question(q: str) -> list[str]:
    qn = re.sub(r"[^\w\s]", " ", (q or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for raw in qn.split():
        w = raw.strip()
        if len(w) < 4 or w in _RHQ_Q_STOPWORDS:
            continue
        if _token_is_schema_noise_keyword(w):
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= 6:
            break
    return out


def _search_terms_for_pack(pack: dict, user_question: str) -> list[str]:
    ec = (pack.get("entity_candidate") or "").strip()
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        t = (s or "").strip()
        if len(t) < 2:
            return
        k = t.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(t)

    if ec and len(ec) >= 2:
        add(ec)
        for w in ec.split():
            wt = w.strip()
            # Skip too-short tokens and common English stopwords ("and", "the",
            # "of", …). Without this, an entity like "Dentons UK And Middle
            # East LLP" produces broad terms that ILIKE-match thousands of
            # unrelated rows and drown out the actual target name in the
            # fetch_cap window.
            if len(wt) < 3:
                continue
            if wt.lower() in _RHQ_Q_STOPWORDS:
                continue
            add(wt)
        for lat in _latin_company_aliases_for_arabic_entity(ec):
            add(lat)
        if out:
            return out[:10]
    return _search_terms_from_question(user_question)

# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------

_LIKELY_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank\s+you|thx|ok|okay|bye|goodbye)\s*[.!…]*\s*$",
    re.I,
)


def _likely_rhq_company_lookup(user_question: str, pack: dict) -> bool:
    q = (user_question or "").strip()
    if len(q) < 3:
        return False
    if _LIKELY_GREETING_RE.match(q):
        return False
    if looks_like_schema_browse_question(q) and not (pack.get("entity_candidate") or "").strip():
        return False
    if (pack.get("entity_candidate") or "").strip():
        return True
    return bool(_search_terms_from_question(q))


_BROWSE_INTENT_TOKENS = frozenset({
    "top", "list", "show", "browse", "all", "any",
    "recent", "latest", "newest", "oldest", "last",
    "first", "few", "many", "most", "best", "highest",
    "lowest", "biggest", "smallest", "largest",
    "every", "each", "some", "compare", "vs", "versus",
})

_NUMERIC_DIGITS_RE = re.compile(r"\b\d+\b")


def _looks_like_browse_or_aggregation(entity: str | None) -> bool:
    """Heuristic: is this extracted 'entity' really a browse / aggregation /
    comparison phrase rather than a real named entity? E.g.:
        - 'top 5 companies by revenue'
        - 'list 10 recent deals'
        - 'compare Apple and Microsoft'
        - 'how many licensed companies'
    We must NOT enforce the named-entity SQL guardrail on these — the
    user isn't looking for one specific entity, they're asking the model
    to assemble a result set."""
    if not entity:
        return False
    words = re.findall(r"\w+", entity.lower(), flags=re.UNICODE)
    if not words:
        return False
    # 4+ tokens AND the first 1-2 tokens are browse-intent words → browse
    if len(words) >= 3 and words[0] in _BROWSE_INTENT_TOKENS:
        return True
    if len(words) >= 4 and words[1] in _BROWSE_INTENT_TOKENS:
        return True
    # contains a digit + browse-intent token → 'top 5 X', '10 recent Y'
    if _NUMERIC_DIGITS_RE.search(entity) and any(
        w in _BROWSE_INTENT_TOKENS for w in words
    ):
        return True
    # 'compare X and Y' / 'X vs Y'
    if "compare" in words or "vs" in words or "versus" in words:
        return True
    return False


def _entity_requires_sql_constraint(entity: str | None) -> bool:
    """A 'real' named entity that deserves a strict SQL constraint AND a
    row-sanity check on results. The previous 6-char threshold was too
    permissive — names like 'Elon', 'Tata', 'Sony', 'Visa' all skipped
    the row-sanity guard entirely, letting smart-search noise like
    SentinelOne come back as if it were Elon. Threshold lowered to
    4 chars (single-token) so any plausibly-named entity is checked.
    Very short tokens (1-3 chars: 'uk', 'us', 'co') are too ambiguous
    to enforce. ALSO: browse / aggregation / comparison phrases like
    'top 5 companies by revenue' are not named entities — the guardrail
    would force a smart-search and return 0; skip them."""
    if not entity or not str(entity).strip():
        return False
    e = entity.strip()
    if _looks_like_browse_or_aggregation(e):
        return False
    if len(e.split()) >= 2:
        return True
    return len(e) >= 4 and any(c.isalpha() for c in e)


def _significant_entity_tokens(entity: str) -> list[str]:
    out: list[str] = []
    for w in re.findall(r"\w+", entity or "", flags=re.UNICODE):
        if len(w) < 2:
            continue
        if len(w) >= 3 or _ARABIC_SCRIPT_RE.search(w):
            out.append(w.lower())
    return out


def _sql_covers_entity(sql: str, params: list, entity: str | None) -> bool:
    if not _entity_requires_sql_constraint(entity):
        return True
    s = (sql or "").lower()
    if "where" not in s:
        return False
    name_cols = (
        "company_name", "ultimate_parent_company", "company_profile",
        "rhq_entity_name", "global_headquarters",
    )
    if not any(c in s for c in name_cols):
        return False
    pl = " ".join(str(p).lower() for p in (params or []))
    el = (entity or "").strip().lower()
    words = re.findall(r"\w+", entity or "", flags=re.UNICODE)
    longest = max((w.lower() for w in words), key=len, default="")
    if longest and len(longest) >= 4 and longest in pl:
        return True
    if el and el.replace(" ", "") in pl.replace("%", "").replace(" ", ""):
        return True
    long_words = [w.lower() for w in words if len(w) > 3]
    if long_words and all(w in pl for w in long_words):
        return True
    toks = _significant_entity_tokens(entity or "")
    if toks and sum(1 for t in toks if t in pl) >= min(len(toks), 2):
        return True
    for lat in _latin_company_aliases_for_arabic_entity(entity):
        if lat.lower() in pl:
            return True
    return False


def _whole_word_in(needle: str, hay: str) -> bool:
    """Whole-word substring check (case-insensitive). Prevents false
    positives like `'elon' in 'sentinelone'` that would let the
    row-entity sanity guard accept clearly-unrelated rows."""
    if not needle or not hay:
        return False
    return re.search(r"\b" + re.escape(needle.lower()) + r"\b", hay.lower()) is not None


def _rows_match_entity_in_cols(
    rows: list[dict], entity: str | None, cols: list[str]
) -> bool:
    """Generic per-table row-entity sanity: does the entity appear as a
    whole-word match anywhere in the listed name/text columns of any
    returned row? Substring matching would let, e.g., `'elon'` match
    `'sentinelone'` and pretend SentinelOne is about Elon — exactly the
    kind of bad result this check is supposed to catch."""
    if not entity or not str(entity).strip() or not rows or not cols:
        return True
    words = [w.lower() for w in re.findall(r"\w+", entity or "", flags=re.UNICODE)]
    longish = [w for w in words if len(w) >= 4]
    anchor = max(longish, key=len) if longish else ""
    ent_phrase = re.sub(r"\s+", " ", entity.strip())
    for r in rows:
        parts = [str(r.get(c) or "") for c in cols]
        blob = re.sub(r"\s+", " ", " ".join(parts))
        if ent_phrase and _whole_word_in(ent_phrase, blob):
            return True
        if anchor and _whole_word_in(anchor, blob):
            return True
        for lat in _latin_company_aliases_for_arabic_entity(entity):
            if _whole_word_in(lat, blob):
                return True
    return False


def _rows_match_entity(rows: list[dict], entity: str | None) -> bool:
    """Row-entity sanity for company_profiles rows. Whole-word match only —
    naive substring matching would let `'elon' in 'sentinelone'` accept a
    SentinelOne row as if it were about Elon. Alias-aware: 'Google' will
    match a row about 'Alphabet, Inc.' because Alphabet is a known alias
    of Google."""
    if not entity or not str(entity).strip() or not rows:
        return True
    ent_phrase = re.sub(r"\s+", " ", entity.strip())
    words = [w.lower() for w in re.findall(r"\w+", entity or "", flags=re.UNICODE)]
    generic = frozenset({"and", "the", "for", "of", "in", "llp", "ltd", "plc",
                          "llc", "uk", "usa", "inc", "corp"})
    longish = [w for w in words if len(w) >= 4 and w not in generic]
    anchor = max(longish, key=len) if longish else ""
    toks = [t for t in _significant_entity_tokens(entity)
            if t not in frozenset({"and", "the", "for"})]
    for r in rows:
        parts = [r.get("company_name") or "", r.get("ultimate_parent_company") or "",
                 r.get("company_profile") or ""]
        blob = re.sub(r"\s+", " ", " ".join(parts))
        if ent_phrase and _whole_word_in(ent_phrase, blob):
            return True
        if anchor and _whole_word_in(anchor, blob):
            return True
        if toks and all(_whole_word_in(t, blob) for t in toks):
            return True
        # Curated company alias map — 'Google' → row 'Alphabet, Inc.' passes
        if matches_any_alias(entity, blob):
            return True
        for lat in _latin_company_aliases_for_arabic_entity(entity):
            if _whole_word_in(lat, blob):
                return True
    return False


def _top_similar_company_names(entity: str, k: int = 3) -> list[str]:
    if not entity or not str(entity).strip():
        return []

    def _fetch_names(conn) -> list[str]:
        source_sql = _table_source_sql(COMPANY_TABLE)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT company_name FROM {source_sql}
                WHERE company_name IS NOT NULL AND btrim(company_name) <> ''
                """
            )
            return [r[0] for r in cur.fetchall() if r and r[0]]

    try:
        conn = get_db()
        names = _fetch_names(conn)
    except Exception as exc:
        # Outage ≠ "no similar names". Callers that treat [] as
        # "entity unknown" must check pack/_retrieval separately.
        try:
            from app.logger import logger as _log
            _log.warning("similar_company_names UNAVAILABLE: %s", exc)
        except Exception:
            pass
        return []

    el = entity.strip().lower()
    latin_q = _latin_company_aliases_for_arabic_entity(entity)
    if latin_q:
        el = latin_q[0].lower()
    scored: list[tuple[float, str]] = []
    for n in names:
        nl = n.lower()
        head = n.split(",")[0].strip().lower()
        r = max(
            difflib.SequenceMatcher(None, el, nl).ratio(),
            difflib.SequenceMatcher(None, el, head).ratio(),
        )
        scored.append((r, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for score, n in scored:
        if n in seen or score < 0.28:
            continue
        out.append(n)
        seen.add(n)
        if len(out) >= k:
            break
    return out

# ---------------------------------------------------------------------------
# OpenAI retry
# ---------------------------------------------------------------------------

def _openai_exc_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _OPENAI_RETRYABLE_TYPES):
        return True
    code = getattr(exc, "status_code", None)
    if code is None:
        return False
    try:
        c = int(code)
    except (TypeError, ValueError):
        return False
    return c == 429 or c >= 500


def _chat_completions_create_with_retry(client: OpenAI, **kwargs):
    # Mask PII/secrets in user/tool message content before egress.
    # System prompts stay intact; business facts are not stripped.
    try:
        from app.services.prompt_masking import mask_messages_for_llm
        if "messages" in kwargs and kwargs["messages"] is not None:
            kwargs = {**kwargs, "messages": mask_messages_for_llm(kwargs["messages"])}
    except Exception:
        pass
    last_exc: BaseException | None = None
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if not _openai_exc_is_retryable(e):
                raise
            last_exc = e
            if attempt + 1 < OPENAI_MAX_RETRIES:
                delay = OPENAI_RETRY_DELAY_SEC * (2 ** attempt) + random.uniform(0, 0.35)
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc

# ---------------------------------------------------------------------------
# Truncation / history helpers
# ---------------------------------------------------------------------------

def _truncate_for_llm(s: str, max_chars: int = MAX_USER_MESSAGE_CHARS) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    suffix = "\n… [message truncated for API size]"
    take = max_chars - len(suffix)
    return (s[:take] + suffix) if take >= 1 else suffix[:max_chars]


def _openai_max_completion_tokens_kw() -> dict:
    return openai_max_completion_tokens_kw()


def _max_history_user_turns() -> int:
    return max_history_user_turns()

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

def _structured_turn_log(payload: dict) -> None:
    # Read env vars at call time so monkeypatching works in tests.
    if os.getenv("MISA_LOG_TURNS", "").strip().lower() not in ("1", "true", "yes"):
        return
    try:
        from app.services.prompt_masking import mask_obj
        payload = mask_obj(payload, for_log=True)
    except Exception:
        pass
    line = json.dumps(payload, ensure_ascii=False, default=str)
    path = (os.getenv("MISA_LOG_FILE") or "").strip()
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
    print(line, file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Tool argument parsing
# ---------------------------------------------------------------------------

def _parse_tool_arguments(raw: str | None) -> dict:
    s = (raw or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    return {}

# ---------------------------------------------------------------------------
# Chat pipeline internals
# ---------------------------------------------------------------------------

def _turn_entity_hint(pack: dict) -> str:
    raw = pack.get("raw") or ""
    cleaned = pack.get("cleaned") or ""
    e = pack.get("entity_candidate")
    bits = [f"raw_user_input={raw!r}", f"cleaned={cleaned!r}", f"entity_candidate={e!r}"]
    body = "TURN_PREPROCESSING (" + ", ".join(bits) + ")."
    if e:
        body += (
            " If you call `query_company_profiles`, include this entity in filters on "
            "`company_name`, `ultimate_parent_company`, or `company_profile` "
            "(use op '=' for substring match on those text fields)."
        )
    return "\n\n" + body


def _assistant_api_dict(msg) -> dict:
    d: dict = {"role": "assistant", "content": msg.content or None}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]
    return d


def _forced_smart_search_tool_result(user_question: str, pack: dict, *, limit: int = 25) -> dict:
    terms = _search_terms_for_pack(pack, user_question)
    if not terms:
        terms = _search_terms_from_question(user_question)
    df, sql, params = run_rhq_company_smart_search(terms, limit)
    from app.database import smart_search_retrieval_failed, smart_search_failure_message
    if smart_search_retrieval_failed(sql):
        err = smart_search_failure_message(sql)
        pack["_degraded"] = "smart_search_retrieval_failed"
        pack["_retrieval"] = {
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "source_name": "company_profiles.smart_search",
            "do_not_claim_zero": True,
            "counts_unavailable": True,
            "error": err,
        }
        return {
            "table": COMPANY_TABLE,
            "filters": {
                "_forced_retrieval_no_tool_call": terms,
                "_db_error": err,
                "_retrieval_status": "SOURCE_UNAVAILABLE",
            },
            "sql": sql,
            "params": params,
            "rows_df": df,
            "row_count": None,  # never 0 on failure — not a verified empty
            "input_trace": dict(pack),
            "sql_entity_check_passed": False,
            "row_entity_sanity_passed": False,
            "closest_names": [],
            "error": err,
            "_retrieval_failed": True,
        }
    entity = pack.get("entity_candidate")
    rows = df.to_dict(orient="records")
    row_sanity = True
    closest: list[str] = []
    if _entity_requires_sql_constraint(entity) and rows:
        row_sanity = _rows_match_entity(rows, entity)
        if not row_sanity:
            closest = _top_similar_company_names(entity, 3)
    elif _entity_requires_sql_constraint(entity) and not rows:
        closest = _top_similar_company_names(entity, 3)
    return {
        "table": COMPANY_TABLE,
        "filters": {"_forced_retrieval_no_tool_call": terms},
        "sql": sql,
        "params": params,
        "rows_df": df,
        "row_count": int(len(df)),
        "input_trace": dict(pack),
        "sql_entity_check_passed": True,
        "row_entity_sanity_passed": row_sanity,
        "closest_names": closest,
    }


def _name_key_for_dedup(name: str) -> str:
    """Normalised key for detecting near-duplicate company names so the
    ambiguity check doesn't fire on entries that are the SAME company
    with minor punctuation/suffix differences ('Alphabet Inc.' vs
    'Alphabet, Inc.' vs 'Alphabet Inc')."""
    n = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    n = re.sub(r"\s+", " ", n).strip()
    # Strip common company suffixes for the dedup key
    n = re.sub(
        r"\b(inc|incorporated|ltd|limited|llc|llp|plc|corp|corporation|"
        r"co|group|holdings|company)\b",
        "", n,
    )
    return re.sub(r"\s+", " ", n).strip()


# ─── Strategic / policy question detector ─────────────────────────────
# When the user asks a STRATEGY question ("how do I bring Chinese
# investment from Europe to KSA, what measure should I take?"), the
# pre-LLM ambiguity detector used to fire because "Chinese" → 5
# companies named China-Something in company_profiles. The bot then
# short-circuited to "which China company did you mean?" — completely
# misreading the question.
#
# Strategy questions are about a TOPIC or POLICY, not a specific named
# entity. They contain verbs of investment-promotion strategy (bring,
# attract, redirect, promote, target), policy/measure framing (what
# measure, what should I take, recommend, advise), or flow language
# (from X to Y, from europe to ksa). When any of these signals are
# present, we suppress the ambiguity short-circuit and let the intent
# classifier route the question to engagement_strategy or
# general_research — both of which compose proper strategic answers.

_STRATEGIC_QUESTION_PATTERNS = [
    # Strategy / promotion verbs paired with investment-like nouns.
    # Suffix forms included — 'targeted'/'attracting' not matching the
    # bare stem let ambiguity short-circuits hijack these questions.
    re.compile(
        r"\b(bring(?:ing|s)?|attract(?:ing|ed|s)?|redirect(?:ing|ed)?|"
        r"promot(?:e|ing|ed)|target(?:ing|ed|s)?|encourag(?:e|ing)|"
        r"increas(?:e|ing)|shift(?:ing)?|convert(?:ing)?|"
        r"captur(?:e|ing|ed)|win(?:ning)?|secur(?:e|ing)|land(?:ing)?|"
        r"grow(?:ing)?)\b.{0,40}"
        r"\b(investment|investments|investor|investors|capital|"
        r"fdi|business|companies)\b",
        re.I | re.DOTALL,
    ),
    # reversed order ("companies to be targeted", "investors we pursue")
    re.compile(
        r"\b(companies|investors?|firms|businesses)\b.{0,40}"
        r"\bto\s+(be\s+)?(target|targeted|attract|attracted|pursue|"
        r"court|engage)",
        re.I | re.DOTALL,
    ),
    # "what measure / what measures / what should I take / what's my approach"
    re.compile(
        r"\bwhat\s+(measure|measures|steps?|actions?|approach|"
        r"strategy|policy|policies|incentives?)\b",
        re.I,
    ),
    # "how do I / how should I / how should we / how can we"
    re.compile(
        r"\bhow\s+(do|should|can|might)\s+(i|we|misa|ksa|saudi)\b",
        re.I,
    ),
    # "should I take / should we do / what should be done"
    re.compile(
        r"\b(should\s+i\s+take|should\s+we\s+do|should\s+be\s+done|"
        r"should\s+misa)\b",
        re.I,
    ),
    # Recommendation / advice framing
    re.compile(
        r"\b(recommend|suggest|advise|propose|plan)\b.{0,40}"
        r"\b(approach|strategy|measure|action|plan|policy)\b",
        re.I | re.DOTALL,
    ),
    # Cross-border flow language ("from europe to ksa", "into saudi")
    re.compile(
        r"\bfrom\s+\w+\s+(to|into)\s+(ksa|saudi|saudi\s+arabia|"
        r"the\s+kingdom|riyadh)\b",
        re.I,
    ),
    # Topic discussion of investment flows / programmes
    re.compile(
        r"\b(investment\s+(flow|policy|promotion|strategy|measures?)|"
        r"capital\s+(flow|attraction)|fdi\s+(policy|strategy|promotion))\b",
        re.I,
    ),
]


def _is_strategic_policy_question(user_question: str) -> bool:
    """True when the question is shaped like an executive policy /
    strategy ask rather than an entity lookup. Used to suppress the
    pre-LLM ambiguity short-circuit so questions like "bring Chinese
    investment from Europe to KSA" don't get hijacked into "which
    China company did you mean?"."""
    if not user_question:
        return False
    for pat in _STRATEGIC_QUESTION_PATTERNS:
        if pat.search(user_question):
            return True
    return False


# ─── Strategic advisory detection ─────────────────────────────────────
# Advisory questions ("market fit for attracting Indian companies",
# "investment case for German manufacturers in KSA") are TOPIC-level
# strategy asks. They never match DB rows, so before this path existed
# they fell through to the general-knowledge fallback — whose prompt
# caps output at ~150 words and bans recommendations — and the user got
# a thin generic paragraph. These questions instead deserve a full
# consultant-grade report (see advisory_system_prompt), optionally
# grounded with real MISA figures for the origin country.
#
# Deliberately narrower than _STRATEGIC_QUESTION_PATTERNS: company-
# specific engagement asks ("how should MISA engage Aramco?") must stay
# on the row-grounded engagement_strategy route, so verbs like "engage"
# / "contact" are NOT in this list.

_ADVISORY_QUESTION_PATTERNS = [
    # "market fit" anywhere is the canonical advisory ask.
    # "market for" is a very common typo / speech-to-text of "market fit"
    # ("make me a market for to attract Indian companies").
    re.compile(r"\bmarket\s+(fit|for)\b|\bfit\s+assessment\b", re.I),
    # "make me a market … attract … companies" (typo-tolerant)
    re.compile(
        r"\bmake\s+(me\s+)?(a\s+)?market\b.{0,60}"
        r"\bat+r+act",
        re.I | re.DOTALL,
    ),
    # attraction verbs + investment-like nouns ("attracting Indian
    # companies", "win Japanese investors", "bring FDI"). Verb suffix
    # forms are spelled out — 'targeted'/'attracting' failing to match
    # the bare stem is exactly how these questions kept leaking into
    # the company disambiguator. Allow common double-letter typos
    # (atrract / attracct).
    re.compile(
        r"\b(at+r+act(?:ing|ed|s)?|bring(?:ing|s)?|draw(?:ing|s)?|"
        r"court(?:ing|ed|s)?|win(?:ning|s)?|captur(?:e|ing|ed|es)|"
        r"target(?:ing|ed|s)?|pursu(?:e|ing|ed|es))\b.{0,60}"
        r"\b(compan(?:y|ies)|invest(?:or|ors|ment|ments)|businesses|"
        r"firms|manufacturers|capital|fdi)\b",
        re.I | re.DOTALL,
    ),
    # reversed order: noun before verb ("companies to be targeted from
    # China", "which investors should we be attracting from Japan")
    re.compile(
        r"\b(compan(?:y|ies)|invest(?:or|ors)|firms|businesses|targets)\b"
        r".{0,40}\b(to|should|would|could|can|will|must)\s+"
        r"(we\s+|misa\s+|i\s+)?(be\s+)?"
        r"(target(?:ing|ed)?|at+r+act(?:ing|ed)?|"
        r"pursu(?:e|ing|ed)|court(?:ing|ed)?|engag(?:e|ing|ed)|focus)",
        re.I | re.DOTALL,
    ),
    # target-list asks: "best/top companies ... from <country>", and
    # any question demanding an investment thesis
    re.compile(
        r"\b(best|top|priority|ideal|promising|key)\s+(\w+\s+){0,2}"
        r"(compan(?:y|ies)|investors?|firms|targets)\b.{0,60}\bfrom\b",
        re.I | re.DOTALL,
    ),
    re.compile(r"\binvestment\s+thes[ie]s\b", re.I),
    # explicit strategy-document framings
    re.compile(
        r"\b(investment\s+(case|attraction|proposition)|"
        r"attraction\s+strategy|value\s+proposition)\b",
        re.I,
    ),
    # "why should X invest (in Saudi)"
    re.compile(r"\bwhy\s+(should|would)\b.{0,50}\binvest", re.I | re.DOTALL),
    # sector/market analysis scoped to Saudi
    re.compile(
        r"\b(sector|market)\s+(analysis|assessment|opportunit\w+)\b"
        r".{0,60}\b(saudi|ksa|kingdom)\b",
        re.I | re.DOTALL,
    ),
    # macro / thematic trend questions ("new global trends impacting
    # investment", "emerging FDI trends"). Without this, the router
    # keyword-searches insight tables and synthesizes "the trends"
    # from whichever 5 rows contain the words — e.g. an answer about
    # global investment trends fixated on EdTech because a couple of
    # education-sector insight rows matched "trends".
    re.compile(
        r"\b(global|emerging|new|latest|macro|current|future)\s+"
        r"(\w+\s+)?trends?\b|"
        r"\btrends?\b.{0,50}\b(investment|investors?|fdi|capital|"
        r"economy|markets?)\b",
        re.I | re.DOTALL,
    ),
    # analytical-synthesis asks ("develop the dynamic between MNCs
    # and asset managers", "how will X be reflected in Y", "the
    # interplay between sovereign funds and ..."). These are THINK
    # questions — treating them as entity lookups dead-ends with
    # 'No record matching "<whole question>" was found'.
    re.compile(
        r"\b(dynamics?|relationship|interplay|interaction|nexus|"
        r"convergence)\s+between\b",
        re.I,
    ),
    re.compile(
        r"\bhow\s+(will|would|might|could)\b.{0,90}\bbe\s+"
        r"(reflected|impacted|affected|shaped|influenced)\b",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"\bdevelop\s+(the|a|an)\s+(dynamic|thesis|narrative|"
        r"analysis|perspective|view)\b",
        re.I,
    ),
    # Corridor / origin-market strategy without an explicit deliverable
    # noun ("investment opportunities from Japan", "FDI strategy for
    # Korea", "how can Saudi attract French manufacturers").
    re.compile(
        r"\b(fdi|foreign\s+direct\s+investment|outbound\s+invest|"
        r"inbound\s+invest|investment\s+opportunit|"
        r"market\s+entry|soft.?landing|localisation\s+partner)\b",
        re.I,
    ),
    re.compile(
        r"\b(how\s+(can|should|do)\s+(saudi|ksa|misa|we)\b.{0,40}"
        r"\bat+r+act)|"
        r"\b(attract(?:ing)?\s+\w+\s+(?:compan|invest|firm))",
        re.I | re.DOTALL,
    ),
]

# Count / browse questions must keep their deterministic routes even if
# they mention attraction verbs ("how many investors did we attract?").
_ADVISORY_EXCLUDE_RE = re.compile(
    r"^\s*(how\s+many|count\s+of|total\s+number|show\s+me|list\s+)", re.I,
)


def _detect_origin_country(user_question: str) -> str | None:
    """Detect the origin (source-market) country named in a question,
    via adjective form ('Japanese'), short alias ('USA', 'UK', 'UAE'),
    or noun form ('Japan'). Saudi Arabia is the destination market,
    never the origin."""
    q = (user_question or "").lower()
    # 1. Adjective map ("Indian" → "India", "American" → "United States" …)
    for adj, noun in _COUNTRY_ADJECTIVE_TO_NOUN.items():
        if noun == "Saudi Arabia":
            continue
        if re.search(rf"\b{re.escape(adj)}\b", q):
            return noun
    # 1b. Bare "US" / "U.S" as a market label — NOT the pronoun "us".
    # Require a market noun after it (companies/firms/…) or a from-phrase.
    if re.search(
        r"\b(?:u\.?s\.?a?\.?)\s+"
        r"(?:compan(?:y|ies)|firms?|corporations?|investors?|"
        r"businesses|enterprises?|market|mncs?|smes?)\b",
        q,
    ) or re.search(
        r"\b(?:from|in|of)\s+(?:the\s+)?u\.?s\.?a?\.?\b",
        q,
    ):
        return "United States"
    # 2. Short aliases — abbreviations not in CANONICAL_COUNTRIES
    #    ("usa", "uk", "uae", "america"). Whole-word match only.
    for alias, noun in _COUNTRY_SHORT_ALIASES.items():
        if noun == "Saudi Arabia":
            continue
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return noun
    # 3. Full noun names from the canonical list ("Japan", "France" …)
    try:
        from app.data.countries import CANONICAL_COUNTRIES
        for name in CANONICAL_COUNTRIES:
            if name == "Saudi Arabia" or len(name) < 4:
                continue
            if re.search(rf"\b{re.escape(name.lower())}\b", q):
                return name
    except Exception:
        pass
    return None


def _is_advisory_question(user_question: str) -> bool:
    """True when the question asks for investment-attraction strategy
    at a topic (country/sector) level, not a row lookup."""
    q = (user_question or "").strip()
    if not q or _ADVISORY_EXCLUDE_RE.search(q):
        return False
    if any(pat.search(q) for pat in _ADVISORY_QUESTION_PATTERNS):
        return True
    # Deliverable-shaped asks ("develop an engagement plan with
    # Japan", "top sectors for France") name the artefact rather than
    # an attraction verb, so the patterns above miss them. When the
    # ask names a recognised advisory deliverable AND an origin
    # country, it is a strategy question — without this, the entity
    # machinery hijacks 'Japan' into a company disambiguation
    # ("did you mean Japan Post Holdings?"). Company-scoped plans
    # ("engagement plan for Aramco") have no origin country and
    # correctly stay on the row-grounded engagement_strategy route.
    if (_detect_advisory_deliverable(q) != "strategy_analysis"
            and _detect_origin_country(q)):
        return True
    # Origin market + investment language → Jul21 advisory, not thin GK
    # or a country overview dump. Skip pure "tell me about Germany".
    try:
        from app.services.jul21_surface import looks_like_corridor_investment_ask
        if looks_like_corridor_investment_ask(q):
            if not re.search(
                r"(?i)^\s*(tell\s+me\s+about|what\s+about|overview\s+of|"
                r"profile\s+of)\s+",
                q,
            ):
                return True
    except Exception:
        pass
    return False


# The advisory document must match the ARTEFACT the user asked for:
# "develop an engagement plan" needs phases/stakeholders/KPIs, not the
# market-fit assessment shape. Ordered — first match wins; plan words
# are checked before assessment words because a question can contain
# both ("plan based on market fit") and the actionable artefact wins.
_ADVISORY_DELIVERABLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("engagement_plan", re.compile(
        r"\b(engagement|action|outreach|attraction|investment)\s+plan\b|"
        r"\broad\s*map\b|\bcampaign\b|"
        r"\b(develop|create|build|make|draft|prepare|design)\b"
        r".{0,30}\bplan\b|"
        r"\bplan\s+(for|to)\s+at+r+act",
        re.I | re.DOTALL,
    )),
    # Market-fit BEFORE company-targeting so "make me a market for to
    # attract Indian companies" is a strategy assessment, not a ranking.
    ("market_fit", re.compile(
        r"\bmarket\s+(fit|for)\b|\bfit\s+assessment\b|"
        r"\bmake\s+(me\s+)?(a\s+)?market\b.{0,80}\bat+r+act|"
        r"\bmarket\s+(entry|attraction)\s+strateg",
        re.I | re.DOTALL,
    )),
    # Company-list + investment thesis (before sector_priorities so
    # "best companies … thesis" does not become a sector essay).
    ("company_targeting", re.compile(
        r"\b(investment\s+thesis|theses)\b|"
        r"\b(best|top|priority|target)\s+compan(?:y|ies)\b.{0,80}\b"
        r"(from|in|of)\b|"
        r"\bcompan(?:y|ies)\s+to\s+(?:be\s+)?(?:target|prioriti[sz]e|"
        r"at+r+act|engage)\b|"
        r"\btarget(?:ed|ing)?\s+(?:list|companies)\b",
        re.I | re.DOTALL,
    )),
    ("sector_priorities", re.compile(
        r"\b(top|priority|key|focus|best|main|promising)\s+sectors?\b|"
        r"\b(which|what)\s+sectors?\b|"
        r"\bsectors?\s+.{0,30}\b(focus|prioriti[sz]e|target)\b",
        re.I | re.DOTALL,
    )),
]


_MARKET_SEGMENT_RE = re.compile(
    r"\b(compan(?:y|ies)|firms?|manufacturers?|investors?|businesses|"
    r"enterprises?|corporations?|startups?|sectors?|industries|"
    r"conglomerates?|multinationals?|mncs?|sme[s]?|"
    r"asset\s+managers?)\b",
    re.IGNORECASE,
)


def _looks_like_market_segment(target: str) -> bool:
    """True when a deep-profile target is actually a market CLASS
    ('German manufacturers', 'trillion-dollar MNCs') rather than a
    single named entity ('Microsoft', 'Tim Cook')."""
    return bool(_MARKET_SEGMENT_RE.search(target or ""))


def _detect_advisory_deliverable(user_question: str) -> str:
    """Classify which advisory artefact the user asked for. Defaults to
    'strategy_analysis' (adaptive structure) when neither a plan nor a
    fit assessment is explicitly requested."""
    q = (user_question or "").strip()
    for label, pat in _ADVISORY_DELIVERABLE_PATTERNS:
        if pat.search(q):
            return label
    return "strategy_analysis"


def _advisory_country_context(user_question: str) -> dict | None:
    """Best-effort MISA-database grounding for the advisory report:
    detect the origin country named in the question (adjective or noun
    form), then pull its Saudi-presence stats (licensed count, RHQ
    count, top companies). Returns None when no country is detected;
    DB errors degrade to a country-only dict so the report still knows
    its subject."""
    country = _detect_origin_country(user_question)
    if not country:
        return None
    ctx: dict = {"origin_country": country}
    try:
        from app.services.engagement_data import fetch_country_saudi_investors
        stats = fetch_country_saudi_investors(country)
        if stats.get("_db_error"):
            # DB/schema failure — NEVER pass zeros (model invents
            # "0 Indian companies licensed" from them).
            ctx["footprint_data_unavailable"] = True
            ctx["retrieval_status"] = (
                stats.get("retrieval_status")
                or "SOURCE_UNAVAILABLE"
            )
            ctx["_db_error"] = stats.get("_db_error")
            ctx["retrieval"] = stats.get("retrieval") or {
                "retrieval_status": ctx["retrieval_status"],
                "counts_unavailable": True,
                "do_not_claim_zero": True,
                "error": stats.get("_db_error"),
            }
            try:
                from app.logger import logger as _log
                _log.warning(
                    "advisory_footprint UNAVAILABLE country=%r err=%s",
                    country, stats.get("_db_error"),
                )
            except Exception:
                pass
        elif (
            int(stats.get("total_licensed") or 0) == 0
            and int(stats.get("total_rhq") or 0) == 0
        ):
            ctx["companies_from_origin_licensed_in_saudi"] = 0
            ctx["companies_from_origin_with_rhq"] = 0
            ctx["retrieval_status"] = "SUCCESS_EMPTY"
            ctx["retrieval_filters"] = {
                "origin_country": country,
                "source": "company_profiles + nationality/origin join",
            }
            ctx["top_rhq_companies"] = []
            ctx["top_licensed_companies"] = []
            ctx["expansion_targets"] = []
        else:
            ctx["companies_from_origin_licensed_in_saudi"] = stats.get(
                "total_licensed")
            ctx["companies_from_origin_with_rhq"] = stats.get("total_rhq")
            ctx["retrieval_status"] = (
                stats.get("retrieval_status") or "SUCCESS_WITH_RESULTS"
            )
            ctx["retrieval_filters"] = {
                "origin_country": country,
                "source": "company_profiles.licensed / is_rhq",
            }
            if stats.get("retrieval"):
                ctx["retrieval"] = stats["retrieval"]
            elif stats.get("retrieval_status"):
                ctx["retrieval"] = {
                    "retrieval_status": stats["retrieval_status"],
                    "source_name": "company_profiles.licensed/is_rhq",
                    "record_count": int(stats.get("total_licensed") or 0),
                }
            ctx["top_rhq_companies"] = [
                {"name": r.get("company_name"), "industry": r.get("industry"),
                 "annual_revenue": r.get("annual_revenue")}
                for r in (stats.get("rhq") or [])[:8]
            ]
            ctx["top_licensed_companies"] = [
                {"name": r.get("company_name"), "industry": r.get("industry"),
                 "annual_revenue": r.get("annual_revenue")}
                for r in (stats.get("licensed_only") or [])[:8]
            ]
            try:
                from app.services.target_ranking import rank_expansion_targets
                from app.logger import logger as _log
                expansion = rank_expansion_targets(stats)
                ctx["expansion_targets"] = expansion
                _log.info(
                    "advisory_footprint country=%r licensed=%s rhq=%s "
                    "expansion_targets=%s",
                    country,
                    ctx.get("companies_from_origin_licensed_in_saudi"),
                    ctx.get("companies_from_origin_with_rhq"),
                    len(expansion),
                )
            except Exception:
                pass
    except Exception as exc:
        # DB unreachable — MISSING data must never masquerade as ZERO.
        ctx["footprint_data_unavailable"] = True
        ctx["retrieval_status"] = "SOURCE_UNAVAILABLE"
        ctx["_db_error"] = str(exc)
        ctx["retrieval"] = {
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "counts_unavailable": True,
            "do_not_claim_zero": True,
            "error": str(exc),
        }
    # Sector distribution — DB evidence for "which sectors convert".
    try:
        from app.services.engagement_data import (
            fetch_country_sector_distribution,
        )
        dist = fetch_country_sector_distribution(country)
        if dist.get("_db_error"):
            ctx["sector_distribution_unavailable"] = True
            ctx["_sector_db_error"] = dist.get("_db_error")
            ctx["sector_retrieval_status"] = dist.get("retrieval_status")
        elif dist.get("sectors"):
            ctx["licensed_sector_distribution"] = dist["sectors"]
    except Exception as exc:
        ctx["sector_distribution_unavailable"] = True
        ctx["_sector_db_error"] = str(exc)
    # Country-level intelligence (vision outlook + strategic
    # opportunities captured by MISA analysts) — the insight layer
    # that keeps the report from reading like generic market prose.
    try:
        from app.services.engagement_data import resolve_country_id
        country_id, _canon = resolve_country_id(country)
        if country_id:
            conn = get_db()
            import psycopg2.extras as _pe
            with conn.cursor(cursor_factory=_pe.RealDictCursor) as cur:
                cur.execute(
                    "SELECT national_vision, diversification_goals, "
                    "five_year_outlook FROM country_vision_outlooks "
                    "WHERE country_profile_id = %s LIMIT 1", (country_id,))
                r = cur.fetchone()
                if r:
                    ctx["origin_country_vision_outlook"] = {
                        k: (str(v)[:600] if v else None)
                        for k, v in dict(r).items()
                    }
                cur.execute(
                    "SELECT category, description FROM "
                    "country_strategic_opportunities "
                    "WHERE country_profile_id = %s LIMIT 5", (country_id,))
                opps = [dict(x) for x in cur.fetchall()]
                if opps:
                    ctx["origin_country_strategic_opportunities"] = [
                        {"category": o.get("category"),
                         "description": str(o.get("description") or "")[:400]}
                        for o in opps
                    ]
    except Exception:
        pass
    return ctx


# ─── Friendly tool error messages ─────────────────────────────────────
# When a tool_call fails (SQL error, network, missing table) the user
# must NEVER see the raw exception. We classify common error shapes
# into user-grade hints and emit those instead. The raw error stays
# in the tool_call's `error_raw` field for debug mode.

# Specific SQL signatures we want to handle with tailored messages.
_SQL_ERROR_PATTERNS: list[tuple[re.Pattern, str]] = [
    # entity name used where a numeric FK is expected — most common
    # symptom of the LLM-routed tool call skipping entity → ID
    # resolution. Hint the user toward a better-shaped question.
    (re.compile(r"invalid input syntax for type bigint", re.I),
     "I couldn't directly look up {entity_label} in `{tbl}` because "
     "that table needs a numeric ID, not a name. Try asking "
     "**\"tell me about {entity_clean}\"** for the full profile "
     "(which I'll join correctly), or be more specific about what "
     "you're looking for."),
    # Numeric out-of-range
    (re.compile(r"out of range|integer overflow", re.I),
     "I couldn't run that lookup against `{tbl}` — the value was "
     "outside the expected range. Try simplifying the question."),
    # Permission / relation missing — backend config issue, never the user's fault
    (re.compile(r"permission denied|relation .* does not exist", re.I),
     "That information isn't available in the current data set."),
    # Connection / transient
    (re.compile(r"connection|timeout|terminated", re.I),
     "The database is briefly unreachable. Please retry in a moment."),
]


def _friendly_tool_error_message(table: str | None, raw_err: str,
                                  entity: str | None) -> str:
    """Map a raw backend error to a user-grade message. Always returns
    something polite — never propagates SQL text or stack traces."""
    tbl = table or "the requested data"
    entity_clean = (entity or "this entity").strip()
    # Strip "tell me about" / question shape from entity for the
    # follow-up suggestion — leave the bare noun.
    entity_clean = re.sub(
        r"^(tell me about|what is|who is|where is|how is)\s+", "",
        entity_clean, flags=re.I,
    ).strip()
    entity_label = f"**\"{entity_clean}\"**" if entity_clean else "that entity"
    for pat, template in _SQL_ERROR_PATTERNS:
        if pat.search(raw_err):
            return template.format(
                tbl=tbl, entity_label=entity_label,
                entity_clean=entity_clean or "the company",
            )
    # Generic fallback — opaque but polite.
    return (
        f"I had trouble pulling data from `{tbl}` for this question. "
        "Try rephrasing, or ask about a specific company by name."
    )


# ─── Sector aggregation direct path ──────────────────────────────────
# "give me the momentum for all the sectors" / "top sectors by
# activity" / "sector breakdown" → run a real SQL aggregation
# against opportunities.sector_name and return ranked results.
#
# Detection regex catches the aggregate / ranking / momentum / overview
# / breakdown / activity / pipeline language for sectors.
_SECTOR_AGGREGATION_PATTERNS = [
    re.compile(
        r"\b(all|every|each|every\s+single|top|leading|ranking|"
        r"rank(ed|ing)?|breakdown|summary|overview|pipeline|"
        r"momentum|activity|distribution|across|by\s+sector)\b.*"
        r"\bsectors?\b",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"\bsectors?\b.*"
        r"\b(all|every|each|top|leading|ranking|rank(ed|ing)?|"
        r"breakdown|summary|overview|pipeline|momentum|activity|"
        r"distribution|ranked\s+by)\b",
        re.I | re.DOTALL,
    ),
    # "what sectors are most active" / "which sectors have most"
    re.compile(
        r"\b(what|which)\s+sectors?\b.*"
        r"\b(most|highest|biggest|largest|leading|top)\b",
        re.I | re.DOTALL,
    ),
]


_SINGLE_SECTOR_NAMES = (
    "ict", "information and communication technology", "technology", "tech",
    "healthcare", "health care", "health", "pharma", "pharmaceutical",
    "energy", "renewable", "renewables", "oil", "gas", "petrochemical",
    "agriculture", "agri", "food", "food processing",
    "tourism", "hospitality", "real estate", "construction", "infrastructure",
    "manufacturing", "industrial", "mining", "metals", "aerospace", "defense",
    "defence", "automotive", "logistics", "transport", "transportation",
    "finance", "financial", "fintech", "banking", "education", "training",
    "media", "entertainment", "telecom", "telecommunications",
)


def _is_sector_aggregation_question(user_question: str) -> bool:
    """True when the question asks about MULTIPLE sectors or sector
    activity in aggregate — not a single named sector. Used to route
    sector_lookup intent through the aggregation path instead of the
    single-sector taxonomy path.

    Guards against false positives: when the question names a SPECIFIC
    sector (e.g. "sector breakdown for ICT"), the user wants a single-
    sector deep-dive, not an aggregation. The single-sector taxonomy
    path handles those.
    """
    if not user_question:
        return False
    q_lower = user_question.lower()
    # If the question explicitly names a single sector, treat as
    # single-sector lookup (not aggregation).
    for sector_name in _SINGLE_SECTOR_NAMES:
        if re.search(rf"\b{re.escape(sector_name)}\b", q_lower):
            return False
    for pat in _SECTOR_AGGREGATION_PATTERNS:
        if pat.search(user_question):
            return True
    return False


# ─── Pre-off-topic intercept handlers ────────────────────────────────
# Three benign cases the off_topic classifier was mislabeling as
# refusal-worthy:
#   1. capability questions ("what can you do for us")
#   2. too-short / punctuation-only inputs ("x", "??!!", " ")
#   3. vague follow-ups ("tell me more") with conversation history
# Each gets a specific helpful response instead of a polite-refusal.

_CAPABILITY_PATTERNS = re.compile(
    r"^\s*("
    r"what\s+(can\s+)?you\s+(do|help)(\s+for\s+(me|us|misa))?|"
    r"how\s+(can|do)\s+you\s+help|"
    r"how\s+(do|does)\s+(you|this)\s+work|"
    r"what\s+(are\s+you|is\s+this)|"
    r"what\s+(topics|areas|domains|things|kinds?)\s+(can|do)|"
    r"capabilities|"
    r"help\s*$|"
    r"menu|"
    r"getting\s+started|"
    r"how\s+to\s+use"
    r")\s*\??\s*$",
    re.I,
)

_VAGUE_FOLLOWUP_PATTERNS = re.compile(
    r"^\s*("
    r"tell\s+me\s+more|"
    r"go\s+on|"
    r"continue|"
    r"keep\s+going|"
    r"expand|"
    r"elaborate|"
    r"more\s+details?|"
    r"more\s+info(rmation)?|"
    r"and\??|"
    r"go\s+deeper|"
    r"dig\s+deeper"
    r")\s*\??\s*$",
    re.I,
)


def _is_capability_question(q: str) -> bool:
    return bool(q and _CAPABILITY_PATTERNS.match(q))


def _is_too_short_or_meaningless(q: str) -> bool:
    """True for empty / 1-char / punctuation-only inputs that the
    classifier would mislabel as off_topic but really just need a
    'what would you like to know?' nudge."""
    if not q:
        return True
    s = q.strip()
    if len(s) < 3:
        return True
    # Only punctuation / symbols, no letters
    if not re.search(r"[A-Za-z؀-ۿ]", s):
        return True
    return False


def _is_vague_followup(q: str) -> bool:
    return bool(q and _VAGUE_FOLLOWUP_PATTERNS.match(q))


def _last_entity_from_history(history: list) -> str | None:
    """Scan the most-recent user turns for a likely entity reference
    (a Capitalised proper-noun phrase). Used to anchor 'tell me more'
    follow-ups to whatever was just discussed."""
    if not history:
        return None
    for msg in reversed(history):
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # Look for "about X" / "of X" / Title Case spans
        m = re.search(r"\babout\s+([A-Z][\w&.,\- ']{2,60})", content)
        if m:
            return m.group(1).strip().rstrip(".,?!:;")
        # Fallback: first 2-3 Title Case word run
        m = re.search(r"\b([A-Z][\w&.]+(?:\s+[A-Z][\w&.]+){0,3})\b", content)
        if m:
            return m.group(1).strip()
    return None


def _rewrite_vague_followup(user_question: str, history: list) -> str | None:
    """Given a vague follow-up like 'tell me more', construct a
    concrete question by pulling the last entity from history. Returns
    None when no entity can be found (caller falls through to off-topic)."""
    entity = _last_entity_from_history(history)
    if not entity:
        return None
    return f"Tell me more about {entity}"


def _capability_response(pack: dict, ui_locale: str, resp_loc: str,
                          user_question: str) -> dict:
    """Friendly capability-overview answer. Lists 4 representative
    example questions so the user can copy/click straight in."""
    pack["_short_circuit"] = "capability"
    answer = (
        "## What I can help with\n\n"
        "I'm MISA's executive intelligence assistant. I answer questions "
        "grounded in MISA's own database (~250K companies, executives, "
        "MISA pipeline opportunities, country macros, RHQ licences, "
        "meeting and engagement history). Examples:\n\n"
        "- **Company briefings** — *Tell me about Apple*\n"
        "- **Executive lookups** — *Who is the CEO of Aramco*\n"
        "- **Country profiles** — *Tell me about Pakistan*\n"
        "- **Sector activity** — *Top sectors by MISA opportunity count*\n"
        "- **Engagement strategy** — *How should MISA attract Chinese investment from Europe*\n"
        "- **Saudi RHQ** — *Does Microsoft have an RHQ in Riyadh*\n"
        "- **Deep profiles** — Type `/profile <Company>` for the full "
        "3-pillar briefing with web grounding.\n\n"
        "I won't answer non-MISA questions (weather, jokes, general "
        "trivia), and I'll explicitly tell you when data is missing "
        "rather than guess."
    )
    return {
        "answer": answer,
        "tool_calls": [{"input_trace": dict(pack)}],
        "error": None,
        "_answer_source": "capability_response",
        "feedback_context": hf.build_feedback_context(
            user_question, ui_locale, resp_loc, pack,
        ),
    }


def _clarification_request_response(pack: dict, ui_locale: str,
                                      resp_loc: str, user_question: str) -> dict:
    """For too-short / punctuation-only inputs, ask what they meant
    and surface a few starter questions."""
    pack["_short_circuit"] = "needs_clarification"
    answer = (
        "## What would you like to know?\n\n"
        "Your message was very short — could you rephrase? I work best "
        "with full questions about MISA's data. A few examples:\n\n"
        "- *Tell me about Apple*\n"
        "- *Top sectors by opportunity count*\n"
        "- *Who is the CEO of Aramco*\n"
        "- *How should MISA attract Chinese investment*"
    )
    return {
        "answer": answer,
        "tool_calls": [{"input_trace": dict(pack)}],
        "error": None,
        "_answer_source": "clarification_request",
        "feedback_context": hf.build_feedback_context(
            user_question, ui_locale, resp_loc, pack,
        ),
    }


# ─── Single-sector opportunity path ──────────────────────────────────
# Detects "opportunities in [sector]" / "most attractive opportunities
# for [sector]" / "what's in the [sector] pipeline" — questions about
# ONE specific sector's MISA opportunity activity. Runs:
#
#   SELECT count, stage breakdown, status breakdown, source breakdown
#   FROM opportunities WHERE sector_name ILIKE '%X%'
#
# Returns aggregated stats. The opportunities table has 3,121 rows
# but NO titles populated — surfaces the count + stage/status mix
# instead of attempting to list named items the DB doesn't have.
# (Rule 7 — gap analysis: surface what's missing.)

_SINGLE_SECTOR_OPP_PATTERNS = [
    # "opportunities in the X sector" / "in/for/across X"
    re.compile(
        r"\b(opportunit(y|ies)|pipeline|deal\s*flow|projects?)\b"
        r".{0,80}\b(in|for|within|across)\s+(?:the\s+)?",
        re.I | re.DOTALL,
    ),
    # "X pipeline" / "X opportunities" — reversed order
    re.compile(
        r"\b(in|for|within|inside)\s+(?:the\s+)?\w+\s+"
        r"(pipeline|opportunit(y|ies)|projects?|deal\s*flow)\b",
        re.I,
    ),
    # "What's in the X pipeline" / "the X opportunities"
    re.compile(
        r"\b(the\s+)?\w+\s+(pipeline|opportunit(y|ies))\b",
        re.I,
    ),
    re.compile(
        r"\b(most\s+attractive|top|key|major|main|leading)\s+"
        r"(opportunit(y|ies)|projects?)\b",
        re.I,
    ),
]


def _is_single_sector_opportunity_question(user_question: str) -> bool:
    """True when the question asks about opportunities in a specific
    sector (not aggregate across all sectors)."""
    if not user_question:
        return False
    # Must NOT be the all-sectors aggregation (those go to the existing
    # sector_aggregation path)
    if _is_sector_aggregation_question(user_question):
        return False
    # Must contain a known sector name to anchor on
    q_lower = user_question.lower()
    if not any(re.search(rf"\b{re.escape(s)}\b", q_lower) for s in _SINGLE_SECTOR_NAMES):
        return False
    # Must look opportunity-shaped
    for pat in _SINGLE_SECTOR_OPP_PATTERNS:
        if pat.search(user_question):
            return True
    return False


def _try_single_sector_opportunity_direct(
    user_question: str, pack: dict, client, model: str,
) -> list | None:
    """Aggregate opportunities for a single sector. Returns synthetic
    tool_call with summary stats. Pulls top 50 raw rows for the
    curator to reference (stage / status / opportunity_id), since
    the title field is empty across the dataset."""
    from app.database import get_db
    from psycopg2.extras import RealDictCursor

    # Extract the sector mention from the question
    q_lower = user_question.lower()
    matched_sector = None
    for s in _SINGLE_SECTOR_NAMES:
        if re.search(rf"\b{re.escape(s)}\b", q_lower):
            matched_sector = s
            break
    if not matched_sector:
        return None

    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Resolve the canonical sector_name as it appears in opportunities
        cur.execute("""
            SELECT sector_name, COUNT(*) AS count
            FROM opportunities
            WHERE sector_name ILIKE %s
            GROUP BY sector_name
            ORDER BY count DESC LIMIT 1
        """, (f"%{matched_sector}%",))
        sector_row = cur.fetchone()
        if not sector_row:
            cur.close()
            return None
        canonical_sector = sector_row["sector_name"]
        total = int(sector_row["count"])

        # Stage breakdown
        cur.execute("""
            SELECT stage, COUNT(*) AS n FROM opportunities
            WHERE sector_name = %s
            GROUP BY stage ORDER BY n DESC NULLS LAST
        """, (canonical_sector,))
        stages = [dict(r) for r in cur.fetchall()]

        # Status breakdown
        cur.execute("""
            SELECT status, COUNT(*) AS n FROM opportunities
            WHERE sector_name = %s
            GROUP BY status ORDER BY n DESC NULLS LAST
        """, (canonical_sector,))
        statuses = [dict(r) for r in cur.fetchall()]

        # Source breakdown (where the opportunities came from)
        cur.execute("""
            SELECT opportunity_source_name, COUNT(*) AS n FROM opportunities
            WHERE sector_name = %s
            GROUP BY opportunity_source_name ORDER BY n DESC NULLS LAST
        """, (canonical_sector,))
        sources = [dict(r) for r in cur.fetchall()]

        # Top 20 raw rows for curator context (opportunity_id, stage,
        # status) — since titles are empty, this is the most useful
        # detail we can offer.
        cur.execute("""
            SELECT opportunity_id, stage, status, opportunity_source_name,
                   created_at
            FROM opportunities
            WHERE sector_name = %s
            ORDER BY created_at DESC NULLS LAST
            LIMIT 20
        """, (canonical_sector,))
        sample_rows = [dict(r) for r in cur.fetchall()]
        cur.close()
    except Exception as e:
        import sys
        print(f"[single_sector_opp] failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None

    pack["_single_sector_opp_mode"] = True
    pack["_sector_canonical"] = canonical_sector
    pack["_sector_total_opps"] = total

    # One synthetic tool_call carrying everything the curator needs
    summary_rows = [{
        "sector": canonical_sector,
        "total_opportunities": total,
        "stage_breakdown": stages,
        "status_breakdown": statuses,
        "source_breakdown": sources,
        "sample_opportunity_ids": [r.get("opportunity_id")
                                     for r in sample_rows if r.get("opportunity_id")][:20],
        "data_note": ("Opportunity titles are not populated in the MISA "
                       "database — only IDs, stages, statuses, and sources "
                       "are available. Surface stats and stage breakdown, "
                       "not a list of named items."),
    }]
    return [_build_engagement_tool_call(
        "opportunities", summary_rows,
        {"_single_sector_opp": True, "sector": canonical_sector,
         "total": total},
        pack,
    )]


# ─── Top-companies-per-sector follow-up handler ────────────────────
# Detects multi-turn follow-ups like "show me the top 10 companies
# from each of these sectors" / "top companies per sector" after a
# previous sector-aggregation turn. Returns a direct answer using
# company_profiles.sector_id joined with sectors table — but is
# transparent about the data gap: only ~870 of 250K+ companies
# in our DB have sector_id tagged. The answer surfaces which
# sectors we CAN provide company-level data for, names the gap
# for the rest, and offers a concrete next step.

_TOP_COMPANIES_PER_SECTOR_PATTERNS = [
    # Must include a multi-sector signal: each / per / across / all sector(s)
    re.compile(r"\bcompanies?\s+(from|in|per|across)\s+each\s+sectors?\b", re.I),
    re.compile(r"\bcompanies?\s+per\s+sector\b", re.I),
    re.compile(r"\bcompanies?\s+across\s+(all|every)\s+sectors?\b", re.I),
    re.compile(r"\b(probable\s+)?top\s+(\d+\s+)?companies?\s+from\s+each\s+(of\s+(these|those|the)\s+)?sectors?\b", re.I),
    re.compile(r"\b(top|leading|main|key)\s+companies?\s+(in|from|across)\s+each\s+(sector|industry)\b", re.I),
    re.compile(r"\b(list|show|give)\s+(me\s+)?(the\s+)?(top|main|leading|key)\s+companies?\s+(from|in|by|per|across)\s+(each|every|all)\s+sectors?\b", re.I),
    # "top companies in each of these sectors" / "across all sectors"
    re.compile(r"\bcompanies?\s+in\s+each\s+of\s+(these|those|the)\s+sectors?\b", re.I),
    re.compile(r"\bcompanies?\s+across\s+sectors?\b", re.I),
    re.compile(r"\bleading\s+companies?\s+across\s+(all\s+)?sectors?\b", re.I),
]


def _is_top_companies_per_sector_question(user_question: str) -> bool:
    """True when the question asks for top companies grouped or split
    across multiple sectors. Often a follow-up after a sector
    aggregation turn ("these sectors", "each of these")."""
    if not user_question:
        return False
    for pat in _TOP_COMPANIES_PER_SECTOR_PATTERNS:
        if pat.search(user_question):
            return True
    return False


def _try_top_companies_per_sector_direct(
    user_question: str, pack: dict, client, model: str,
) -> list | None:
    """For each sector that has companies tagged in company_profiles,
    pull the top N by annual_revenue (then market_cap as tiebreaker).
    Returns a synthetic tool_call with one row per (sector, company)
    pair, plus a final "data gap" row that the curator surfaces
    explicitly.

    Returns None on failure → caller falls through to normal path.
    """
    from app.database import get_db
    from psycopg2.extras import RealDictCursor
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Get the TOP sectors by company count (only sectors WITH
        # companies appear; sectors with all-NULL sector_id are absent).
        cur.execute("""
            SELECT s.name AS sector_name, COUNT(cp.id) AS company_count
            FROM company_profiles cp
            JOIN sectors s ON cp.sector_id = s.id
            GROUP BY s.name
            ORDER BY company_count DESC
            LIMIT 10
        """)
        sectors_with_data = list(cur.fetchall())
        if not sectors_with_data:
            cur.close()
            return None
        # For each of those sectors, top 10 companies by revenue
        rows = []
        for s in sectors_with_data:
            cur.execute("""
                SELECT cp.company_name, cp.annual_revenue, cp.market_cap,
                       cp.role, cp.licensed, cp.is_rhq
                FROM company_profiles cp
                JOIN sectors s ON cp.sector_id = s.id
                WHERE s.name = %s
                  AND cp.company_name IS NOT NULL
                  AND cp.company_name != ''
                ORDER BY cp.annual_revenue DESC NULLS LAST,
                         cp.market_cap DESC NULLS LAST
                LIMIT 10
            """, (s["sector_name"],))
            for r in cur.fetchall():
                # Canonical licensing markers: licensed / is_rhq booleans.
                _is_licensed = bool(r.get("licensed"))
                _is_rhq = bool(r.get("is_rhq"))
                rows.append({
                    "sector": s["sector_name"],
                    "sector_company_count_in_db": s["company_count"],
                    "company_name": r["company_name"],
                    "annual_revenue": r.get("annual_revenue"),
                    "market_cap": r.get("market_cap"),
                    "rhq_licensed": _is_rhq,
                    "misa_licensed": _is_licensed,
                })
        # Also surface a "data gap" indicator: number of total companies
        # in DB with NO sector tag (so the curator can explain why some
        # sectors have no entries).
        cur.execute("""
            SELECT COUNT(*) AS untagged
            FROM company_profiles WHERE sector_id IS NULL
        """)
        gap_row = cur.fetchone()
        cur.close()
    except Exception as e:
        import sys
        print(f"[top_companies_per_sector] failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None

    if not rows:
        return None

    pack["_top_companies_per_sector_mode"] = True
    pack["_sectors_covered"] = [s["sector_name"] for s in sectors_with_data]
    pack["_untagged_companies"] = int(gap_row["untagged"]) if gap_row else 0

    return [_build_engagement_tool_call(
        "company_profiles",
        rows,
        {"_top_companies_per_sector": True,
         "sectors_with_data": [s["sector_name"] for s in sectors_with_data],
         "untagged_companies_count": int(gap_row["untagged"]) if gap_row else 0},
        pack,
    )]


def _try_sector_aggregation_direct(
    user_question: str, pack: dict, client, model: str,
) -> list | None:
    """Direct SQL aggregation for sector-level questions. Runs:

      SELECT sector_name, COUNT(*) AS opportunity_count
      FROM opportunities
      WHERE sector_name IS NOT NULL AND sector_name != ''
      GROUP BY sector_name
      ORDER BY opportunity_count DESC
      LIMIT 15

    Returns a list of tool_calls (just one — a single synthetic table
    with the aggregated rows) that the curator composes into a ranked
    sector activity briefing.

    Returns None on any failure so the caller falls through to the
    regular sector_lookup taxonomy path.
    """
    from app.database import get_db
    from psycopg2.extras import RealDictCursor
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT sector_name,
                   COUNT(*) AS opportunity_count
            FROM opportunities
            WHERE sector_name IS NOT NULL AND sector_name != ''
            GROUP BY sector_name
            ORDER BY opportunity_count DESC
            LIMIT 15
        """)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
    except Exception as e:
        # Surface the failure on stderr — silent fall-through here used
        # to mask the cursor-factory bug for a full debug cycle.
        import sys
        print(f"[sector_agg] direct path failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None
    if not rows:
        return None

    pack["_sector_aggregation_mode"] = True
    pack["_sector_aggregation_count"] = len(rows)
    pack["_sector_aggregation_rows"] = rows

    # Build a single synthetic tool_call shaped like the rest of the
    # pipeline expects. The curator sees this as one table of rows
    # (sector_name + opportunity_count) and composes a ranked list.
    return [_build_engagement_tool_call(
        "opportunities",
        rows,
        {"_sector_aggregation": True, "group_by": "sector_name"},
        pack,
    )]


def _format_sector_opportunity_briefing(rows: list[dict]) -> str:
    """Deterministic Jul21-lite sector activity brief from opportunity counts.

    Avoids thin textbook GK when the curator compresses aggregation rows.
    """
    total = sum(int(r.get("opportunity_count") or 0) for r in rows)
    lines = [
        "# Sector Opportunity Priorities (MISA pipeline)",
        "",
        "## Strategic Context",
        "",
        "MISA's live opportunities pipeline is the evidence base for where "
        "investment-attraction capacity is converting today. The ranking "
        "below is ordered by opportunity count in the MISA database — use "
        "it to prioritise desk coverage, roadshows, and account targeting "
        "against Vision 2030 demand anchors (SDAIA / LEAP for digital, "
        "NUPCO for health, NEOM and industrial zones for localisation).",
        "",
        f"**{len(rows)} sectors** carry tagged opportunities "
        f"(**{total:,}** opportunities in total in this cut).",
        "",
        "## Sector Ranking",
        "",
        "| Rank | Sector | MISA opportunities | Priority | Saudi demand anchor |",
        "|---|---|---|---|---|",
    ]
    # Simple demand-anchor heuristic by sector keywords
    def _anchor(sector: str) -> str:
        s = (sector or "").casefold()
        if any(k in s for k in ("ict", "tech", "digital", "software", "telecom")):
            return "SDAIA / LEAP"
        if any(k in s for k in ("health", "pharma", "life", "bio")):
            return "NUPCO / localisation"
        if any(k in s for k in ("energy", "oil", "gas", "power", "renew", "water")):
            return "NEOM / energy transition"
        if any(k in s for k in ("construct", "infra", "real estate", "engineer")):
            return "Giga-projects / NIDLP"
        if any(k in s for k in ("financ", "bank", "insur")):
            return "Financial sector development"
        if any(k in s for k in ("tour", "hospital", "entertain")):
            return "Tourism / entertainment vision"
        if any(k in s for k in ("agro", "food", "agricult")):
            return "Food security / NIDLP"
        if any(k in s for k in ("petro", "chem", "mining", "metal")):
            return "Industrial / PIF zones"
        return "Vision 2030 sector programme"

    for i, r in enumerate(rows[:15], 1):
        sector = str(r.get("sector_name") or "—")
        count = int(r.get("opportunity_count") or 0)
        priority = "Tier 1" if i <= 5 else ("Tier 2" if i <= 10 else "Tier 3")
        lines.append(
            f"| {i} | {sector} | {count:,} | {priority} | {_anchor(sector)} |"
        )

    # Jul21 depth: numbered deep-dives for every Tier-1 sector.
    lines += ["", "## Tier-1 Sector Deep-Dives", ""]
    for i, r in enumerate(rows[:5], 1):
        sector = str(r.get("sector_name") or "sector")
        count = int(r.get("opportunity_count") or 0)
        anchor = _anchor(sector)
        lines += [
            f"### {i}. {sector}",
            "",
            f"**Why it ranks.** MISA's opportunities table carries "
            f"**{count:,}** tagged opportunities in {sector} — evidence "
            f"that attraction capacity is converting in this corridor.",
            "",
            f"**Saudi demand driver.** Anchor outreach to **{anchor}** "
            f"and name the concrete buyer / programme in every account "
            f"conversation.",
            "",
            f"**MISA plays.** (1) Desk sprint on the top opportunity "
            f"accounts within 90 days. (2) Pair each account with a "
            f"named Saudi counterpart under {anchor}. (3) Use LEAP / "
            f"FII / sector exhibitions as commitment forcing functions.",
            "",
        ]

    lines += [
        "",
        "## Recommended Next Actions for MISA",
        "",
    ]
    for r in rows[:5]:
        sector = str(r.get("sector_name") or "sector")
        count = int(r.get("opportunity_count") or 0)
        lines.append(
            f"- Stand up a **{sector}** desk sprint on the top "
            f"opportunity accounts ({count:,} tagged) — map each to a "
            f"named Saudi demand anchor ({_anchor(sector)}) and a 90-day "
            f"outreach calendar."
        )
    lines += [
        "- Publish a one-pager of this ranking for IPA / chamber briefings "
        "ahead of LEAP / FII.",
        "- Flag any Tier-1 sector with thin RHQ coverage for conversion plays.",
        "",
        "_Source: MISA `opportunities` aggregated by `sector_name`._",
        "",
    ]
    return "\n".join(lines)

def _detect_ambiguous_candidates(rows: list[dict], entity: str) -> list[dict] | None:
    """Detect when smart-search returned multiple DISTINCT companies
    whose names all plausibly match the user's entity — in which case
    we should ask for clarification instead of profiling the top one.

    Distinctness is on a normalised name key (suffix-stripped), so
    'Alphabet Inc.' / 'Alphabet, Inc.' / 'Alphabet Inc' all count as
    one entity and DON'T trigger clarification — they're the same
    company recorded with minor spelling variants.

    Heuristics:
      - 2-5 candidates with DISTINCT normalised names
      - The user's entity is short (≤2 tokens) — long multi-word
        queries usually pin down one entity already
      - The top candidate's fuzzy score isn't significantly higher
        than the next candidate's (within 0.10)
    """
    import difflib
    if not entity or not rows or len(rows) < 2:
        return None
    if len(entity.split()) > 2:
        return None
    e = entity.strip().lower()
    if len(e) < 3:
        return None
    # Score each row's head name against the entity; dedup on
    # normalised name key so 'Alphabet Inc.' and 'Alphabet, Inc.' don't
    # both appear as candidates.
    scored: list[tuple[float, dict]] = []
    seen_keys: set[str] = set()
    for r in rows:
        name = (r.get("company_name") or "").strip()
        if not name:
            continue
        key = _name_key_for_dedup(name)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        head = name.split(",", 1)[0].strip().lower()
        ratio = max(
            difflib.SequenceMatcher(None, e, head).ratio(),
            difflib.SequenceMatcher(None, e, name.lower()).ratio(),
        )
        for word in head.split():
            if len(word) < 3:
                continue
            r2 = difflib.SequenceMatcher(None, e, word).ratio()
            if r2 > ratio:
                ratio = r2
        scored.append((ratio, r))
    if len(scored) < 2:
        return None
    scored.sort(key=lambda x: -x[0])
    top, second = scored[0], scored[1]
    if top[0] < 0.55:
        return None
    if (top[0] - second[0]) > 0.10:
        return None
    candidates: list[dict] = []
    for score, r in scored[:5]:
        if score < 0.45:
            break
        candidates.append({
            "name":   r.get("company_name") or "",
            "hq":     r.get("global_headquarters") or r.get("rhq_country") or "",
            "sector": r.get("sector") or "",
        })
    if len(candidates) < 2:
        return None
    return candidates


def _format_filter_summary(filters: dict) -> str:
    """Short human-readable summary of the filters used in a count, for
    the deterministic count answer. Skips internal trace markers."""
    parts: list[str] = []
    for k, v in (filters or {}).items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            op = v.get("op", "=")
            val = v.get("value")
            parts.append(f"{k} {op} {val!r}")
        else:
            parts.append(f"{k} = {v!r}")
    if not parts:
        return "no filters (entire table)"
    return ", ".join(parts)


def _format_count_only_answer(count_results: list, pack: dict) -> str:
    """Deterministic answer for COUNT(*)-only tool results. Skips
    OpenAI — counts are exact and don't need narration."""
    lines: list[str] = ["## Count"]
    for tc in count_results:
        n = tc.get("_count_only_result")
        if n is None:
            err = tc.get("_count_only_error") or "filter invalid"
            lines.append(
                f"- `{tc.get('table')}`: count unavailable — {err}"
            )
            continue
        filt_desc = _format_filter_summary(tc.get("filters") or {})
        lines.append(
            f"- **{n:,}** records in `{tc.get('table')}` "
            f"({filt_desc})"
        )
    if len(count_results) == 1 and count_results[0].get("_count_only_result") is not None:
        n = count_results[0]["_count_only_result"]
        tbl = count_results[0].get("table")
        filt = _format_filter_summary(count_results[0].get("filters") or {})
        return (
            f"## Answer\n"
            f"**{n:,}** records in `{tbl}` ({filt}).\n\n"
            f"_Source: live `SELECT COUNT(*)` against the MISA database._"
        )
    return "\n".join(lines)


def _compose_local_commentary(
    tool_calls_executed: list,
    user_question: str,
    pack: dict,
    *,
    response_locale: str = "en",
) -> str:
    """Compose then run the shared finalize gate (one voice, all paths)."""
    raw = _compose_local_commentary_raw(
        tool_calls_executed, user_question, pack,
        response_locale=response_locale,
    )
    try:
        from app.services.answer_finalize import finalize_answer
        return finalize_answer(raw, user_question=user_question, pack=pack)
    except Exception:
        return raw


def _compose_local_commentary_raw(
    tool_calls_executed: list,
    user_question: str,
    pack: dict,
    *,
    response_locale: str = "en",
) -> str:
    # COUNT-only short-circuit: if every successful tool call is a
    # count-only result, render a deterministic answer without calling
    # OpenAI. Saves a round-trip and is 100% accurate / consistent.
    count_results = [
        tc for tc in tool_calls_executed
        if not tc.get("error") and "_count_only_result" in tc
    ]
    non_count_results = [
        tc for tc in tool_calls_executed
        if not tc.get("error") and "_count_only_result" not in tc
    ]
    if count_results and not non_count_results:
        return _format_count_only_answer(count_results, pack)

    err_lines: list[str] = []
    merged_rows: list[dict] = []
    seen_ids: set = set()
    # For multi-tool-call turns, primary_table drives which curation template
    # is used. Take the FIRST table that returned rows (typically the "anchor"
    # table the model chose — e.g. country_profiles for a country question
    # that also pulled supporting company_profiles rows), not whichever was
    # processed last.
    primary_table = COMPANY_TABLE
    primary_locked = False
    sql_ok = True
    row_sanity = True
    closest: list[str] = []
    entity = pack.get("entity_candidate")
    entity_lookup_required = _entity_requires_sql_constraint(entity)

    for tc in tool_calls_executed:
        tbl = tc.get("table")
        if tc.get("error"):
            # NEVER LEAK RAW SQL / BACKEND ERRORS TO USER.
            # Until 2026-06: raw psycopg2 strings like
            # 'invalid input syntax for type bigint: "Apple" LINE 1: ...'
            # were rendered into the user-facing answer verbatim, looking
            # completely broken and exposing implementation detail. We
            # now classify the error into a friendly category and emit
            # a user-grade message instead. The raw error is preserved
            # in the trace (visible only in debug mode) for diagnostics.
            raw_err = str(tc.get("error") or "")
            tc["error_raw"] = raw_err  # keep for trace / audit
            tc["error"] = None         # so downstream doesn't re-render
            friendly = _friendly_tool_error_message(tbl, raw_err, entity)
            err_lines.append(friendly)
            continue
        if tbl and not primary_locked:
            primary_table = tbl
            primary_locked = True
        if tc.get("sql_entity_check_passed") is False:
            sql_ok = False
        if tc.get("row_entity_sanity_passed") is False:
            row_sanity = False
        cnames = tc.get("closest_names") or []
        if cnames:
            closest = list(cnames)
        df = tc.get("rows_df")
        if df is None or df.empty:
            continue
        for r in df.to_dict(orient="records"):
            rid = r.get("id")
            if rid is not None and rid in seen_ids:
                continue
            if rid is not None:
                seen_ids.add(rid)
            merged_rows.append(r)

    # Risk-20-5: enforce the per-turn aggregate row budget across all
    # query_table calls before these rows drive curation. Truncation is
    # audited inside the helper; normal turns are far under the cap, so the
    # answer is unchanged.
    try:
        from app.services.curation import cap_rows_for_turn
        merged_rows, _was_trunc = cap_rows_for_turn(
            merged_rows, context="chat_engine.merged_rows")
        if _was_trunc:
            pack["_truncated"] = True
            pack["_truncation_reason"] = "row_budget"
    except Exception:
        pass

    entity_matched = sql_ok and row_sanity

    # AMBIGUITY DETECTION: if the user named a specific entity AND we
    # got back multiple distinct candidates with similar fuzzy scores,
    # ask the user to clarify rather than silently picking the top
    # match. This is the rule-5 "if multiple possible matches exist,
    # do not guess" behaviour.
    #
    # GUARD: skip the short-circuit entirely when the question shape
    # is strategic/policy (e.g. "bring Chinese investment from Europe
    # to KSA, what measure should I take?"). In those cases the user
    # is asking about a TOPIC, not an entity — firing clarification on
    # "Chinese" → 5 China-named companies is a misread. Let the intent
    # classifier route to engagement_strategy / general_research instead.
    if (entity and _entity_requires_sql_constraint(entity)
            and primary_table == COMPANY_TABLE and len(merged_rows) >= 2
            and not _is_strategic_policy_question(user_question)
            and not _is_self_reference_entity(entity)):
        candidates = _detect_ambiguous_candidates(merged_rows, entity)
        if candidates:
            # Full company-profile asks: auto-pick the top candidate and
            # continue — don't ship a clarification stub as the answer.
            if re.search(
                r"(?i)\b(company\s+(?:profile|briefing)|"
                r"briefing\s+on|profile\s+of|"
                r"tell\s+me\s+about|brief\s+me\s+on)\b",
                user_question or "",
            ):
                top_name = (candidates[0] or {}).get("name") or ""
                if top_name:
                    pack["_auto_picked_ambiguous"] = top_name
                    pack["entity_candidate"] = top_name
                    # Keep only rows for the top distinct company (fuzzy).
                    top_key = _name_key_for_dedup(top_name)
                    filtered = [
                        r for r in merged_rows
                        if _name_key_for_dedup(
                            str(r.get("company_name") or r.get("name") or "")
                        ) == top_key
                        or top_name.casefold() in str(
                            r.get("company_name") or r.get("name") or ""
                        ).casefold()
                    ]
                    if filtered:
                        merged_rows = filtered
            else:
                ent_clean = entity.strip()
                cand_lines = "\n".join(
                    f"  - **{c['name']}**" +
                    (f" (HQ {c['hq']})" if c.get("hq") else "") +
                    (f" — {c['sector']}" if c.get("sector") else "")
                    for c in candidates
                )
                return (
                    f"## Multiple possible matches for \"{ent_clean}\"\n\n"
                    f"I found several records that could match your query. "
                    f"Please tell me which one you meant:\n\n"
                    f"{cand_lines}\n\n"
                    f"_Reply with the exact name and I'll pull the full "
                    f"profile, including FK-linked AI insights, executives, "
                    f"and MENA presence._"
                )

    # Auto-enrich primary rows with FK-linked supplementary tables
    # (company_ai_insights, executives, competitors, country_insights,
    # key_indicators, trade_partners, etc.). Best-effort — failures are
    # silently swallowed so a single broken sub-query never blocks the
    # primary answer. Each enriched row gets a `_related` key with
    # human-labelled lists of cleaned related rows.
    try:
        from app.services.record_enrichment import (
            enrich_records, supports_enrichment,
        )
        if supports_enrichment(primary_table) and merged_rows:
            merged_rows = enrich_records(primary_table, merged_rows)
    except Exception:
        pass  # enrichment is best-effort

    # 1) Rows found → Jul21 path first: Azure/OpenAI narrative over
    #    privacy-filtered fact cards (MISA_NARRATIVE_CLOUD). Deterministic
    #    templates are fallback only when curation fails / is disabled.
    if merged_rows and CHAT_CURATION_ENABLED:
        client = get_openai_client()
        if client is not None:
            try:
                insight = curate_company_insights(
                    merged_rows,
                    user_question,
                    locale=response_locale,
                    entity_candidate=entity,
                    entity_matched=entity_matched,
                    table=primary_table,
                    client=client,
                    model=OPENAI_MODEL,
                    intent=pack.get("_intent"),
                    depth=pack.get("_depth"),
                )
                if insight:
                    pack["_answer_source"] = "curated"
                    # Person briefs: layer question-only / web Background under
                    # ## Background (Role stays MISA-authoritative). Matches
                    # Jul21 person-brief richness without sending DB JSON to web.
                    try:
                        import re as _re
                        if _re.search(r"(?m)^##\s+Role\b", insight or ""):
                            from app.services.hybrid_briefing import enrich_db_briefing
                            enriched = enrich_db_briefing(
                                insight,
                                user_question,
                                entity_hint=entity or entity_matched or "",
                                include_docs=False,
                                include_web=True,
                            )
                            if enriched.get("answer"):
                                insight = enriched["answer"]
                                pack["_answer_source"] = "curated+public_bg"
                                if enriched.get("web_sources"):
                                    pack["_web_sources"] = enriched["web_sources"]
                    except Exception:
                        pass
                    # DURABLE CONTRACT GATE: if curated narrative is thin /
                    # wrong shape, fall through to deterministic templates
                    # instead of shipping a regression (ops-less Apple brief,
                    # Role-only CEO, etc.).
                    try:
                        from app.services.answer_contracts import soft_check_answer
                        from app.logger import logger as _log
                        try:
                            _viol = soft_check_answer(
                                insight,
                                intent=pack.get("_intent"),
                                user_question=user_question,
                                db_context=pack.get("_advisory_db_context")
                                or pack.get("_db_context"),
                            )
                        except Exception as _sc_exc:
                            # Fail closed — never ship past a broken gate.
                            _viol = [f"soft_check_exception:{type(_sc_exc).__name__}"]
                            _log.warning(
                                "answer_contract: soft_check raised %s — "
                                "falling back to templates",
                                _sc_exc,
                            )
                        if _viol:
                            pack["_contract_violations"] = _viol
                            _log.warning(
                                "answer_contract: curated shape failed %s — "
                                "falling back to templates",
                                _viol,
                            )
                            insight = None
                    except Exception:
                        insight = None
                    if insight:
                        # A successful answer makes per-tool-call error hints
                        # internal plumbing — showing "I couldn't look up X in
                        # `fdi_data`" above a correct answer reads broken in
                        # production. The raw errors stay in the tool_call
                        # trace for debug mode.
                        return insight
            except Exception:
                pass
        try:
            from app.services.db_briefing import render_db_briefing
            db_brief = render_db_briefing(
                merged_rows,
                intent=pack.get("_intent"),
                table=primary_table,
                user_question=user_question,
                locale=response_locale,
                force=True,
            )
            if db_brief:
                try:
                    from app.services.hybrid_briefing import enrich_db_briefing
                    enriched = enrich_db_briefing(
                        db_brief,
                        user_question,
                        entity_hint=entity or "",
                    )
                    if enriched.get("answer"):
                        pack["_answer_source"] = "hybrid_db"
                        pack["_db_briefing"] = "deterministic+web+docs"
                        if enriched.get("web_sources"):
                            pack["_web_sources"] = enriched["web_sources"]
                        if enriched.get("doc_sources"):
                            pack["_doc_sources"] = enriched["doc_sources"]
                        return enriched["answer"]
                except Exception:
                    pass
                pack["_answer_source"] = "db"
                pack["_db_briefing"] = "deterministic"
                return db_brief
        except Exception:
            pass

    # 2) No rows in the DB → prefer Jul21 advisory for corridor asks,
    # else labelled general knowledge.
    if not merged_rows and CHAT_FALLBACK_ENABLED:
        try:
            from app.services.jul21_surface import looks_like_corridor_investment_ask
            if looks_like_corridor_investment_ask(user_question):
                _client = get_openai_client()
                if _client is not None:
                    adv = _run_advisory_path(
                        user_question,
                        pack,
                        response_locale or "en",
                        response_locale or "en",
                        _client,
                    )
                    if adv and adv.get("answer") and "Response withheld" not in (
                        adv.get("answer") or ""
                    ):
                        return adv["answer"]
        except Exception:
            pass
        client = get_openai_client()
        if client is not None:
            answer = general_knowledge_answer(
                user_question,
                locale=response_locale,
                client=client,
                model=OPENAI_MODEL,
            )
            if answer:
                return answer

    # 3) Deterministic local commentary (used when OpenAI is unavailable/errors).
    body = generate_commentary(
        merged_rows,
        primary_table,
        user_question,
        entity_candidate=entity,
        entity_lookup_required=entity_lookup_required,
        sql_entity_check_passed=sql_ok,
        row_entity_sanity_passed=row_sanity,
        closest_names=closest,
        locale=response_locale,
    )
    if err_lines and not merged_rows:
        return "\n\n".join(err_lines)
    # Rows produced an answer → tool-call error hints are internal
    # plumbing; they stay in the trace, not the user-facing text.
    return body


_FOLLOWUP_HINT_RE = re.compile(
    r"\b("
    # Command / continuation words
    r"more|further|continue|continue\s+with|carry\s+on|elaborate|"
    r"expand|deeper|deep[\s-]?dive|"
    r"research|investigate|analyze|analyse|analysis|"
    r"plan|strategy|brief|briefing|memo|"
    r"why|how|when|where|"
    r"more\s+(?:about|on)|tell\s+me\s+more|go\s+on|"
    # Pronouns (subject + object + possessive). 'they' was missing
    # previously and broke obvious follow-ups like "are they in saudi".
    r"they|them|their|theirs|"
    r"he|she|him|her|hers|"
    r"this|that|these|those|it|its|"
    # Demonstrative noun phrases pointing back at a prior entity
    r"the\s+(?:company|country|entity|deal|investor|record|firm|"
    r"organization|organisation)"
    r")\b",
    re.I,
)


_ENTITY_RESOLVER_PROMPT = """You are an entity-resolution assistant for a Saudi business-intelligence chatbot.

Given the conversation so far and the user's CURRENT question, identify the SPECIFIC named entity (company, country, or person) the question is about, and what KIND of entity it is.

CONVERSATION SO FAR:
{history}

CURRENT QUESTION: {question}

Return ONLY a JSON object with this exact shape:
{{
  "entity": "<canonical name, or null>",
  "entity_type": "<company|country|person|topic|null>",
  "parent_entity": "<parent company/org, or null>",
  "is_followup": <true|false>,
  "is_new_topic": <true|false>,
  "reasoning": "<one short sentence>"
}}

PRINCIPLES (apply your own judgement, not keyword lists):
- Pronouns (they/them/their/he/she/him/her/it/its/this/that/these/those, "the company", "the firm") refer to the most recently named subject in the conversation. Resolve them.
- Geographies (Saudi, MENA, UAE, GCC, Riyadh, …), corporate jargon (RHQ, CEO, CFO, AI, HQ, office, license, revenue, headcount, …) and generic biz nouns are NEVER entities — they are filters or attributes. Capitalisation does NOT make them entities.
- If the user names a NEW proper entity (Apple, Microsoft, Alphabet, Saudi Aramco, Pakistan, Vision 2030, an individual person), that is the entity and is_new_topic=true. A new proper name always wins over a prior subject.
- entity_type "person" = individual human (Sundar Pichai, Tim Cook, Mohammed bin Salman, a CEO/founder/exec named by name). Person names are typically two or three Capitalised tokens with no corporate suffix (Inc/Ltd/LLC).
- entity_type "company" = a corporation, firm, brand, holding (Apple, Alphabet, Aramco, Berkshire Hathaway).
- entity_type "country" = a sovereign country / geography (Saudi Arabia, Pakistan, UAE).
- entity_type "topic" = the question is generic/conceptual (Vision 2030, "renewable energy investments", "tech giants").
- parent_entity: ONLY for persons — the company/org they're known to lead or work at, if obvious from conversation OR widely known public knowledge ("Sundar Pichai" → "Alphabet"; "Tim Cook" → "Apple"). null otherwise.
- If the question is genuinely topical / generic with no specific entity AND no prior subject to inherit, return entity=null, entity_type=null.
- Output the entity as the user would naturally say it ("Alphabet", "Apple", "Saudi Aramco", "Sundar Pichai"). Use the form that appeared in conversation when possible.
- Be conservative on entity: if you are not sure there is a specific entity, return null. But fill in entity_type whenever you DO return an entity."""


def _resolve_entity_with_llm(
    user_question: str,
    history: list,
    client,
    model: str,
) -> dict:
    """Ask the LLM to resolve what entity the user is asking about,
    given the conversation history. Replaces hand-rolled noise-token
    heuristics — the LLM already knows that RHQ is jargon and Saudi
    is geography, so we let it judge.

    Returns a dict: {entity, is_followup, is_new_topic, reasoning}.
    On any failure returns {entity: None, ...} so the caller falls
    through to the existing path.
    """
    if client is None or not history:
        return {"entity": None, "is_followup": False,
                "is_new_topic": False, "reasoning": "no history or no client"}
    # Compact the history — last 6 turns, content truncated to 600 chars.
    snippet_lines: list[str] = []
    for h in history[-6:]:
        role = (h.get("role") or "").upper()
        content = (h.get("content") or "")[:600]
        if role and content:
            snippet_lines.append(f"{role}: {content}")
    prompt = _ENTITY_RESOLVER_PROMPT.format(
        history="\n\n".join(snippet_lines) or "(none)",
        question=user_question.strip(),
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
        )
        import json as _json
        raw = (resp.choices[0].message.content or "{}").strip()
        data = _json.loads(raw)
        ent = (data.get("entity") or "").strip() or None
        ent_type = (data.get("entity_type") or "").strip().lower() or None
        if ent_type not in {"company", "country", "person", "topic"}:
            ent_type = None
        parent = (data.get("parent_entity") or "").strip() or None
        return {
            "entity": ent,
            "entity_type": ent_type,
            "parent_entity": parent,
            "is_followup": bool(data.get("is_followup")),
            "is_new_topic": bool(data.get("is_new_topic")),
            "reasoning": str(data.get("reasoning") or "")[:200],
        }
    except Exception as e:
        return {"entity": None, "entity_type": None, "parent_entity": None,
                "is_followup": False, "is_new_topic": False,
                "reasoning": f"resolver error: {e}"}


def _inherit_entity_from_history(
    current_pack: dict, history: list
) -> tuple[dict, str | None]:
    """LEGACY heuristic kept as a deterministic fallback for when the
    LLM resolver is unavailable (no client) or fails. The primary path
    in _chat_execute now uses _resolve_entity_with_llm above — see
    the docstring there for the principles applied.

    Returns (possibly-augmented pack, name-of-inherited-entity-or-None).
    """
    if not history:
        return current_pack, None
    raw_q = (current_pack.get("raw") or "").strip()
    if not _FOLLOWUP_HINT_RE.search(raw_q):
        return current_pack, None
    cur_ent = (current_pack.get("entity_candidate") or "").strip()
    if cur_ent:
        from app.services.curation import _looks_like_proper_entity
        if _looks_like_proper_entity(cur_ent) and len(cur_ent.split()) <= 4:
            return current_pack, None

    # Walk history backward, find the most recent user turn with a real entity
    for h in reversed(history):
        if h.get("role") != "user":
            continue
        prev_content = h.get("content") or ""
        if not prev_content:
            continue
        prev_pack = clean_user_question(prev_content)
        prev_ent = (prev_pack.get("entity_candidate") or "").strip()
        if not prev_ent:
            continue
        # Don't inherit a follow-up-shaped previous entity
        if _FOLLOWUP_HINT_RE.fullmatch(prev_ent):
            continue
        new_pack = dict(current_pack)
        new_pack["entity_candidate"] = prev_ent
        new_pack["_inherited_from_history"] = prev_ent
        return new_pack, prev_ent
    return current_pack, None


# Executive tables, in priority order. Searched by smart_search() against
# each table's name-like columns. Stop on first hit.
_PERSON_LOOKUP_TABLES = (
    "company_executives",
    "executives",
    "rhq_topexecutives",
    "contacts",
)


# ─── Executive-lookup intent: dedicated direct path ─────────────────

# Question shapes that indicate the user wants forward-looking news /
# succession info, NOT the current state. These trigger an OpenAI web
# search augmentation that produces a "## What's Reported" section
# AFTER the DB-grounded current-state answer. Examples that match:
#   "who will follow Tim Cook as the next CEO of Apple"
#   "who is the next CEO of Apple"
#   "Apple's incoming CEO"
#   "Tim Cook's successor"
#   "who is taking over from Tim Cook"
#   "next chairman of Saudi Aramco"
_FORWARD_LOOKING_EXEC_RE = re.compile(
    r"\b("
    r"will\s+(?:follow|succeed|replace|take\s+over|become|be)|"
    r"next\s+(?:ceo|chairman|chair|chief|president|coo|cfo|cto|leader)|"
    r"new\s+(?:ceo|chairman|chair|chief|president|coo|cfo|cto|leader)|"
    r"incoming\s+(?:ceo|chairman|chair|chief|president|coo|cfo|cto)|"
    r"upcoming\s+(?:ceo|chairman|chair|chief|president|coo|cfo|cto)|"
    r"future\s+(?:ceo|chairman|chair|chief|president|coo|cfo|cto)|"
    r"successor|succession|"
    r"taking\s+over|step(?:s|ping)?\s+down|"
    r"replace(?:s|d|ment)?\s+(?:by|with|as)|"
    r"transition(?:s|ing)?\s+(?:to|from|out)|"
    r"after\s+(?:tim\s+cook|sundar\s+pichai|the\s+current\s+ceo)"
    r")\b",
    re.I,
)


def _is_forward_looking_exec_question(question: str) -> bool:
    """Heuristic flag for executive questions about future/succession
    rather than current state. Drives a web-search augmentation that
    appends a sourced 'What's Reported' section."""
    return bool(_FORWARD_LOOKING_EXEC_RE.search(question or ""))


# Current government / cabinet office-holder questions. The company_executives
# table lags royal decrees (e.g. Saudi Investment Minister changed Feb 2026).
# These MUST be web-verified; DB rows are supporting context only.
# Public / cabinet officeholders — NOT private-company C-suite.
# "Who is the CEO of Apple?" must stay on MISA company_executives;
# "Who is the Minister of Investment?" must prefer live web.
_CURRENT_OFFICEHOLDER_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"\bwho\s+(?:is|are)\s+(?:the\s+)?"
    r"(?:current\s+|acting\s+|incumbent\s+)?"
    r"(?:saudi\s+(?:arabian?\s+)?)?"
    r"(?:minister|deputy\s+minister|secretary|governor|"
    r"director\s+general)\b"
    r"|"
    r"\b(?:current|incumbent|acting)\s+"
    r"(?:saudi\s+(?:arabian?\s+)?)?"
    r"(?:minister|investment\s+minister|energy\s+minister|"
    r"finance\s+minister|foreign\s+minister)\b"
    r"|"
    r"\bminister\s+of\s+(?:investment|energy|finance|foreign\s+affairs|"
    r"commerce|industry|health|defence|defense|interior|economy)\b"
    r"|"
    r"\b(?:وزير|وزيرة)\b"  # Arabic 'minister'
    r")"
)


def _is_current_officeholder_question(question: str) -> bool:
    """True when the user asks who currently holds a public office / cabinet role.

    Stale MISA executive rows have repeatedly answered these incorrectly
    (e.g. naming a former Minister of Investment as current). Live web
    must lead; the DB is demoted to supporting context.

    Corporate C-suite asks (CEO/CFO of Apple, Aramco, etc.) are NOT
    officeholder questions — ``company_executives`` is authoritative.
    """
    return bool(_CURRENT_OFFICEHOLDER_RE.search(question or ""))


_EXEC_EXTRACT_PROMPT = """Extract the EXECUTIVE-LOOKUP target from this question.

Question: {question}

Return JSON with these fields (use null where not applicable):
  - "person_name":  the named individual being asked about, or null
                    (e.g. "Tim Cook", "Sundar Pichai", "Khalid Al-Falih")
  - "company":      the company OR government body whose executive / minister
                    is being asked about, or null
                    (e.g. "Apple", "Saudi Aramco", "Ministry of Investment",
                    "Government of Saudi Arabia")
  - "role":         the role being asked about, normalised
                    (one of: CEO, Chairman, Founder, CFO, COO, President,
                    Minister, Executive, or null if generic)

Examples:
  "Who is the CEO of Apple?"        → {{"person_name": null, "company": "Apple", "role": "CEO"}}
  "Tell me about Tim Cook"          → {{"person_name": "Tim Cook", "company": null, "role": null}}
  "Who chairs Saudi Aramco?"        → {{"person_name": null, "company": "Saudi Aramco", "role": "Chairman"}}
  "Who is the Minister of Investment of Saudi Arabia?" → {{"person_name": null, "company": "Ministry of Investment", "role": "Minister"}}
  "Sundar Pichai background"        → {{"person_name": "Sundar Pichai", "company": null, "role": null}}
  "Apple leadership"                → {{"person_name": null, "company": "Apple", "role": "Executive"}}"""


def _extract_exec_target(user_question: str, client, model: str) -> dict:
    """Pull {person_name, company, role} out of an executive_lookup
    question. Returns {} on failure so the caller falls through.

    LRU+TTL cached on (question, model) — same question text from a
    later turn skips the ~1.3s LLM call.
    """
    from app.services.llm_cache import cached_call

    def _do_extract():
        try:
            import json as _json
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": _EXEC_EXTRACT_PROMPT.format(question=user_question)}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=120,
            )
            d = _json.loads(resp.choices[0].message.content or "{}")
            return {
                "person_name": (d.get("person_name") or "").strip() or None,
                "company":     (d.get("company") or "").strip() or None,
                "role":        (d.get("role") or "").strip() or None,
            }
        except Exception:
            return None  # cached_call doesn't cache None — retry on next turn

    result = cached_call(
        "extract_exec_target",
        ((user_question or "").strip(), model),
        _do_extract,
    )
    return result or {}


# ─── Saudi RHQ / licensing aggregate ────────────────────────────────

# Questions that should bypass the LLM and the auxiliary rhq_licenses
# table entirely, going straight to a deterministic count + breakdown
# from the canonical company_profiles.licensed / is_rhq flags. Catches
# "how many RHQ licences do we have?", "total licensed companies in
# saudi", "how many rhq", "tell me the active MISA licenses", etc.
# The regex is tolerant of typos like "licens" / "licence" / "license".
# IMPORTANT: do NOT route these to `rhq_licenses` (small auxiliary table
# ~661 rows) — that path returns 0 rows / "no reliable information".
_SAUDI_LICENSING_COUNT_RE = re.compile(
    r"(?:"
    # Explicit count asks
    r"\b(how\s+many|count|total|number\s+of)\b.{0,40}\b"
    r"(rhq|licen[cs]e?d?|licen[cs]es?|licensing)\b"
    r"|"
    # "active/current MISA licenses", "tell me the active MISA licences"
    r"\b(active|current)\s+(misa\s+)?(rhq\s+)?licen[cs]e?s?\b"
    r"|"
    r"\b(misa\s+)?(rhq\s+)?licen[cs]e?\s+(count|total|numbers?|snapshot)\b"
    r")",
    re.IGNORECASE,
)


# "which companies hold an RHQ license", "show me the MISA license
# holders", "who is licensed by MISA" — a request to LIST license
# holders (no origin country). Routed to the deterministic licensing
# briefing with a holders table instead of entity smart-search (which
# has no entity to match and used to fall through to general knowledge).
_SAUDI_LICENSE_LIST_RE = re.compile(
    r"\b(which|list|show(?:\s+me)?|who|name)\b"
    r".{0,60}\b(rhq\s+licen[cs]e|licen[cs]e\s+holders?|"
    r"licen[cs]ed\s+by\s+misa|misa\s+licen[cs]e|hold\w*\s+an?\s+rhq)",
    re.IGNORECASE,
)


def _is_saudi_licensing_count_question(question: str) -> bool:
    q = question or ""
    if not (
        _SAUDI_LICENSING_COUNT_RE.search(q)
        or _SAUDI_LICENSE_LIST_RE.search(q)
    ):
        return False
    # Keep country company-list asks on their own path
    # ("tell me the Indian active companies").
    if _is_country_company_list_question(q) and _detect_origin_country(q):
        return False
    return True


# "list / show / tell me the <country> [active|licensed|rhq] companies" —
# a request to LIST a country's companies, not a count and not an entity
# lookup. Without this, "tell me the indian active companies" gets
# "indian active" extracted as an entity and hijacked into disambiguation.
_COUNTRY_COMPANY_LIST_RE = re.compile(
    r"\b(list|show|give\s+me|tell\s+me|who\s+are|which\s+are|name)\b"
    r".{0,30}\b(compan(?:y|ies)|firms|investors|businesses)\b",
    re.IGNORECASE,
)


def _is_country_company_list_question(question: str) -> bool:
    """True for 'list/show/tell me the <country> companies' style asks
    that also name an origin country (adjective or noun)."""
    q = question or ""
    if not _COUNTRY_COMPANY_LIST_RE.search(q):
        return False
    return _detect_origin_country(q) is not None


def _format_country_licensing_answer(country: str, stats: dict) -> str:
    """Direct, deterministic answer to 'how many licensed / RHQ
    companies from <country>' — states the country's own numbers
    instead of the global aggregate.

    NEVER renders zeros when retrieval failed (`_db_error` /
    counts_unavailable) — that was the India false-zero class.
    """
    if stats.get("_db_error") or (
        isinstance(stats.get("retrieval"), dict)
        and stats["retrieval"].get("do_not_claim_zero")
    ) or stats.get("footprint_data_unavailable"):
        err = stats.get("_db_error") or (
            (stats.get("retrieval") or {}).get("error")
        ) or "unknown"
        status = (
            stats.get("retrieval_status")
            or (stats.get("retrieval") or {}).get("retrieval_status")
            or "SOURCE_UNAVAILABLE"
        )
        return (
            f"## {country}-origin companies in Saudi Arabia\n\n"
            f"Internal MISA footprint data for **{country}** could not be "
            f"retrieved (`{status}`"
            + (f": {err}" if err and err != "unknown" else "")
            + "). This is **not** a verified zero — do not conclude that "
            "no licensed or RHQ companies exist.\n\n"
            "_Source: `company_profiles.licensed` / `is_rhq` (unavailable)._"
        )

    licensed    = int(stats.get("total_licensed") or 0)
    rhq         = int(stats.get("total_rhq") or 0)
    non_lic     = int(stats.get("total_non_licensed") or 0)
    non_lic_rhq = int(stats.get("total_non_licensed_rhq") or 0)
    total       = licensed + non_lic

    if (
        licensed == 0 and rhq == 0
        and stats.get("retrieval_status") in (
            "SUCCESS_EMPTY", "zero_records",
        )
    ):
        return (
            f"## {country}-origin companies in Saudi Arabia\n\n"
            f"The queried MISA source returned **0** verified licensed "
            f"records and **0** RHQ records for **{country}** "
            f"(source: `company_profiles.licensed` / `is_rhq`; "
            f"origin filters applied). This is a successful empty "
            f"result, not a retrieval failure.\n"
        )

    if non_lic:
        rhq_note = (
            f" (including **{non_lic_rhq:,} with RHQ status**)"
            if non_lic_rhq else ""
        )
        unlicensed_clause = (
            f", and **{non_lic:,} are present but unlicensed**{rhq_note}."
        )
    else:
        unlicensed_clause = "."
    lines = [
        f"## {country}-origin companies in Saudi Arabia",
        "",
        f"**{total:,} companies** from {country} are present in Saudi Arabia — "
        f"**{licensed:,} hold an active MISA licence** (of which **{rhq:,} hold "
        f"Regional Headquarters (RHQ) status**)" + unlicensed_clause,
    ]
    tops = stats.get("rhq") or []
    if tops:
        lines += ["", "### Top RHQ holders"]
        for r in tops[:8]:
            name = r.get("company_name") or "—"
            ind = r.get("industry")
            lines.append(f"- **{name}**" + (f" — {ind}" if ind else ""))
    lic_only = stats.get("licensed_only") or []
    if lic_only:
        lines += ["", "### Other licensed companies (no RHQ)"]
        for r in lic_only[:8]:
            name = r.get("company_name") or "—"
            ind = r.get("industry")
            lines.append(f"- **{name}**" + (f" — {ind}" if ind else ""))
    non_lic_rows = stats.get("non_licensed") or []
    if non_lic_rows:
        lines += ["", "### Unlicensed companies (not MISA-registered)"]
        for r in non_lic_rows[:8]:
            name = r.get("company_name") or "—"
            ind = r.get("industry")
            lines.append(f"- **{name}**" + (f" — {ind}" if ind else ""))

    # Jul21-lite closing: country-correct trade bodies + named next moves
    # so count answers are not a bare census.
    try:
        from app.services.advisory_structured import _default_trade_bodies
        bodies = _default_trade_bodies(country)[:5]
        if bodies:
            lines += [
                "",
                "## Investment & Trade Bodies to Engage",
                "",
                "| Organisation | Type | Role in engagement |",
                "|---|---|---|",
            ]
            for b in bodies:
                lines.append(
                    f"| {b.get('organisation')} | {b.get('type')} | "
                    f"{b.get('role')} |"
                )
        lead_names = []
        for r in (tops or [])[:3]:
            n = r.get("company_name")
            if n:
                lead_names.append(str(n))
        lines += ["", "## Recommended Next Moves for MISA"]
        if lead_names:
            lines.append(
                f"- Run RHQ expansion account reviews with "
                f"**{lead_names[0]}**"
                + (f" and **{lead_names[1]}**" if len(lead_names) > 1 else "")
                + " — map Vision 2030 / giga-project demand (NEOM, SDAIA, "
                "NUPCO as relevant)."
            )
        ipa = (bodies[0].get("organisation") if bodies else None) or (
            f"the national IPA of {country}"
        )
        lines.append(
            f"- Brief **{ipa}** on Saudi corridor offers for the top "
            f"licensed/{country} RHQ accounts above."
        )
        lines.append(
            f"- Publish a one-pager of the **{licensed:,}** licensed / "
            f"**{rhq:,}** RHQ footprint for desk targeting ahead of LEAP / FII."
        )
    except Exception:
        pass

    lines += ["", "_Source: licensed companies keyed by shareholder\\_country\\_name; "
              "unlicensed companies keyed by country\\_profile\\_name._"]
    return "\n".join(lines)


def _licensing_question_focus(question: str) -> str:
    """What the licensing-count question is really asking about:
    'rhq'      -> lead with the 727 RHQ figure,
    'licensed' -> lead with the 95,671 licensed figure,
    'both'     -> the combined snapshot (default)."""
    q = (question or "").lower()
    has_rhq = "rhq" in q or "regional head" in q or "headquarter" in q
    has_lic = bool(re.search(r"licen[cs]e", q))
    # 'RHQ' always denotes the RHQ figure, even in 'RHQ licences'.
    if has_rhq:
        return "rhq"
    if has_lic:
        return "licensed"
    return "both"


def _format_saudi_licensing_briefing(summary: dict, focus: str = "both") -> str:
    """Deterministic executive briefing for the RHQ / licensing-count
    question. No LLM — counts are exact and the structure is fixed.
    `focus` decides which number leads the answer so 'how many licenses'
    and 'how many RHQ licenses' get the right headline.

    Never renders zeros when retrieval failed (`_db_error` /
    counts_unavailable).
    """
    if summary.get("_db_error") or summary.get("counts_unavailable") or (
        isinstance(summary.get("retrieval"), dict)
        and summary["retrieval"].get("do_not_claim_zero")
    ):
        try:
            from app.schemas.quality_response import licensing_fallback_message
            return licensing_fallback_message(
                status=summary.get("retrieval_status") or "SOURCE_UNAVAILABLE",
                error=str(summary.get("_db_error") or "")[:200],
            )
        except Exception:
            return (
                "## Licensing Snapshot\n\n"
                "Internal MISA licensing aggregates could not be retrieved. "
                "This is **not** a verified zero.\n"
            )

    total_lic = summary.get("total_licensed") or 0
    total_rhq = summary.get("total_rhq") or 0
    rhq_rows = summary.get("rhq_by_country") or []
    lic_rows = summary.get("licensed_by_country") or []
    # Build the RHQ-by-country table. Drop "Other / Unspecified" from
    # the headline view but mention it in a footer line so the totals
    # reconcile.
    rhq_named = [r for r in rhq_rows if r["country"] != "Other / Unspecified"]
    rhq_other = next((r["n"] for r in rhq_rows
                      if r["country"] == "Other / Unspecified"), 0)
    lic_named = [r for r in lic_rows if r["country"] != "Other / Unspecified"
                 and r["country"] != "Saudi Arabia"]
    lic_saudi = next((r["n"] for r in lic_rows
                      if r["country"] == "Saudi Arabia"), 0)
    lic_other = next((r["n"] for r in lic_rows
                      if r["country"] == "Other / Unspecified"), 0)

    # Compute share for the top-3 RHQ countries to underline concentration.
    top3 = sum(r["n"] for r in rhq_named[:3])
    top3_pct = round(100 * top3 / total_rhq, 1) if total_rhq else 0

    lines: list[str] = []
    if focus == "licensed":
        title = "## Licensing Snapshot"
    elif focus == "rhq":
        title = "## Saudi RHQ Snapshot"
    else:
        title = "## Licensing & RHQ Snapshot"
    lines.append(title)
    lines.append("")
    if focus == "licensed":
        lines.append(
            f"**{total_lic:,} companies hold an active MISA licence** in "
            f"the database. Of these, **{total_rhq:,}** also hold a Saudi "
            f"Regional Headquarters (RHQ) licence."
        )
    elif focus == "rhq":
        lines.append(
            f"**{total_rhq:,} companies hold an active Saudi Regional "
            f"Headquarters (RHQ) licence**, out of a broader pool of "
            f"**{total_lic:,} MISA-licensed companies** in the database."
        )
    else:
        lines.append(
            f"**{total_lic:,} companies hold an active MISA licence** in "
            f"the database. Of these, **{total_rhq:,}** hold a Saudi "
            f"Regional Headquarters (RHQ) licence."
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 📊 Top Origin Countries — RHQ Licence Holders")
    lines.append("")
    lines.append("| Rank | Origin Country | RHQ Companies |")
    lines.append("|---|---|---|")
    for i, r in enumerate(rhq_named[:10], start=1):
        lines.append(f"| {i} | {r['country']} | **{r['n']}** |")
    if rhq_other:
        lines.append(
            f"\n_Plus {rhq_other} additional RHQ companies whose origin "
            f"nationality isn't tagged in the database._"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 🇸🇦 Licensed Pool (Broader)")
    lines.append("")
    if lic_saudi:
        lines.append(
            f"The full licensed pool includes **{lic_saudi:,}** domestic "
            f"Saudi licences, plus foreign-origin companies below."
        )
    else:
        lines.append(
            f"Canonical total: **{total_lic:,}** licensed companies "
            f"(`licensed = true`). Foreign-origin nationality tags "
            f"(where available) are shown below."
        )
    lines.append("")
    if lic_named:
        lines.append("Top non-Saudi origin countries (tagged nationality):")
        for r in lic_named[:8]:
            lines.append(f"- **{r['country']}**: {r['n']:,}")
    if lic_other:
        lines.append(
            f"- _No nationality tag in supporting tables_: {lic_other:,}"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 🇸🇦 Strategic Read")
    if len(rhq_named) >= 3:
        lines.append(
            f"- The top three origin countries (**{rhq_named[0]['country']}**, "
            f"**{rhq_named[1]['country']}**, "
            f"**{rhq_named[2]['country']}**) "
            f"account for **{top3_pct}%** of all RHQ licences."
        )
    elif rhq_named:
        lines.append(
            f"- Leading RHQ origin: **{rhq_named[0]['country']}** "
            f"({rhq_named[0]['n']} companies)."
        )
    lines.append(
        "- The licensed-but-not-RHQ pool is the upgrade pipeline: "
        f"**{total_lic - total_rhq:,} licensed entities** that have NOT "
        "yet taken the RHQ step."
    )
    if rhq_other:
        lines.append(
            f"- **Data hygiene flag:** {rhq_other} RHQ-flagged company "
            "rows have no origin nationality tag — worth a "
            "data-quality pass to enable cleaner geographic targeting."
        )
    lines.append("")
    lines.append(
        "_Sources: `company_profiles.licensed` and `company_profiles.is_rhq` "
        "— MISA's canonical licensing markers. Counts via live "
        "`SELECT COUNT(*) WHERE licensed = true` / `is_rhq = true`._"
    )
    return "\n".join(lines)


# ─── company_profile direct path (correlator-backed) ───────────────

def _filter_company_ids_by_name_match(ids: list[int], target: str,
                                       min_word_overlap: int = 1) -> list[int]:
    """Filter a list of company_profiles.id values to those whose
    company_name shares at least one significant word (3+ chars,
    not a stopword) with `target`, OR matches any alias of `target`
    from the curated alias map. This guards against fuzzy-search
    contamination — e.g. Google's resolved IDs sometimes include
    'Mandiant, Inc. (FireEye)' because of trigram overlap, and we
    don't want the correlator to pick that as primary."""
    if not ids or not target:
        return ids
    from app.database import get_db
    from app.services.alias_resolver import expand_aliases
    import psycopg2.extras as _ex
    import re as _re

    aliases = expand_aliases(target) or [target]
    alias_lc = {a.lower().strip() for a in aliases}
    stopwords = {"the", "and", "of", "for", "in", "on", "with", "&", "or", "to",
                 "inc", "ltd", "llc", "plc", "corp", "co", "group", "holdings",
                 "company", "limited", "incorporated", "corporation"}
    target_words = {
        w for w in _re.findall(r"[a-z0-9]{3,}", target.lower())
        if w not in stopwords
    }
    # Include alias words too — "Google" expands to "Alphabet", so
    # rows whose name contains "Alphabet" pass.
    for a in aliases:
        for w in _re.findall(r"[a-z0-9]{3,}", a.lower()):
            if w not in stopwords:
                target_words.add(w)

    conn = get_db()
    with conn.cursor(cursor_factory=_ex.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, company_name FROM company_profiles WHERE id = ANY(%s)",
            (ids,),
        )
        rows = cur.fetchall()
    kept: list[int] = []
    for r in rows:
        name_lc = (r["company_name"] or "").lower()
        # Pass if any alias-as-whole appears in the name.
        if any(a in name_lc for a in alias_lc if len(a) >= 3):
            kept.append(r["id"])
            continue
        # Pass if name shares at least min_word_overlap content words.
        name_words = {
            w for w in _re.findall(r"[a-z0-9]{3,}", name_lc)
            if w not in stopwords
        }
        if len(target_words & name_words) >= min_word_overlap:
            kept.append(r["id"])
    return kept


def _try_company_profile_correlated(
    user_question: str, pack: dict, client, model: str,
) -> list | None:
    """NEW direct path for company_profile intent. Replaces the LLM-
    routed flow that pulled only the company_profiles row + maybe FK
    enrichment. Instead, runs the correlator to pull EVERYTHING
    FK-related to the company in parallel:
      - primary profile
      - executives
      - competitors
      - geographic revenues
      - news / ai_insights / business_units / global_presences
      - misa_contacts
      - opportunities
      - meetings + engagements + notes (engagement history)
      - strategic investors
    The curator then weaves these into ONE cross-referenced briefing
    instead of a single-table snapshot."""
    from app.services.engagement_data import resolve_company_ids
    from app.services.correlator import correlate_company, bundle_summary_for_prompt

    target = _extract_entity_from_question(user_question, client, model)
    if not target:
        return None
    ids, canon = resolve_company_ids(target)
    if not ids:
        return None
    # Filter resolved IDs to those whose name actually matches the
    # target (or one of its aliases). Without this, fuzzy contaminants
    # like 'Mandiant, Inc.' in Google's resolved set OR completely
    # unrelated 'Vikoma International Ltd.' in 'Acme Foo Bar Holdings'
    # search end up driving the primary row in the correlator.
    filtered_ids = _filter_company_ids_by_name_match(ids, target)
    if not filtered_ids:
        # No real match — let the LLM-routed flow handle it via
        # classify_match (which produces the honest "No record matching
        # X was found" message for true non-matches).
        return None
    ids = filtered_ids

    bundle = correlate_company(ids)
    if not bundle.get("primary"):
        return None
    pack["_correlated_target"] = canon or target
    pack["_correlated_ids"] = ids[:10]
    pack["_correlator_elapsed_ms"] = bundle.get("_meta", {}).get("elapsed_ms")

    # Emit ONE tool_call per non-empty section. The curator's prompt
    # (see _CORRELATED_BRIEFING_NOTE in intent_router) tells it to
    # weave them into a single answer with cross-references — not
    # render each as its own section.
    summary = bundle_summary_for_prompt(bundle)
    primary = summary.get("primary") or {}
    tcs: list = [_build_engagement_tool_call(
        "company_profiles", [primary],
        {"_company_correlated": True, "_target": canon or target},
        pack,
    )]
    # Always attach briefing-critical FK sections (execs / geo / opps /
    # financials). Pruning these at simple_fact was the root cause of
    # "HQ ask is fine but company brief loses Operational Detail" —
    # SSE kept full folds, JSON path did not. Heavy sections (news,
    # meetings, …) still skip at simple_fact for latency.
    _depth = pack.get("_depth") or ""
    _BRIEFING_CRITICAL = (
        ("executives",             "company_executives"),
        ("geographic_revenues",    "company_geographic_revenues"),
        ("financial_performances", "company_financial_performances"),
        ("opportunities",          "opportunities"),
    )
    _HEAVY_SECTIONS = (
        ("competitors",            "company_competitors"),
        ("global_presences",       "company_global_presences"),
        ("business_units",         "company_business_units"),
        ("news",                   "company_news"),
        ("ai_insights",            "company_ai_insights"),
        ("misa_contacts",          "misa_contact_details"),
        ("strategic_investors",    "strategic_investors"),
        ("meetings",               "meetings"),
    )
    section_tables = list(_BRIEFING_CRITICAL)
    if _depth != "simple_fact":
        section_tables.extend(_HEAVY_SECTIONS)
    for key, table_label in section_tables:
        rows = summary.get(key) or []
        if not rows:
            continue
        tcs.append(_build_engagement_tool_call(
            table_label, rows,
            {"_company_correlated": True, "_target": canon or target,
             "_correlator_section": key},
            pack,
        ))
    return tcs


# ─── country_profile direct path ────────────────────────────────────

def _force_country_licensing_summary(answer: str, tool_calls: list) -> str:
    """Post-process the country_profile answer to inject the
    ground-truth licensing summary line. The LLM curator can't see
    filter metadata (only row data goes into the prompt) so it
    invents numbers from the rendered (top-10 truncated) rows.
    This replaces any LLM-emitted 'N companies from X are licensed'
    line with the correct totals carried on the tool_call filters.

    Idempotent: if the answer doesn't have a licensing section, no-op."""
    if not answer or not tool_calls:
        return answer
    # Find the totals from any tool_call carrying them.
    total_lic = None; total_rhq = None; country = None
    for tc in tool_calls:
        f = tc.get("filters") or {}
        if "_total_licensed" in f and "_total_rhq" in f:
            total_lic = int(f["_total_licensed"])
            total_rhq = int(f["_total_rhq"])
            country = f.get("_target") or country
            break
    if total_lic is None or country is None:
        return answer
    # Build the correct summary line (singular/plural aware).
    noun = "company" if total_lic == 1 else "companies"
    hold = "holds" if total_rhq == 1 else "hold"
    correct = (
        f"**{total_lic} {noun} from {country} "
        f"{'is' if total_lic == 1 else 'are'} licensed in "
        f"Saudi Arabia ({total_rhq} of those {hold} a Regional HQ "
        f"licence).**"
    )
    import re as _re
    # Match any "**N companies from X are licensed in Saudi Arabia
    # (M of those hold a Regional HQ licence).**" line — bold or
    # not — and replace with the correct one. Catches "10 of 6"
    # and similar LLM hallucinations.
    pat = _re.compile(
        r"\*{0,2}\d+\s+(?:company|companies)\s+from\s+[^.]+?(?:licensed in Saudi[^.]+|are\s+licensed[^.]+|hold[^.]+regional[^.]+|hq[^.]+licence[^.]+)\.\s*\*{0,2}",
        _re.IGNORECASE,
    )
    if pat.search(answer):
        answer = pat.sub(correct, answer, count=1)
    else:
        # No existing line found — prepend it under the company
        # section header if present.
        hdr_pat = _re.compile(r"^(##\s+[^\n]*Companies[^\n]*Active[^\n]*\n)",
                              _re.MULTILINE)
        m = hdr_pat.search(answer)
        if m:
            insert_at = m.end()
            answer = answer[:insert_at] + correct + "\n\n" + answer[insert_at:]
    # CONSISTENCY: when the ground truth is non-zero, the LLM's
    # "_No companies from X are recorded as licensed..._" line (written
    # because the truncated row payload looked empty) directly
    # contradicts the injected summary — drop it.
    if total_lic and total_lic > 0:
        none_pat = _re.compile(
            r"^_?\*?_?No companies from [^\n]*?(?:licensed|recorded)"
            r"[^\n]*$\n?",
            _re.IGNORECASE | _re.MULTILINE,
        )
        answer = none_pat.sub("", answer)
    return answer



_COUNTRY_EXTRACT_PROMPT = """Extract the COUNTRY this question is about.

Question: {question}

Return JSON: {{"country": "<country name, or null>"}}

Examples:
  "pakistan"                                          → {{"country": "Pakistan"}}
  "tell me about germany"                             → {{"country": "Germany"}}
  "which Indian companies have invested in Saudi"     → {{"country": "India"}}
  "Pakistani companies with RHQ licences"             → {{"country": "Pakistan"}}
  "Show me German companies in Saudi"                 → {{"country": "Germany"}}
  "How many UK firms have an RHQ in KSA?"             → {{"country": "United Kingdom"}}
  "Egypt FDI outlook"                                 → {{"country": "Egypt"}}
  "Tell me about Apple"                               → {{"country": null}}

Map nationality adjectives to country names (Indian→India, Pakistani→Pakistan,
German→Germany, British/UK/English→United Kingdom, American→United States,
Chinese→China, Japanese→Japan, Korean→South Korea, Brazilian→Brazil, etc.).
Return null if no specific country is named."""


def _extract_country_from_question(
    user_question: str, client, model: str,
) -> str | None:
    """Country-specific entity extractor. Handles bare names
    ('pakistan'), 'tell me about X', and nationality adjectives
    ('Indian companies' → 'India'). The executive extractor can't
    handle these because its 'company' field excludes countries.

    LRU+TTL cached on (question, model)."""
    if not user_question:
        return None
    from app.services.llm_cache import cached_call

    def _do_extract():
        try:
            import json as _json
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": _COUNTRY_EXTRACT_PROMPT.format(
                               question=user_question)}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=60,
            )
            d = _json.loads(resp.choices[0].message.content or "{}")
            return (d.get("country") or "").strip() or None
        except Exception:
            return None

    return cached_call(
        "extract_country",
        (user_question.strip(), model),
        _do_extract,
    )


def _try_country_profile_direct(
    user_question: str, pack: dict, client, model: str,
) -> list | None:
    """Bundle country macros + Vision 2030 outlook + rhq_licenses
    (companies from that country with active Saudi RHQ licences)
    in ONE payload. Without this, the LLM was routing to
    company_profiles, getting both a direct company row AND an
    auto-enriched 'notable companies' list — same companies twice.
    With ONE consolidated source we avoid the dup AND surface real
    Saudi-licensing data."""
    from app.services.engagement_data import fetch_country_profile_bundle
    # Use the country-specific extractor — the executive extractor
    # returns null for bare country names ("pakistan") and nationality
    # adjectives ("Indian companies").
    target = _extract_country_from_question(user_question, client, model)
    if not target:
        # Fall back to the generic extractor for unusual phrasing.
        target = _extract_entity_from_question(user_question, client, model)
    if not target:
        return None
    bundle = fetch_country_profile_bundle(target)
    si = bundle.get("saudi_investors") or {}
    has_si = bool(
        si.get("rhq") or si.get("licensed_only") or si.get("non_licensed")
        or si.get("_db_error") or si.get("do_not_claim_zero")
    )
    if not bundle.get("country_profile") and not has_si:
        return None

    pack["_country_target"] = bundle.get("_canonical_name") or target
    pack["_country_id"] = bundle.get("_country_id")

    tcs: list = []
    cp = bundle.get("country_profile")
    if cp:
        tcs.append(_build_engagement_tool_call(
            "country_profiles", [cp],
            {"_country_direct": True, "_target": bundle["_canonical_name"]},
            pack,
        ))
    if bundle.get("vision_outlook"):
        tcs.append(_build_engagement_tool_call(
            "country_vision_outlooks", [bundle["vision_outlook"]],
            {"_country_direct": True, "_target": bundle["_canonical_name"]},
            pack,
        ))
    if bundle.get("strategic_opportunities"):
        tcs.append(_build_engagement_tool_call(
            "country_strategic_opportunities", bundle["strategic_opportunities"],
            {"_country_direct": True, "_target": bundle["_canonical_name"]},
            pack,
        ))
    # Saudi-presence buckets, sourced from the CANONICAL
    # company_profiles.licensed / is_rhq booleans (NOT the auxiliary
    # rhq_licenses table — that's only 661 rows vs 95,671 licensed
    # / 727 RHQ on company_profiles).
    #
    # Two separate tool_calls so the curator can render them as
    # two clearly-labelled subsections. The summary line is computed
    # DETERMINISTICALLY here and passed as a verbatim string the LLM
    # MUST lift unchanged — past attempts at letting the model count
    # from the filter values produced invented numbers (e.g., "10 of
    # 6 RHQ" instead of the real "39 of 34"). The model is good at
    # rendering rows; it's bad at arithmetic on hidden metadata.
    total_lic     = si.get("total_licensed", 0)
    total_rhq     = si.get("total_rhq", 0)
    total_non_lic = si.get("total_non_licensed", 0)
    total_non_lic_rhq = si.get("total_non_licensed_rhq", 0)
    total_present = total_lic + total_non_lic
    country = bundle["_canonical_name"]
    if si.get("_db_error") or si.get("do_not_claim_zero"):
        # Do not invent a "0 companies" summary line on retrieval failure.
        pack["_degraded"] = "country_profile_footprint_unavailable"
        pack["_retrieval"] = si.get("retrieval") or {
            "do_not_claim_zero": True,
            "counts_unavailable": True,
        }
        # Still return macros/outlook if present; footprint sections skipped.
        if not tcs:
            return None
        # Attach an explicit limitation tool_call so curation cannot invent zeros.
        tcs.append(_build_engagement_tool_call(
            "company_profiles_footprint_unavailable",
            [],
            {
                "_country_direct": True,
                "_target": country,
                "_db_error": si.get("_db_error"),
                "_retrieval_status": si.get("retrieval_status"),
                "_summary_line": (
                    f"Internal MISA footprint data for **{country}** could "
                    f"not be retrieved — this is **not** a verified zero."
                ),
            },
            pack,
        ))
        return tcs
    if total_non_lic:
        _rhq_note = (
            f" (including {total_non_lic_rhq} with RHQ status)"
            if total_non_lic_rhq else ""
        )
        unlicensed_note = (
            f", and {total_non_lic} are present but unlicensed{_rhq_note}"
        )
    else:
        unlicensed_note = ""
    summary_line = (
        f"**{total_present} companies from {country} are present in "
        f"Saudi Arabia — {total_lic} hold a MISA licence ({total_rhq} "
        f"of those hold a Regional HQ licence){unlicensed_note}.**"
    )
    if si.get("rhq"):
        tcs.append(_build_engagement_tool_call(
            "company_profiles_rhq", si["rhq"],
            {"_country_direct": True,
             "_target": country,
             "_saudi_rhq_companies": True,
             "_total_rhq": total_rhq,
             "_total_licensed": total_lic,
             "_total_non_licensed": total_non_lic,
             "_summary_line": summary_line},
            pack,
        ))
    if si.get("licensed_only"):
        tcs.append(_build_engagement_tool_call(
            "company_profiles_licensed", si["licensed_only"],
            {"_country_direct": True,
             "_target": country,
             "_saudi_licensed_companies": True,
             "_total_rhq": total_rhq,
             "_total_licensed": total_lic,
             "_total_non_licensed": total_non_lic,
             "_summary_line": summary_line},
            pack,
        ))
    if si.get("non_licensed"):
        tcs.append(_build_engagement_tool_call(
            "company_profiles_unlicensed", si["non_licensed"],
            {"_country_direct": True,
             "_target": country,
             "_saudi_non_licensed_companies": True,
             "_total_non_licensed": total_non_lic,
             "_total_non_licensed_rhq": total_non_lic_rhq,
             "_summary_line": summary_line},
            pack,
        ))
    return tcs if tcs else None


# ─── relationship_intelligence + opportunity_alignment direct paths ──

# ─── Self-reference entity blocklist ─────────────────────────────────
# When the LLM extracts an entity that is actually the system's OWN
# org name ("MISA", "the ministry", "ksa", "we", "us"), we must NOT
# treat it as a company/country to look up. Otherwise we trigger:
#   - smart_search on company_profiles for "MISA" → zero rows
#   - "No record matching MISA was found in the MISA database"
# which is absurd. This blocklist nulls those out so the question
# falls through to its proper intent route (engagement_strategy,
# general_research, etc.) with country / sector data instead.
#
# Normalised against lowercase + stripped + punctuation-removed input.
_SELF_REFERENCE_ENTITIES: frozenset[str] = frozenset({
    # Ministry / org-self names
    "misa", "m.i.s.a", "misas",
    "ministry", "the ministry", "ministry of investment",
    "ministry of investment of saudi arabia",
    "ministry of investment saudi arabia",
    "investment ministry",
    # Country-as-subject (NOT target) when paired with strategy verbs
    "ksa", "saudi", "saudi arabia", "the kingdom", "kingdom",
    "kingdom of saudi arabia",
    # Pronouns / first-person references
    "we", "us", "our", "ourselves", "ours",
    "i", "me", "my", "myself",
})


def _is_self_reference_entity(entity: str | None) -> bool:
    """True when the extracted 'entity' is actually the system's own
    org name (MISA / the ministry) or a first-person subject pronoun.
    Used to suppress no-match flows that would otherwise produce
    'No record matching MISA was found in the MISA database'."""
    if not entity:
        return False
    e = entity.strip().lower()
    # Strip trailing punctuation, possessives, articles
    e = re.sub(r"[^\w\s\.]", "", e).strip()
    e = re.sub(r"^(the|a|an)\s+", "", e).strip()
    e = re.sub(r"'s$", "", e).strip()
    return e in _SELF_REFERENCE_ENTITIES


def _extract_entity_from_question(
    user_question: str, client, model: str,
) -> str | None:
    """Pull the company / country name out of a relationship or
    opportunity question. Reuses the executive-target extractor (which
    already returns {person, company, role}) — we just take the
    company field. Returns None on failure, empty extraction, or
    when the LLM extracted a self-reference (MISA, the ministry,
    ksa, we, us) — those are subjects of strategic questions, not
    entities to look up in company_profiles."""
    try:
        tgt = _extract_exec_target(user_question, client, model)
    except Exception:
        return None
    extracted = (tgt or {}).get("company") or (tgt or {}).get("person_name")
    if _is_self_reference_entity(extracted):
        # The LLM pulled "MISA" or "we" — return None so the caller
        # falls through to intent-driven routing (engagement_strategy
        # answers the question with country/sector data, no entity
        # lookup needed).
        return None
    return extracted


def _build_engagement_tool_call(table: str, rows: list[dict], filters: dict,
                                pack: dict) -> dict:
    """Construct a tool_call dict from a list of plain row dicts, in
    the shape the curator's compose_local_commentary expects.
    Treats the rows as if they came from a smart_search DataFrame."""
    import pandas as pd
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return {
        "table": table,
        "filters": filters,
        "sql": "(direct query via engagement_data.py)",
        "params": [],
        "rows_df": df,
        "row_count": int(len(df)),
        "input_trace": dict(pack),
        "sql_entity_check_passed": True,
        "row_entity_sanity_passed": True,
        "closest_names": [],
    }


def _try_relationship_intelligence_direct(
    user_question: str, pack: dict, client, model: str,
) -> list | None:
    """Direct engagement-history query. Resolves entity → company_ids
    (across ALL dup rows since meetings may link to any variant), then
    pulls meetings + engagements + notes + misa contacts + recent
    interactions. Returns tool_calls list or None when entity can't
    be resolved (caller falls through to normal flow)."""
    from app.services.engagement_data import (
        resolve_company_ids, fetch_engagement_history,
    )
    target = _extract_entity_from_question(user_question, client, model)
    if not target:
        return None
    ids, canon = resolve_company_ids(target)
    if not ids:
        return None
    pack["_engagement_target"] = canon or target
    pack["_engagement_company_ids"] = ids[:10]
    data = fetch_engagement_history(ids)

    # Build a tool_calls list — one per non-empty bucket — so the
    # curator sees structured engagement data instead of a generic
    # company_profiles row. The intent directive in intent_router
    # already tells the model to lead with engagement history; the
    # rows become the evidence.
    tcs: list = []
    if data["meetings"]:
        # Meetings carry their own _engagements + _notes inline.
        tcs.append(_build_engagement_tool_call(
            "meetings", data["meetings"],
            {"_relationship_direct": True, "_target": canon or target,
             "_company_ids": ids[:5]},
            pack,
        ))
    if data["contacts"]:
        tcs.append(_build_engagement_tool_call(
            "misa_contact_details", data["contacts"],
            {"_relationship_direct": True, "_target": canon or target},
            pack,
        ))
    if data["interactions"]:
        tcs.append(_build_engagement_tool_call(
            "latest_interactions", data["interactions"],
            {"_relationship_direct": True, "_target": canon or target},
            pack,
        ))

    if not tcs:
        # No engagement records anywhere — return a synthetic empty
        # tool_call so the curator sees "we searched, we found nothing"
        # rather than thinking it's a normal company_profile lookup
        # that returned no rows. The directive in intent_router
        # tells the model to emit the explicit "No engagement history
        # found in the current database for X." line in that case.
        tcs.append(_build_engagement_tool_call(
            "meetings", [],
            {"_relationship_direct": True, "_relationship_no_records": True,
             "_target": canon or target},
            pack,
        ))
    return tcs


def _try_opportunity_alignment_direct(
    user_question: str, pack: dict, client, model: str,
) -> list | None:
    """Direct opportunity / Vision-2030 query. Joins:
       - opportunities (company OR country keyed)
       - focused_sectors / suggested_opportunities
       - country_vision_outlooks (the Vision 2030-style mapping layer
         for 195 countries — national_vision / diversification_goals
         / five_year_outlook)
       - country_strategic_opportunities
    For company entities we still pull country_vision_outlooks for
    SAUDI ARABIA as the default reference frame (since the audience
    is the Saudi Ministry of Investment)."""
    from app.services.engagement_data import (
        resolve_company_ids, resolve_country_id,
        fetch_opportunity_alignment,
    )
    target = _extract_entity_from_question(user_question, client, model)
    if not target:
        return None
    ids, canon = resolve_company_ids(target)
    # Also try country lookup in case the user asked about a country.
    country_id, country_name = resolve_country_id(target)
    if not ids and not country_id:
        return None

    # MISA-relevance default: also pull Saudi Arabia's vision outlook
    # when the target is a company — it's the audience's reference
    # frame for "what aligns with Vision 2030".
    saudi_id = None
    if ids and not country_id:
        sid, _ = resolve_country_id("Saudi Arabia")
        saudi_id = sid

    data = fetch_opportunity_alignment(ids if ids else None, country_id)
    if saudi_id and not data["country_vision_outlooks"]:
        # Pull Saudi vision as default reference.
        saudi_data = fetch_opportunity_alignment(None, saudi_id)
        data["country_vision_outlooks"] = saudi_data["country_vision_outlooks"]
        if not data["country_strategic_opportunities"]:
            data["country_strategic_opportunities"] = saudi_data["country_strategic_opportunities"]

    pack["_opportunity_target"] = canon or country_name or target

    tcs: list = []
    if data["opportunities"]:
        tcs.append(_build_engagement_tool_call(
            "opportunities", data["opportunities"],
            {"_opportunity_direct": True, "_target": canon or country_name or target},
            pack,
        ))
    if data["focused_sectors"]:
        tcs.append(_build_engagement_tool_call(
            "focused_sectors", data["focused_sectors"],
            {"_opportunity_direct": True, "_target": canon or country_name or target},
            pack,
        ))
    if data["suggested_opportunities"]:
        tcs.append(_build_engagement_tool_call(
            "suggested_opportunities", data["suggested_opportunities"],
            {"_opportunity_direct": True, "_target": canon or country_name or target},
            pack,
        ))
    if data["country_vision_outlooks"]:
        tcs.append(_build_engagement_tool_call(
            "country_vision_outlooks", data["country_vision_outlooks"],
            {"_opportunity_direct": True, "_target": canon or country_name or target},
            pack,
        ))
    if data["country_strategic_opportunities"]:
        tcs.append(_build_engagement_tool_call(
            "country_strategic_opportunities", data["country_strategic_opportunities"],
            {"_opportunity_direct": True, "_target": canon or country_name or target},
            pack,
        ))
    if not tcs:
        # No opportunity data — let normal flow fall through.
        return None
    return tcs


def _try_executive_lookup_direct(
    user_question: str, pack: dict, client, model: str,
    prefetched_target: dict | None = None,
) -> list | None:
    """Direct executive-lookup query.

    1. {person_name, company, role} — passed in via `prefetched_target`
       when available (the chat engine runs the extractor in parallel
       with intent classification to hide latency), else extracted now.
    2. If person_name → run the person-lookup flow (existing).
    3. If company → search company_executives for that company; if a
       role was specified (CEO/Chairman), filter executives whose
       title matches. Attach parent-company row for context.
    4. Returns the tool_calls list on success, None on no-match (so
       caller falls through to the normal LLM-routed path).
    """

    target = prefetched_target if prefetched_target else _extract_exec_target(
        user_question, client, model,
    )
    if not target:
        return None

    person = target.get("person_name")
    company = target.get("company")
    role = target.get("role")
    pack["_exec_lookup_target"] = target

    # Branch A: a specific person was named — use the existing person
    # flow, which already does executives-table search + parent-company
    # context enrichment.
    if person:
        return _try_person_direct_query(
            person_name=person,
            parent_entity=company,  # may be None
            pack=pack,
        )

    # Branch B: company + role (or just company). Find executives of
    # that company by FK (company_profile_id), NOT by smart-search on
    # the executives table — that table's text columns don't include
    # the parent company name, so a name-based search would always
    # return zero rows.
    if not company:
        return None
    company_terms = expand_aliases(company) or [company]

    # Step 1: resolve company → company_profile.id(s)
    company_ids: list = []
    company_df_for_context = None
    try:
        df_co, sql_co, params_co = run_rhq_company_smart_search(company_terms, 3)
        if df_co is not None and not df_co.empty:
            company_ids = [int(x) for x in df_co["id"].tolist() if x is not None]
            company_df_for_context = df_co
    except Exception:
        pass
    if not company_ids:
        return None

    # Step 2: pull executives keyed off those company IDs from the
    # candidate executive tables. Use a direct SQL query because
    # smart_search on those tables hits text columns that don't
    # contain the company name.
    import psycopg2.extras
    import pandas as pd
    from app.database import get_db

    exec_df = None
    exec_sql_used = ""
    exec_params_used: list = []
    exec_table_used: str | None = None
    # Each entry: (table_name, fk_column). We try in priority order.
    fk_tables = (
        ("company_executives", "company_profile_id"),
        ("executives",         "company_profile_id"),
        ("rhq_topexecutives",  "company_profile_id"),
        ("contacts",           "company_profile_id"),
    )
    for table_name, fk_col in fk_tables:
        try:
            conn = get_db()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Verify the column exists before querying — some
                # tables in dev DBs lack the FK.
                cur.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name=%s AND column_name=%s",
                    (table_name, fk_col),
                )
                if not cur.fetchone():
                    continue
                q = (
                    f"SELECT * FROM {table_name} "
                    f"WHERE {fk_col} = ANY(%s) LIMIT 50"
                )
                cur.execute(q, (company_ids,))
                rows = cur.fetchall()
        except Exception:
            continue
        if not rows:
            continue
        df = pd.DataFrame([dict(r) for r in rows])
        # If a role was specified, filter to rows whose title-like
        # column matches. Cover the most common column names.
        if role:
            role_lc = role.lower()
            kept = []
            for r in df.to_dict(orient="records"):
                title_blob = " ".join(
                    str(r.get(k, "") or "").lower()
                    for k in ("title", "role", "position", "designation",
                              "current_title", "exec_title", "job_title")
                )
                hit = (
                    role_lc in title_blob
                    or (role_lc == "ceo" and "chief executive" in title_blob)
                    or (role_lc == "chairman" and (
                        "chair" in title_blob or "chairperson" in title_blob))
                    or (role_lc == "founder" and "founder" in title_blob)
                )
                if hit:
                    kept.append(r)
            if kept:
                df = pd.DataFrame(kept)
            # If filtering wiped everything, fall back to ALL execs
            # so the user at least sees the company's leadership;
            # the curation prompt will note no exact role match.
        exec_df = df
        exec_sql_used = f"SELECT * FROM {table_name} WHERE {fk_col} = ANY(:ids)"
        exec_params_used = [company_ids]
        exec_table_used = table_name
        break

    if exec_df is None or exec_df.empty:
        return None
    exec_rows = exec_df
    exec_sql = exec_sql_used
    exec_params = exec_params_used
    exec_table = exec_table_used

    tool_calls = [{
        "table": exec_table,
        "filters": {
            "_executive_lookup_direct": True,
            "_target_company": company,
            "_target_role": role,
        },
        "sql": exec_sql, "params": exec_params, "rows_df": exec_rows,
        "row_count": int(len(exec_rows)),
        "input_trace": dict(pack),
        "sql_entity_check_passed": True,
        "row_entity_sanity_passed": True,
        "closest_names": [],
    }]
    # Attach the company profile rows for curator context. We already
    # fetched them when resolving the IDs; reuse that DataFrame instead
    # of double-querying.
    if company_df_for_context is not None and not company_df_for_context.empty:
        tool_calls.append({
            "table": COMPANY_TABLE,
            "filters": {
                "_executive_lookup_context": True,
                "_target_company": company,
            },
            "sql": "(reused from company_ids resolution)",
            "params": [],
            "rows_df": company_df_for_context,
            "row_count": int(len(company_df_for_context)),
            "input_trace": dict(pack),
            "sql_entity_check_passed": True,
            "row_entity_sanity_passed": True,
            "closest_names": [],
        })
    # CORRELATOR: for named-person bios, a light company bundle helps
    # Strategic Read. For "who is the CEO of <Company>?" role lookups,
    # the full meetings/contacts/opportunities dump causes local models
    # to emit a company briefing instead of a person brief — skip it.
    _role_only = bool(role) and not person
    if not _role_only:
        try:
            from app.services.correlator import (
                correlate_company, bundle_summary_for_prompt,
            )
            bundle = correlate_company(company_ids)
            summary = bundle_summary_for_prompt(bundle)
            # Add tool_calls for each non-empty correlator section that we
            # didn't already include via the executive lookup itself.
            for key, table_label in [
                ("opportunities",          "opportunities"),
                ("misa_contacts",          "misa_contact_details"),
                ("meetings",               "meetings"),
                ("ai_insights",            "company_ai_insights"),
                ("news",                   "company_news"),
                ("strategic_investors",    "strategic_investors"),
            ]:
                rows = summary.get(key) or []
                if not rows:
                    continue
                tool_calls.append(_build_engagement_tool_call(
                    table_label, rows,
                    {"_exec_correlator_context": True,
                     "_target_company": company,
                     "_correlator_section": key},
                    pack,
                ))
        except Exception:
            # Correlator is augmentation only — never block the answer.
            pass
    return tool_calls


def _try_person_direct_query(
    person_name: str, parent_entity: str | None, pack: dict,
) -> list | None:
    """Search the executive tables for `person_name`. Returns a tool-call
    payload ready for the local commentary composer, or None if no rows
    found across any of the tables.

    The chat engine routes here when the LLM resolver classifies the
    entity as a person — bypassing the company smart-search + fuzzy
    clarification path that would surface garbage like 'Standard
    Chartered' for 'Sundar Pichai'."""
    from app.database import smart_search
    name = (person_name or "").strip()
    if not name:
        return None
    # Try aliases for the parent (so "Sundar Pichai" of "Alphabet" also
    # surfaces records keyed under "Google Regional Office").
    parent_terms = expand_aliases(parent_entity) if parent_entity else []
    tool_calls: list = []
    for table in _PERSON_LOOKUP_TABLES:
        try:
            df, sql, params = smart_search(table, [name], 10)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        tool_calls.append({
            "table": table,
            "filters": {
                "_person_direct_query": True,
                "_person_name": name,
                "_parent_entity": parent_entity,
            },
            "sql": sql, "params": params, "rows_df": df,
            "row_count": int(len(df)),
            "input_trace": dict(pack),
            "sql_entity_check_passed": True,
            "row_entity_sanity_passed": True,
            "closest_names": [],
        })
        # Stop at first hit — additional tables would be duplicates.
        break
    if not tool_calls:
        return None
    # Optionally enrich with the parent company's profile row so the
    # curator can ground the person in their company context.
    if parent_entity and parent_terms:
        try:
            df_p, sql_p, params_p = run_rhq_company_smart_search(parent_terms, 3)
            if df_p is not None and not df_p.empty:
                tool_calls.append({
                    "table": COMPANY_TABLE,
                    "filters": {
                        "_person_parent_context": True,
                        "_person_name": name,
                        "_parent_entity": parent_entity,
                    },
                    "sql": sql_p, "params": params_p, "rows_df": df_p,
                    "row_count": int(len(df_p)),
                    "input_trace": dict(pack),
                    "sql_entity_check_passed": True,
                    "row_entity_sanity_passed": True,
                    "closest_names": [],
                })
        except Exception:
            pass
    return tool_calls


# ─── Web-search augmentation for forward-looking exec questions ─────

_EXEC_NEWS_PROMPT = """You are a corporate-news analyst. The user asked a forward-looking question:

QUESTION: {question}

Use the WEB SOURCES below to give the executive a DIRECT answer
first, then supporting evidence.

WEB SOURCES (numbered for citation):
{web_evidence}

EXACT output format (no preamble):

## What's Reported (Live Web)

**<ONE-SENTENCE DIRECT ANSWER>** [web:N]

The first line MUST be a bold, single sentence that names the
specific person / outcome the user asked about, with a citation.
Examples of strong opening lines:
  - "**John Ternus is reported as Tim Cook's expected successor as
     Apple CEO, with the transition planned for April 2026.** [web:1]"
  - "**Microsoft is intensifying succession planning but no successor
     has been publicly named.** [web:2]"
  - "**No specific successor has been confirmed.** [web:3]"

DO NOT open with "Tim Cook is expected to step down" (that's the
PREMISE, not the answer). Open with the NAMED SUCCESSOR or, when
no name is reported, an honest "no name confirmed" statement.

Then a heading and 2-4 supporting bullets:

### Supporting reporting
- <observation> [web:N]
- <observation> [web:N]

Each supporting bullet ≤ 25 words; every substantive claim cited.

If WEB SOURCES is empty or the sources don't actually address the
question, output ONLY:

## What's Reported (Live Web)
*No reliable web sources found for this forward-looking question.*

Hard rules:
  - Cite EVERY substantive claim with [web:N].
  - Do NOT invent dates or names that aren't in the sources.
  - If sources disagree, the opening sentence should note that
    ("...with two candidates reported: X and Y") rather than
    arbitrarily picking one.
  - Keep it tight: bullets ≤ 25 words each."""


_EXEC_CURRENT_HOLDER_PROMPT = """You are a MISA investment-intelligence briefing writer.
Answer who CURRENTLY holds a public / government / cabinet office.

QUESTION: {question}

WEB EVIDENCE (numbered — cite as [web:N]):
{web_evidence}

INTERNAL DRAFT FROM MISA TABLES (for your eyes only — never quote
table/column names, never say "database draft", "may lag", or
"outdated record"):
{db_answer}

Write like a calm boardroom brief. The reader is a senior MISA officer.
Sound sure when sources agree; sound careful when they do not.
Never sound like a system log, a QA checklist, or a disclaimer page.

EXACT output format (no preamble, no meta commentary):

## Role

**<Full name> is Saudi Arabia's <office>, appointed <date if known>.** [web:N]

- <one supporting fact: prior role / succession / mandate> [web:N]
- <one supporting fact if useful; else omit> [web:N]

_Sources: public reporting._

Hard rules:
  - Lead with the CURRENT holder. Web evidence wins over the internal draft.
  - Prefer the most recent dated reporting (a 2026 replacement beats an older appointment order).
  - Cite substantive claims with [web:N].
  - Do NOT invent names or dates absent from WEB EVIDENCE.
  - FORBIDDEN phrases / headings (never write these):
      "Verified facts", "Live Web", "officeholder", "MISA record",
      "may lag", "appears outdated", "database draft", "company_executives",
      "web:1" as prose, any `table.column` source path.
  - Do not tell the reader you are correcting the system. Just give the right answer.
  - If sources conflict, one short careful sentence — not a self-audit.
"""


def _augment_exec_answer_with_web(
    db_answer: str, user_question: str, client, model: str,
    *, lead_with_web: bool = False,
    capture_sources: list | None = None,
    mode: str = "succession",
) -> str:
    """Augment the DB-grounded executive answer with a cited live-web section.

    mode:
      - "succession" — forward-looking successor reporting
      - "current_office" — verify who currently holds a cabinet / public office
        (web leads; DB is treated as possibly stale)

    Order:
      - lead_with_web=False: web section APPENDED
      - lead_with_web=True: web section PREPENDED (or replaces for current_office)
    """
    # docs_only mode: never consult the live web on the exec path either.
    from app import config as _cfg
    if getattr(_cfg, "DOCUMENTS_WEB_MODE", "hybrid") == "docs_only":
        return db_answer

    try:
        from app.services import web_search
        # Prefer recent cabinet/appointment reporting. Do NOT bias toward
        # "royal decree" alone — that surfaces historic appointment orders
        # (e.g. 2020 O/255 for Al-Falih) over 2026 replacement coverage.
        q = user_question
        if mode == "current_office":
            q = (
                f"{user_question.strip()} — who currently holds this office "
                f"as of 2026? Prefer the most recent cabinet reshuffle or "
                f"replacement reporting; treat older appointment decrees as historical."
            )
        results = web_search.search(q, max_results=6)
    except Exception:
        results = []
    if capture_sources is not None and results:
        capture_sources.clear()
        # UI Sources panel: skip the synthesis stub (no URL).
        capture_sources.extend(
            r for r in results if (r.get("url") or "").startswith("http")
        )

    if not results:
        # Empty web lane is noise — never required. Return the DB brief alone.
        return db_answer

    try:
        from app.services import web_search
        from app.services.style_guide import STYLE_GUIDE_PROMPT
        evidence = web_search.format_for_prompt(results)
        from app.config import openai_max_completion_tokens_kw
        compose_client, compose_model = client, model
        if mode == "current_office":
            # db_answer is Postgres-grounded prose — must not leave the
            # machine under residency strict (compose on Ollama).
            from app.services.llm_residency import resolve_data_completion_client
            compose_client, compose_model = resolve_data_completion_client(
                client, preferred_model=model,
            )
            user_content = STYLE_GUIDE_PROMPT + "\n\n" + _EXEC_CURRENT_HOLDER_PROMPT.format(
                question=user_question,
                web_evidence=evidence,
                db_answer=(db_answer or "")[:2500],
            )
        else:
            user_content = STYLE_GUIDE_PROMPT + "\n\n" + _EXEC_NEWS_PROMPT.format(
                question=user_question, web_evidence=evidence,
            )
        resp = compose_client.chat.completions.create(
            model=compose_model,
            messages=[{"role": "user", "content": user_content}],
            **_det_kw(),
            **openai_max_completion_tokens_kw(),
        )
        section = (resp.choices[0].message.content or "").strip()
        if not section:
            return db_answer
        if mode == "current_office":
            return _polish_officeholder_answer(section)
        if lead_with_web:
            return f"{section}\n\n{db_answer}"
        return f"{db_answer}\n\n{section}"
    except Exception:
        return db_answer


def _polish_officeholder_answer(answer: str) -> str:
    """Strip trust-killing meta tone and schema leaks from officeholder briefs."""
    if not answer:
        return answer
    import re as _re
    text = answer
    # Drop forbidden meta headings the model sometimes still emits.
    text = _re.sub(
        r"(?im)^#{1,3}\s*(Current officeholder.*|Verified facts|MISA record.*)\s*$",
        "",
        text,
    )
    # Soften leftover self-audit sentences about the DB lagging.
    text = _re.sub(
        r"(?im)^.*\b(appears outdated|database draft|may lag|MISA record)\b.*$",
        "",
        text,
    )
    # Never leak table.column paths in the reader-facing footer.
    text = _re.sub(
        r"(?i)_?Sources?:\s*[^.\n]*\bcompany_executives\b[^.\n]*\.?",
        "_Sources: public reporting._",
        text,
    )
    text = _re.sub(
        r"(?i)\b(?:company_executives|company_profiles|executives|contacts)"
        r"\.[a-z_]+\b",
        "",
        text,
    )
    # If we nuked the footer, restore a clean one.
    if not _re.search(r"(?i)_Sources:", text):
        text = text.rstrip() + "\n\n_Sources: public reporting._"
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + ("\n" if answer.endswith("\n") else "")


def _run_advisory_path(user_question: str, pack: dict, ui_locale: str,
                       resp_loc: str, client) -> dict | None:
    """Compose the full advisory-report response (deliverable-shaped
    document, DB grounding, deterministic validation). Returns the
    chat result dict, or None on generation failure so the caller can
    fall through to the normal pipeline.

    Called from TWO gates: the deterministic regex fast-path (obvious
    phrasings, saves the ~2.6s classifier wait) and the LLM intent
    classifier (intent == 'strategic_advisory') — which is what makes
    NOVEL phrasings route correctly without anyone maintaining regex
    lists."""
    from app.config import ADVISORY_MODEL, OPENAI_MODEL
    db_ctx = _advisory_country_context(user_question)
    deliverable = _detect_advisory_deliverable(user_question)
    answer = strategic_advisory_answer(
        user_question,
        db_context=db_ctx,
        deliverable=deliverable,
        locale=resp_loc,
        client=client,
        model=ADVISORY_MODEL,
    )
    if not answer and ADVISORY_MODEL != OPENAI_MODEL:
        # The advisory tier may be unavailable on this API key — a
        # thinner report from the chat model still beats losing the
        # advisory structure entirely.
        answer = strategic_advisory_answer(
            user_question,
            db_context=db_ctx,
            deliverable=deliverable,
            locale=resp_loc,
            client=client,
            model=OPENAI_MODEL,
        )
    if not answer:
        return None
    if db_ctx:
        pack["_advisory_db_context"] = db_ctx
        pack["_retrieval"] = db_ctx.get("retrieval") or {
            "retrieval_status": db_ctx.get("retrieval_status"),
        }
    # Deterministic post-generation guard: strip fabricated footprint
    # sections (no DB context) and rebuild footprint counts that don't
    # match the database. Code-enforced — prompt instructions drift.
    try:
        from app.services.response_validator import (
            validate_advisory_answer,
        )
        answer, _adv_fixes = validate_advisory_answer(
            answer, db_ctx, deliverable=deliverable,
        )
        if _adv_fixes:
            pack["_advisory_validation_fixes"] = _adv_fixes
        try:
            from app.services.advisory_enrichment import (
                enrich_advisory_deliverable,
            )
            answer, _enr_fixes = enrich_advisory_deliverable(
                answer, deliverable=deliverable, db_context=db_ctx,
            )
            if _enr_fixes:
                pack["_advisory_enrichment_fixes"] = _enr_fixes
            # Repair pass: if required Jul21 sections still missing after
            # the first enrich, re-run scaffolds once (idempotent).
            try:
                from app.services.answer_contracts import (
                    advisory_deliverable_violations,
                )
                _miss = advisory_deliverable_violations(
                    answer, deliverable=deliverable,
                )
                if _miss:
                    pack["_advisory_section_gaps"] = _miss
                    answer, _repair = enrich_advisory_deliverable(
                        answer, deliverable=deliverable, db_context=db_ctx,
                    )
                    if _repair:
                        pack.setdefault(
                            "_advisory_enrichment_fixes", [],
                        ).extend(
                            [f"repair:{x}" for x in _repair]
                        )
                    _miss2 = advisory_deliverable_violations(
                        answer, deliverable=deliverable,
                    )
                    if _miss2:
                        pack["_advisory_section_gaps_after_repair"] = _miss2
            except Exception:
                pass
        except Exception:
            pass
        try:
            from app.services.quality_gate import run_quality_gate
            answer, _qg_issues, _qg_fixes = run_quality_gate(
                answer,
                question=user_question,
                db_context=db_ctx,
                retrieval_meta=(db_ctx or {}).get("retrieval"),
                deliverable=deliverable,
            )
            if _qg_fixes:
                pack["_quality_gate_fixes"] = _qg_fixes
            if _qg_issues:
                pack["_quality_gate_issues"] = [
                    i.get("code") for i in _qg_issues
                ]
                try:
                    from app.logger import logger as _log
                    _log.warning(
                        "quality_gate issues=%s fixes=%s",
                        pack["_quality_gate_issues"], _qg_fixes,
                    )
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
    try:
        from app.services.answer_finalize import finalize_answer
        pack["_answer_source"] = "strategic_advisory"
        pack["_advisory_deliverable"] = deliverable
        pack["_short_circuit"] = "strategic_advisory"
        answer = finalize_answer(
            answer, user_question=user_question, pack=pack,
        )
    except Exception:
        pass
    # Fail-closed: after finalize, if the Jul21 advisory shape is still
    # missing (LLM sometimes emits a named-company Engagement Recommendation
    # stub), force enrichment scaffolds again.
    try:
        from app.services.answer_contracts import advisory_deliverable_violations
        from app.services.advisory_enrichment import enrich_advisory_deliverable
        _gaps = advisory_deliverable_violations(
            answer, deliverable=deliverable,
        )
        if _gaps:
            pack["_advisory_post_finalize_gaps"] = _gaps
            answer, _pf = enrich_advisory_deliverable(
                answer, deliverable=deliverable, db_context=db_ctx,
            )
            if _pf:
                pack.setdefault("_advisory_enrichment_fixes", []).extend(
                    [f"post_finalize:{x}" for x in _pf]
                )
    except Exception:
        pass
    pack["_short_circuit"] = "strategic_advisory"
    pack["_advisory_deliverable"] = deliverable
    if db_ctx:
        pack["_advisory_origin_country"] = db_ctx.get("origin_country")
        pack["_advisory_retrieval_status"] = db_ctx.get("retrieval_status")
        pack["_advisory_licensed"] = db_ctx.get(
            "companies_from_origin_licensed_in_saudi")
        pack["_advisory_rhq"] = db_ctx.get("companies_from_origin_with_rhq")
        pack["_advisory_expansion_n"] = len(
            db_ctx.get("expansion_targets") or [])
        if db_ctx.get("footprint_data_unavailable"):
            pack["_advisory_footprint_unavailable"] = True
            pack["_advisory_db_error"] = db_ctx.get("_db_error")
        try:
            from app.logger import logger as _log
            _log.info(
                "advisory_path deliverable=%s country=%s status=%s "
                "licensed=%s rhq=%s expansion=%s filters=%s",
                deliverable,
                db_ctx.get("origin_country"),
                db_ctx.get("retrieval_status")
                or ("error" if db_ctx.get("footprint_data_unavailable")
                    else "none"),
                db_ctx.get("companies_from_origin_licensed_in_saudi"),
                db_ctx.get("companies_from_origin_with_rhq"),
                len(db_ctx.get("expansion_targets") or []),
                db_ctx.get("retrieval_filters"),
            )
        except Exception:
            pass
    # Lift deliverable + DB context onto the result so JSON polish,
    # SSE, PDF/DOCX, and finalize all see the same contract — not only
    # the in-path validator.
    return {
        "answer": answer,
        "tool_calls": [{"input_trace": dict(pack)}],
        "error": None,
        "_answer_source": "strategic_advisory",
        "_advisory_deliverable": deliverable,
        "_advisory_db_context": db_ctx,
        "_short_circuit": "strategic_advisory",
        "feedback_context": hf.build_feedback_context(
            user_question, ui_locale, resp_loc, pack,
        ),
    }


# ---------------------------------------------------------------------------
# DB-query security guards (Risk-20-1 / Risk-20-6)

_count_only_limiter = RateLimiter(
    max_requests=COUNT_ONLY_RATE_LIMIT[0],
    window_seconds=COUNT_ONLY_RATE_LIMIT[1],
)


def _reject_table_and_audit(table: str) -> dict:
    """Tool-error payload for a disallowed/unknown `table`, plus a security
    event naming it (Risk-20-1). The returned dict is byte-for-byte what the
    inline literal produced before, so the model — and therefore the user's
    answer — sees exactly what it saw previously; only the audit trail is new.
    """
    try:
        from app.services.audit_log import emit_security_event
        emit_security_event({
            "event": "query_table_blocked",
            "table": table or "(empty)",
        })
    except Exception:
        pass
    return {"error": f"table {table!r} is not allowed or unknown"}


def _audit_blocked_columns(table: str, filters: dict) -> None:
    """Record filter columns the model asked for that the catalog does not
    expose (Risk-20-1 blocked-column logging).

    Pure observation: the SQL builder already drops unknown/denied columns
    silently, and this function does not touch `filters` — it only compares
    against the same catalog the builder uses and logs the delta. Sensitive
    columns (password/token/passport/...) never appear in `filterable`, so a
    request for one lands here. Keys starting with "_" are internal trace
    markers added server-side, not model input, so they are excluded.
    """
    try:
        requested = {
            str(k) for k in (filters or {}) if not str(k).startswith("_")
        }
        if not requested:
            return
        info = get_table_info(table) or {}
        allowed = set(info.get("filterable") or ())
        blocked = sorted(requested - allowed)
        if not blocked:
            return
        from app.services.audit_log import emit_security_event
        emit_security_event({
            "event": "query_table_column_blocked",
            "table": table,
            "columns": blocked,
        })
    except Exception:
        pass


def _count_only_guard(table: str, filters: dict) -> bool:
    """Gate + log a count-only query (Risk-20-6 inference/enumeration).

    Counting is the one way to learn about rows without reading their
    values, so repeated counts with varying filters can map out restricted
    data indirectly. Every call is logged for visibility; the rate limit
    (per authenticated identity) only engages under scripted probing — the
    default is far above what conversation can reach.

    Returns True if the call may proceed. Fails OPEN on any internal error:
    a bug in the limiter must never break a legitimate count.
    """
    from app.services.audit_log import emit_security_event, get_audit_user
    try:
        allowed, _retry_after = _count_only_limiter.check(get_audit_user())
    except Exception:
        allowed = True
    try:
        emit_security_event({
            "event": "count_only_query",
            "table": table,
            "filter_cols": sorted(
                str(k) for k in (filters or {}) if not str(k).startswith("_")
            ),
            "rate_limited": not allowed,
        })
    except Exception:
        pass
    return allowed


def _chat_execute(user_question: str, history: list, ui_locale: str = "en") -> dict:
    from app.config import OPENAI_MODEL

    pack = clean_user_question(user_question)
    resp_loc = _effective_response_locale(ui_locale, user_question)
    client = get_openai_client()

    # Structured intent + pipeline trace (Phase 1/7) — before any retrieval.
    try:
        from app.services.query_intent import build_query_intent
        from app.services.pipeline_trace import new_trace
        _qi = build_query_intent(
            user_question,
            history,
            entity_candidate=pack.get("entity_candidate"),
        )
        pack["_query_intent"] = _qi.to_dict()
        pack["_pipeline_trace"] = new_trace()
        pack["_pipeline_trace"].intent = _qi.to_log_dict()
    except Exception:
        pack["_query_intent"] = {"task_type": "unknown"}

    if client is None:
        return {
            "answer": "",
            "tool_calls": [],
            "error": "OPENAI_API_KEY not configured.",
            "feedback_context": hf.build_feedback_context(user_question, ui_locale, resp_loc, pack),
        }

    # PROMPT-ATTACK GUARD (deterministic, pre-LLM/DB).
    # Refuse obvious instruction-override / system-prompt-extraction /
    # schema-dump / internal-note-disclosure / role-hijack attempts
    # before any model or database work. Defense in depth on top of the
    # curation privacy filter + provenance guardrails downstream. Runs
    # first so the cheapest, most common attacks never reach a model.
    from app.services import prompt_guard as _pg
    _attack, _attack_category = _pg.detect_prompt_attack(user_question)
    if _attack:
        pack["_short_circuit"] = "prompt_injection"
        pack["_prompt_attack_category"] = _attack_category
        # Match this module's logging style (stderr line, SIEM-friendly).
        # Category only — never the user's raw text — so the log doesn't
        # become a store of attack payloads / any PII they contain.
        print(
            f"[prompt_guard] refused turn (category={_attack_category})",
            file=sys.stderr, flush=True,
        )
        return {
            "answer": _pg.refusal_reply(resp_loc),
            "tool_calls": [{"input_trace": dict(pack)}],
            "error": None,
            "_answer_source": "prompt_guard_refusal",
            "feedback_context": hf.build_feedback_context(
                user_question, ui_locale, resp_loc, pack,
            ),
        }

    # DOCUMENT RETRIEVAL (library → optional web complement → else DB).
    # Strong library hits answer from documents. In hybrid mode (default)
    # we also consult the live web and merge both with dual provenance.
    # docs_first keeps the legacy exclusive short-circuit; docs_only never
    # calls the web. Explicit "only from the document" overrides hybrid.
    from app import config as _doc_cfg
    # A multi-page advisory deliverable (market fit, engagement plan,
    # sector priorities, targeting, corridor strategy) must NOT be
    # pre-empted by an incidental document match — e.g. an uploaded
    # internet-penetration report matching "Saudi Arabia" in a German
    # market-fit question. Only an explicit "from the document" ask
    # lets a document win over the advisory path.
    from app.services.document_ingest import wants_docs_only as _wants_docs_only
    _skip_docs_for_advisory = (
        _is_advisory_question(user_question)
        and not _wants_docs_only(user_question)
    )
    if getattr(_doc_cfg, "DOCUMENTS_ENABLED", False) and not _skip_docs_for_advisory:
        try:
            from app.services.audit_log import get_audit_user
            from app.services.document_ingest import (
                compose_document_answer,
                compose_hybrid_document_web_answer,
                should_augment_docs_with_web,
            )
            from app.services.document_store import get_document_store
            _doc_user = get_audit_user()
            if _doc_user and _doc_user not in ("unknown", "anonymous", "invalid-token"):
                _hits = get_document_store().retrieve(user_question, _doc_user)
                _enough_probe = compose_document_answer(
                    user_question, _hits, footer="",
                )
                if _enough_probe.get("enough"):
                    _web_results: list = []
                    if should_augment_docs_with_web(user_question):
                        try:
                            from app.services import web_search as _ws
                            _web_results = _ws.search(user_question, max_results=5) or []
                        except Exception:
                            _web_results = []
                    _composed = compose_hybrid_document_web_answer(
                        user_question, _hits, _web_results,
                    )
                    _src = _composed.get("answer_source") or "document"
                    pack["_short_circuit"] = (
                        "document_library_hybrid" if _src == "hybrid"
                        else "document_library"
                    )
                    pack["_document_hit_count"] = len(_hits)
                    _doc_answer = _composed["answer"]
                    try:
                        from app.services.answer_finalize import finalize_answer
                        pack["_answer_source"] = _src
                        pack["_keep_citations"] = True
                        _doc_answer = finalize_answer(
                            _doc_answer,
                            user_question=user_question,
                            pack=pack,
                        )
                    except Exception:
                        pass
                    return {
                        "answer": _doc_answer,
                        "tool_calls": [{
                            "input_trace": dict(pack),
                            "table": "_documents",
                            "row_count": len(_hits),
                            "filters": {
                                "_document_first": True,
                                "_docs_web_mode": getattr(
                                    _doc_cfg, "DOCUMENTS_WEB_MODE", "hybrid"
                                ),
                                "_web_augmented": _src == "hybrid",
                            },
                        }],
                        "error": None,
                        "_answer_source": _src,
                        "web_sources": _composed.get("web_sources") or [],
                        "doc_sources": _composed.get("doc_sources") or [],
                        "feedback_context": hf.build_feedback_context(
                            user_question, ui_locale, resp_loc, pack,
                        ),
                    }
        except Exception:
            # Document path must never take the whole chat down.
            pass

    # DEEP-PROFILE MODE (explicit opt-in ONLY).
    # Triggered by '/profile X' or 'deep profile of X'. Natural
    # "briefing on X" / "company profile for X" use the Jul21 company
    # path — not this heavier 3-pillar mode.
    from app.services import deep_profile as _dp
    deep_target = _dp.is_deep_profile_request(user_question)
    # GUARD: deep-profile is for a SPECIFIC NAMED ENTITY (a company,
    # person, country). Skip market segments and corridor advisories.
    if deep_target and (
        _looks_like_market_segment(deep_target)
        or (_is_advisory_question(user_question)
            and _detect_origin_country(user_question))
    ):
        deep_target = None
    # GUARD: CEO / person asks and sector briefings must never be stolen
    # by deep-profile even if a future trigger regresses.
    if deep_target:
        try:
            from app.services.db_briefing import _question_looks_like_person
            if _question_looks_like_person(user_question):
                deep_target = None
        except Exception:
            pass
    if deep_target and re.search(
        r"(?i)\bsector\s+(?:briefing|overview|opportunit)|"
        r"\bopportunit(?:y|ies)\s+in\b",
        user_question or "",
    ):
        deep_target = None
    if deep_target:
        # Pass entity_type=None — compose_deep_profile runs its own
        # bare-name classifier, because the conversational resolver
        # doesn't work on stand-alone '/profile X' input (no history
        # to anchor pronoun resolution to).
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")
        result = _dp.compose_deep_profile(
            target=deep_target,
            entity_type=None,
            parent_entity=None,
            today=today,
        )
        # Prepend the warning so the user always sees the mode banner
        # AND the privacy disclosure before the profile body.
        body = result["answer"]
        warn = result.get("warning") or ""
        if warn:
            body = f"> {warn.replace(chr(10), chr(10) + '> ')}\n\n{body}"
        # Reflect the trigger + classifier decision in the pack trace.
        pack["_deep_profile_mode"] = True
        pack["_deep_profile_target"] = deep_target
        pack["_deep_profile_entity_type"] = result.get("entity_type")
        pack["_deep_profile_parent"] = result.get("parent_entity")
        try:
            from app.services.answer_finalize import finalize_answer
            pack["_answer_source"] = "deep_profile"
            pack["_keep_citations"] = True
            body = finalize_answer(
                body, user_question=user_question, pack=pack,
            )
        except Exception:
            pass
        return {
            "answer": body,
            "tool_calls": result["tool_calls"],
            "error": None,
            "web_sources": result.get("web_results") or [],
            "feedback_context": hf.build_feedback_context(
                user_question, ui_locale, resp_loc, pack,
            ),
        }

    # STRATEGIC ADVISORY SHORT-CIRCUIT.
    # Topic-level attraction-strategy questions ("market fit for
    # attracting Indian companies to Saudi Arabia", "investment case
    # for German manufacturers in KSA") have no matching DB rows, so
    # the normal pipeline used to dump them into the 150-word
    # general-knowledge fallback — a thin generic paragraph. Route them
    # instead to a dedicated consultant-grade report path, grounded
    # (best-effort) with the origin country's real MISA figures:
    # licensed count, RHQ count, top companies. Runs BEFORE intent
    # classification both to save the ~2.6s classifier call and because
    # the LLM classifier tends to mislabel these as off_topic.
    if _is_advisory_question(user_question):
        _adv_result = _run_advisory_path(
            user_question, pack, ui_locale, resp_loc, client)
        if _adv_result:
            return _adv_result
        # Advisory generation failed (OpenAI error) — fall through to
        # the normal pipeline rather than returning nothing.

    # SECTOR AGGREGATION — early short-circuit (before intent classifier).
    # "momentum across all sectors" is often mislabelled as general
    # research / off-topic and falls to thin GK. Aggregation detection
    # is deterministic; do not wait on the classifier.
    if _is_sector_aggregation_question(user_question):
        try:
            tcs = _try_sector_aggregation_direct(
                user_question, pack, client, OPENAI_MODEL,
            )
        except Exception:
            tcs = None
        if tcs is not None:
            rows = pack.get("_sector_aggregation_rows") or []
            if rows:
                answer = _format_sector_opportunity_briefing(rows)
                try:
                    from app.services.answer_finalize import finalize_answer
                    answer = finalize_answer(
                        answer, user_question=user_question, pack=pack,
                    )
                except Exception:
                    pass
                return {
                    "answer": answer, "tool_calls": tcs, "error": None,
                    "_answer_source": "sector_aggregation",
                    "feedback_context": hf.build_feedback_context(
                        user_question, ui_locale, resp_loc, pack,
                    ),
                }

    # INTENT CLASSIFICATION (LLM-driven, intent-first) + speculative
    # EXECUTIVE-TARGET EXTRACTION in parallel.
    #
    # The classifier and the exec-target extractor are independent LLM
    # calls on the same input. Profiled at ~2.6s and ~1.3s respectively
    # when run sequentially. Running them in parallel cuts the
    # exec-question pre-route from ~4s to the classifier's ~2.6s alone.
    # If the intent turns out NOT to be executive_*, the extractor's
    # result is discarded — a deliberate "speculative work" trade-off,
    # paying ~1.3s of LLM time we'd otherwise pay sequentially anyway,
    # in exchange for a guaranteed 1.3s wall-clock saving on the common
    # case (executive questions).
    from concurrent.futures import ThreadPoolExecutor
    from app.services.intent_router import classify_intent as _classify

    def _safe_classify():
        try:
            return _classify(user_question, history or [], client, OPENAI_MODEL)
        except Exception as _e:
            return {"intent": "general_research", "confidence": 0.0,
                    "reasoning": f"intent classifier crash: {_e}"}

    def _safe_extract():
        try:
            return _extract_exec_target(user_question, client, OPENAI_MODEL)
        except Exception:
            return {}

    with ThreadPoolExecutor(max_workers=2) as _pool:
        fut_intent = _pool.submit(_safe_classify)
        fut_exec = _pool.submit(_safe_extract)
        intent_meta = fut_intent.result()
        # Stash the speculative result; only consumed by the
        # executive-lookup branch below.
        _speculative_exec_target = fut_exec.result()

    pack["_intent"] = intent_meta.get("intent")
    pack["_intent_confidence"] = intent_meta.get("confidence")
    pack["_intent_reasoning"] = intent_meta.get("reasoning")

    # Engagement-plan phrasing must not stay as broad_topic / entity_lookup
    # — that made curation emit Executive Briefing instead of the
    # Engagement Recommendation → Snapshot → MENA → Strategic Read shape.
    try:
        from app.services.db_briefing import _question_looks_like_engagement_plan
        if _question_looks_like_engagement_plan(user_question):
            if pack.get("_intent") != "engagement_strategy":
                pack["_intent_overridden_from"] = pack.get("_intent")
                pack["_intent"] = "engagement_strategy"
                pack["_intent_reasoning"] = (
                    (pack.get("_intent_reasoning") or "")
                    + " | forced engagement_strategy from question shape"
                ).strip(" |")
    except Exception:
        pass

    # DEPTH DETECTION (Tier 3 commit 1) — intent says WHAT the user
    # wants; depth says HOW MUCH. Same intent + different depth →
    # different answer shape (1-line fact vs 10-section briefing).
    # Pure regex, no LLM call.
    from app.services.depth_detector import detect_depth
    _depth, _depth_trigger = detect_depth(user_question)
    pack["_depth"] = _depth
    pack["_depth_trigger"] = _depth_trigger

    # PRE-OFF-TOPIC INTERCEPTS — before refusing as off-topic, check
    # for benign cases that the LLM classifier tends to mislabel:
    #   - Capability questions ("what can you do") → describe what
    #     the bot is for + 4 example questions. Useful onboarding,
    #     never a refusal.
    #   - Too-short / punctuation-only inputs ("x", "??!!") → ask
    #     for clarification with example questions, don't refuse.
    #   - Vague follow-ups ("tell me more") with history → extract
    #     the last user-mentioned entity and pivot to a deeper
    #     question on it.
    # These three were the only NON-correct refusals in the stress
    # battery (the security / off-topic / emotional refusals were
    # all the right call).
    _q_stripped = (user_question or "").strip()

    if _is_capability_question(_q_stripped):
        return _capability_response(pack, ui_locale, resp_loc, user_question)
    if _is_too_short_or_meaningless(_q_stripped):
        return _clarification_request_response(pack, ui_locale, resp_loc, user_question)
    if _is_vague_followup(_q_stripped) and history:
        rewritten = _rewrite_vague_followup(user_question, history)
        if rewritten and rewritten != user_question:
            # Re-route the rewritten question through the full pipeline
            # — much cleaner than trying to special-case "tell me more"
            # in every downstream branch.
            return chat(rewritten, history, ui_locale)

    # LLM-CLASSIFIED ADVISORY ROUTE. The regex fast-path earlier only
    # recognises phrasings someone anticipated; the classifier
    # understands ANY phrasing ("if MISA wants more money flowing in
    # from Brazil, what's the smartest move?"). This is the structural
    # answer to the routing whack-a-mole: new advisory phrasings route
    # correctly without maintaining pattern lists. Deliberately placed
    # AFTER the vague-followup rewrite, and guarded against pronoun
    # follow-ups: "make me a plan for them" (with history) must inherit
    # the prior entity downstream, not become a generic report.
    _pronoun_followup = bool(history) and bool(re.search(
        r"\b(them|they|it|its|this|that|these|those|him|her)\b",
        user_question or "", re.I))
    if (intent_meta.get("intent") == "strategic_advisory"
            and not _pronoun_followup):
        _adv_result = _run_advisory_path(
            user_question, pack, ui_locale, resp_loc, client)
        if _adv_result:
            return _adv_result
        # Generation failure → continue down the normal pipeline.

    # OFF-TOPIC SHORT-CIRCUIT.
    # If the classifier labelled the input as off_topic (emotional
    # venting, vulgarity, opinion-seeking, conversational meta,
    # harmful content, off-domain), return a polite redirect
    # IMMEDIATELY — before any DB lookup, web search, or curation.
    # This prevents the system from attaching a "Strategic Read for
    # MISA" to a frustration statement like "i hate tim cook so
    # much", which would read as endorsement of the input.
    if intent_meta.get("intent") == "off_topic":
        # FACTUAL off-topic ("what is the capital of France") gets the
        # labelled general-knowledge answer, not the canned redirect —
        # the redirect is written for emotional/opinion/vulgar inputs
        # and reads absurd against a plain factual question. The
        # deterministic GK-shape check separates the two.
        if looks_like_general_knowledge_question(user_question):
            gk = general_knowledge_answer(
                user_question, locale=resp_loc,
                client=client, model=OPENAI_MODEL,
            )
            if gk:
                pack["_short_circuit"] = "off_topic_gk"
                return {
                    "answer": gk,
                    "tool_calls": [{"input_trace": dict(pack)}],
                    "error": None,
                    "_answer_source": "off_topic_fallback",
                    "feedback_context": hf.build_feedback_context(
                        user_question, ui_locale, resp_loc, pack,
                    ),
                }
        from app.services.intent_router import OFF_TOPIC_REPLY
        pack["_short_circuit"] = "off_topic"
        # Synthesise a single empty tool_call carrying just the pack
        # trace so build_debug_payload (which reads from the first
        # tool_call's input_trace) can still surface intent + reasoning
        # when debug=true. Has no `table` so it's filtered out of the
        # rows / trace lists shown to the user.
        return {
            "answer": OFF_TOPIC_REPLY,
            "tool_calls": [{"input_trace": dict(pack)}],
            "error": None,
            "_answer_source": "off_topic_redirect",
            "feedback_context": hf.build_feedback_context(
                user_question, ui_locale, resp_loc, pack,
            ),
        }

    # CLEAN-ENTITY EXTRACTION for non-generic intents.
    # The cleaner often grabs too much for question-shaped inputs:
    # "What is Apple presence in Saudi?" → entity_candidate = "Apple
    # presence in Saudi". For intent-driven flows, that breaks smart-
    # search. Re-extract just the COMPANY using a one-shot LLM call,
    # then override pack["entity_candidate"] so the normal pipeline
    # operates on the clean entity. Only fires for the 3 intents
    # where the cleaner reliably overshoots; company_profile already
    # works (the entity sits at the start of the question) and
    # executive_lookup has its own direct path below.
    _intent_now = intent_meta.get("intent")
    if _intent_now in {"saudi_presence", "engagement_strategy",
                       "financial_lookup", "country_profile",
                       "relationship_intelligence",
                       "opportunity_alignment"}:
        try:
            tgt = _extract_exec_target(user_question, client, OPENAI_MODEL)
        except Exception:
            tgt = {}
        clean_company = (tgt.get("company") or "").strip()
        # SELF-REFERENCE GUARD: if the LLM extracted "MISA" / "the
        # ministry" / "ksa" / "we", that's the SUBJECT of a strategy
        # question, not a target entity. Null it out so we don't trigger
        # "No record matching MISA was found" on the engagement_strategy
        # flow. Also clear pack["entity_candidate"] if the cleaner had
        # already grabbed the same self-reference.
        if clean_company and _is_self_reference_entity(clean_company):
            clean_company = ""
            cur_ent = (pack.get("entity_candidate") or "").strip()
            if _is_self_reference_entity(cur_ent):
                pack["entity_candidate"] = ""
        if clean_company:
            cur_ent = (pack.get("entity_candidate") or "").strip()
            # Only override if the clean company differs (avoid no-op
            # churn). The cleaner's value being LONGER than the
            # extracted company is the typical signal that overshoot
            # happened.
            if (not cur_ent
                    or cur_ent.lower() != clean_company.lower()):
                pack["entity_candidate"] = clean_company
                pack["_intent_clean_entity"] = clean_company

    # COUNTRY COMPANY LIST — "tell me the Indian active companies",
    # "list the German licensed firms", etc. A request to LIST a
    # country's companies. Deterministic, keyed on HQ country +
    # licensed/is_rhq. Runs before entity extraction/ambiguity so the
    # country adjective isn't mistaken for a company name.
    if (not history) and _is_country_company_list_question(user_question):
        _list_country = _detect_origin_country(user_question)
        if _list_country:
            try:
                from app.services.engagement_data import (
                    fetch_country_saudi_investors,
                )
                stats = fetch_country_saudi_investors(_list_country)
                if stats.get("_db_error"):
                    answer = _format_country_licensing_answer(
                        _list_country, stats)
                    pack["_country_company_list"] = _list_country
                    pack["_degraded"] = "country_list_retrieval_failed"
                    tcs = [{
                        "table": "company_profiles",
                        "filters": {"_db_error": stats.get("_db_error"),
                                    "_retrieval_status": stats.get(
                                        "retrieval_status")},
                        "sql": None, "params": [], "rows_df": None,
                        "row_count": 0,
                        "input_trace": dict(pack),
                        "error": stats.get("_db_error"),
                    }]
                    return {
                        "answer": answer, "tool_calls": tcs, "error": None,
                        "_answer_source": "db",
                        "feedback_context": hf.build_feedback_context(
                            user_question, ui_locale, resp_loc, pack,
                        ),
                    }
                answer = _format_country_licensing_answer(_list_country, stats)
                pack["_country_company_list"] = _list_country
                tcs = [{
                    "table": "company_profiles",
                    "filters": {"_country_company_list": _list_country,
                                "_total_licensed": stats.get("total_licensed"),
                                "_total_rhq": stats.get("total_rhq")},
                    "sql": "SELECT COUNT(*) FROM company_profiles WHERE "
                           "licensed = true [AND is_rhq = true] "
                           "AND origin nationality match",
                    "params": [_list_country], "rows_df": None,
                    "row_count": int(stats.get("total_licensed") or 0),
                    "input_trace": dict(pack),
                    "sql_entity_check_passed": True,
                    "row_entity_sanity_passed": True,
                }]
                return {
                    "answer": answer, "tool_calls": tcs, "error": None,
                    "_answer_source": "db",
                    "feedback_context": hf.build_feedback_context(
                        user_question, ui_locale, resp_loc, pack,
                    ),
                }
            except Exception:
                pass  # fall through to normal flow on any failure

    # SAUDI RHQ / LICENSING aggregate questions.
    # "how many RHQ licences do we have?" / "total licensed companies"
    # / "count of RHQ" — bypass the LLM tool-routing entirely and
    # answer deterministically from company_profiles.licensed/is_rhq
    # (canonical source: 727 RHQ / 95,671 licensed; rhq_licenses is
    # an auxiliary view with only 661 rows). Output is a rich
    # executive briefing — snapshot + top-HQ-country table +
    # licensed pool + strategic read, all sourced live.
    if _is_saudi_licensing_count_question(user_question):
        # COUNTRY-SPECIFIC first: "how many licences from India origin"
        # must answer India's number, not the global aggregate. Only
        # the generic (no-country) phrasing gets the full breakdown.
        _lic_country = _detect_origin_country(user_question)
        if _lic_country:
            try:
                from app.services.engagement_data import (
                    fetch_country_saudi_investors,
                )
                stats = fetch_country_saudi_investors(_lic_country)
                answer = _format_country_licensing_answer(_lic_country, stats)
                pack["_saudi_licensing_count"] = True
                pack["_licensing_country"] = _lic_country
                if stats.get("_db_error"):
                    pack["_degraded"] = "country_licensing_retrieval_failed"
                tcs = [{
                    "table": "company_profiles",
                    "filters": {
                        "_licensing_country": _lic_country,
                        "_total_licensed": stats.get("total_licensed"),
                        "_total_rhq": stats.get("total_rhq"),
                        "_db_error": stats.get("_db_error"),
                        "_retrieval_status": stats.get("retrieval_status"),
                    },
                    "sql": (
                        None if stats.get("_db_error") else
                        "SELECT COUNT(*) FROM company_profiles WHERE "
                        "licensed = true [AND is_rhq = true] "
                        "AND shareholder/nationality origin match"
                    ),
                    "params": [_lic_country], "rows_df": None,
                    "row_count": (
                        None if stats.get("_db_error")
                        else int(stats.get("total_licensed") or 0)
                    ),
                    "input_trace": dict(pack),
                    "sql_entity_check_passed": not bool(
                        stats.get("_db_error")),
                    "row_entity_sanity_passed": not bool(
                        stats.get("_db_error")),
                    "error": stats.get("_db_error"),
                }]
                try:
                    from app.services.quality_gate import run_quality_gate
                    _db_ctx = {
                        "origin_country": _lic_country,
                        "companies_from_origin_licensed_in_saudi":
                            stats.get("total_licensed"),
                        "companies_from_origin_with_rhq":
                            stats.get("total_rhq"),
                        "footprint_data_unavailable": bool(
                            stats.get("_db_error")
                            or stats.get("counts_unavailable")
                        ),
                        "retrieval_status": stats.get("retrieval_status"),
                        "retrieval": stats.get("retrieval"),
                    }
                    answer, _iss, _fixes = run_quality_gate(
                        answer,
                        question=user_question,
                        db_context=_db_ctx,
                        retrieval_meta=stats.get("retrieval"),
                        hard_block=True,
                    )
                    if _fixes:
                        pack["_quality_gate_fixes"] = _fixes
                except Exception:
                    pass
                return {
                    "answer": answer, "tool_calls": tcs, "error": None,
                    "_answer_source": "db",
                    "_retrieval": stats.get("retrieval"),
                    "feedback_context": hf.build_feedback_context(
                        user_question, ui_locale, resp_loc, pack,
                    ),
                }
            except Exception as exc:
                # Country-scoped licensing ask must NEVER fall through to
                # the global aggregate (that was a wrong-answer class).
                pack["_saudi_licensing_count"] = True
                pack["_licensing_country"] = _lic_country
                pack["_degraded"] = "country_licensing_exception"
                from app.services.retrieval_status import (
                    classify_exception, failure, user_facing_retrieval_message,
                )
                rr = failure(
                    classify_exception(exc),
                    source_name="company_profiles.licensed/is_rhq",
                    error=str(exc),
                    filters={"origin_country": _lic_country},
                )
                pack["_retrieval"] = rr.to_context_dict()
                answer = (
                    f"## {_lic_country}-origin companies in Saudi Arabia\n\n"
                    + user_facing_retrieval_message(rr)
                )
                return {
                    "answer": answer,
                    "tool_calls": [{
                        "table": "company_profiles",
                        "filters": {
                            "_licensing_country": _lic_country,
                            "_db_error": str(exc),
                            "_retrieval_status": rr.status.value,
                        },
                        "sql": None, "params": [], "rows_df": None,
                        "row_count": None,
                        "input_trace": dict(pack),
                        "error": str(exc),
                    }],
                    "error": None,
                    "_answer_source": "db",
                    "feedback_context": hf.build_feedback_context(
                        user_question, ui_locale, resp_loc, pack,
                    ),
                }
        try:
            from app.services.engagement_data import fetch_saudi_licensing_summary
            summary = fetch_saudi_licensing_summary()
            answer = _format_saudi_licensing_briefing(
                summary, focus=_licensing_question_focus(user_question))
            # List asks ("which companies hold an RHQ license") get the
            # actual holder table, revenue-ranked, appended to the counts.
            if _SAUDI_LICENSE_LIST_RE.search(user_question or ""):
                from app.services.engagement_data import (
                    fetch_rhq_license_holders,
                )
                holders = fetch_rhq_license_holders(limit=15)
                if holders:
                    lines = [
                        "", "## RHQ License Holders (top by revenue)", "",
                        "| Company Name | Industry | Annual Revenue (USD) |",
                        "|---|---|---|",
                    ]
                    for h in holders:
                        rev = h.get("annual_revenue")
                        try:
                            rev = f"{float(rev):,.0f}" if rev else "N/A"
                        except (TypeError, ValueError):
                            rev = "N/A"
                        lines.append(
                            f"| {(h.get('company_name') or '—')} "
                            f"| {(h.get('industry') or 'Unclassified')} "
                            f"| {rev} |"
                        )
                    lines.append("")
                    lines.append(
                        "_Sources: `rhq_company.rhq_license_status` / "
                        "`company_profiles.is_rhq`._"
                    )
                    answer = answer.rstrip() + "\n" + "\n".join(lines)
            pack["_saudi_licensing_count"] = True
            pack["_retrieval"] = summary.get("retrieval")
            try:
                from app.services.quality_gate import run_quality_gate
                answer, _, _fixes = run_quality_gate(
                    answer, question=user_question,
                    retrieval_meta=summary.get("retrieval"),
                )
                if _fixes:
                    pack["_quality_gate_fixes"] = _fixes
            except Exception:
                pass
            # Single synthetic tool_call so debug=true / trace shows
            # the source attribution.
            tcs = [{
                "table": "company_profiles",
                "filters": {"_saudi_licensing_count": True,
                            "_total_rhq": summary.get("total_rhq"),
                            "_total_licensed": summary.get("total_licensed"),
                            "_retrieval_status": summary.get(
                                "retrieval_status")},
                "sql": "SELECT COUNT(*) FROM company_profiles "
                       "WHERE licensed = true [AND is_rhq = true]",
                "params": [], "rows_df": None,
                "row_count": summary.get("total_licensed") or 0,
                "input_trace": dict(pack),
                "sql_entity_check_passed": True,
                "row_entity_sanity_passed": True,
            }]
            return {
                "answer": answer, "tool_calls": tcs, "error": None,
                "_answer_source": "db",
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }
        except Exception:
            pass  # fall through to normal flow on any failure

    # COMPANY_PROFILE — direct correlator-backed path (Tier 2).
    # Replaces the LLM-routed company_profile flow that pulled only
    # the company_profiles row. Now uses the correlator to fan out
    # parallel queries across 13 FK-related tables (executives,
    # competitors, opportunities, MISA contacts, meetings, news,
    # ai_insights, etc.) in ~15ms. The curator weaves them into one
    # cross-referenced briefing.
    if intent_meta.get("intent") == "company_profile":
        try:
            tcs = _try_company_profile_correlated(
                user_question, pack, client, OPENAI_MODEL,
            )
        except Exception:
            tcs = None
        if tcs is not None:
            answer = _compose_local_commentary(
                tcs, user_question, pack, response_locale=resp_loc,
            )
            return {
                "answer": answer, "tool_calls": tcs, "error": None,
                "_answer_source": pack.get("_answer_source"),
                "_web_sources": pack.get("_web_sources"),
                "_doc_sources": pack.get("_doc_sources"),
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }
        # Fall through to LLM-routed flow on resolver miss.

    # COUNTRY_PROFILE — direct DB path.
    # Bundles country macros + Vision 2030 outlook + the canonical
    # company_profiles.licensed / is_rhq buckets in ONE payload.
    if intent_meta.get("intent") == "country_profile":
        try:
            tcs = _try_country_profile_direct(
                user_question, pack, client, OPENAI_MODEL,
            )
        except Exception:
            tcs = None
        if tcs is not None:
            answer = _compose_local_commentary(
                tcs, user_question, pack, response_locale=resp_loc,
            )
            # POST-PROCESS: rewrite the licensing summary line to the
            # ground-truth count. The model can't see filter values
            # (only row data is in the curation payload) so it
            # invents numbers based on rendered rows (which are
            # top-10 truncations). Force the real totals here.
            answer = _force_country_licensing_summary(answer, tcs)
            return {
                "answer": answer, "tool_calls": tcs, "error": None,
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }

    # RELATIONSHIP_INTELLIGENCE — direct DB path.
    # "previous meetings with Apple" / "what engagements have we had
    # with Tesla" / "MISA contacts at Aramco" — the user wants prior
    # interaction history, not a company snapshot. Pulls from the
    # engagements / meetings / meeting_notes / misa_contact_details
    # tables (engagement_data.py). When NO records found, returns
    # honest "no engagement history" message instead of inventing
    # meetings — the highest-stakes anti-hallucination requirement
    # for this intent.
    if intent_meta.get("intent") == "relationship_intelligence":
        try:
            tcs = _try_relationship_intelligence_direct(
                user_question, pack, client, OPENAI_MODEL,
            )
        except Exception:
            tcs = None
        if tcs is not None:
            # Honest "no records found" message when the entity exists
            # but has zero meetings / engagements / contacts. Previously
            # this fell through to general-knowledge fallback, which
            # made it sound like we hadn't looked — misleading the
            # executive about a real data gap. (NDMO / Rule 7 — surface
            # data absences explicitly.)
            no_records = any(
                (tc.get("filters") or {}).get("_relationship_no_records")
                for tc in tcs
            )
            if no_records:
                target = pack.get("_engagement_target") or "the requested entity"
                answer = (
                    f"## Engagement History — {target}\n\n"
                    f"**No engagement history found** for {target} in "
                    f"MISA's internal records (meetings, engagements, "
                    f"contact assignments, or recent interactions).\n\n"
                    f"This usually means either the entity has no recorded "
                    f"MISA engagements yet, or its records are filed under "
                    f"a different legal name. Try the full registered company "
                    f"name and re-ask."
                )
                return {
                    "answer": answer, "tool_calls": tcs, "error": None,
                    "_answer_source": "relationship_no_records",
                    "feedback_context": hf.build_feedback_context(
                        user_question, ui_locale, resp_loc, pack,
                    ),
                }
            answer = _compose_local_commentary(
                tcs, user_question, pack, response_locale=resp_loc,
            )
            return {
                "answer": answer, "tool_calls": tcs, "error": None,
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }
        # Fall through to normal flow if direct path failed for any
        # reason — curator still gets the relationship_intelligence
        # intent hint and the company_profiles row(s).

    # OPPORTUNITY_ALIGNMENT — direct DB path.
    # "why is Apple relevant to MISA" / "how does X align with
    # Vision 2030" — joins company.opportunities + focused_sectors
    # + country_vision_outlooks. The country_vision_outlooks table
    # IS the Vision 2030-style mapping layer (195 countries, with
    # national_vision / diversification_goals / five_year_outlook
    # text).
    if intent_meta.get("intent") == "opportunity_alignment":
        try:
            tcs = _try_opportunity_alignment_direct(
                user_question, pack, client, OPENAI_MODEL,
            )
        except Exception:
            tcs = None
        if tcs is not None:
            answer = _compose_local_commentary(
                tcs, user_question, pack, response_locale=resp_loc,
            )
            return {
                "answer": answer, "tool_calls": tcs, "error": None,
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }

    # SINGLE-SECTOR OPPORTUNITY path — "Identify the most attractive
    # opportunities in the [X] sector" / "What's in the ICT pipeline?".
    # Aggregates opportunities for a single named sector. Without this,
    # the entity extractor pulled "[X] sector" as a company name and
    # the question dead-ended on "No record matching '[X] sector'".
    # 18 such failures out of the 155-question battery — biggest
    # single-pattern fix.
    if _is_single_sector_opportunity_question(user_question):
        try:
            tcs = _try_single_sector_opportunity_direct(
                user_question, pack, client, OPENAI_MODEL,
            )
        except Exception:
            tcs = None
        if tcs is not None:
            answer = _compose_local_commentary(
                tcs, user_question, pack, response_locale=resp_loc,
            )
            return {
                "answer": answer, "tool_calls": tcs, "error": None,
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }

    # TOP-COMPANIES-PER-SECTOR follow-up handler. Detects
    # "show me the top companies from each of these sectors" style
    # multi-turn questions and runs a direct cross-sector ranking
    # join, with explicit data-gap disclosure for sectors where
    # company_profiles doesn't have sector_id populated.
    # Runs BEFORE sector aggregation so the more-specific intent
    # wins routing.
    if _is_top_companies_per_sector_question(user_question):
        try:
            tcs = _try_top_companies_per_sector_direct(
                user_question, pack, client, OPENAI_MODEL,
            )
        except Exception:
            tcs = None
        if tcs is not None:
            answer = _compose_local_commentary(
                tcs, user_question, pack, response_locale=resp_loc,
            )
            return {
                "answer": answer, "tool_calls": tcs, "error": None,
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }

    # SECTOR AGGREGATION — direct path for "all sectors" / sector
    # ranking / momentum-style questions. Until now, sector_lookup
    # intent only queried the `sectors` taxonomy table (which has
    # names, not activity) — so "give me momentum for all the
    # sectors" returned 0 rows and fell back to general OpenAI
    # knowledge (a useless textbook paragraph).
    #
    # The right answer is to aggregate `opportunities.sector_name`
    # — that's where MISA's actual sector activity lives (600
    # Agriculture opportunities, 508 Petrochemical, etc.).
    if (intent_meta.get("intent") == "sector_lookup"
            and _is_sector_aggregation_question(user_question)):
        try:
            tcs = _try_sector_aggregation_direct(
                user_question, pack, client, OPENAI_MODEL,
            )
        except Exception:
            tcs = None
        if tcs is not None:
            # Prefer deterministic Jul21-lite briefing over curator compression.
            rows = pack.get("_sector_aggregation_rows") or []
            if rows:
                answer = _format_sector_opportunity_briefing(rows)
                try:
                    from app.services.answer_finalize import finalize_answer
                    answer = finalize_answer(
                        answer, user_question=user_question, pack=pack,
                    )
                except Exception:
                    pass
            else:
                answer = _compose_local_commentary(
                    tcs, user_question, pack, response_locale=resp_loc,
                )
            return {
                "answer": answer, "tool_calls": tcs, "error": None,
                "_answer_source": "sector_aggregation",
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }

    # EXECUTIVE-LOOKUP & SUCCESSION INTENTS — direct force-query.
    # When the user asks "Who is the CEO of Apple?" / "Tell me about
    # Tim Cook" / "Who chairs Saudi Aramco?" / "Who will follow Tim
    # Cook?", the answer should LEAD with the person, not bury them
    # in a company snapshot. This path extracts {person, company,
    # role} via a focused LLM call, then queries company_executives
    # (filtered by role when specified) and attaches the company
    # profile as context. For executive_succession intent, the
    # web-augmentation runs unconditionally (it's the whole point of
    # the intent); for executive_lookup it runs only if the question
    # also matches the forward-looking shape regex.
    if intent_meta.get("intent") in ("executive_lookup", "executive_succession"):
        try:
            # Consume the speculative extraction kicked off in parallel
            # with the classifier — avoids a sequential second LLM call.
            pack["_intent"] = intent_meta.get("intent")
            pack.setdefault("_query_intent", intent_meta)
            exec_tcs = _try_executive_lookup_direct(
                user_question, pack, client, OPENAI_MODEL,
                prefetched_target=_speculative_exec_target,
            )
        except Exception:
            exec_tcs = None
        if exec_tcs is not None:
            answer = _compose_local_commentary(
                exec_tcs, user_question, pack, response_locale=resp_loc,
            )
            # WEB AUGMENTATION for forward-looking exec questions
            # ("who will follow Tim Cook", "Apple's next CEO",
            # "Tim Cook's successor"). The DB has the CURRENT state;
            # the future succession is news, which needs web grounding.
            # Append a separate '## What's Reported (Live Web)' section
            # with cited sources, keeping DB facts and web facts cleanly
            # separated so the reader sees which is verified-current vs
            # reported-upcoming.
            #
            # Fires UNCONDITIONALLY for executive_succession intent (it's
            # the whole point of that intent), and additionally for any
            # executive_lookup question that matches the forward-looking
            # regex as a safety net (the classifier might miss subtle
            # succession phrasing).
            # WEB AUGMENTATION:
            #  - executive_succession / forward-looking → successor news
            #  - current cabinet/minister officeholder → live web MUST lead
            #    (MISA executives table lags royal decrees; e.g. Saudi
            #    Investment Minister changed Feb 2026)
            is_officeholder = _is_current_officeholder_question(user_question)
            should_augment = (
                intent_meta.get("intent") == "executive_succession"
                or _is_forward_looking_exec_question(user_question)
                or is_officeholder
            )
            exec_web_sources: list = []
            if should_augment:
                pack["_exec_web_augmented"] = True
                lead_with_web = (
                    intent_meta.get("intent") == "executive_succession"
                    or is_officeholder
                )
                from app.config import ADVISORY_MODEL as _ADV
                answer = _augment_exec_answer_with_web(
                    answer, user_question, client, _ADV or OPENAI_MODEL,
                    lead_with_web=lead_with_web,
                    capture_sources=exec_web_sources,
                    mode="current_office" if is_officeholder else "succession",
                )
            try:
                from app.services.answer_finalize import finalize_answer
                pack["_intent"] = intent_meta.get("intent") or pack.get("_intent")
                pack.setdefault("_query_intent", intent_meta)
                answer = finalize_answer(
                    answer, user_question=user_question, pack=pack,
                )
            except Exception:
                pass
            return {
                "answer": answer, "tool_calls": exec_tcs, "error": None,
                "web_sources": exec_web_sources,
                "_answer_source": pack.get("_answer_source") or "curated",
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }
        # No DB match — for current cabinet/minister questions, still
        # answer from live web rather than falling through to stale GK.
        if _is_current_officeholder_question(user_question):
            exec_web_sources: list = []
            from app.config import ADVISORY_MODEL as _ADV
            answer = _augment_exec_answer_with_web(
                "", user_question, client, _ADV or OPENAI_MODEL,
                lead_with_web=True,
                capture_sources=exec_web_sources,
                mode="current_office",
            )
            if answer.strip():
                try:
                    from app.services.answer_finalize import finalize_answer
                    pack["_intent"] = "executive_lookup"
                    pack["_answer_source"] = "web_officeholder"
                    answer = finalize_answer(
                        answer, user_question=user_question, pack=pack,
                    )
                except Exception:
                    pass
                return {
                    "answer": answer,
                    "tool_calls": [],
                    "error": None,
                    "web_sources": exec_web_sources,
                    "feedback_context": hf.build_feedback_context(
                        user_question, ui_locale, resp_loc, pack,
                    ),
                }
        # No DB match — fall through to normal flow with intent hint
        # still set, so curation can still lead with "Not available in
        # the current database." for the role/person.

    # ENTITY REFERENCE RESOLUTION (LLM-driven).
    # Ask the LLM "what entity is this question about, and what TYPE?".
    # The LLM already knows RHQ is jargon, Saudi is geography, Apple
    # is a company, Sundar Pichai is a person — far more general than
    # hand-rolled keyword lists. We run this on every turn (even with
    # no history) because entity_type also routes first-turn person
    # lookups. Falls back to the legacy heuristic if the call fails.
    inherited: str | None = None
    resolved_ent: str = ""
    resolved_type: str | None = None
    resolved_parent: str | None = None
    resolver_meta: dict = _resolve_entity_with_llm(
        user_question, history or [], client, OPENAI_MODEL,
    )
    if resolver_meta.get("entity"):
        resolved_ent = (resolver_meta.get("entity") or "").strip()
        resolved_type = resolver_meta.get("entity_type")
        resolved_parent = resolver_meta.get("parent_entity")
        is_followup = bool(resolver_meta.get("is_followup"))
        is_new_topic = bool(resolver_meta.get("is_new_topic"))
        cur_ent = (pack.get("entity_candidate") or "").strip()
        # Honour the resolver when:
        #   (a) it identifies an entity AND it's a follow-up referencing prior, OR
        #   (b) the cleaner found nothing, OR
        #   (c) the cleaner extracted something noisy that doesn't equal the
        #       resolver's answer (e.g. cleaner: "give me more details about
        #       sundar pichai", resolver: "Sundar Pichai").
        if resolved_ent and is_followup and not is_new_topic:
            new_pack = dict(pack); new_pack["entity_candidate"] = resolved_ent
            new_pack["_inherited_from_history"] = resolved_ent
            new_pack["_resolver_reasoning"] = resolver_meta.get("reasoning", "")
            new_pack["_resolver_entity_type"] = resolved_type
            new_pack["_resolver_parent"] = resolved_parent
            pack = new_pack
            inherited = resolved_ent
        elif resolved_ent and (not cur_ent or cur_ent.lower() != resolved_ent.lower()):
            new_pack = dict(pack); new_pack["entity_candidate"] = resolved_ent
            new_pack["_resolver_reasoning"] = resolver_meta.get("reasoning", "")
            new_pack["_resolver_entity_type"] = resolved_type
            new_pack["_resolver_parent"] = resolved_parent
            pack = new_pack
            if is_followup and not is_new_topic:
                inherited = resolved_ent
    # Safety net: if the resolver was unavailable or returned nothing,
    # fall back to the deterministic regex-based heuristic.
    if inherited is None and history:
        pack, inherited = _inherit_entity_from_history(pack, history)

    # PERSON ROUTING: if the resolver classified the entity as a person
    # (Sundar Pichai, Tim Cook, MBS), skip company smart-search entirely
    # and run a direct query against the executive tables for that name.
    # Falls through to the normal LLM flow when no rows are found, so
    # general-knowledge persons (with parent_entity context attached to
    # the pack for the curator) still get a useful answer.
    if resolved_type == "person" and resolved_ent:
        tc_person = _try_person_direct_query(
            resolved_ent, resolved_parent, pack,
        )
        if tc_person is not None:
            answer = _compose_local_commentary(
                tc_person, user_question, pack, response_locale=resp_loc,
            )
            return {
                "answer": answer, "tool_calls": tc_person, "error": None,
                "_answer_source": pack.get("_answer_source") or "curated",
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack
                ),
            }
        # No DB rows for this person — annotate the pack so the curator
        # answers from general knowledge with the parent-company hint.
        pack["_person_no_db_match"] = True

    # FORCED FOLLOW-UP QUERY: when entity was inherited from a previous
    # user turn, do NOT trust the LLM with a "use this entity in your
    # filter" hint — the current user message ("how can I engage with
    # them") doesn't contain the entity name, and the model routinely
    # ignores per-turn hints under that condition, falling into a
    # generic smart-search that returns junk. Run the query ourselves:
    # smart-search company_profiles for the inherited entity + its
    # aliases, then curate normally.
    if inherited and inherited.strip():
        ent = inherited.strip()
        terms = expand_aliases(ent) or [ent]
        try:
            df_fu, sql_fu, params_fu = run_rhq_company_smart_search(terms, 5)
        except Exception:
            df_fu = None; sql_fu = ""; params_fu = []
        if df_fu is not None and not df_fu.empty:
            tc_followup = [{
                "table": COMPANY_TABLE,
                "filters": {
                    "_inherited_from_history": ent,
                    "_followup_direct_query": True,
                    "_alias_terms": terms,
                },
                "sql": sql_fu, "params": params_fu, "rows_df": df_fu,
                "row_count": int(len(df_fu)),
                "input_trace": dict(pack),
                "sql_entity_check_passed": True,
                "row_entity_sanity_passed": True,
                "closest_names": [],
            }]
            answer = _compose_local_commentary(
                tc_followup, user_question, pack, response_locale=resp_loc,
            )
            return {
                "answer": answer, "tool_calls": tc_followup, "error": None,
                "_answer_source": "db",
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }
        # If no rows even for the inherited entity, fall through to
        # the normal LLM-routed flow (rare; usually means the
        # inherited entity isn't in our DB).

    # Off-topic short-circuit: "what is the capital of France" / "weather in
    # Dubai" / etc. Without this, keyword collisions ("capital" → "Capital
    # Management") send the chat into the DB for a noisy match it then
    # presents confidently. Route straight to the OpenAI fallback, which
    # by prompt is required to label its answer "NOT from the MISA database".
    if looks_like_general_knowledge_question(user_question) and client is not None:
        try:
            from app.services.jul21_surface import looks_like_corridor_investment_ask
            if looks_like_corridor_investment_ask(user_question):
                _adv = _run_advisory_path(
                    user_question, pack, ui_locale, resp_loc, client)
                if _adv:
                    return _adv
        except Exception:
            pass
        ans = general_knowledge_answer(
            user_question, locale=resp_loc, client=client, model=OPENAI_MODEL,
        )
        if ans:
            try:
                from app.services.answer_finalize import finalize_answer
                pack["_answer_source"] = "off_topic_fallback"
                ans = finalize_answer(
                    ans, user_question=user_question, pack=pack,
                )
            except Exception:
                pass
            return {
                "answer": ans, "tool_calls": [], "error": None,
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
                "_answer_source": "off_topic_fallback",
            }
        # Fall through if fallback failed.

    # DETERMINISTIC AMBIGUITY CHECK (pre-LLM) — for short entity-lookup
    # queries like "alpha", we hit the DB ourselves and run the
    # ambiguity rules without depending on the LLM's table choice or
    # smart-search depth. If multiple distinct candidates score
    # similarly high (and none is an exact-name dominator), emit a
    # clarification message immediately. Same query, same DB, same
    # decision — every time.
    # Skip for follow-ups (history-inherited entity is intentional),
    # off-topic (already handled above), and explicitly multi-intent
    # questions (count / comparison / browse).
    if (not history) and (not inherited):
        ent_for_ambig = (pack.get("entity_candidate") or "").strip()
        intent_for_ambig = detect_intent(user_question, history)
        if (intent_for_ambig in ("entity_lookup", "broad_topic", "unknown")
                and is_short_entity_query(ent_for_ambig)):
            try:
                candidates = discover_candidates(ent_for_ambig)
                ambiguous = detect_ambiguity(ent_for_ambig, candidates)
            except Exception:
                ambiguous = None
            if ambiguous:
                # Company-profile asks ("company profile for Hitachi")
                # should not dead-end on a clarification stub — auto-pick
                # the top-scoring candidate and continue into Jul21 brief.
                top = ambiguous[0] if ambiguous else None
                top_name = (top or {}).get("name") or ""
                if (
                    top_name
                    and re.search(
                        r"(?i)\b(company\s+(?:profile|briefing)|"
                        r"briefing\s+on|profile\s+of|"
                        r"tell\s+me\s+about|brief\s+me\s+on)\b",
                        user_question or "",
                    )
                ):
                    pack["entity_candidate"] = top_name
                    pack["_auto_picked_ambiguous"] = top_name
                    pack["_ambiguous_alternates"] = [
                        c.get("name") for c in ambiguous[1:4]
                        if c.get("name")
                    ]
                else:
                    return {
                        "answer": format_clarification(
                            ent_for_ambig, ambiguous,
                        ),
                        "tool_calls": [],
                        "error": None,
                        "_answer_source": "clarification",
                        "feedback_context": hf.build_feedback_context(
                            user_question, ui_locale, resp_loc, pack,
                        ),
                    }

    # Pure-browse short-circuit: "show me companies" / "list deals" etc.
    # These are routinely mishandled by the LLM router (it sends an empty
    # filter, the engine smart-searches the question text, returns 0 rows,
    # and the OpenAI fallback then lists random world companies as if from
    # the DB). Run the browse directly against the right table and skip
    # OpenAI tool-call routing entirely.
    browse = detect_pure_browse(user_question)
    if browse and not history:
        table, order_by = browse
        try:
            df, sql, params = generate_query_and_run_query(
                table=table, filters={}, order_by=order_by,
                descending=True, limit=10,
            )
            tc_result = [{
                "table": table,
                "filters": {"_pure_browse_shortcircuit": {"order_by": order_by, "limit": 10}},
                "sql": sql,
                "params": params,
                "rows_df": df,
                "row_count": int(len(df)),
                "input_trace": dict(pack),
                "sql_entity_check_passed": True,
                "row_entity_sanity_passed": True,
                "closest_names": [],
            }]
            answer = _compose_local_commentary(
                tc_result, user_question, pack, response_locale=resp_loc,
            )
            return {
                "answer": answer, "tool_calls": tc_result, "error": None,
                "feedback_context": hf.build_feedback_context(
                    user_question, ui_locale, resp_loc, pack,
                ),
            }
        except Exception:
            # Fall through to normal routing on any failure here.
            pass

    messages = [{"role": "system", "content": system_prompt() + _turn_entity_hint(pack)}]

    user_msgs: list[dict] = []
    for h in history:
        if h.get("role") == "user" and h.get("content"):
            user_msgs.append({"role": "user", "content": _truncate_for_llm(h["content"])})
    m_hist = _max_history_user_turns()
    if m_hist > 0 and len(user_msgs) > m_hist:
        user_msgs = user_msgs[-m_hist:]
    messages.extend(user_msgs)
    messages.append({"role": "user", "content": _truncate_for_llm(user_question)})

    tool_calls_executed: list = []
    validation_round = 0

    try:
        while True:
            response = _chat_completions_create_with_retry(
                client,
                model=OPENAI_MODEL,
                messages=messages,
                tools=_build_tools(),
                tool_choice="auto",
                temperature=0.0,  # routing should be deterministic — variance here yields different table picks for the same question
                **_openai_max_completion_tokens_kw(),
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                if _likely_rhq_company_lookup(user_question, pack):
                    tool_calls_executed = [_forced_smart_search_tool_result(user_question, pack)]
                    return {
                        "answer": _compose_local_commentary(
                            tool_calls_executed, user_question, pack, response_locale=resp_loc
                        ),
                        "tool_calls": tool_calls_executed,
                        "error": None,
                        "feedback_context": hf.build_feedback_context(
                            user_question, ui_locale, resp_loc, pack
                        ),
                    }
                empty = "لا يوجد رد." if resp_loc == "ar" else "(no response)"
                return {
                    "answer": (msg.content or "").strip() or empty,
                    "tool_calls": tool_calls_executed,
                    "error": None,
                    "feedback_context": hf.build_feedback_context(
                        user_question, ui_locale, resp_loc, pack
                    ),
                }

            temp_ok: list = []
            pending_tool_msgs: list = []
            round_has_sql_validation_failure = False
            last_limit = 25

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = _parse_tool_arguments(tc.function.arguments)
                # New unified tool: `query_table` with table in args.
                # Backwards-compat: legacy `query_<table>` still recognised.
                if fn_name == "query_table":
                    table = str(args.get("table") or "").strip()
                else:
                    table = fn_name.replace("query_", "")
                limit = int(args.get("limit", 25) or 25)
                last_limit = limit

                # Reject unknown / denied tables outright.
                if not table or (table != COMPANY_TABLE and not is_allowed_table(table)):
                    pending_tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(_reject_table_and_audit(table)),
                    })
                    continue

                try:
                    filters = _coerce_filters_mapping(args.get("filters"))
                    # CODE ENFORCER: never count/aggregate via the auxiliary
                    # rhq_licenses table — rewrite to company_profiles.
                    try:
                        from app.services.source_policy import (
                            rewrite_aggregate_licensing_query,
                        )
                        table, filters, _lic_notes = (
                            rewrite_aggregate_licensing_query(
                                table,
                                count_only=bool(args.get("count_only")),
                                filters=filters,
                                question=user_question,
                            )
                        )
                        if _lic_notes:
                            pack.setdefault("_source_rewrites", []).extend(
                                _lic_notes
                            )
                    except Exception:
                        pass
                    # Strip noise filters like `company_name = 'MISA'` — the
                    # ministry is the audience, not a company in the data.
                    filters = _strip_self_reference_filters(filters)
                    # Backstop: model may set company_name=<country-adjective>
                    # — rewrite to global_headquarters=<noun country> server-
                    # side so cross-geo queries actually filter on origin.
                    filters = _normalise_country_adjective_filters(filters)
                    # Also: when the model passes a country NAME to a
                    # country-FK integer column (country_id /
                    # country_profile_id), resolve the name to the int id
                    # server-side. This makes "deals in Pakistan",
                    # "opportunities in Egypt", "meetings about Saudi"
                    # answerable across the ~45 tables that store country
                    # by FK rather than by name.
                    filters = _normalise_country_id_filters(table, filters)
                    # Risk-20-1: log any filter column the catalog doesn't
                    # expose (denied/sensitive/hallucinated). Observation
                    # only — the builder already drops these; `filters` is
                    # untouched, so behaviour is identical either way.
                    _audit_blocked_columns(table, filters)
                    terms = _search_terms_for_pack(pack, user_question)

                    # COUNT mode: answer "how many", "total number of",
                    # "count of" questions without the LIMIT 100 cap.
                    if bool(args.get("count_only")):
                        from app.database import count_table_rows
                        # Risk-20-6: gate the enumeration vector. Logs every
                        # count; only blocks scripted probing well above
                        # conversational volume.
                        if not _count_only_guard(table, filters):
                            pending_tool_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({
                                    "ok": False,
                                    "error": (
                                        "Too many count-only requests in a short "
                                        "window. Ask for the underlying data "
                                        "directly, or retry shortly."
                                    ),
                                }),
                            })
                            continue
                        try:
                            n, sql_c, params_c = count_table_rows(table, filters)
                        except ValueError as ve:
                            # All requested filters were invalid for this
                            # table — surface as an honest 0-count with the
                            # error in trace, rather than counting the
                            # whole table and misleading the user.
                            temp_ok.append({
                                "table": table,
                                "filters": dict(filters,
                                                _invalid_filter_for_count=str(ve)),
                                "sql": f"-- count blocked: {ve}",
                                "params": [],
                                "rows_df": pd.DataFrame(),
                                "row_count": 0,
                                "input_trace": dict(pack),
                                "sql_entity_check_passed": True,
                                "row_entity_sanity_passed": True,
                                "closest_names": [],
                                "_count_only_result": None,
                                "_count_only_error": str(ve),
                            })
                            pending_tool_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({"ok": False,
                                                       "error": str(ve)}),
                            })
                            continue
                        temp_ok.append({
                            "table": table,
                            "filters": dict(filters, _count_only=True),
                            "sql": sql_c,
                            "params": params_c,
                            "rows_df": pd.DataFrame([{"count": n}]),
                            "row_count": 1,
                            "input_trace": dict(pack),
                            "sql_entity_check_passed": True,
                            "row_entity_sanity_passed": True,
                            "closest_names": [],
                            "_count_only_result": n,
                        })
                        pending_tool_msgs.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"ok": True, "count": n}),
                        })
                        continue  # done with this tool call

                    if table == COMPANY_TABLE:
                        trace_filters = dict(filters)

                        # If the model passed ANY filters, run them — even
                        # if none are name-search columns. The previous logic
                        # only honoured name-search filters and otherwise
                        # hijacked to a keyword smart-search, which broke
                        # cross-geo queries like
                        # `global_headquarters=Pakistan` (the filter was
                        # silently ignored in favour of question keywords).
                        if filters:
                            df, sql, params = generate_query_and_run_query(
                                table=table,
                                filters=filters,
                                order_by=args.get("order_by"),
                                descending=args.get("descending", True),
                                limit=limit,
                            )
                            # Only fall back to fuzzy keyword search when
                            # the filter targeted a name column (typo cover).
                            # For non-name filters (global_headquarters,
                            # sector, rhq_country, …) an empty result is a
                            # truthful answer; don't dilute it with noise.
                            if (df.empty and terms
                                    and _rhq_filters_include_name_search(filters)):
                                df, sql, params = run_rhq_company_smart_search(terms, limit)
                                trace_filters["_smart_fallback_after_empty"] = terms
                        elif terms:
                            df, sql, params = run_rhq_company_smart_search(terms, limit)
                            trace_filters["_keyword_search_injected"] = terms
                        else:
                            df = pd.DataFrame()
                            sql = "-- company_profiles: blocked empty browse"
                            params = []

                        entity = pack.get("entity_candidate")
                        # Skip the named-entity SQL guardrail for clearly
                        # geographic queries: if the filters target a
                        # country/HQ column (global_headquarters / rhq_country
                        # / ultimate_parent_company set to a country) or the
                        # filters were rewritten from a country adjective,
                        # the "entity_candidate" extracted from the question
                        # is a query intent ("Pakistani companies have …"),
                        # not a real company name to validate against.
                        geo_intent = (
                            "_rewritten_company_name_adjective" in trace_filters
                            or any(k in trace_filters for k in (
                                "global_headquarters", "rhq_country"
                            ))
                        )
                        if (not geo_intent
                                and _entity_requires_sql_constraint(entity)
                                and not _sql_covers_entity(sql, params, entity)):
                            round_has_sql_validation_failure = True
                            pending_tool_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({
                                    "validation_error": True,
                                    "message": (
                                        f'The user is asking about the entity "{entity}". '
                                        "The SQL MUST include a filter such as "
                                        "`company_name ILIKE %s` with a parameter that "
                                        "contains that name."
                                    ),
                                }),
                            })
                            continue

                        rows = df.to_dict(orient="records")
                        row_sanity = True
                        closest: list[str] = []
                        if _entity_requires_sql_constraint(entity) and rows:
                            row_sanity = _rows_match_entity(rows, entity)
                            if not row_sanity:
                                closest = _top_similar_company_names(entity, 3)

                        temp_ok.append({
                            "table": table,
                            "filters": trace_filters,
                            "sql": sql,
                            "params": params,
                            "rows_df": df,
                            "row_count": int(len(df)),
                            "input_trace": dict(pack),
                            "sql_entity_check_passed": True,
                            "row_entity_sanity_passed": row_sanity,
                            "closest_names": closest,
                        })
                        pending_tool_msgs.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"ok": True, "row_count": int(len(df))}),
                        })

                    else:
                        # Generic path for any introspected business table.
                        # Mirrors the company_profiles flow: structured filters
                        # first; fall back to a generic smart-search across the
                        # table's name/text columns when filters return empty or
                        # the user named a specific entity but didn't filter.
                        trace_filters = dict(filters)
                        ncols = name_columns(table)
                        entity = pack.get("entity_candidate")

                        # Validate filter columns against the table's allowed
                        # set. If the model passed only columns that don't
                        # exist for this table (e.g. country_profile_id on
                        # `deals`), the query builder would silently drop them
                        # and return unfiltered rows — misleading. Surface
                        # this as a clean "no records" with a trace marker so
                        # curation reports honestly instead.
                        info = get_table_info(table)
                        allowed_cols = set((info or {}).get("filterable", []))
                        user_facing_cols = {
                            k for k in filters.keys() if not k.startswith("_")
                        }
                        if user_facing_cols and allowed_cols and not (
                            user_facing_cols & allowed_cols
                        ):
                            trace_filters["_dropped_unknown_filters"] = sorted(
                                user_facing_cols
                            )
                            df = pd.DataFrame()
                            sql = (
                                f"-- {table}: all filter columns "
                                f"({sorted(user_facing_cols)}) are not in this "
                                "table's filterable list; refusing to return "
                                "unfiltered rows."
                            )
                            params = []
                            temp_ok.append({
                                "table": table,
                                "filters": trace_filters,
                                "sql": sql,
                                "params": params,
                                "rows_df": df,
                                "row_count": 0,
                                "input_trace": dict(pack),
                                "sql_entity_check_passed": True,
                                "row_entity_sanity_passed": True,
                                "closest_names": [],
                            })
                            pending_tool_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({
                                    "ok": True,
                                    "row_count": 0,
                                    "note": (
                                        f"None of the requested filter "
                                        f"columns exist on `{table}`. "
                                        "Allowed: "
                                        + ", ".join(sorted(allowed_cols)[:10])
                                    ),
                                }),
                            })
                            continue

                        if filters:
                            df, sql, params = generate_query_and_run_query(
                                table=table,
                                filters=filters,
                                order_by=args.get("order_by"),
                                descending=args.get("descending", True),
                                limit=limit,
                            )
                            # Entity-defining filters (a resolved country
                            # FK) must NOT be silently dropped on empty:
                            # zero rows for "Korea" is the truthful
                            # answer. Falling through to keyword search
                            # returns OTHER countries' rows which the
                            # curator then mis-attributes (e.g. Saudi
                            # aggregate fdi_data presented as South
                            # Korea's FDI). Typo-cover fallback stays for
                            # name-style filters only.
                            _entity_defining = (
                                "_resolved_country_fk" in trace_filters
                                or any(
                                    "country" in str(k).lower()
                                    and str(k).lower().endswith("_id")
                                    for k in (filters or {})
                                )
                            )
                            if df.empty and terms and not _entity_defining:
                                df, sql, params = smart_search(table, terms, limit)
                                trace_filters["_smart_fallback_after_empty"] = terms
                        elif entity and ncols and terms and not args.get("order_by"):
                            # Entity-style question, no filters and no explicit
                            # browse order → fall back to keyword search across
                            # the table's name/text columns.
                            df, sql, params = smart_search(table, terms, limit)
                            trace_filters["_keyword_search_injected"] = terms
                        else:
                            df, sql, params = generate_query_and_run_query(
                                table=table,
                                filters={},
                                order_by=args.get("order_by"),
                                descending=args.get("descending", True),
                                limit=limit,
                            )

                        row_sanity = True
                        if _entity_requires_sql_constraint(entity) and ncols and not df.empty:
                            row_sanity = _rows_match_entity_in_cols(
                                df.to_dict(orient="records"), entity, ncols
                            )

                        temp_ok.append({
                            "table": table,
                            "filters": trace_filters,
                            "sql": sql,
                            "params": params,
                            "rows_df": df,
                            "row_count": int(len(df)),
                            "input_trace": dict(pack),
                            "sql_entity_check_passed": True,
                            "row_entity_sanity_passed": row_sanity,
                            "closest_names": [],
                        })
                        pending_tool_msgs.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"ok": True, "row_count": int(len(df))}),
                        })

                except Exception as e:
                    temp_ok.append({
                        "table": table,
                        "filters": args.get("filters", {}),
                        "sql": "ERROR",
                        "params": [],
                        "rows_df": pd.DataFrame(),
                        "row_count": 0,
                        "error": str(e),
                        "input_trace": dict(pack),
                        "sql_entity_check_passed": False,
                        "row_entity_sanity_passed": False,
                        "closest_names": [],
                    })
                    pending_tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": str(e)}),
                    })

            if round_has_sql_validation_failure:
                if validation_round < 2:
                    messages.append(_assistant_api_dict(msg))
                    messages.extend(pending_tool_msgs)
                    validation_round += 1
                    continue

                messages.append(_assistant_api_dict(msg))
                messages.extend(pending_tool_msgs)

                entity_fb = pack.get("entity_candidate")
                terms_fb: list[str] = (
                    [entity_fb.strip()]
                    if entity_fb and str(entity_fb).strip()
                    else _search_terms_for_pack(pack, user_question)
                )
                df_fb, sql_fb, params_fb = run_rhq_company_smart_search(terms_fb, last_limit)
                rows_fb = df_fb.to_dict(orient="records")
                row_sanity_fb = True
                closest_fb: list[str] = []
                if _entity_requires_sql_constraint(entity_fb) and rows_fb:
                    row_sanity_fb = _rows_match_entity(rows_fb, entity_fb)
                    if not row_sanity_fb:
                        closest_fb = _top_similar_company_names(entity_fb, 3)
                tool_calls_executed.append({
                    "table": COMPANY_TABLE,
                    "filters": {"_forced_entity_smart_search_after_retries": terms_fb},
                    "sql": sql_fb,
                    "params": params_fb,
                    "rows_df": df_fb,
                    "row_count": int(len(df_fb)),
                    "input_trace": dict(pack),
                    "sql_entity_check_passed": True,
                    "row_entity_sanity_passed": row_sanity_fb,
                    "closest_names": closest_fb,
                })
            else:
                tool_calls_executed.extend(temp_ok)

            break

    except Exception as e:
        return {
            "answer": "",
            "tool_calls": tool_calls_executed,
            "error": f"OpenAI call failed: {e}",
            "feedback_context": hf.build_feedback_context(user_question, ui_locale, resp_loc, pack),
        }

    return {
        "answer": _compose_local_commentary(
            tool_calls_executed, user_question, pack, response_locale=resp_loc
        ),
        "tool_calls": tool_calls_executed,
        "error": None,
        "_answer_source": pack.get("_answer_source"),
        "_web_sources": pack.get("_web_sources"),
        "_doc_sources": pack.get("_doc_sources"),
        "feedback_context": hf.build_feedback_context(user_question, ui_locale, resp_loc, pack),
    }


def build_debug_payload(user_question: str, result: dict) -> dict:
    """Construct the `?debug=true` trace payload from a chat() result.

    Returns a flat dict capturing what the engine did on this turn:
    intent, entity resolution, aliases, tables searched, evidence rows,
    fuzzy/fallback markers, and resolver reasoning. Safe to surface to
    API callers — does NOT include raw row data (only counts), so it
    won't bloat the response or leak privacy-filterable fields.

    Used by the chat router when the request had `debug=true`. Built
    here (not in the router) so the logic stays close to the pipeline
    that produced it.
    """
    debug: dict = {
        "question": user_question,
        "answer_chars": len(result.get("answer") or ""),
        "error": result.get("error"),
    }
    tcs = result.get("tool_calls") or []

    # Intent + resolver state — pulled from the first tool_call's
    # input_trace, which is a snapshot of the cleaner's pack dict
    # (intent/resolver fields are added to pack during _chat_execute).
    pack_trace: dict = {}
    for tc in tcs:
        it = tc.get("input_trace")
        if isinstance(it, dict):
            pack_trace = it
            break
    debug["intent"] = pack_trace.get("_intent")
    debug["intent_confidence"] = pack_trace.get("_intent_confidence")
    debug["intent_reasoning"] = pack_trace.get("_intent_reasoning")
    debug["entity_resolved"] = pack_trace.get("entity_candidate")
    debug["entity_type"] = pack_trace.get("_resolver_entity_type")
    debug["parent_entity"] = pack_trace.get("_resolver_parent")
    debug["resolver_reasoning"] = pack_trace.get("_resolver_reasoning")
    debug["inherited_from_history"] = pack_trace.get("_inherited_from_history")
    debug["intent_clean_entity"] = pack_trace.get("_intent_clean_entity")
    debug["deep_profile_mode"] = bool(pack_trace.get("_deep_profile_mode"))
    debug["exec_web_augmented"] = bool(pack_trace.get("_exec_web_augmented"))
    # For executive_lookup / executive_succession, the {person, company,
    # role} extracted by _extract_exec_target is more informative than
    # the cleaner's raw entity_candidate (which often includes the
    # "CEO of" prefix). Surface it when present.
    debug["exec_lookup_target"] = pack_trace.get("_exec_lookup_target")

    # Aliases — prefer the clean target company over the cleaner's raw
    # entity (which may include role/jargon prefixes).
    alias_seed = (
        debug["intent_clean_entity"]
        or (debug["exec_lookup_target"] or {}).get("company")
        or debug["entity_resolved"]
    )
    debug["aliases_found"] = []
    if alias_seed:
        try:
            from app.services.alias_resolver import expand_aliases
            debug["aliases_found"] = expand_aliases(alias_seed)
        except Exception:
            pass

    # Tables searched + row counts (one entry per tool call).
    debug["tables_searched"] = []
    for tc in tcs:
        tbl = tc.get("table")
        if not tbl:
            continue
        filters = tc.get("filters") or {}
        debug["tables_searched"].append({
            "table": tbl,
            "row_count": int(tc.get("row_count") or 0),
            "trace_markers": sorted([k for k in filters.keys()
                                     if k.startswith("_")]),
            "filter_cols": sorted([k for k in filters.keys()
                                   if not k.startswith("_")]),
        })

    # Aggregate counters
    debug["evidence_row_count_total"] = sum(
        int(tc.get("row_count") or 0) for tc in tcs
    )
    debug["evidence_table_count"] = len({
        tc.get("table") for tc in tcs if tc.get("table")
    })
    return debug


# ─── Response cache (OPT-IN) ──────────────────────────────────────────
# Even at temperature 0 + fixed seed, OpenAI is only best-effort
# deterministic, so the SAME question can still reword slightly between
# runs. This cache returns the byte-identical answer for a repeated
# single-turn question within the TTL.
#
# DEFAULT OFF. The determinism controls (temperature 0 + seed) already
# give strong consistency without side effects. The cache adds two
# risks not worth carrying by default: (1) it freezes whatever the
# first answer was — a one-off weak answer would be served for the
# whole TTL; (2) it makes the quality/regression batteries flaky by
# serving stale answers. Enable with MISA_RESPONSE_CACHE=true when
# byte-identical repeats matter and the staleness tradeoff is accepted.
_RESPONSE_CACHE: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_RESPONSE_CACHE_LOCK = Lock()


def _response_cache_settings() -> tuple[bool, int, int]:
    import os
    enabled = (os.getenv("MISA_RESPONSE_CACHE", "false") or "").strip().lower() \
        in ("1", "true", "yes", "on")
    try:
        ttl = int(os.getenv("MISA_RESPONSE_CACHE_TTL", "900") or "900")
    except ValueError:
        ttl = 900
    return enabled, ttl, 500


def _response_cache_key(user_question: str, ui_locale: str) -> str:
    norm = re.sub(r"\s+", " ", (user_question or "").strip().lower())
    return hashlib.sha256(f"{ui_locale}\x1f{norm}".encode()).hexdigest()


def _response_cache_get(key: str) -> dict | None:
    enabled, ttl, _ = _response_cache_settings()
    if not enabled:
        return None
    with _RESPONSE_CACHE_LOCK:
        hit = _RESPONSE_CACHE.get(key)
        if not hit:
            return None
        ts, val = hit
        if (time.time() - ts) > ttl:
            _RESPONSE_CACHE.pop(key, None)
            return None
        _RESPONSE_CACHE.move_to_end(key)
        # Return a shallow copy so callers can't mutate the cached dict.
        out = dict(val)
        out["_from_cache"] = True
        return out


def _response_cache_put(key: str, result: dict) -> None:
    enabled, _, cap = _response_cache_settings()
    if not enabled:
        return
    with _RESPONSE_CACHE_LOCK:
        _RESPONSE_CACHE[key] = (time.time(), dict(result))
        _RESPONSE_CACHE.move_to_end(key)
        while len(_RESPONSE_CACHE) > cap:
            _RESPONSE_CACHE.popitem(last=False)


def _is_cacheable(user_question: str, history: list, result: dict) -> bool:
    # Only cache clean, single-turn, successful answers. Follow-ups
    # depend on history; errors/clarifications should re-run.
    if history:
        return False
    if not result or result.get("error"):
        return False
    ans = result.get("answer") or ""
    if len(ans) < 40:
        return False
    src = result.get("_answer_source") or ""
    if src in (
        "clarification", "off_topic_fallback",
        "off_topic_redirect", "prompt_guard_refusal",
    ):
        return False
    if result.get("_short_circuit") in ("off_topic", "prompt_injection"):
        # Never cache a refusal — every attack attempt should hit the
        # guard fresh so it's logged/telemetered, not served from cache.
        return False
    # Never freeze raw multi-match listings or retrieval-trace dumps —
    # those are not Jul21 briefs and poison the cache.
    if re.search(
        r"(?i)Multiple possible matches|Your search matched|"
        r"Open \*\*Retrieval trace\*\*|Retrieval trace",
        ans,
    ):
        return False
    # Company-profile asks must cache only full Jul21 shapes.
    if re.search(
        r"(?i)\b(company\s+(?:profile|briefing)|briefing\s+on|"
        r"profile\s+of)\b",
        user_question or "",
    ):
        if "Executive Briefing" not in ans and "Snapshot of Operations" not in ans:
            return False
    # Corridor engagement / market-fit asks must cache full Jul21 advisory
    # shape — never freeze a mis-routed named-company Engagement Recommendation.
    if re.search(
        r"(?i)\b(engagement\s+plan|market\s+fit|attract\w*\s+\w+\s+compan)",
        user_question or "",
    ):
        if not re.search(r"(?i)Strategic Context", ans):
            return False
        if re.search(r"(?i)engagement\s+plan", user_question or ""):
            if not (
                re.search(r"(?i)Phase\s*1|Phased\s+Roadmap", ans)
                and re.search(r"(?i)KPI", ans)
            ):
                return False
        # Named-company engagement stub is never a corridor plan.
        if re.search(
            r"(?im)^##\s+Engagement Recommendation\b",
            ans,
        ) and not re.search(r"(?i)Phase\s*1|Phased\s+Roadmap", ans):
            return False
    return True


def _attach_quality_meta(result: dict | None) -> dict:
    """Lift pack quality/intent fields onto the top-level result for the API."""
    if not result:
        return result or {}
    pack: dict = {}
    for tc in (result.get("tool_calls") or []):
        trace = tc.get("input_trace")
        if isinstance(trace, dict) and trace:
            pack = trace
            break
    for k in (
        "_query_intent", "_quality_gate", "_quality_eval", "_retrieval",
        "_retrieval_status", "_pipeline_trace", "_data_limitations",
        "_truncated", "_degraded",
        "_advisory_deliverable", "_advisory_db_context",
        "_advisory_validation_fixes", "_quality_gate_fixes",
        "_quality_gate_issues", "_short_circuit",
    ):
        if result.get(k) is None and pack.get(k) is not None:
            result[k] = pack[k]
    # Normalize client-facing snapshots
    if result.get("_pipeline_trace") is not None and hasattr(
        result["_pipeline_trace"], "trace_id"
    ):
        result["_trace_id"] = result["_pipeline_trace"].trace_id
    elif not result.get("_trace_id"):
        result["_trace_id"] = None
    return result


def chat(user_question: str, history: list, ui_locale: str = "en") -> dict:
    """Entry point: runs the chat pipeline and emits a structured turn log."""
    turn_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    outcome = "ok"
    err_type: str | None = None
    result: dict | None = None
    _cache_key = _response_cache_key(user_question, ui_locale) if not history else None
    if _cache_key:
        cached = _response_cache_get(_cache_key)
        if cached is not None:
            return cached
    try:
        result = _chat_execute(user_question, history, ui_locale)
        result = _attach_quality_meta(result)
        if result.get("error"):
            outcome = "result_error"
        if _cache_key and _is_cacheable(user_question, history, result):
            _response_cache_put(_cache_key, result)
        return result
    except Exception as e:
        outcome = "exception"
        err_type = type(e).__name__
        raise
    finally:
        payload = {
            "event": "chat_turn",
            "turn_id": turn_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            "outcome": outcome,
            "error_type": err_type,
            "ui_locale": ui_locale,
            "question_len": len(user_question or ""),
            "question_sha256_16": hashlib.sha256(
                (user_question or "").encode("utf-8", errors="replace")
            ).hexdigest()[:16],
        }
        if result is not None:
            tcs = result.get("tool_calls") or []
            payload["tool_calls_n"] = len(tcs)
            payload["answer_len"] = len(result.get("answer") or "")
            payload["had_error_field"] = bool(result.get("error"))

            # Pipeline-aware diagnostics: intent + extracted entity +
            # confidence scores + failure_type. These let an operator
            # grep the log for low-confidence turns or for a particular
            # failure mode.
            try:
                from app.services.query_intent import build_query_intent
                qi = build_query_intent(user_question, history)
                payload["query_intent"] = qi.to_log_dict()
                payload["detected_intent"] = qi.legacy_intent_label
            except Exception:
                try:
                    payload["detected_intent"] = detect_intent(
                        user_question, history
                    )
                except Exception:
                    payload["detected_intent"] = "unknown"
            if result.get("_quality_eval"):
                payload["quality_eval"] = {
                    k: result["_quality_eval"].get(k)
                    for k in ("score", "pass")
                }
            if result.get("_quality_gate"):
                payload["quality_gate"] = result.get("_quality_gate")
            try:
                pack_for_log = clean_user_question(user_question)
                payload["extracted_entity"] = pack_for_log.get(
                    "entity_candidate"
                )
            except Exception:
                payload["extracted_entity"] = None
            # Pipeline diagnostics: what the engine did and where the
            # answer came from. Useful for debugging "why did the model
            # pick this table" / "did fuzzy fire" / "was it a fallback".
            payload["tables_queried"] = [tc.get("table") for tc in tcs if tc.get("table")]
            payload["row_count_total"] = sum(
                int(tc.get("row_count") or 0) for tc in tcs
            )
            # Distinguish verified-empty from failed-retrieval zeros
            payload["retrieval_failures_n"] = sum(
                1 for tc in tcs
                if (tc.get("filters") or {}).get("_db_error")
                or (tc.get("error"))
            )
            payload["filter_cols_used"] = sorted({
                k
                for tc in tcs
                for k in (tc.get("filters") or {}).keys()
                if not k.startswith("_")  # skip internal trace markers
            })
            # Detect server-side rewrites / fallbacks via trace markers.
            trace_markers: list[str] = []
            for tc in tcs:
                for k in (tc.get("filters") or {}).keys():
                    if k.startswith("_"):
                        trace_markers.append(k)
            payload["trace_markers"] = sorted(set(trace_markers))
            payload["fuzzy_match_fired"] = any(
                "_smart_fallback_after_empty" in m or "_keyword_search_injected" in m
                for m in trace_markers
            )
            payload["entity_matched"] = all(
                tc.get("row_entity_sanity_passed", True) for tc in tcs
            ) if tcs else True
            # answer_source heuristic — set explicitly by short-circuit
            # paths via result["_answer_source"], else inferred.
            if result.get("_answer_source"):
                payload["answer_source"] = result["_answer_source"]
            elif not tcs:
                # No tool call: greeting or pure-conversation
                payload["answer_source"] = "conversational"
            elif payload["row_count_total"] == 0:
                # Rows empty → answer came from OpenAI general-knowledge fallback
                payload["answer_source"] = "fallback"
            else:
                payload["answer_source"] = "db"

            # Convenience booleans for log filtering
            payload["db_answer_found"] = (
                payload["row_count_total"] > 0 and
                payload["answer_source"] in ("db", "count_only",
                                              "clarification",
                                              "deterministic_no_match")
            )
            payload["openai_fallback_used"] = payload["answer_source"] in (
                "fallback", "off_topic_fallback"
            )

            # Confidence scores
            try:
                # Re-derive classification for confidence — best-effort,
                # not all paths populate this. Try _classification from
                # result (if set), else infer from entity sanity flags.
                cls = result.get("_classification")
                if cls is None:
                    if tcs and payload["row_count_total"] > 0:
                        cls = "exact" if payload.get("entity_matched") else None
                payload["confidence_scores"] = compute_confidence(
                    tool_calls_executed=tcs,
                    classification=cls,
                    answer_source=payload["answer_source"],
                    intent=payload.get("detected_intent", "unknown"),
                )
            except Exception:
                payload["confidence_scores"] = None

            # Failure-type classification — for grep-friendly triage
            failure_type = None
            if outcome == "exception":
                failure_type = "server_exception"
            elif outcome == "result_error":
                failure_type = "result_error_field"
            elif payload["row_count_total"] == 0 and payload["answer_source"] == "fallback":
                failure_type = "no_db_match"
            elif payload.get("trace_markers") and any(
                m in payload["trace_markers"]
                for m in ("_dropped_unknown_filters",
                          "_invalid_filter_for_count")
            ):
                failure_type = "invalid_filter"
            payload["failure_type"] = failure_type
        _structured_turn_log(payload)
