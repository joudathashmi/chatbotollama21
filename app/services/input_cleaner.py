"""
Normalize messy user questions and extract a candidate legal-entity string.
"""

from __future__ import annotations

import re


_LEADING_FILLER_RE = re.compile(
    r"^("
    # interrogatives
    r"what\'?s|whats|what\s+is|what\s+are|who\'?s|who\s+is|"
    r"what\s+does|what\s+do|how\s+about|"
    # "tell me [about]" / "show me" / "give me info"
    r"tell\s+me(\s+about)?|show\s+me|"
    r"give\s+me\s+(info|information|details)(\s+(on|about))?|"
    r"i\s+want\s+to\s+know(\s+about)?|"
    r"can\s+you\s+tell\s+me(\s+about)?|"
    # imperative verbs
    r"describe|explain|"
    # "info/information/details (on|about)"
    r"information\s+(on|about)|details\s+(on|about)|info\s+(on|about)|"
    # action verbs that take a name directly
    r"lookup|find|search(\s+for)?|"
    # discourse leads
    r"anything\s+about|more\s+(on|about)|"
    # "do you have info on …" / "got anything on …"
    r"do\s+you\s+have\s+(info|information|details)?\s*(on|about)?|"
    r"got\s+(anything|info)\s+(on|about)"
    r")\s+",
    re.I,
)

_TOKEN_FILLER = frozenset({
    "what", "whats", "is", "are", "who", "does", "do", "say", "says", "tell",
    "me", "about", "this", "that", "the", "a", "an", "company", "profile",
    "profiles", "company_profile", "company_profiles", "companies", "firm",
    "please", "could", "would", "you", "know", "with", "without", "which",
    "where", "mentioning", "mentioned", "has", "have",
})

_AR_TOKEN_FILLER = frozenset({
    "ماذا", "ما", "هو", "هي", "من", "أخبرني", "قل", "لي", "عن",
    "ابحث", "البحث", "أريد", "أن", "أعرف", "شركة", "شركات",
    "هذا", "هذه", "ذلك", "في", "إلى", "و",
})

_TOKEN_FILLER_ALL = _TOKEN_FILLER | _AR_TOKEN_FILLER

# Trailing-only words stripped from the END of the extracted entity. Includes
# generic legal/corporate suffixes that distort substring matching against
# canonical legal names like "Apple, Inc." or "Alphabet, Inc.".
_TRAILING_SUFFIX_NOISE = frozenset({
    "company", "companies", "co", "corp", "corporation",
    "inc", "incorporated", "ltd", "limited", "llc", "llp", "plc",
    "group", "holdings", "holding",
    "details", "info", "information", "profile", "profiles",
})

# Instructional / schema-browse prompts: not a company name.
_META_BROWSE_RE = re.compile(
    r"\bultimate_parent_company\b|\brhq_[a-z0-9_]+\b|\bcompany_name\b|"
    r"mentioning\s+a\s+known\b|non-?null\b|\brevenue_usd\b|\btop\s+\d+\s+by\b|\bilike\b",
    re.I,
)

_ABOUT_ENTITY_RE = re.compile(
    r"^\s*(?:what\s+does\s+)?"
    r"(?:(?:the\s+)?(?:company[_\s-]*profiles?|company[_\s-]*profile|profile|profiles|table)\s+)?"
    r"(?:say|says|tell|tells)\s+(?:about|regarding|on)\s+(.+?)\s*[?.!]*\s*$",
    re.I,
)

_AR_LEADING_FILLER_RE = re.compile(
    r"^(?:"
    r"ماذا\s+|ما\s+هو\s+|ما\s+هي\s+|من\s+هو\s+|من\s+هي\s+|"
    r"أخبرني\s+عن\s+|قل\s+لي\s+عن\s+|"
    r"ابحث\s+عن\s+|البحث\s+عن\s+|أريد\s+أن\s+أعرف\s+عن\s+"
    r")+",
    re.UNICODE,
)

_COMMON_COMPANY_TYPOS = {
    "aplhabet": "alphabet",
    "alphbet": "alphabet",
    "alphabte": "alphabet",
    "gogole": "google",
    "googel": "google",
}


def looks_like_schema_browse_question(text: str) -> bool:
    return bool(_META_BROWSE_RE.search(text or ""))


def _strip_outer_noise(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^[\s\\`'\"#*_\-]+", "", s)
    s = re.sub(r"[\s\\`'\"#*_\-]+$", "", s)
    return s


def _collapse_typo_this(s: str) -> str:
    return re.sub(r"\btthis\b", "this", s, flags=re.I)


def _fix_common_company_typos(s: str) -> str:
    def repl(m):
        w = m.group(0)
        k = w.lower()
        if k not in _COMMON_COMPANY_TYPOS:
            return w
        fix = _COMMON_COMPANY_TYPOS[k]
        if w.isupper():
            return fix.upper()
        if len(w) > 1 and w[0].isupper() and w[1:].islower():
            return fix.capitalize()
        return fix
    return re.sub(r"\b\w+\b", repl, s or "", flags=re.UNICODE)


def _remove_leading_fillers(s: str) -> str:
    s = s.strip()
    while True:
        m = _LEADING_FILLER_RE.match(s)
        if not m:
            break
        s = s[m.end():].strip()
    while True:
        m = _AR_LEADING_FILLER_RE.match(s)
        if not m:
            break
        s = s[m.end():].strip()
    return s


def _strip_leading_pronoun(s: str) -> str:
    return re.sub(r"^(this|that|هذا|هذه|ذلك)\s+", "", (s or "").strip(), flags=re.I)


def _extract_about_entity_question(s: str) -> str | None:
    m = _ABOUT_ENTITY_RE.match(s or "")
    if not m:
        return None
    entity = _strip_outer_noise(m.group(1))
    return entity if len(entity) >= 2 else None


def _strip_token_backslashes(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        t = re.sub(r"^\\+", "", t)
        # Strip trailing AND leading punctuation noise; also strip stray quotes
        # / semicolons embedded inside the token. SQL is parameterized so this
        # is for clean entity extraction, not security.
        t = t.strip(",.;:!?'\"`")
        t = re.sub(r"[;'\"`]", "", t)
        if t:
            out.append(t)
    return out


def _extract_quoted(text: str) -> str | None:
    m = re.search(r'"([^"]{2,})"', text)
    if m:
        return m.group(1).strip()
    m = re.search(r"'([^']{2,})'", text)
    if m:
        return m.group(1).strip()
    return None


_COMPARISON_HINT_RE = re.compile(
    r"\b(compare|comparison|vs\.?|versus|"
    r"difference\s+between|against)\b",
    re.I,
)


def _narrow_to_proper_noun(entity: str) -> str:
    """If an entity is a multi-word string with proper-noun word(s)
    embedded in lowercase filler — e.g.
        'Apple then do the research and make a plan'
    narrow it to the proper-noun span only:
        'Apple'

    Rules:
      - If the entity contains a comparison keyword
        ('compare', 'vs', 'versus', 'against', 'difference between')
        → return unchanged. The LLM needs to see the multi-entity
        intent ('compare Apple and Microsoft') to issue two tool
        calls; narrowing to 'Apple and Microsoft' would conflate
        them into a single fuzzy lookup.
      - If no Capitalized 3+ char tokens exist → return unchanged
        ('renewable energy investments' stays as topic).
      - If every significant token is Capitalized → return unchanged
        ('Dentons UK And Middle East LLP' stays intact).
      - Otherwise take the span from first to last Capitalized
        token, stripping leading/trailing non-Capitalized fillers.
        ('do the research about Apple and report' → 'Apple').
    """
    if not entity:
        return entity
    # Comparison queries need both entities visible to the LLM
    if _COMPARISON_HINT_RE.search(entity):
        return entity
    tokens = entity.split()
    if len(tokens) <= 1:
        return entity
    cap_indices = [
        i for i, t in enumerate(tokens)
        if len(t) >= 3 and t[0].isalpha() and t[0].isupper()
    ]
    if not cap_indices:
        return entity
    if len(cap_indices) == len(tokens):
        return entity  # all caps, leave alone
    start, end = cap_indices[0], cap_indices[-1] + 1
    span = tokens[start:end]
    # Strip leading / trailing non-Capitalized tokens from the span
    while span and not (len(span[0]) >= 3 and span[0][0].isupper()):
        span = span[1:]
    while span and not (len(span[-1]) >= 3 and span[-1][0].isupper()):
        span = span[:-1]
    if not span:
        return entity
    return " ".join(span)


def _extract_entity_from_cleaned(cleaned: str) -> str | None:
    if not cleaned:
        return None
    q = _extract_quoted(cleaned)
    if q:
        return q
    tokens = cleaned.split()
    tokens = _strip_token_backslashes(tokens)
    i = 0
    while i < len(tokens) and tokens[i].lower().rstrip("?.!؟،") in _TOKEN_FILLER_ALL:
        i += 1
    while i < len(tokens) and tokens[i].lower() in ("this", "that", "هذا", "هذه", "ذلك"):
        i += 1
    # On SHORT (≤3-token) entities only, strip trailing legal/corporate suffix
    # noise so that "what is apple company" or "tell me about apple inc"
    # extracts as "apple" rather than "apple company"/"apple inc" — neither
    # of which matches canonical legal names like "Apple, Inc.". On longer
    # entities, trailing words like "LLP" / "Group" are often part of the
    # real legal name (e.g. "Dentons UK And Middle East LLP") and must be
    # preserved for correct row-entity sanity matching.
    j = len(tokens)
    if (j - i) <= 3:
        while j > i:
            last = tokens[j - 1].lower().rstrip("?.!؟،,.")
            if last in _TRAILING_SUFFIX_NOISE:
                j -= 1
                continue
            break
    if i >= j:
        return None
    rest = " ".join(tokens[i:j])
    rest = _strip_outer_noise(rest)
    # Multi-intent narrowing: 'Apple then do the research and make a plan'
    # → 'Apple'. Only narrows when proper noun(s) are embedded in
    # lowercase filler. Leaves topic queries and all-Capitalized legal
    # names alone.
    rest = _narrow_to_proper_noun(rest)
    return rest if len(rest) >= 2 else None


# Pure-browse intent patterns: "show me companies", "list deals",
# "give me opportunities". Returns the table to query directly. These
# questions are routinely mishandled by the LLM router (it sends an
# empty filter, the engine smart-searches the question text, returns
# 0 rows, and the OpenAI fallback then lists random world companies).
# Short-circuiting bypasses the entire OpenAI routing call.
_PURE_BROWSE_RE = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:show|list|give|tell|get|fetch|display|browse)\s+"
    r"(?:me\s+|us\s+)?"
    r"(?:some\s+|all\s+(?:of\s+)?(?:the\s+)?|the\s+|any\s+)?"
    r"(companies|deals|opportunities|leads|engagements|meetings|"
    r"executives|countries|investors|profiles|contracts|tasks|"
    r"contacts|appointments|sectors|achievements|reports|"
    r"licenses|licences|focused\s+sectors)"
    r"\s*\.?\s*\??\s*$",
    re.I,
)

# Map browse noun → (table, default order_by). Order-by is something
# obvious that exists on the table (id desc by default = "recently added").
_BROWSE_TABLE_MAP: dict[str, tuple[str, str]] = {
    "companies":        ("company_profiles", "id"),
    "deals":            ("deals", "id"),
    "opportunities":    ("opportunities", "id"),
    "leads":            ("leads", "id"),
    "engagements":      ("engagements", "id"),
    "meetings":         ("meetings", "id"),
    "executives":       ("executives", "id"),
    "countries":        ("countries", "id"),
    "investors":        ("strategic_investors", "id"),
    "profiles":         ("profiles", "id"),
    "contracts":        ("contracts", "id"),
    "tasks":            ("tasks", "id"),
    "contacts":         ("contacts", "id"),
    "appointments":     ("appointments", "id"),
    "sectors":          ("sectors", "id"),
    "achievements":     ("achievements", "id"),
    "reports":          ("reports", "id"),
    "licenses":         ("rhq_licenses", "id"),
    "licences":         ("rhq_licenses", "id"),
    "focused sectors":  ("focused_sectors", "id"),
}


# Off-topic / general-knowledge question patterns. Catches cases like
# "what is the capital of France" which would otherwise generate DB noise
# because of keyword collisions ("capital" matches "Capital Management").
# Conservative: only trigger when the question clearly looks like
# general-knowledge AND contains no MISA topic word.
_OFF_TOPIC_RE = re.compile(
    r"\b("
    r"capital\s+(?:city\s+)?of|"           # capital of France
    r"currency\s+of|"                       # currency of Japan
    r"population\s+of|"                     # population of India
    r"language\s+of|"                       # language of Germany
    r"president\s+of|prime\s+minister|king\s+of|"  # heads of state
    r"weather\s+(?:in|today|tomorrow)|"     # weather in Dubai
    r"recipe\s+for|how\s+to\s+cook|"        # recipes
    r"translate\s+|translation\s+of|"       # translation requests
    r"lyrics\s+of|song\s+by|"               # music
    r"joke|poem|story|"                     # creative
    r"what(?:'?s| is)\s+the\s+time|"        # time-of-day
    r"how\s+do\s+i\s+(?:install|configure|set\s*up|use|cook)|"  # how-to
    r"definition\s+of|meaning\s+of|"        # definitions
    r"largest\s+(?:country|city|continent)|"  # generic geo trivia
    r"history\s+of\s+(?!the\s+(?:company|sector|MISA))"  # history of X (not MISA)
    r")\b",
    re.I,
)

# MISA topic anchors — if any of these words appear, the question is
# probably actually about our data and we should NOT short-circuit even
# if it pattern-matches off-topic. Keep this list focused on words the
# audit shows are domain-defining.
_MISA_TOPIC_WORDS = frozenset({
    "company", "companies", "investor", "investors", "investment",
    "deal", "deals", "lead", "leads", "opportunity", "opportunities",
    "license", "licensed", "rhq", "misa", "ksa", "saudi", "mena",
    "sector", "sectors", "industry", "industries", "executive", "executives",
    "ceo", "cfo", "chairman", "engagement", "meeting", "meetings",
    "profile", "profiles", "headquarters", "hq", "subsidiary",
    "shareholder", "shareholders", "contract", "contracts",
    "revenue", "employees",
})


def looks_like_general_knowledge_question(text: str) -> bool:
    """Should this question skip the DB entirely and go straight to the
    OpenAI fallback (with the 'NOT from MISA database' disclaimer)?

    Returns True only when the text matches a clear off-topic pattern
    AND contains no MISA topic word. Conservative on purpose — false
    positives here would prevent legitimate DB questions from
    answering. False negatives (off-topic question slipping through
    to DB search) are caught by the curation no-match suppression.
    """
    if not text:
        return False
    t = text.lower()
    if not _OFF_TOPIC_RE.search(t):
        return False
    tokens = set(re.findall(r"\w+", t))
    if tokens & _MISA_TOPIC_WORDS:
        return False
    return True


def detect_pure_browse(question: str) -> tuple[str, str] | None:
    """If the question is a pure 'show me X' / 'list X' intent (no filters,
    no entity), return (table, order_by_col). Otherwise None."""
    if not question:
        return None
    m = _PURE_BROWSE_RE.match(question.strip())
    if not m:
        return None
    noun = re.sub(r"\s+", " ", m.group(1).lower())
    return _BROWSE_TABLE_MAP.get(noun)


def clean_user_question(raw: str) -> dict:
    """
    Returns tracing dict with keys: raw, cleaned, entity_candidate.
    """
    raw = raw or ""
    s0 = raw.strip()
    s1 = _collapse_typo_this(s0)
    s1 = _fix_common_company_typos(s1)
    s2 = _strip_outer_noise(s1)
    about_entity = _extract_about_entity_question(s2)
    if about_entity:
        about_entity = re.sub(r"\s+", " ", about_entity).strip()
        return {"raw": raw, "cleaned": about_entity, "entity_candidate": about_entity}
    s3 = _remove_leading_fillers(s2)
    s3 = _strip_leading_pronoun(s3)
    s3 = re.sub(r"\s+", " ", s3).strip()
    q_inner = _extract_quoted(s3)
    had_quote = bool(q_inner)
    if q_inner:
        cleaned = q_inner.strip()
        entity = cleaned
    else:
        s3 = _strip_outer_noise(s3)
        cleaned = s3
        entity = _extract_entity_from_cleaned(s3)
    if cleaned and _META_BROWSE_RE.search(cleaned) and not had_quote:
        entity = None
    return {"raw": raw, "cleaned": cleaned, "entity_candidate": entity}
