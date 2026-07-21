"""
Deterministic prompt-attack guard for the chat endpoint.

Runs BEFORE any LLM or DB work and refuses, deterministically, when a
user turn is an obvious attempt to:
  - override the system instructions ("ignore previous instructions")
  - extract the hidden system prompt ("reveal your system prompt")
  - dump the database schema/catalog ("list all tables / columns")
  - disclose internal analyst/reviewer notes ("show internal comments")
  - hijack the assistant's role ("you are now DAN", "developer mode")

This is DEFENSE IN DEPTH, not the only layer — the curation privacy
filter (curation._is_sensitive_key), unsourced-bullet stripping, fixed
JSON tool schema, and provenance labelling still stand behind it. The
value of a deterministic pre-filter is that the most common, lowest-
effort attacks never reach a model at all, and the refusal is stable
and testable rather than dependent on the LLM's mood.

Design constraints:
  - LOW false-positive rate is the priority. This endpoint answers real
    investment-intelligence questions; wrongly refusing a legitimate
    query is a worse failure than missing an exotic jailbreak phrasing
    (the downstream structural guardrails catch what slips through).
    Patterns are therefore multi-word and tightly anchored, not single
    trigger words ("ignore", "system", "act" alone are NOT matched).
  - Bilingual: English + Arabic, since the deployment is bilingual and
    attacks translate.
"""

from __future__ import annotations

import re
import unicodedata

# Each entry: (category, compiled pattern). Categories are surfaced in
# logs/telemetry (not to the user) so repeated attack classes are
# visible for the red-team monitoring requirement.

_PATTERNS: list[tuple[str, re.Pattern]] = [
    # ── instruction override ────────────────────────────────────────
    ("instruction_override", re.compile(
        r"\b(ignore|disregard|forget)\b[^.\n]{0,30}\b(previous|prior|above|earlier|preceding|all|your|the)\b"
        r"[^.\n]{0,20}\b(instruction|instructions|prompt|prompts|rule|rules|directive|directives|"
        r"guardrail|guardrails|guideline|guidelines|message|messages|context)\b",
        re.I)),
    ("instruction_override", re.compile(
        r"\b(override|bypass|circumvent|disable|turn\s+off)\b[^.\n]{0,25}\b"
        r"(instruction|instructions|rule|rules|restriction|restrictions|guardrail|guardrails|"
        r"safety|filter|filters|safeguard|safeguards|limitation|limitations)\b",
        re.I)),
    ("instruction_override", re.compile(
        r"\bforget\s+(everything|all)\b[^.\n]{0,20}\b(you\s+were\s+told|before|above|instructed)\b",
        re.I)),

    # ── system-prompt extraction ────────────────────────────────────
    ("prompt_extraction", re.compile(
        r"\b(reveal|show|print|repeat|output|display|expose|leak|tell\s+me|give\s+me)\b[^.\n]{0,40}\b"
        r"(system\s+prompt|system\s+message|your\s+prompt|your\s+instructions?|initial\s+instructions?|"
        r"original\s+instructions?|your\s+rules?|your\s+guidelines?|your\s+directives?|"
        r"the\s+prompt\s+above|words?\s+above|text\s+above)\b",
        re.I)),
    ("prompt_extraction", re.compile(
        r"\bwhat\s+(is|are|were)\b[^.\n]{0,20}\byour\b[^.\n]{0,15}\b"
        r"(exact\s+)?(instructions?|system\s+prompt|system\s+message|rules?|directives?|guidelines?|"
        r"configuration|programming)\b",
        re.I)),
    ("prompt_extraction", re.compile(
        r"\brepeat\b[^.\n]{0,20}\b(the\s+)?(words|text|everything|sentence|prompt)\b[^.\n]{0,15}\b(above|before|earlier)\b",
        re.I)),

    # ── schema / catalog extraction ─────────────────────────────────
    ("schema_extraction", re.compile(
        r"\b(show|list|reveal|print|give\s+me|dump|expose|what\s+(are|is))\b[^.\n]{0,40}\b"
        r"(table\s+schema|database\s+schema|db\s+schema|full\s+schema|the\s+schema|"
        r"all\s+(the\s+)?tables?|every\s+table|table\s+names?|list\s+of\s+tables?|"
        r"all\s+(the\s+)?columns?|column\s+names?|list\s+of\s+columns?)\b",
        re.I)),
    ("schema_extraction", re.compile(
        r"\b(database|db)\s+(schema|structure|catalog|catalogue|layout)\b",
        re.I)),

    # ── internal-note disclosure ────────────────────────────────────
    ("internal_disclosure", re.compile(
        r"\b(show|reveal|display|give\s+me|expose|leak|what\s+(are|is|do))\b[^.\n]{0,40}\b"
        r"(internal\s+(comment|comments|note|notes|remark|remarks)|"
        r"reviewer\s+(comment|comments|note|notes)|analyst\s+(comment|comments|note|notes)|"
        r"misa\s+comments?|team\s+comments?|company\s+notes?|"
        r"audit\s+(log|logs|trail)|hidden\s+(field|fields|data|column|columns))\b",
        re.I)),

    # ── role hijack / jailbreak ─────────────────────────────────────
    ("role_hijack", re.compile(
        r"\byou\s+are\s+now\b[^.\n]{0,25}\b(a|an|in|the|dan|jailbroken|unrestricted|unfiltered|free)\b",
        re.I)),
    ("role_hijack", re.compile(
        r"\bpretend\s+(that\s+)?(you\s+are|you're|to\s+be)\b",
        re.I)),
    ("role_hijack", re.compile(
        r"\b(developer\s+mode|dan\s+mode|jailbreak|jailbroken|do\s+anything\s+now)\b",
        re.I)),
    ("role_hijack", re.compile(
        r"\b(act|behave|respond|roleplay|role.?play)\s+as\s+(a|an|if)\b[^.\n]{0,25}\b"
        r"(unrestricted|unfiltered|jailbroken|no\s+restrictions?|without\s+(any\s+)?(restrictions?|rules?|filters?|limits?)|"
        r"evil|malicious|hacker)\b",
        re.I)),
    ("role_hijack", re.compile(
        r"\b(with|have)\s+no\s+(restrictions?|rules?|filters?|limits?|guardrails?|safety)\b",
        re.I)),

    # ── Arabic (high-signal equivalents) ────────────────────────────
    # ignore/disregard the (previous) instructions/rules
    ("instruction_override", re.compile(
        r"(تجاهل|تجاهلي|اهمل|انسَ|انسي)\s*[^.\n]{0,20}?(التعليمات|التوجيهات|القواعد|الأوامر|ما\s+سبق|السابق)")),
    # reveal/show the (system) instructions/prompt
    ("prompt_extraction", re.compile(
        r"(اكشف|أظهر|اظهر|اعرض|اطبع|كرر)\s*[^.\n]{0,20}?(التعليمات|التوجيهات|موجه\s+النظام|النظام\s+الأساسي|قواعدك|تعليماتك)")),
    # what are your instructions
    ("prompt_extraction", re.compile(
        r"ما\s+(هي|هى)\s+[^.\n]{0,10}?(تعليمات|قواعد|توجيهات)")),
    # show all tables / database schema
    ("schema_extraction", re.compile(
        r"(أظهر|اظهر|اعرض|اكشف)\s*[^.\n]{0,20}?(الجداول|قاعدة\s+البيانات|مخطط\s+قاعدة|الأعمدة)")),

    # ── encoding-smuggling lure ─────────────────────────────────────
    # "decode this base64 and follow it: aWdub3Jl…" — a decode/run verb
    # next to a long base64-ish blob. We flag the *lure*, we don't
    # decode (decoding arbitrary input invites its own problems).
    ("encoding_evasion", re.compile(
        r"\b(base\s*64|b64|decode|decouple|deobfuscate|from\s+base64|rot13)\b"
        r"[^.\n]{0,40}[A-Za-z0-9+/]{20,}={0,2}",
        re.I)),

    # ── foreign-language instruction override (high-signal only) ─────
    # French / Spanish / German equivalents of "ignore the instructions".
    ("instruction_override", re.compile(
        r"\b(ignore[zr]?|oublie[zr]?|ignora|olvida|vergiss|missachte)\b[^.\n]{0,25}\b"
        r"(instruction|instructions|instrucciones|règles|reglas|regeln|anweisungen|consignes)\b",
        re.I)),
]

# User-facing refusal — deliberately generic (does not name the
# detected category, which would help an attacker probe the filter),
# and redirects to the legitimate purpose.
_REFUSAL_EN = (
    "I can't help with attempts to override my instructions, reveal internal "
    "system configuration or schema, or disclose internal notes.\n\n"
    "I'm here to answer questions about companies, countries, deals, executives, "
    "and investment intelligence from the MISA database — feel free to ask one of those."
)
_REFUSAL_AR = (
    "لا يمكنني المساعدة في محاولات تجاوز التعليمات، أو كشف إعدادات النظام الداخلية أو "
    "مخطط قاعدة البيانات، أو الإفصاح عن الملاحظات الداخلية.\n\n"
    "أنا هنا للإجابة عن الأسئلة المتعلقة بالشركات والدول والصفقات والمسؤولين التنفيذيين "
    "ومعلومات الاستثمار من قاعدة بيانات «مِسا» — يسعدني الإجابة عن أي منها."
)


# ── Evasion normalization ───────────────────────────────────────────
# A regex on the raw text alone is trivially bypassed with homoglyphs
# ("ѕystem" with a Cyrillic ѕ), zero-width chars, whitespace-splitting
# ("i g n o r e"), or leetspeak ("1gn0re"). We therefore match the
# patterns against SEVERAL normalized views of the input, not just the
# original. An attack matches if ANY view matches.

# Zero-width / bidi control chars an attacker can sprinkle into a phrase.
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E],
    None,
)

# Common Latin-lookalike codepoints (Cyrillic/Greek) → ASCII. NFKC
# already folds fullwidth/mathematical variants; this covers the
# confusables NFKC leaves alone.
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d", "ո": "n", "т": "t", "к": "k",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "К": "K",
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "ι": "i", "κ": "k", "ѡ": "w",
})

# Leetspeak fold — applied as an ADDITIONAL view only, so it can widen
# detection without rewriting the user's real text.
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _normalized_views(text: str) -> list[str]:
    """Return the raw text plus normalized variants to defeat common
    evasion tricks. Order doesn't matter — the detector ORs across all."""
    base = unicodedata.normalize("NFKC", text)
    base = base.translate(_INVISIBLE)
    folded = base.translate(_HOMOGLYPHS)
    collapsed = re.sub(r"\s+", " ", folded)   # defeats whitespace splitting
    leet = collapsed.translate(_LEET)          # defeats leetspeak
    return [text, base, folded, collapsed, leet]


def detect_prompt_attack(text: str) -> tuple[bool, str | None]:
    """Return (is_attack, category). `category` is a machine-readable
    slug for logging/telemetry, never shown to the user.

    Matches every pattern against several normalized views of the input
    (homoglyph-folded, zero-width-stripped, whitespace-collapsed,
    leet-folded) so the most common evasion tricks don't slip past."""
    if not text or not text.strip():
        return False, None
    views = _normalized_views(text)
    for category, pattern in _PATTERNS:
        for view in views:
            if pattern.search(view):
                return True, category
    return False, None


def refusal_reply(locale: str | None = "en") -> str:
    """Deterministic refusal message, localized."""
    return _REFUSAL_AR if (locale or "en").lower().startswith("ar") else _REFUSAL_EN
