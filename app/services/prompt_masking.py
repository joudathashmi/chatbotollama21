"""Fast deterministic prompt masking — secrets/PII only.

Design goals (must NOT hurt answer quality or latency):
  - Mask only high-confidence secret / contact patterns (email, phone,
    API keys, bearer tokens, password assignments). Never mask company
    names, license counts, sectors, or other MISA business facts.
  - Pure regex, compiled once — no LLM, no network, O(n) in string length.
  - Applied at LLM *egress* and *log* boundaries only. Final user-facing
    answers are not rewritten by this module (answers already go through
    quality gates; stripping emails from a contact brief would hurt UX —
    contact fields are filtered by key via curation._safe_row instead).

Enable/disable: MISA_PROMPT_MASKING=1 (default on) / 0 to disable.
"""

from __future__ import annotations

import os
import re
from typing import Any


def masking_enabled() -> bool:
    raw = (os.getenv("MISA_PROMPT_MASKING") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


# High-precision patterns only — avoid false positives on company names /
# Vision 2030 / entity IDs that are legitimately in answers.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
)
_PHONE_RE = re.compile(
    # Labeled contact numbers OR international (+…) form only —
    # never bare digit runs (those are often revenues / IDs / years).
    r"(?i)(?:(?:tel|phone|mobile|cell|fax|whatsapp)\s*[:.#]?\s*)"
    r"(\+?[\d\s\-().]{8,22})"
    r"|"
    r"(\+\d{1,3}[\s\-.(]?\d{1,4}[\s\-.)]?\d{3,4}[\s\-.]?\d{3,4})",
)
_PHONE_DIGIT_MIN = 8

_API_KEY_RE = re.compile(
    r"\b(sk-[A-Za-z0-9_\-]{16,}|AIza[0-9A-Za-z_\-]{20,}|"
    r"ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9\-]{10,})\b",
)
_BEARER_RE = re.compile(
    r"(?i)\b(bearer\s+)[A-Za-z0-9\-._~+/]+=*",
)
_PASSWORD_ASSIGN_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|private[_-]?key)\s*[=:]\s*([^\s,;\"']{4,})",
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b",
)
_SYSTEM_PROMPT_LEAK_RE = re.compile(
    r"(?i)(You are a senior investment-promotion strategist|"
    r"SOURCE HIERARCHY \(mandatory|"
    r"ROLE_NOTE|"
    r"Retrieved records from .{0,40}\(privacy-filtered JSON\))",
)


def _mask_phone_match(m: re.Match) -> str:
    raw = m.group(0)
    digits = re.sub(r"\D", "", raw)
    if len(digits) < _PHONE_DIGIT_MIN:
        return raw
    return f"[PHONE_MASKED…{digits[-2:]}]"


def mask_text(text: str, *, for_log: bool = False) -> str:
    """Mask secrets/PII in a string. Idempotent and fast."""
    if not text or not masking_enabled():
        return text
    s = text
    s = _EMAIL_RE.sub("[EMAIL_MASKED]", s)
    s = _API_KEY_RE.sub("[API_KEY_MASKED]", s)
    s = _JWT_RE.sub("[JWT_MASKED]", s)
    s = _BEARER_RE.sub(r"\1[TOKEN_MASKED]", s)
    s = _PASSWORD_ASSIGN_RE.sub(
        lambda m: f"{m.group(1)}=[REDACTED]", s,
    )
    s = _PHONE_RE.sub(_mask_phone_match, s)
    if for_log:
        # Extra: collapse obvious system-prompt dumps in log lines
        s = _SYSTEM_PROMPT_LEAK_RE.sub("[SYSTEM_PROMPT_MASKED]", s)
    return s


def mask_obj(obj: Any, *, for_log: bool = False, _depth: int = 0) -> Any:
    """Recursively mask strings in dict/list structures (egress / logs)."""
    if not masking_enabled() or _depth > 12:
        return obj
    if isinstance(obj, str):
        return mask_text(obj, for_log=for_log)
    if isinstance(obj, dict):
        return {
            k: mask_obj(v, for_log=for_log, _depth=_depth + 1)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [mask_obj(x, for_log=for_log, _depth=_depth + 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(mask_obj(x, for_log=for_log, _depth=_depth + 1) for x in obj)
    return obj


def mask_messages_for_llm(messages: list[dict]) -> list[dict]:
    """Copy chat messages with user/tool content masked for egress.

    System messages are left intact (they are our prompts, not user PII).
    Does not mutate the input list.
    """
    if not masking_enabled() or not messages:
        return messages
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = (m.get("role") or "").lower()
        if role == "system":
            out.append(m)
            continue
        nm = dict(m)
        if isinstance(nm.get("content"), str):
            nm["content"] = mask_text(nm["content"])
        out.append(nm)
    return out


def scrub_system_prompt_leak(answer: str) -> str:
    """If the model echoed system-prompt boilerplate, strip it from the
    user-visible answer. Conservative — only known leak markers."""
    if not answer or not masking_enabled():
        return answer
    if not _SYSTEM_PROMPT_LEAK_RE.search(answer):
        return answer
    # Remove lines that look like prompt dumps
    lines = []
    for ln in answer.splitlines():
        if _SYSTEM_PROMPT_LEAK_RE.search(ln):
            continue
        lines.append(ln)
    return "\n".join(lines)
