"""World-class chat context: state card, topic-shift, trimmed history.

Sessions store the full archive for the UI. This module decides what
actually enters the model prompt so old threads don't pollute answers.

Evidence (docs / DB / web) remains the source of truth — history is only
discourse state (who we're talking about, follow-ups).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app import config

_TOPIC_SHIFT_RE = re.compile(
    r"(?i)\b("
    r"new\s+topic|different\s+(topic|subject|question)|"
    r"forget\s+(the\s+)?(previous|prior|last|earlier)|"
    r"start\s+over|starting\s+fresh|unrelated|"
    r"switch\s+to|instead\s+(tell|ask|show)|"
    r"now\s+(tell|ask|show|about)|changing\s+(the\s+)?subject|"
    r"another\s+(company|country|question)|never\s+mind\s+that"
    r")\b"
)

_ABOUT_ENTITY_RE = re.compile(
    r"(?i)\b(?:about|on|regarding|for|of)\s+"
    r"([A-Z][\w&.,\-']+(?:\s+[A-Z][\w&.,\-']+){0,4})"
)

_TITLE_CASE_RE = re.compile(
    r"\b([A-Z][\w&.]+(?:\s+[A-Z][\w&.]+){0,3})\b"
)

_STOP_ENTITIES = frozenset({
    "I", "I'm", "The", "A", "An", "What", "Who", "Where", "When", "How",
    "Tell", "Show", "Please", "Thanks", "Hello", "Hi", "MISA", "CEO",
    "RHQ", "PDF", "MoU", "Saudi", "Arabia",
})

_PRONOUNS = frozenset({
    "them", "they", "it", "this", "that", "these", "those",
    "him", "her", "he", "she", "we", "us", "you",
})


@dataclass
class StateCard:
    """Compact discourse state persisted on the session."""

    active_entity: str | None = None
    entity_type: str | None = None  # company | country | person | topic
    last_intent: str | None = None
    last_answer_source: str | None = None
    topic_shift: bool = False
    summary: str = ""
    turn_count: int = 0
    recent_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "StateCard":
        if not data or not isinstance(data, dict):
            return cls()
        rq = data.get("recent_questions") or []
        if not isinstance(rq, list):
            rq = []
        return cls(
            active_entity=(data.get("active_entity") or None),
            entity_type=(data.get("entity_type") or None),
            last_intent=(data.get("last_intent") or None),
            last_answer_source=(data.get("last_answer_source") or None),
            topic_shift=bool(data.get("topic_shift")),
            summary=str(data.get("summary") or ""),
            turn_count=int(data.get("turn_count") or 0),
            recent_questions=[str(x)[:120] for x in rq][-8:],
        )

    def prompt_block(self) -> str:
        """Tiny block prepended for the model — not evidence."""
        if not self.active_entity and not self.summary:
            return ""
        lines = ["[Conversation state — discourse only, not evidence]"]
        if self.active_entity:
            typ = f" ({self.entity_type})" if self.entity_type else ""
            lines.append(f"- Active entity: {self.active_entity}{typ}")
        if self.summary:
            lines.append(f"- Summary: {self.summary}")
        if self.last_answer_source:
            lines.append(f"- Last answer source: {self.last_answer_source}")
        lines.append(
            "- Prefer fresh documents / database / web over prior assistant text."
        )
        return "\n".join(lines)


def extract_entity_candidate(question: str) -> str | None:
    q = (question or "").strip()
    if not q:
        return None
    m = _ABOUT_ENTITY_RE.search(q)
    if m:
        cand = m.group(1).strip().rstrip(".,?!:;")
        # IGNORECASE makes [A-Z] match lowercase pronouns — reject those.
        if cand and cand.lower() not in _PRONOUNS:
            first = cand.split()[0]
            if first not in _STOP_ENTITIES and first.lower() not in _PRONOUNS:
                return cand
    # Prefer multi-word Title Case (company names)
    for m in _TITLE_CASE_RE.finditer(q):
        cand = m.group(1).strip()
        first = cand.split()[0]
        if first in _STOP_ENTITIES or first.lower() in _PRONOUNS:
            continue
        if len(cand) >= 3:
            return cand
    return None


def is_explicit_topic_shift(question: str) -> bool:
    return bool(question and _TOPIC_SHIFT_RE.search(question))


def entities_differ(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    na = re.sub(r"\s+", " ", a.strip().lower())
    nb = re.sub(r"\s+", " ", b.strip().lower())
    if na == nb:
        return False
    # One contains the other → same family (Apple vs Apple Inc.)
    if na in nb or nb in na:
        return False
    return True


def detect_topic_shift(question: str, state: StateCard) -> bool:
    if is_explicit_topic_shift(question):
        return True
    cand = extract_entity_candidate(question)
    if cand and state.active_entity and entities_differ(cand, state.active_entity):
        # New named entity while one was active → topic shift
        return True
    return False


def _truncate_msg(content: str, limit: int) -> str:
    text = (content or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def prompt_history_user_turns() -> int:
    raw = getattr(config, "SESSIONS_PROMPT_HISTORY_TURNS", None)
    if raw is None:
        return min(8, config.max_history_user_turns())
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 8


def prompt_assistant_char_cap() -> int:
    return max(80, int(getattr(config, "SESSIONS_PROMPT_ASSISTANT_CHARS", 400)))


def build_rolling_summary(state: StateCard, question: str) -> str:
    """Cheap deterministic summary — no LLM required."""
    parts: list[str] = []
    if state.active_entity:
        typ = f" ({state.entity_type})" if state.entity_type else ""
        parts.append(f"Focus: {state.active_entity}{typ}.")
    recent = list(state.recent_questions or [])[-3:]
    if question and (not recent or recent[-1] != question[:120]):
        recent = (recent + [question[:120]])[-3:]
    if recent:
        parts.append("Recent asks: " + " | ".join(recent))
    if state.last_answer_source:
        parts.append(f"Last source: {state.last_answer_source}.")
    summary = " ".join(parts).strip()
    return summary[:400]


def prepare_prompt_history(
    question: str,
    history: list[dict] | None,
    state: StateCard | None = None,
) -> tuple[list[dict], StateCard, dict[str, Any]]:
    """Return (history_for_model, updated_state, meta).

    On topic shift, prior turns are dropped from the prompt (archive
    remains in the session DB). Otherwise keep a short trimmed window
    plus an updated state card.
    """
    state = StateCard.from_dict(state.to_dict() if state else None)
    history = list(history or [])
    meta: dict[str, Any] = {
        "topic_shift": False,
        "prompt_turns_kept": 0,
        "dropped_prior": False,
    }

    shifted = detect_topic_shift(question, state)
    if shifted:
        meta["topic_shift"] = True
        meta["dropped_prior"] = True
        # Keep state entity only if the new question names one; else clear.
        new_ent = extract_entity_candidate(question)
        state = StateCard(
            active_entity=new_ent,
            entity_type=None,
            last_intent=None,
            last_answer_source=None,
            topic_shift=True,
            summary="",
            turn_count=state.turn_count,
            recent_questions=[],
        )
        effective: list[dict] = []
    else:
        # Pair-preserving trim: keep last N user turns + their following assistant
        max_users = prompt_history_user_turns()
        asst_cap = prompt_assistant_char_cap()
        # Walk from end, collect up to max_users user messages (+ assistants after them)
        kept_rev: list[dict] = []
        users = 0
        for msg in reversed(history):
            role = (msg.get("role") or "").strip()
            content = msg.get("content") or ""
            if not content:
                continue
            if role == "user":
                if max_users and users >= max_users:
                    break
                users += 1
                kept_rev.append({"role": "user", "content": _truncate_msg(content, 800)})
            elif role == "assistant":
                if users == 0 and not kept_rev:
                    # trailing assistant without user yet — skip
                    continue
                kept_rev.append({
                    "role": "assistant",
                    "content": _truncate_msg(content, asst_cap),
                })
            else:
                continue
        effective = list(reversed(kept_rev))
        meta["prompt_turns_kept"] = users
        # Soft-update entity from question if present
        cand = extract_entity_candidate(question)
        if cand:
            state.active_entity = cand
        state.topic_shift = False

    state.summary = build_rolling_summary(state, question)
    # Inject state as a synthetic system-facing user preface? Better: return
    # in meta and let chat_engine prepend. We attach as a leading assistant
    # note filtered out of "user_msgs only" paths — instead stash on state
    # and have prepare return a history that starts with a compact user hint
    # only when we have an active entity (follow-up case).
    if state.active_entity and not shifted and effective:
        # Prepend a single compact orientation message (as user) for resolvers
        # that only read user turns — keep it short.
        orient = (
            f"(Continuing about {state.active_entity}"
            + (f", {state.entity_type}" if state.entity_type else "")
            + ". Prior answers are not evidence.)"
        )
        if not any(
            (m.get("content") or "").startswith("(Continuing about")
            for m in effective
        ):
            effective = [{"role": "user", "content": orient}] + effective

    return effective, state, meta


def update_state_after_turn(
    state: StateCard,
    question: str,
    *,
    answer_source: str | None = None,
    intent: str | None = None,
    entity: str | None = None,
    entity_type: str | None = None,
) -> StateCard:
    state = StateCard.from_dict(state.to_dict())
    state.turn_count = int(state.turn_count or 0) + 1
    q_short = (question or "").strip()[:120]
    rq = list(state.recent_questions or [])
    if q_short and (not rq or rq[-1] != q_short):
        rq.append(q_short)
    state.recent_questions = rq[-8:]
    if answer_source:
        state.last_answer_source = answer_source
    if intent:
        state.last_intent = intent
    ent = entity or extract_entity_candidate(question) or state.active_entity
    if ent:
        state.active_entity = ent
    if entity_type:
        state.entity_type = entity_type
    state.topic_shift = False
    state.summary = build_rolling_summary(state, question)
    return state
