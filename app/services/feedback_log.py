"""
Feedback log.

The continuous-quality-loop foundation. Every turn the user thumbs-up
or thumbs-down (with optional comment) is appended to a JSON-lines file
on disk. A weekly review script extracts the low-confidence + thumbs-
down turns, promotes the worst cases into golden tests, and surfaces
trends.

Design choices:
  - JSONL on disk, not a real DB. Simple to inspect with `tail`, easy
    to back up, no schema migration. The volume is small (one record
    per thumbs click, not per turn) so disk is fine.
  - Append-only. We never overwrite or delete; the analyst's task is
    to read & curate, not to "fix" the log.
  - Schema is deliberately wide — store the full debug payload
    alongside the feedback so a reviewer doesn't need to replay the
    turn to understand why the answer came out the way it did.
  - Path configurable via MISA_FEEDBACK_LOG env, default
    ./feedback.jsonl in CWD. Gitignored.

Records look like:
  {
    "ts": "2026-06-08T12:34:56Z",
    "verdict": "down",           # "up" | "down"
    "comment": "missed the CFO",  # optional, free text from user
    "question": "Who is the CEO of Apple?",
    "answer_head": "## CEO\n- ...",    # first 800 chars of answer
    "answer_chars": 1240,
    "intent": "executive_lookup",
    "intent_confidence": 1.0,
    "entity_resolved": "Apple",
    "tables_searched": ["company_executives", "company_profiles"],
    "row_count_total": 2,
    "ui_locale": "en"
  }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from threading import Lock


_LOG_LOCK = Lock()


def _log_path() -> str:
    return (os.getenv("MISA_FEEDBACK_LOG") or "feedback.jsonl").strip()


def append_feedback(record: dict) -> dict:
    """Append a feedback record to the JSONL log. Auto-stamps timestamp
    and clamps long fields. Returns the persisted record (with the
    timestamp populated) so the API can echo it back.

    Failure-mode: if the file can't be written, log to stderr and
    return the record with an 'error' key — the API should NOT 500
    over a feedback write failure.
    """
    rec = dict(record or {})
    rec.setdefault("ts", datetime.now(timezone.utc).isoformat())
    # Clamp text fields so a runaway answer can't blow up the log.
    if isinstance(rec.get("answer_head"), str):
        rec["answer_head"] = rec["answer_head"][:800]
    if isinstance(rec.get("comment"), str):
        rec["comment"] = rec["comment"][:2000]
    if isinstance(rec.get("question"), str):
        rec["question"] = rec["question"][:1500]
    path = _log_path()
    try:
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        rec["_persist_error"] = f"{type(e).__name__}: {e}"
        import sys
        print(f"[feedback_log] failed to write: {e}", file=sys.stderr)
    return rec


def read_all() -> list[dict]:
    """Read every feedback record from disk. Used by the review script."""
    path = _log_path()
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return out
