"""
Append-only JSONL log for human-in-the-loop review / future training exports.

Rows and SQL parameters are not stored here — only high-level retrieval metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Three levels up from app/services/ reaches the project root.
ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATH = ROOT / "data" / "human_feedback.jsonl"


def feedback_path() -> Path:
    raw = (os.getenv("HUMAN_FEEDBACK_JSONL") or "").strip()
    return Path(raw) if raw else DEFAULT_PATH


def build_feedback_context(
    user_question: str,
    ui_locale: str,
    response_locale: str,
    pack: dict,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "user_question": (user_question or "")[:8000],
        "ui_locale": ui_locale,
        "response_locale": response_locale,
        "cleaned": pack.get("cleaned"),
        "entity_candidate": pack.get("entity_candidate"),
    }


def summarize_tool_calls_for_training(tool_calls: list | None) -> list[dict[str, Any]]:
    return [
        {
            "table": tc.get("table"),
            "row_count": tc.get("row_count"),
            "had_error": bool(tc.get("error")),
            "row_entity_sanity_passed": tc.get("row_entity_sanity_passed"),
        }
        for tc in (tool_calls or [])
    ]


def append_training_record(record: dict[str, Any]) -> Path:
    path = feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
