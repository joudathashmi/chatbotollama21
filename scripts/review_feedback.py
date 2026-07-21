#!/usr/bin/env python3
"""
Weekly quality-loop review tool.

Reads feedback.jsonl (the user thumbs up/down log) and prints:
  - Volume summary (counts of up/down, % thumbs-down)
  - All thumbs-DOWN turns with full context (intent, entity, tables,
    comment if any)
  - Low-confidence thumbs-UP turns (intent_confidence < 0.7) — these
    are at-risk cases worth promoting into golden tests even when the
    user accepted the answer
  - A starter golden-test JSON block for each thumbs-down turn,
    pre-filled, ready to paste into tests/golden_cases.json

Usage:
  ./venv/bin/python scripts/review_feedback.py             # read default ./feedback.jsonl
  MISA_FEEDBACK_LOG=/path/to/log ./venv/bin/python scripts/review_feedback.py

Designed to be run weekly. Output is plain text for easy capture into
a Notion / Confluence doc as the curation pass record.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.services.feedback_log import read_all


def fmt_record(r: dict) -> str:
    out = []
    out.append(f"  ts:        {r.get('ts')}")
    out.append(f"  verdict:   {r.get('verdict')}")
    out.append(f"  question:  {r.get('question')}")
    if r.get("comment"):
        out.append(f"  comment:   {r['comment']}")
    out.append(f"  intent:    {r.get('intent')}  "
               f"(conf={r.get('intent_confidence')})")
    out.append(f"  entity:    {r.get('entity_resolved')}")
    out.append(f"  tables:    {r.get('tables_searched')}")
    out.append(f"  rows:      {r.get('row_count_total')}")
    if r.get("answer_head"):
        head = r["answer_head"][:280].replace("\n", " ")
        out.append(f"  answer:    {head}…")
    return "\n".join(out)


def golden_template(r: dict, qid: str) -> str:
    """Pre-fill a golden-case JSON block from a thumbs-down record.
    The reviewer fills in expected_keywords / forbidden_keywords and
    pastes into tests/golden_cases.json."""
    tables = r.get("tables_searched") or []
    return json.dumps({
        "question_id": qid,
        "user_question": r.get("question") or "",
        "expected_behavior": (
            "Reviewer note from thumbs-down: "
            + (r.get("comment") or "(no comment provided)")
        ),
        "expected_source": "DB",
        "expected_company": r.get("entity_resolved"),
        "expected_tables": tables,
        "expected_keywords": ["TODO: list keywords that MUST appear"],
        "forbidden_keywords": ["TODO: list strings that MUST NOT appear"],
    }, indent=2)


def main() -> int:
    records = read_all()
    if not records:
        print("No feedback records found. Click thumbs up/down on a turn to start logging.")
        return 0

    verdicts = Counter(r.get("verdict") for r in records)
    total = sum(verdicts.values())
    pct_down = (verdicts.get("down", 0) / total * 100) if total else 0

    print("=" * 70)
    print("FEEDBACK SUMMARY")
    print("=" * 70)
    print(f"Total records: {total}")
    print(f"  up:    {verdicts.get('up', 0)}")
    print(f"  down:  {verdicts.get('down', 0)}  ({pct_down:.1f}%)")

    by_intent = Counter(r.get("intent") for r in records)
    print()
    print("By intent:")
    for intent, n in by_intent.most_common():
        print(f"  {intent or '(none)':30} {n}")

    # Thumbs-down deep dive
    downs = [r for r in records if r.get("verdict") == "down"]
    if downs:
        print()
        print("=" * 70)
        print(f"THUMBS-DOWN TURNS ({len(downs)})")
        print("=" * 70)
        for i, r in enumerate(downs, start=1):
            print()
            print(f"--- #{i} ---")
            print(fmt_record(r))
            print()
            print("  ↓ Starter golden-test case (paste into tests/golden_cases.json):")
            qid = f"G{100+i:03d}-feedback-{(r.get('entity_resolved') or 'review').lower().replace(' ','-')[:30]}"
            print()
            for line in golden_template(r, qid).splitlines():
                print("    " + line)

    # At-risk thumbs-up turns (low classifier confidence)
    risky_ups = [
        r for r in records
        if r.get("verdict") == "up"
        and isinstance(r.get("intent_confidence"), (int, float))
        and r["intent_confidence"] < 0.7
    ]
    if risky_ups:
        print()
        print("=" * 70)
        print(f"LOW-CONFIDENCE THUMBS-UP TURNS ({len(risky_ups)})")
        print("=" * 70)
        print("(User accepted, but classifier wasn't sure — worth a closer look.)")
        for r in risky_ups:
            print()
            print(fmt_record(r))

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print("Next steps:")
    print("  1. For each thumbs-down turn above, decide if it's a fixable")
    print("     bug or a tricky-but-acceptable edge case.")
    print("  2. Paste the starter golden-test JSON into tests/golden_cases.json,")
    print("     fill in expected_keywords / forbidden_keywords.")
    print("  3. Run: ./venv/bin/python scripts/run_golden_tests.py")
    print("  4. If the new case fails, that's the bug to fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
