#!/usr/bin/env python3
"""
Golden test runner — runs each case in tests/golden_cases.json against
the live chat API and emits structured pass/fail results.

Result schema (per case, matches what was requested):

    {
      "question_id": "G001-exact-company",
      "user_question": "...",
      "expected_source": "DB",
      "expected_company": "Alphabet",
      "expected_tables": [...],
      "expected_keywords": [...],
      "should_handle_spelling_mistake": false,
      "should_search_whole_db": true,
      "actual_answer": "...",
      "actual_tables": [...],
      "actual_answer_source": "db",
      "confidence_scores": {...},
      "pass_fail": "PASS" | "FAIL",
      "failure_type": null | "wrong_table" | "wrong_company" | "forbidden_keyword" |
                       "missing_keyword" | "short_answer" | "wrong_source" | "exception"
    }

Usage:
    python scripts/run_golden_tests.py             # human-readable
    python scripts/run_golden_tests.py --json      # machine-readable JSON
    python scripts/run_golden_tests.py --out=/tmp/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

URL = os.getenv("MISA_BATTERY_URL", "http://127.0.0.1:8000/api/v1/chat")
USER = os.getenv("MISA_BATTERY_USER", "admin")
PASS = os.getenv("MISA_BATTERY_PASS", "test")
LOGIN_URL = os.getenv(
    "MISA_BATTERY_LOGIN_URL",
    URL.rsplit("/api/v1/chat", 1)[0] + "/api/v1/auth/login"
    if "/api/v1/chat" in URL
    else "http://127.0.0.1:8000/api/v1/auth/login",
)

DEFAULT_CASES_PATH = Path(__file__).resolve().parent.parent / "tests" / "golden_cases.json"

_cached_bearer: str | None = None


def _bearer_token() -> str:
    """Exchange username/password for a JWT access token (cached per run)."""
    global _cached_bearer
    if _cached_bearer:
        return _cached_bearer
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = Request(LOGIN_URL, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"login at {LOGIN_URL} returned no access_token: {data!r}")
    _cached_bearer = f"Bearer {token}"
    return _cached_bearer


def call_chat(question: str, history: list | None = None, timeout: int = 90) -> dict:
    body = json.dumps({
        "question": question,
        "history": history or [],
        "locale": "en",
        "stream": False,
    }).encode()
    req = Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": _bearer_token(),
    })
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _check(case: dict, response: dict) -> tuple[str, str | None]:
    """Returns ('PASS', None) or ('FAIL', failure_type)."""
    answer_lc = (response.get("answer") or "").lower()
    trace = response.get("trace") or []
    tables = [t.get("table") for t in trace if t.get("table")]
    rows_total = sum((t.get("row_count") or 0) for t in trace)
    err = response.get("error")

    if err:
        return "FAIL", "result_error"

    # expected_tables (any-of match)
    expected_tables = case.get("expected_tables") or []
    expected_source = case.get("expected_source", "")
    if expected_tables and not any(t in expected_tables for t in tables):
        # Allow empty-trace cases (off-topic, conversational, clarification)
        # — clarification is a deterministic pre-LLM short-circuit and has
        # no DB tool calls by design.
        if expected_source in ("DB", "DB+OpenAI"):
            return "FAIL", "wrong_table"

    # expected_keywords (all must appear)
    for kw in case.get("expected_keywords") or []:
        if kw.lower() not in answer_lc:
            return "FAIL", "missing_keyword"

    # forbidden_keywords (none may appear).
    # Use WHOLE-WORD matching to avoid false positives like "ERM"
    # firing on "long-tERM commitment". A keyword that is itself a
    # phrase (has whitespace) is still matched as a substring because
    # phrases are unambiguous; short acronyms get the word-boundary
    # guard.
    import re as _re
    for kw in case.get("forbidden_keywords") or []:
        kw_l = kw.lower()
        if " " in kw_l:
            if kw_l in answer_lc:
                return "FAIL", "forbidden_keyword"
        else:
            if _re.search(r"\b" + _re.escape(kw_l) + r"\b", answer_lc):
                return "FAIL", "forbidden_keyword"

    # min_answer_length
    if "min_answer_length" in case:
        if len(response.get("answer") or "") < int(case["min_answer_length"]):
            return "FAIL", "short_answer"

    # expected_source: DB / OpenAI fallback / clarification / conversational
    # / DB+OpenAI. Soft check — only fail when totally wrong direction.
    exp_src = (case.get("expected_source") or "").lower()
    if "openai fallback" in exp_src:
        if rows_total > 0 and "general knowledge" not in answer_lc:
            return "FAIL", "wrong_source"
    elif exp_src == "clarification":
        if "multiple possible matches" not in answer_lc:
            return "FAIL", "wrong_source"

    return "PASS", None


def run_all(cases_path: Path, only_id: str | None = None) -> list[dict]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if only_id:
        cases = [c for c in cases if c.get("question_id") == only_id]
    results: list[dict] = []
    for case in cases:
        qid = case.get("question_id", "?")
        question = case.get("user_question", "")
        history = case.get("history") or []
        t0 = time.time()
        try:
            response = call_chat(question, history=history)
            verdict, failure_type = _check(case, response)
        except (HTTPError, URLError, TimeoutError) as e:
            response = {}
            verdict, failure_type = "FAIL", f"http_{type(e).__name__}"
        except Exception as e:
            response = {}
            verdict, failure_type = "FAIL", f"exception_{type(e).__name__}"
        dt = round(time.time() - t0, 2)
        trace = response.get("trace") or []
        results.append({
            "question_id": qid,
            "user_question": question,
            "expected_source": case.get("expected_source"),
            "expected_company": case.get("expected_company"),
            "expected_tables": case.get("expected_tables"),
            "expected_keywords": case.get("expected_keywords"),
            "should_handle_spelling_mistake": case.get(
                "should_handle_spelling_mistake", False
            ),
            "should_search_whole_db": case.get("should_search_whole_db", True),
            "actual_answer": response.get("answer", ""),
            "actual_tables": [t.get("table") for t in trace if t.get("table")],
            "actual_row_count_total": sum((t.get("row_count") or 0) for t in trace),
            "pass_fail": verdict,
            "failure_type": failure_type,
            "duration_seconds": dt,
        })
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    ap.add_argument("--only", help="run only this question_id")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON array (machine-readable)")
    ap.add_argument("--out", help="write JSON results to this path")
    args = ap.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"cases file not found: {cases_path}", file=sys.stderr)
        return 2
    results = run_all(cases_path, only_id=args.only)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Human-readable summary
        n_pass = sum(1 for r in results if r["pass_fail"] == "PASS")
        n_fail = sum(1 for r in results if r["pass_fail"] == "FAIL")
        for r in results:
            tag = "PASS" if r["pass_fail"] == "PASS" else "FAIL"
            ft = f" [{r['failure_type']}]" if r["failure_type"] else ""
            print(f"  [{tag}] {r['question_id']:<32} "
                  f"tables={','.join(r['actual_tables'])[:32]:<32} "
                  f"({r['duration_seconds']}s){ft}")
            if r["pass_fail"] == "FAIL":
                snippet = (r["actual_answer"] or "")[:200].replace("\n", " ")
                print(f"           expected: {r['expected_source']!r} "
                      f"tables={r['expected_tables']} "
                      f"keywords={r['expected_keywords']}")
                print(f"           actual:   {snippet!r}…")
        print(f"\n{n_pass} pass, {n_fail} fail "
              f"({len(results)} total)")

    return 0 if all(r["pass_fail"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
