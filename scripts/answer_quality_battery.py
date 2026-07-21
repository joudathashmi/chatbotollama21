#!/usr/bin/env python3
"""
LLM-judged answer QUALITY battery for the MISA chat pipeline.

The regression battery (chat_regression_battery.py) checks answers for
known failure CLASSES via string assertions. This battery checks the
thing string assertions can't: is the answer DEEP, INSIGHTFUL, and
MISA-RELEVANT — or generic filler?

For every question in the bank it:
  1. Calls the live /api/v1/chat endpoint.
  2. Sends (question, answer, per-case quality criteria) to an OpenAI
     judge model with a strict rubric.
  3. Collects a structured verdict: score 1-10 + named defects.

A case PASSES at score >= 7 with no critical defects. Exit code is
non-zero if any case fails — wire into CI next to the regression
battery. Run:

    python3 scripts/answer_quality_battery.py            # full bank
    python3 scripts/answer_quality_battery.py entity     # one category

Set MISA_BATTERY_URL / MISA_BATTERY_USER / MISA_BATTERY_PASS to
override the target, and MISA_JUDGE_MODEL to change the judge
(default gpt-4o). Requires OPENAI_API_KEY in the environment / .env.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from urllib.request import Request, urlopen

# Make `app` importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

URL = os.getenv("MISA_BATTERY_URL", "http://127.0.0.1:8000/api/v1/chat")
USER = os.getenv("MISA_BATTERY_USER", "admin")
PASS = os.getenv("MISA_BATTERY_PASS", "test")
AUTH = "Basic " + base64.b64encode(f"{USER}:{PASS}".encode()).decode()
JUDGE_MODEL = os.getenv("MISA_JUDGE_MODEL", "gpt-4o")
PASS_SCORE = int(os.getenv("MISA_QUALITY_PASS_SCORE", "7"))


# --- question bank ---------------------------------------------------------
# (category, question, case-specific criteria the judge must check)
# Every entry represents a SCENARIO CLASS, not just one question — when a
# user reports a new generic-answer example, add its class here.

BANK: list[tuple[str, str, str]] = [
    ("entity:fund",
     "What is the China National IC fund (Big Fund)",
     "Must explain what the fund is AND spell out the investment-"
     "attraction implications for MISA: which companies/sub-sectors the "
     "fund capitalises and which of those MISA should court into Saudi "
     "Arabia. A 2-3 sentence definition with no implications section is "
     "an automatic fail."),

    ("entity:company",
     "Tell me about Alphabet and its MENA presence",
     "Must state concrete record facts (sector, HQ, scale, RHQ/Saudi "
     "presence fields) and close with a Strategic Read for MISA that "
     "names specific engagement angles — not 'explore opportunities'."),

    ("advisory:market-fit",
     "what is the market fit for attracting Indian companies to Saudi Arabia",
     "Must be a full tiered market-fit assessment (sector table with "
     "priorities, deep-dives, MISA targeting recommendations) and cite "
     "MISA database figures for the existing Indian footprint."),

    ("advisory:engagement-plan",
     "Develop an engagement plan for attracting investment from France "
     "to Saudi Arabia",
     "Must be an OPERATIONAL PLAN: measurable objectives, phased "
     "roadmap with month ranges, stakeholder/channel map naming real "
     "French bodies (e.g. MEDEF, Business France, CCI France), KPIs, "
     "risks. Every action bullet needs a named anchor. A market "
     "assessment instead of a plan is a fail."),

    ("advisory:sector-priorities",
     "what are the top sectors that I should be focusing on for "
     "attracting investors from Germany",
     "Must rank sectors with EVIDENCE: lead with what MISA's own data "
     "shows converts (sector distribution of German licensees) when "
     "available, name anchor companies, include a de-prioritisation "
     "('what NOT to focus on'), and anchored next moves."),

    ("country:profile",
     "tell me about Pakistan",
     "Must ground in the MISA country record (indicators, trade "
     "partners, reforms where present), list company-level records "
     "honestly, and close with an Engagement Read naming specific "
     "sectors/programmes — no generic 'strong potential' prose."),

    ("comparison",
     "compare Apple and Microsoft in terms of Saudi presence",
     "Must compare the two on RECORD fields side-by-side (RHQ status, "
     "presence type, headcount where present) and state which record "
     "fields are missing rather than papering over gaps."),

    ("data:fdi",
     "What is the size of outflow FDI from South Korea? How much is it "
     "inflow to Saudi Arabia?",
     "GROUND TRUTH: the MISA database has NO Korea-specific FDI series "
     "— fdi_data holds only Saudi Arabia's aggregate FDI totals. Judge "
     "attribution by the WORDING: a figure presented explicitly as "
     "Saudi Arabia's total ('Saudi Arabia's FDI inflow in 2024 was SAR "
     "119.2B') is CORRECT and must not be flagged; misattribution "
     "(automatic fail) is ONLY wording that claims a figure as "
     "Korea's ('inflow from South Korea was SAR X'). The Korea "
     "sub-questions should be answered with clearly-labelled general "
     "knowledge (order of magnitude suffices) or an honest gap "
     "statement. Formatting: currency code used consistently (never "
     "'$X SAR'), years stated. CRITICAL: no internal plumbing "
     "messages ('I couldn't directly look up X in `table`', 'needs a "
     "numeric ID', 'try asking...'). The italic transparency footers "
     "are intended UX, not plumbing."),
]


# --- plumbing ---------------------------------------------------------------

def call_chat(q: str, timeout: int = 120) -> dict:
    body = json.dumps({
        "question": q, "history": [], "locale": "en", "stream": False,
    }).encode()
    req = Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": AUTH,
    })
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


_JUDGE_SYSTEM = """You are a demanding quality reviewer for a Ministry of
Investment (MISA, Saudi Arabia) intelligence chatbot. You receive a user
question, the chatbot's answer, and case-specific criteria. Grade the
ANSWER against this rubric:

1. DIRECT: does it actually answer the question asked (right deliverable
   type, right subject)?
2. DEPTH: is it substantive for the question's ambition? Strategy /
   profile / definitional questions need structured, multi-section
   answers; only narrow factual questions may be short.
3. MISA RELEVANCE: does it end with concrete investment-attraction
   implications or actions for MISA where applicable?
4. SPECIFICITY: are claims anchored to named programmes, agencies,
   events, companies, or figures? Penalise filler ("strengthen bilateral
   relations", "explore opportunities", "leverage synergies", slogans).
5. HONESTY: no invented precise statistics; missing data acknowledged;
   general-knowledge content labelled when the DB had nothing.
   IMPORTANT CALIBRATION: figures AND status facts (RHQ status,
   presence type, headcounts, GDP, FDI) attributed to the MISA
   database / record ARE the system of record — do NOT mark them
   incorrect merely because they differ from your training knowledge
   (vintages differ, and the DB sees non-public licensing data). An
   answer carrying the line "All figures per the MISA record unless
   labelled general knowledge" (or attributing figures inline) counts
   as attributed. Only flag: (a) figures presented as live current
   statistics with NO record attribution anywhere, and (b) genuine
   INTERNAL contradictions — e.g. a line saying 1 company is licensed
   followed by a line saying none are recorded. NOT a contradiction:
   "1 licensed, 0 of those hold an RHQ licence" (that is a breakdown).
   NUMBER FORMAT CALIBRATION: 'SAR 110.1B', '$4.2B', 'SAR 3.4M' is the
   house style — do NOT flag it as a formatting defect. Only flag a
   symbol mixed with a mismatched code ('$110.1B SAR').
   INTENDED UX (do not flag as plumbing/defects): the italic lines
   '_Internal records do not currently show: <topic>._',
   '_All figures per the MISA record unless labelled general
   knowledge._', '_Sources: <tables>._', and the labelled
   '*General knowledge — not sourced from the MISA database.*'
   passages are deliberate transparency features. 'Plumbing' means
   error text about tables/IDs/lookups ('I couldn't directly look up
   X in `table`', 'needs a numeric ID', 'try asking...').
6. CORRECTNESS RED FLAGS: mentions of SAGIA (defunct since 2020),
   advising MISA to "coordinate with the investment authority" (MISA IS
   the authority), broken markdown/numbering.

Return STRICT JSON only:
{"score": <1-10>, "pass": <bool>, "critical_defects": ["..."],
 "minor_defects": ["..."], "one_line_summary": "..."}
score >= 7 AND no critical defects => pass true. Be strict: a generic
answer that would fit any country/company pair scores <= 4."""


def judge(question: str, answer: str, criteria: str) -> dict:
    from openai import OpenAI
    from app.config import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)
    user = (
        f"USER QUESTION:\n{question}\n\n"
        f"CASE-SPECIFIC CRITERIA:\n{criteria}\n\n"
        f"CHATBOT ANSWER:\n{answer}"
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": _JUDGE_SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def main() -> int:
    only = sys.argv[1].lower() if len(sys.argv) > 1 else None
    cases = [c for c in BANK if not only or c[0].startswith(only)]
    print(f"MISA answer QUALITY battery — {len(cases)} cases "
          f"(judge: {JUDGE_MODEL}, pass >= {PASS_SCORE})")
    print(f"target: {URL}\n")

    results, n_fail = [], 0
    for category, q, criteria in cases:
        t0 = time.time()
        try:
            answer = (call_chat(q).get("answer") or "").strip()
            if not answer:
                raise RuntimeError("empty answer")
            verdict = judge(q, answer, criteria)
        except Exception as e:
            verdict = {"score": 0, "pass": False,
                       "critical_defects": [f"battery error: {e}"],
                       "minor_defects": [], "one_line_summary": "error"}
            answer = ""
        ok = bool(verdict.get("pass")) and int(verdict.get("score", 0)) >= PASS_SCORE
        n_fail += (not ok)
        dt = time.time() - t0
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {category:28s} score={verdict.get('score')}/10 "
              f"({dt:.0f}s) — {verdict.get('one_line_summary','')[:90]}")
        for d in (verdict.get("critical_defects") or []):
            print(f"         CRITICAL: {d}")
        for d in (verdict.get("minor_defects") or [])[:3]:
            print(f"         minor:    {d}")
        results.append({"category": category, "question": q, "ok": ok,
                        "verdict": verdict,
                        "answer_words": len(answer.split()),
                        "answer": answer})

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "answer_quality_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{len(cases) - n_fail}/{len(cases)} passed — report: {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
