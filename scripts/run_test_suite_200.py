"""
Run the 200-question MISA test suite from
~/Downloads/MISA_Executive_Chatbot_Test_Suite_200_Questions.xlsx
against the live /api/v1/chat endpoint.

The spreadsheet has 155 unique question stems but many are template
placeholders ("global CEO Case 1", "country #1", "sector #1", etc.).
This runner substitutes those placeholders with REAL values from the
MISA DB so each question becomes a runnable, meaningful test:

  - 5 real entity scenarios run as-is (Apple/Microsoft/NVIDIA/Google/Amazon)
  - "global CEO Case N" → real exec name from company_executives
  - "country #N"        → real country name from countries table
  - "sector #N"         → real sector name from sectors table
  - "opportunity #N"    → real opportunity title from opportunities table
  - "Follow-up #N"      → multi-turn with synthesised prior turn
  - "Misspelled #N"     → typo variant of a real known company
  - "Info unavailable #N" → meta-question about behaviour (run as-is)

For repeated scenarios (e.g. Apple ×10), runs the FIRST instance only
to measure quality and skip 9 duplicates — saves ~75% of runtime.

Output: /tmp/test_suite_200_results.json + console summary.
"""
from __future__ import annotations

import base64
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
import psycopg2
from psycopg2.extras import RealDictCursor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import DB_CONFIG


API = "http://127.0.0.1:8000/api/v1/chat"
AUTH = base64.b64encode(b"admin:test").decode()
SOURCE = "/Users/joudathashmi/Downloads/MISA_Executive_Chatbot_Test_Suite_200_Questions.xlsx"


# ─── Pull substitution values from DB ────────────────────────────────

def load_substitutions() -> dict:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT DISTINCT ON (name) name, position
        FROM company_executives
        WHERE name IS NOT NULL AND name != ''
          AND (position ILIKE '%CEO%' OR position ILIKE '%Chief Executive%')
        LIMIT 50
    """)
    ceos = [r["name"] for r in cur.fetchall()]

    cur.execute("""
        SELECT name FROM countries
        WHERE name IS NOT NULL
          AND name NOT IN ('Saudi Arabia', 'KSA')
        ORDER BY id LIMIT 40
    """)
    countries = [r["name"] for r in cur.fetchall()]

    cur.execute("SELECT name FROM sectors ORDER BY id LIMIT 30")
    sectors = [r["name"] for r in cur.fetchall()]

    cur.execute("""
        SELECT title FROM opportunities
        WHERE title IS NOT NULL AND title != ''
        ORDER BY id LIMIT 30
    """)
    opps = [r["title"][:120] for r in cur.fetchall()]

    cur.execute("""
        SELECT company_name FROM company_profiles
        WHERE company_name IS NOT NULL AND length(company_name) BETWEEN 5 AND 30
        ORDER BY random() LIMIT 15
    """)
    real_companies = [r["company_name"] for r in cur.fetchall()]

    cur.close(); conn.close()
    return {"ceos": ceos, "countries": countries, "sectors": sectors,
            "opps": opps, "real_companies": real_companies}


def inject_typo(name: str) -> str:
    """Make a plausible 1-char typo of a company name."""
    if len(name) < 4:
        return name
    # swap 2 adjacent middle letters
    i = len(name) // 2
    return name[:i] + name[i+1] + name[i] + name[i+2:]


# Follow-up multi-turn scenarios — each is (prior_question, prior_answer_stub, followup)
_FOLLOWUP_SCENARIOS = [
    ("Tell me about Apple",
     "Apple Inc — Executive Briefing. Apple is a leading consumer-tech firm headquartered in Cupertino.",
     "How should we engage with them"),
    ("Tell me about Microsoft",
     "Microsoft Corp — Executive Briefing. HQ Redmond, RHQ in Riyadh.",
     "Show me their competitors"),
    ("Top sectors by opportunity count",
     "Top sectors: Agriculture (600), Petrochemical (508), Real Estate (248)...",
     "Top 10 companies from each of these sectors"),
    ("Tell me about Saudi Aramco",
     "Saudi Aramco — energy company.",
     "Brief me on engagement history with them"),
    ("Tell me about Pakistan",
     "Pakistan country profile.",
     "What MISA opportunities exist there"),
    ("Tell me about Tim Cook",
     "Tim Cook is the CEO of Apple Inc.",
     "Who would succeed him"),
    ("Compare Apple and Microsoft for KSA engagement",
     "Apple Inc vs Microsoft Corp comparison for Saudi Arabia.",
     "Which one should we approach first"),
    ("Tell me about Google",
     "Alphabet Inc — parent of Google.",
     "What are their Saudi investments"),
    ("Show me investors interested in renewable energy",
     "Several investors are active in renewable energy in MENA.",
     "Which ones are German"),
    ("What measures should we take to attract Chinese FDI",
     "MISA should leverage SEZs, tax incentives, and PIF partnerships.",
     "Recommend specific Chinese companies to target"),
]


# ─── Substitute placeholders into runnable questions ────────────────

def substitute_question(q: str, subs: dict) -> tuple[str, list]:
    """Returns (rewritten_question, history). History is non-empty for
    follow-up scenarios; empty for everything else."""
    # 1. Global CEO Case N
    m = re.search(r"global CEO.*\(Case (\d+)\)", q, re.I)
    if m:
        idx = (int(m.group(1)) - 1) % max(1, len(subs["ceos"]))
        if subs["ceos"]:
            ceo = subs["ceos"][idx]
            return (f"Prepare an executive briefing on {ceo} before a ministerial meeting.", [])
        return (q, [])

    # 2. country #N
    m = re.search(r"country #(\d+)", q, re.I)
    if m:
        idx = (int(m.group(1)) - 1) % max(1, len(subs["countries"]))
        if subs["countries"]:
            country = subs["countries"][idx]
            return (f"What should MISA know before engaging investors from {country}?", [])
        return (q, [])

    # 3. sector #N
    m = re.search(r"sector #(\d+)", q, re.I)
    if m:
        idx = (int(m.group(1)) - 1) % max(1, len(subs["sectors"]))
        if subs["sectors"]:
            sector = subs["sectors"][idx]
            return (f"Identify the most attractive opportunities in the {sector} sector.", [])
        return (q, [])

    # 4. opportunity #N
    m = re.search(r"opportunity #(\d+)", q, re.I)
    if m:
        idx = (int(m.group(1)) - 1) % max(1, len(subs["opps"]))
        if subs["opps"]:
            opp = subs["opps"][idx]
            return (f"Which companies are the best fit for the '{opp}' opportunity and why?", [])
        return (q, [])

    # 5. Follow-up test #N → multi-turn
    m = re.search(r"Follow-up question sequence test #(\d+)", q, re.I)
    if m:
        idx = (int(m.group(1)) - 1) % len(_FOLLOWUP_SCENARIOS)
        prev_q, prev_a, followup = _FOLLOWUP_SCENARIOS[idx]
        return (followup, [
            {"role": "user", "content": prev_q},
            {"role": "assistant", "content": prev_a},
        ])

    # 6. Misspelled entity test #N
    m = re.search(r"misspelled entity test #(\d+)", q, re.I)
    if m:
        idx = (int(m.group(1)) - 1) % max(1, len(subs["real_companies"]))
        if subs["real_companies"]:
            real = subs["real_companies"][idx]
            return (f"Tell me about {inject_typo(real)}", [])
        return (q, [])

    # 7. "What should you do when information is unavailable for scenario #N"
    #    Run as-is — it's a meta-question about behaviour
    return (q, [])


# ─── Runner ─────────────────────────────────────────────────────────

RED_FLAGS = [
    (re.compile(r"invalid input syntax|LINE 1:|psycopg2|bigint", re.I), "raw_sql_error"),
    (re.compile(r"^\s*$"), "empty_answer"),
    (re.compile(r"Multiple possible matches", re.I), "ambiguity_clarification"),
    (re.compile(r"No record matching", re.I), "no_match_dead_end"),
    (re.compile(r"I'm an executive-intelligence assistant.*don't have a useful", re.I),
     "off_topic_refusal"),
]
MEDIOCRE_FLAGS = [
    (re.compile(r"General knowledge\s*[—-]\s*not sourced", re.I),
     "general_knowledge_fallback"),
    (re.compile(r"No engagement history found", re.I),
     "honest_no_records"),
]


def classify(answer: str, status: int) -> tuple[str, list]:
    flags = []
    if status != 200:
        return "broken", [f"http_{status}"]
    if not answer or len(answer.strip()) < 30:
        return "broken", ["too_short"]
    for pat, name in RED_FLAGS:
        if pat.search(answer):
            flags.append(name)
    if flags:
        return "broken", flags
    for pat, name in MEDIOCRE_FLAGS:
        if pat.search(answer):
            flags.append(name)
    if flags:
        # honest_no_records is GOOD behaviour (Rule 7), not mediocre
        if flags == ["honest_no_records"]:
            return "good", flags
        return "mediocre", flags
    if len(answer) < 200:
        return "mediocre", ["short_answer"]
    return "good", []


def run_one(q: str, history: list) -> dict:
    t0 = time.time()
    try:
        r = requests.post(
            API,
            json={"question": q, "history": history, "locale": "en",
                  "stream": False, "debug": True},
            headers={"Authorization": f"Basic {AUTH}"},
            timeout=180,
        )
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {"question": q, "status": r.status_code,
                    "elapsed_s": round(elapsed, 2), "label": "broken",
                    "flags": [f"http_{r.status_code}"]}
        d = r.json()
        answer = d.get("answer") or ""
        dbg = d.get("debug") or {}
        label, flags = classify(answer, r.status_code)
        return {
            "question": q,
            "status": r.status_code,
            "elapsed_s": round(elapsed, 2),
            "answer_chars": len(answer),
            "answer": answer,  # full text
            "answer_head": answer.splitlines()[0][:140] if answer else "",
            "intent": dbg.get("intent"),
            "label": label,
            "flags": flags,
        }
    except Exception as e:
        return {"question": q, "status": 0,
                "elapsed_s": round(time.time() - t0, 2), "label": "broken",
                "flags": [f"exception:{type(e).__name__}"]}


def main():
    df = pd.read_excel(SOURCE)
    print(f"Loaded {len(df)} questions from spreadsheet")
    subs = load_substitutions()
    print(f"Substitutions: {len(subs['ceos'])} CEOs, {len(subs['countries'])} countries, "
          f"{len(subs['sectors'])} sectors, {len(subs['opps'])} opportunities")

    # Dedup: only run each unique stem ONCE (test suite has 5 scenarios
    # × 10 copies each — running 10× wastes time without new signal)
    def strip_scenario(q):
        return re.sub(r'\s*\([Ss]cenario\s+#?\d+\)\s*$', '', q).strip()
    df['base'] = df['Question'].apply(strip_scenario)
    unique_df = df.drop_duplicates(subset='base').reset_index(drop=True)
    print(f"After dedup: {len(unique_df)} unique questions to run\n")

    results = []
    for i, row in unique_df.iterrows():
        original = row['Question']
        rewritten, history = substitute_question(original, subs)
        was_substituted = rewritten != original
        result = run_one(rewritten, history)
        result["original_question"] = original
        result["was_substituted"] = was_substituted
        result["has_history"] = bool(history)
        results.append(result)
        emoji = {"good": "✅", "mediocre": "🟡", "broken": "🔴"}[result["label"]]
        flags_s = (" / " + ", ".join(result["flags"])) if result["flags"] else ""
        sub_indicator = "↻" if was_substituted else " "
        h_indicator = "+H" if history else "  "
        print(f"  {i+1:3d}/{len(unique_df)} {emoji}{sub_indicator}{h_indicator} {result['elapsed_s']:5.1f}s  "
              f"{rewritten[:80]!r}{flags_s}")
        Path("/tmp/test_suite_200_results.json").write_text(
            json.dumps(results, indent=2, default=str))

    print(f"\nFull results: /tmp/test_suite_200_results.json\n")

    by_label = defaultdict(int)
    flag_counter = defaultdict(int)
    for r in results:
        by_label[r["label"]] += 1
        for f in r.get("flags", []):
            flag_counter[f] += 1

    print("═" * 80)
    print("SUMMARY")
    print("═" * 80)
    total = len(results)
    print(f"Unique questions run: {total} (deduped from 200)")
    for lbl in ("good", "mediocre", "broken"):
        n = by_label[lbl]
        pct = 100 * n / total if total else 0
        print(f"  {lbl:10s} {n:3d}  ({pct:.0f}%)")
    print()
    print("Failure / flag counts:")
    for f, n in sorted(flag_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {f}")
    elapsed = [r["elapsed_s"] for r in results if r.get("elapsed_s")]
    if elapsed:
        elapsed.sort()
        p50 = elapsed[len(elapsed) // 2]
        p95 = elapsed[int(len(elapsed) * 0.95)]
        print(f"\nLatency: p50 {p50:.1f}s · p95 {p95:.1f}s · max {max(elapsed):.1f}s")


if __name__ == "__main__":
    main()
