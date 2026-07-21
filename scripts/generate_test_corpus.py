#!/usr/bin/env python3
"""
Auto-evaluator: sample real entities from the DB, template questions
across all 10 intents, run them through the live chat API, and
auto-evaluate structural quality.

This is the answer to "we need 500-1000 golden cases" without manual
curation. Instead of writing each case by hand, we:

  1. Sample diverse companies / countries / executives from DB
  2. Generate questions per intent (~10-15 templates per entity)
  3. Add multi-turn follow-up sequences (pronoun resolution)
  4. POST each to /api/v1/chat with 4-way concurrency
  5. Run STRUCTURAL assertions per intent:
       - intent-specific headings present
       - intent-specific content patterns
       - universal data-hygiene check (no leftover backend noise)
  6. Output:
       - console summary (pass/fail by intent, top failure modes)
       - generated_results.jsonl (full results, one per question)
       - generated_failures.json (starter golden-case stubs for fails)

NOT meant to replace the curated golden suite — that's the
behavioural pin. This is the WIDE coverage that catches drift on
real entities the curated suite doesn't reach.

Usage:
  ./venv/bin/python scripts/generate_test_corpus.py
  ./venv/bin/python scripts/generate_test_corpus.py --count 200
  ./venv/bin/python scripts/generate_test_corpus.py --concurrency 6

Output ends up at:
  /tmp/generated_results.jsonl
  /tmp/generated_failures.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import psycopg2.extras  # noqa: E402
from app.database import get_db  # noqa: E402


CHAT_URL = "http://127.0.0.1:8000/api/v1/chat"
AUTH = ("admin", "test")


# ─── Entity sampling ─────────────────────────────────────────────────

def sample_entities(
    n_companies: int = 30, n_countries: int = 10, n_executives: int = 15,
):
    """Pull a diverse mix from DB. Skips entities with non-printable
    chars in their names (mostly noisy Arabic-only rows that the
    smart-search can't reliably route)."""
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Bias toward companies that actually have rich profiles —
        # prefer rows with non-null employee_count or annual_revenue
        # so we test the system on entities it can answer about. Pure
        # random sampling picks ~70% noise entities (local Arabic-named
        # contractors with no profile data) where the system correctly
        # falls back to GK and most structural asserts don't apply.
        cur.execute("""
            SELECT company_name FROM company_profiles
            WHERE company_name IS NOT NULL
              AND length(company_name) > 4
              AND length(company_name) < 70
              AND company_name ~ '^[A-Za-z]'
              AND (employee_count IS NOT NULL OR annual_revenue IS NOT NULL
                   OR ceo IS NOT NULL OR market_cap IS NOT NULL)
            ORDER BY RANDOM() LIMIT %s
        """, (n_companies,))
        companies = [r["company_name"] for r in cur.fetchall()]

        cur.execute("""
            SELECT country_name FROM country_profiles
            WHERE country_name IS NOT NULL
              AND length(country_name) > 3
            ORDER BY RANDOM() LIMIT %s
        """, (n_countries,))
        countries = [r["country_name"] for r in cur.fetchall()]

        # %% escapes a literal % for psycopg2 parameter substitution.
        # Use a subquery so ORDER BY RANDOM() can mix with DISTINCT.
        cur.execute("""
            SELECT name FROM (
                SELECT DISTINCT name FROM company_executives
                WHERE name IS NOT NULL
                  AND length(name) > 5
                  AND name ~ '^[A-Z][a-z]'
                  AND name NOT LIKE '%%Holding%%'
                  AND name NOT LIKE '%%Group%%'
                  AND name NOT LIKE '%%Inc%%'
                  AND name NOT LIKE '%%Corp%%'
                  AND name NOT LIKE '%%Ltd%%'
            ) t
            ORDER BY RANDOM() LIMIT %s
        """, (n_executives,))
        executives = [r["name"] for r in cur.fetchall()]
    return companies, countries, executives


# ─── Question templates per intent ───────────────────────────────────

# (intent_label, question_template) — {ENT} is the entity placeholder
COMPANY_TEMPLATES = [
    ("executive_lookup",    "Who is the CEO of {ENT}?"),
    ("executive_lookup",    "Who chairs {ENT}?"),
    ("executive_lookup",    "Who runs {ENT}?"),
    ("executive_succession", "Who is the next CEO of {ENT}?"),
    ("executive_succession", "{ENT} succession plans"),
    ("company_profile",     "Tell me about {ENT}"),
    ("company_profile",     "{ENT} profile"),
    ("company_profile",     "What does {ENT} do?"),
    ("saudi_presence",      "Does {ENT} have an RHQ in Saudi?"),
    ("saudi_presence",      "{ENT} presence in Saudi Arabia"),
    ("saudi_presence",      "{ENT} MENA footprint"),
    ("engagement_strategy", "How should MISA engage {ENT}?"),
    ("engagement_strategy", "Suggest an engagement plan for {ENT}"),
    ("engagement_strategy", "Who should we contact at {ENT}?"),
    ("financial_lookup",    "What is {ENT}'s revenue?"),
    ("financial_lookup",    "How many employees does {ENT} have?"),
    ("relationship_intelligence", "Previous meetings with {ENT}"),
    ("relationship_intelligence", "What engagements have we had with {ENT}?"),
    ("opportunity_alignment", "Why is {ENT} relevant to MISA?"),
    ("opportunity_alignment", "How does {ENT} align with Vision 2030?"),
]

COUNTRY_TEMPLATES = [
    ("country_profile",     "Tell me about {ENT}"),
    ("country_profile",     "What is the economy of {ENT}?"),
    ("country_profile",     "{ENT} FDI outlook"),
]

PERSON_TEMPLATES = [
    ("executive_lookup",    "Who is {ENT}?"),
    ("executive_lookup",    "Tell me about {ENT}"),
]


def generate_questions(companies, countries, executives, cap=None):
    """Materialise the full question list. Returns list of dicts."""
    out = []
    for c in companies:
        for intent, tpl in COMPANY_TEMPLATES:
            out.append({"intent": intent, "entity": c, "entity_kind": "company",
                        "question": tpl.replace("{ENT}", c)})
    for c in countries:
        for intent, tpl in COUNTRY_TEMPLATES:
            out.append({"intent": intent, "entity": c, "entity_kind": "country",
                        "question": tpl.replace("{ENT}", c)})
    for p in executives:
        for intent, tpl in PERSON_TEMPLATES:
            out.append({"intent": intent, "entity": p, "entity_kind": "person",
                        "question": tpl.replace("{ENT}", p)})
    # Multi-turn follow-up sequences: prior turn establishes the
    # entity, follow-up references via pronoun. Exercises the
    # entity-inheritance + intent-routing paths together.
    followups = []
    for c in companies[:8]:
        followups.append([
            {"role": "user", "content": f"tell me about {c}"},
            {"role": "user", "content": "who is their CEO"},
        ])
        followups.append([
            {"role": "user", "content": f"tell me about {c}"},
            {"role": "user", "content": "are they in saudi"},
        ])
        followups.append([
            {"role": "user", "content": f"tell me about {c}"},
            {"role": "user", "content": "how should we engage them"},
        ])
    for seq in followups:
        # Reconstruct the last turn as a separate test case;
        # earlier turns become the history. We mock the assistant
        # replies with a one-line summary that NAMES THE ENTITY, so
        # the chat engine's resolver can anchor on it (a totally
        # placeholder reply gives the resolver nothing to extract).
        prior_entity = seq[0]["content"].replace("tell me about ", "").strip()
        history = []
        last = None
        for turn in seq:
            if last is not None:
                history.append({"role": "user", "content": last})
                history.append({
                    "role": "assistant",
                    "content": (
                        f"{prior_entity} is a company we just discussed. "
                        f"Here is a brief profile of {prior_entity}..."
                    ),
                })
            last = turn["content"]
        out.append({
            "intent": "followup",
            "entity": prior_entity,
            "entity_kind": "followup",
            "question": last,
            "history": history,
        })
    if cap and len(out) > cap:
        # Stratified sub-sample: keep ratio across intents.
        from collections import defaultdict
        buckets = defaultdict(list)
        for q in out:
            buckets[q["intent"]].append(q)
        per_bucket = max(1, cap // len(buckets))
        out = []
        for intent_name, qs in buckets.items():
            out.extend(qs[:per_bucket])
    return out


# ─── Assertions per intent ───────────────────────────────────────────

# Universal forbidden strings — data-hygiene rules apply to ALL turns.
# These are stripped by app/services/curation.py:_scrub_backend_noise
# so seeing them in output is a regression.
UNIVERSAL_FORBIDDEN = [
    "(High)", "(Medium)", "(Low)", "(Unknown)",
    "[DB]", "[gk]", "[inferred]",
    "Source: DB", "**Source:** DB",
    "_(general knowledge)_",
    "Not available in the current database",
]


def _has_any(text: str, needles: list[str]) -> str | None:
    """Returns the first needle that appears in text, or None."""
    tl = text.lower()
    for n in needles:
        if n.lower() in tl:
            return n
    return None


def evaluate_answer(case: dict, answer: str, response: dict) -> dict:
    """Apply structural assertions appropriate to the case's intent.
    Returns {pass: bool, failure_modes: list[str], note: str}."""
    if not answer or not answer.strip():
        return {"pass": False, "failure_modes": ["empty_answer"], "note": ""}

    failures: list[str] = []
    # Universal hygiene check
    hit = _has_any(answer, UNIVERSAL_FORBIDDEN)
    if hit:
        failures.append(f"hygiene:{hit}")

    intent = case["intent"]
    al = answer.lower()
    first_500 = al[:500]

    # Skip per-intent structural asserts when the answer is one of
    # the system's CORRECT fallback shapes:
    #   - GK-only reply (entity not in DB → freeform general knowledge)
    #   - Honest no-match ("No record matching X was found")
    #   - Clarification card ("Multiple possible matches for ...")
    # The briefing structure doesn't apply to these. The universal
    # hygiene check above still ran, so backend-noise leaks are still
    # caught here.
    is_correct_fallback = (
        "not sourced from the misa database" in first_500
        or "general knowledge" in first_500[:200]
        or "does not appear to refer" in first_500
        or "is not a widely recognized" in first_500
        or "no record matching" in first_500
        or "multiple possible matches for" in first_500
    )
    if is_correct_fallback:
        return {"pass": not failures, "failure_modes": failures,
                "note": "correct_fallback (no DB hit)"}

    # Intent-specific checks
    if intent == "executive_lookup":
        # Must NOT lead with company snapshot when a person is the focus
        if first_500.startswith("## snapshot") and "ceo" not in first_500[:150]:
            failures.append("exec_buried_under_snapshot")
        # Should have a person-focused header somewhere near top
        if not any(h in first_500 for h in ["## ceo", "## chair", "## founder",
                                            "## executive", "## leadership", "## person"]):
            # Not catastrophic; just note if first heading isn't person-shaped
            if "name:" not in first_500:
                failures.append("no_person_header_near_top")

    elif intent == "executive_succession":
        if "what's reported" not in al and "leading candidates" not in al \
                and "successor" not in al and "next" not in first_500:
            failures.append("no_succession_section")

    elif intent == "company_profile":
        if "executive briefing" not in al and "## snapshot" not in al:
            failures.append("no_briefing_heading")
        # Strategic read should appear somewhere
        if "strategic" not in al:
            failures.append("no_strategic_section")

    elif intent == "country_profile":
        country = case["entity"].lower()
        if country not in al:
            failures.append("country_name_missing")
        if "snapshot" not in al and "outlook" not in al:
            failures.append("no_country_structure")

    elif intent == "saudi_presence":
        if "saudi" not in al and "mena" not in al and "rhq" not in al:
            failures.append("no_saudi_mention")

    elif intent == "engagement_strategy":
        if "engagement" not in al and "engage" not in al and "approach" not in al:
            failures.append("no_engagement_focus")

    elif intent == "financial_lookup":
        if not re.search(r"\$|\bbillion\b|\bmillion\b|\bemployees?\b|\bn/?a\b|\d[\d,]{2,}",
                         al):
            failures.append("no_number_or_metric")

    elif intent == "relationship_intelligence":
        # Must either show engagement records OR honestly say none
        if not any(s in al for s in [
            "engagement", "meeting", "contact", "no engagement", "no record",
            "previous", "history",
        ]):
            failures.append("no_relationship_signal")

    elif intent == "opportunity_alignment":
        if not any(s in al for s in ["vision 2030", "strategic fit", "align",
                                     "opportunity", "misa"]):
            failures.append("no_alignment_signal")

    elif intent == "followup":
        # Follow-up should produce SOMETHING relevant — the FIRST WORD
        # of the prior entity should show up in the answer. Using the
        # first word (not first 8 chars) so long company names like
        # "Velosi Asset Integrity And Engineering..." check on
        # "velosi" not "velosi a".
        ent = (case.get("entity") or "").strip()
        first_word = ent.split()[0].lower() if ent else ""
        if first_word and len(first_word) >= 3 and first_word not in al[:1500]:
            failures.append("followup_lost_entity")

    return {
        "pass": not failures,
        "failure_modes": failures,
        "note": "",
    }


# ─── Network calls ───────────────────────────────────────────────────

def call_chat(question: str, history: list | None = None) -> dict:
    """POST /api/v1/chat (JSON mode, debug=true). Returns parsed
    JSON or {'error': ...} on failure."""
    import base64
    body = json.dumps({
        "question": question,
        "history": history or [],
        "locale": "en",
        "stream": False,
        "debug": True,
    }).encode("utf-8")
    auth = base64.b64encode(b"admin:test").decode()
    req = Request(
        CHAT_URL, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth}",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ─── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100,
                    help="cap on total questions (default 100)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel chat calls (default 4)")
    ap.add_argument("--out-results", default="/tmp/generated_results.jsonl")
    ap.add_argument("--out-failures", default="/tmp/generated_failures.json")
    args = ap.parse_args()

    print(f"sampling entities from DB...")
    companies, countries, executives = sample_entities()
    print(f"  {len(companies)} companies, {len(countries)} countries, "
          f"{len(executives)} executives")

    print(f"generating questions (cap={args.count})...")
    cases = generate_questions(companies, countries, executives, cap=args.count)
    print(f"  {len(cases)} cases generated")

    print(f"hitting chat API ({args.concurrency} workers, "
          f"~{len(cases) * 8 // args.concurrency}s expected)...")
    t0 = time.time()
    results = []
    done = 0

    def run_one(c):
        resp = call_chat(c["question"], c.get("history"))
        return c, resp

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, c) for c in cases]
        for f in as_completed(futures):
            done += 1
            c, resp = f.result()
            answer = resp.get("answer") or ""
            verdict = evaluate_answer(c, answer, resp)
            rec = {
                **c,
                "answer_head": answer[:600],
                "answer_chars": len(answer),
                "error": resp.get("error"),
                "verdict_pass": verdict["pass"],
                "failure_modes": verdict["failure_modes"],
                "debug": resp.get("debug"),
            }
            results.append(rec)
            mark = "✓" if verdict["pass"] else "✗"
            modes = ",".join(verdict["failure_modes"]) if verdict["failure_modes"] else ""
            if done % 10 == 0 or not verdict["pass"]:
                print(f"  [{done}/{len(cases)}] {mark} {c['intent']:24} "
                      f"{c['question'][:60]:60} {modes}")
    elapsed = time.time() - t0
    print(f"done in {elapsed:.0f}s")

    # Write all results
    with open(args.out_results, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {args.out_results}")

    # Summary
    from collections import Counter
    passed = sum(1 for r in results if r["verdict_pass"])
    print()
    print("=" * 70)
    print(f"SUMMARY: {passed}/{len(results)} pass "
          f"({passed * 100 // max(1, len(results))}%)")
    print("=" * 70)
    by_intent = {}
    for r in results:
        d = by_intent.setdefault(r["intent"], {"pass": 0, "fail": 0})
        d["pass" if r["verdict_pass"] else "fail"] += 1
    for intent_name, d in sorted(by_intent.items()):
        total = d["pass"] + d["fail"]
        pct = d["pass"] * 100 // max(1, total)
        print(f"  {intent_name:28} {d['pass']:3}/{total:3}  ({pct}%)")
    print()
    print("Top failure modes:")
    modes_counter: Counter = Counter()
    for r in results:
        for m in r.get("failure_modes") or []:
            modes_counter[m] += 1
    for m, n in modes_counter.most_common(10):
        print(f"  {n:4}  {m}")

    # Write starter golden cases for fails (capped to first 30)
    fails = [r for r in results if not r["verdict_pass"]][:30]
    golden_stubs = []
    for i, r in enumerate(fails, start=1):
        ent = (r.get("entity") or "?").replace(" ", "-")[:30]
        golden_stubs.append({
            "question_id": f"GEN{i:03d}-{r['intent']}-{ent}",
            "user_question": r["question"],
            "history": r.get("history", []),
            "expected_behavior": (
                f"AUTO-GENERATED. Intent={r['intent']}. "
                f"Failure modes={r['failure_modes']}. "
                f"Answer head: {r['answer_head'][:180]}…"
            ),
            "expected_source": "DB",
            "expected_company": r.get("entity") if r.get("entity_kind") == "company" else None,
            "expected_tables": [],
            "expected_keywords": ["TODO: fill in from answer-shape rule"],
            "forbidden_keywords": [m for m in (r.get("failure_modes") or [])
                                   if m.startswith("hygiene:")],
        })
    with open(args.out_failures, "w", encoding="utf-8") as f:
        json.dump(golden_stubs, f, indent=2, ensure_ascii=False, default=str)
    print()
    print(f"wrote {args.out_failures} ({len(golden_stubs)} starter cases)")
    print()
    print("Next steps:")
    print(f"  1. Review {args.out_failures} — curate the worst into")
    print(f"     tests/golden_cases.json")
    print(f"  2. Re-run scripts/run_golden_tests.py to lock the fixes in")
    print(f"  3. Re-run this script after prompt tweaks to confirm")
    print(f"     no broader regression on the wider corpus")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
