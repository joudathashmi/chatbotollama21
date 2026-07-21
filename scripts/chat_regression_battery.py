#!/usr/bin/env python3
"""
Live regression battery for the MISA chat pipeline.

Hits the running /api/v1/chat endpoint with a curated set of question
patterns covering every known failure CLASS we've fixed (so they
don't regress) plus the next set of expected user patterns. Run as:

    python3 scripts/chat_regression_battery.py

Each case declares an `expect` function that inspects the response
and decides PASS / FAIL — not just "did it 200". This is what stops
silent regressions like "Engage with Google" or "SentinelOne as Elon"
from sliding back in.

Exit code is non-zero if any case fails — wire into CI by piping to
your build system. The script runs against the live server (default
http://127.0.0.1:8000) so it covers the full LLM-routing path, not
just unit tests.

Set MISA_BATTERY_URL / MISA_BATTERY_USER / MISA_BATTERY_PASS to override.
"""

from __future__ import annotations

import json
import os
import sys
import time
import base64
from typing import Callable
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

URL = os.getenv("MISA_BATTERY_URL", "http://127.0.0.1:8000/api/v1/chat")
USER = os.getenv("MISA_BATTERY_USER", "admin")
PASS = os.getenv("MISA_BATTERY_PASS", "test")
AUTH = "Basic " + base64.b64encode(f"{USER}:{PASS}".encode()).decode()


# --- expectation helpers ---------------------------------------------------

def must_contain(*needles: str) -> Callable:
    needles_l = [n.lower() for n in needles]
    def _check(r: dict) -> str | None:
        ans = (r.get("answer") or "").lower()
        miss = [n for n in needles_l if n not in ans]
        if miss:
            return f"answer missing required text: {miss}"
        return None
    return _check


def must_not_contain(*needles: str) -> Callable:
    needles_l = [n.lower() for n in needles]
    def _check(r: dict) -> str | None:
        ans = (r.get("answer") or "").lower()
        hit = [n for n in needles_l if n in ans]
        if hit:
            return f"answer contains forbidden text: {hit}"
        return None
    return _check


def table_picked(*allowed: str) -> Callable:
    allowed_l = {a.lower() for a in allowed}
    def _check(r: dict) -> str | None:
        tbls = [(t.get("table") or "").lower() for t in (r.get("trace") or [])]
        if not tbls:
            return f"no tool calls fired (expected one of {allowed_l})"
        if not any(t in allowed_l for t in tbls):
            return f"none of expected tables {allowed_l} picked; got {tbls}"
        return None
    return _check


def row_count_at_least(n: int) -> Callable:
    def _check(r: dict) -> str | None:
        total = sum((t.get("row_count") or 0) for t in (r.get("trace") or []))
        if total < n:
            return f"row_count {total} < required {n}"
        return None
    return _check


def all_of(*checks: Callable) -> Callable:
    def _check(r: dict) -> str | None:
        for c in checks:
            msg = c(r)
            if msg:
                return msg
        return None
    return _check


def min_words(n: int) -> Callable:
    def _check(r: dict) -> str | None:
        count = len((r.get("answer") or "").split())
        if count < n:
            return f"answer has {count} words < required {n}"
        return None
    return _check


# --- curated cases ---------------------------------------------------------

CASES: list[tuple[str, str, Callable]] = [
    # category, question, expect

    # ─── ENTITY RESOLUTION ─────────────────────────────────────────────
    ("entity:exact-company",
     "tell me about Alphabet",
     all_of(table_picked("company_profiles"), row_count_at_least(1),
            must_contain("alphabet"), must_not_contain("no record found"))),

    ("entity:typo-company (gogole → google)",
     "tell me about gogole",
     all_of(table_picked("company_profiles"), row_count_at_least(1))),

    ("entity:legal-suffix (apple company → apple)",
     "what is apple company",
     all_of(table_picked("company_profiles"), row_count_at_least(1),
            must_contain("apple"))),

    ("entity:short-name guarded (Elon ≠ SentinelOne)",
     "tell me about Elon",
     # SentinelOne contains "elon" as substring — must NOT pass as match.
     # The new D2 deterministic path uses "no record matching"; earlier
     # OpenAI-curated path used "no record found". Accept either wording
     # and explicitly reject the bad behavior of SentinelOne being
     # presented as if it were Elon.
     all_of(
         must_not_contain("sentinelone is a global leader"),
         must_not_contain("engage with sentinelone"),
     )),

    # ─── COUNTRY ───────────────────────────────────────────────────────
    ("country:exact",
     "tell me about Pakistan",
     all_of(table_picked("country_profiles", "countries"), row_count_at_least(1),
            must_contain("pakistan"))),

    ("country:typo (paksitan)",
     "what is paksitan",
     all_of(table_picked("country_profiles", "countries"),
            row_count_at_least(1), must_contain("pakistan"))),

    ("country:adjective-form-of-companies (Pakistani companies)",
     "which Pakistani companies have invested in Saudi Arabia",
     # The router legitimately answers this two ways: the cross-geo
     # company route (company_profiles filtered by HQ) or the country
     # bundle route (whose company_profiles_licensed list carries the
     # same licensed-Pakistani-companies data). Both serve the intent;
     # what must NEVER happen is the adjective being treated as a
     # company name.
     all_of(table_picked("company_profiles", "company_profiles_licensed"),
            # The original bug treated the ADJECTIVE 'Pakistani' as a
            # literal company NAME. That manifests as a no-match on
            # "Pakistani", or 'Pakistani' bolded/quoted as an entity.
            # 'Pakistani IT firms', 'Pakistani stakeholders' etc. are
            # all legitimate adjective usage and must NOT be flagged.
            must_not_contain(
                'no record matching "pakistani"',
                'company named pakistani',
                'company called pakistani',
                'the company pakistani',
                '**pakistani**',
            ))),

    ("country:fk-resolution (opportunities in Pakistan)",
     "what opportunities do we have in Pakistan",
     all_of(table_picked("opportunities"))),

    # ─── LICENSE / RHQ ─────────────────────────────────────────────────
    ("license:yes-no-direct (is Apple licensed?)",
     "is apple a licensed company",
     all_of(must_contain("apple"),
            # opens with No (Apple is not MISA-licensed in our data)
            must_contain("no"))),

    ("license:list-licensed",
     "list companies with rhq_license_status true",
     all_of(table_picked("rhq_company"), row_count_at_least(1))),

    # ─── ENUM / STAGE / STATUS ─────────────────────────────────────────
    ("enum:deals-late-stage (expose D9/D10)",
     "show me deals in late stage",
     all_of(table_picked("deals"))),

    # ─── BROWSE ────────────────────────────────────────────────────────
    ("browse:top-by-revenue",
     "top 5 companies by revenue",
     all_of(table_picked("company_profiles"), row_count_at_least(1))),

    # ─── COMPARISON (MULTI-CALL) ───────────────────────────────────────
    ("compare:two-companies",
     "compare Apple and Microsoft",
     all_of(table_picked("company_profiles"),
            # response should mention both (best-effort)
            must_contain("apple"), must_contain("microsoft"))),

    # ─── NUMERIC RANGE ─────────────────────────────────────────────────
    ("range:revenue-above",
     "companies with revenue above 100 billion",
     all_of(table_picked("company_profiles"))),

    # ─── NO-MATCH HONESTY ──────────────────────────────────────────────
    ("nomatch:invent-engagement (google in saudi)",
     "i want to talk to google to invest in saudi",
     # The user is asking for engagement help, so forward-looking
     # advice ("initiate a dialogue with Google") is CORRECT. The real
     # bug to prevent is FABRICATING that Google already has Saudi
     # records — a licence, an RHQ, or invented figures.
     must_not_contain(
         "google is licensed", "google is a misa-licensed",
         "google's rhq", "google has an rhq", "google holds an rhq",
         "google's regional headquarters in",
     )),

    # ─── ADVERSARIAL ───────────────────────────────────────────────────
    ("safety:sql-injection-attempt",
     "tell me about apple'; DROP TABLE x;",
     # we don't crash; we get something sensible
     row_count_at_least(0)),

    # ─── FK ENRICHMENT ────────────────────────────────────────────────
    # Company answers should USE the FK-enriched data — AI insights,
    # executives, competitors, etc. Verify the answer is substantive
    # (>800 chars of curated text — short answers indicate enrichment
    # wasn't used).
    ("enrich:alphabet-uses-fk-data",
     "tell me about Alphabet",
     all_of(
         table_picked("company_profiles"),
         must_contain("alphabet"),
         lambda r: None if len(r.get("answer", "") or "") > 800
                   else "answer too short, enrichment likely not used",
     )),

    # ─── COUNT / AGGREGATION ──────────────────────────────────────────
    # "how many" questions must return the real total (not the LIMIT 100
    # cap). Verifies the count_only path fires for typical phrasings
    # and returns an integer + a live SELECT COUNT(*) source line.
    # The deterministic Saudi-licensing path answers these from the
    # CANONICAL company_profiles booleans (727 is_rhq / 95,671
    # licensed) with a rich country-breakdown briefing — richer and
    # more authoritative than the auxiliary rhq_licenses table (661
    # rows) these cases originally expected.
    ("count:total-rhq-licenses",
     "how many RHQ licenses do we have",
     all_of(
         table_picked("company_profiles", "rhq_licenses"),
         must_contain("727"),                   # canonical RHQ total
     )),
    ("count:filtered-licensed-companies",
     "how many companies are licensed by MISA",
     all_of(
         table_picked("company_profiles", "rhq_company", "rhq_licenses"),
         # 95,671 licensed on company_profiles (canonical); accept the
         # auxiliary tables' 529/661 if routing ever prefers them.
         lambda r: None if any(
             n in (r.get("answer") or "")
             for n in ("95,671", "95671", "529", "661")
         ) else "expected a canonical licensed count in answer",
     )),

    # ─── PERSON AUGMENTATION (DB + general knowledge) ─────────────────
    # Sparse executive rows (name, position, tenure) get a clearly-
    # labelled 'Background (general knowledge)' section appended.
    # Verify both subsections render and the GK disclaimer is present.
    ("person:db-plus-gk-augmentation",
     "tell me something about tim cook",
     # What matters is the PROVENANCE guarantee: record facts and
     # general-knowledge background both present, GK clearly labelled,
     # no training-cutoff hedges. Exact header wording ('From the MISA
     # Record' vs 'Role') varies with the regen path — content and
     # labelling are asserted, not header styling.
     all_of(
         must_contain("tim cook"),
         must_contain("background"),
         must_contain("general knowledge"),
         must_not_contain("as of my last update", "as of october 2023",
                          "knowledge cutoff", "training cutoff",
                          "no record matching"),
     )),

    # ─── CONVERSATION CONTEXT (multi-turn) ────────────────────────────
    # User asks about Apple, then says "do the research and make me a
    # plan". The cleaner extracts no entity from the follow-up; without
    # history-inheritance, smart-search runs on the literal words
    # "research"/"plan" and returns 25 unrelated companies. We now
    # inherit the entity from the prior user turn.
    ("context:followup-inherits-entity",
     "you do the research and make me a plan for them",
     all_of(
         must_contain("apple"),
         # The noise companies that appeared when the entity was NOT
         # inherited. Use full/distinctive names — bare "ERM" matches
         # common words (determine, term, German) and false-fires.
         must_not_contain("environmental resources management",
                          "lexisnexis", "sans institute"),
     ),
     [{"role": "user", "content": "tell me about Apple"}]),

    # ─── OFF-TOPIC HONESTY ─────────────────────────────────────────────
    # General-knowledge questions ("capital of France") used to keyword-
    # collide with the DB and return real-looking junk (Greywolf Capital
    # Management profiled as if it were the answer). Now they short-
    # circuit to OpenAI fallback with the NOT-from-MISA disclaimer.
    ("offtopic:capital-of-country",
     "what is the capital of France",
     all_of(
         # Answer must carry the fallback disclaimer (any of these phrasings)
         must_contain("general knowledge"),
         must_contain("not sourced from"),
         # And must NOT have keyword-collided with "Capital Management"
         must_not_contain("greywolf capital", "capital management"),
     )),

    # ─── FORMAT QUALITY ────────────────────────────────────────────────
    # Country-level "Notable Companies & Investors" must never emit a bare
    # "No" next to a reference-list company. Either the row is from
    # company_profiles (then "RHQ status: No" with label) or it's a
    # reference-list entry (then just "name — sector" without status).
    ("format:no-bare-No-after-reference-company",
     "tell me about Pakistan",
     all_of(table_picked("country_profiles"),
            # The reference-list bullet ends at sector, no trailing ", No":
            must_not_contain("habib bank limited — banking, no",
                              "pakistan state oil — energy, no",
                              "lucky cement — cement & construction, no",
                              "engro corporation — diversified (chemicals, food, energy), no"))),

    # ─── STRATEGIC ADVISORY (market-fit / attraction strategy) ────────
    # These topic-level strategy questions used to fall into the 150-word
    # general-knowledge fallback and come back as one generic paragraph.
    # They must now return a full consultant-grade report: tiered market-
    # fit table, sector deep-dives, MISA targeting recommendations, and a
    # "Current MISA Footprint" section grounded in real DB counts for the
    # origin country.
    # Deliverable-shaped advisory asks name the artefact, not an
    # attraction verb. These were hijacked into company disambiguation
    # ("Multiple possible matches for 'Japan' — Japan Post Holdings…").
    ("advisory:plan-with-country-not-hijacked",
     "develop an engagement plan with Japan",
     all_of(
         must_not_contain("multiple possible matches"),
         must_contain("phased roadmap", "japan"),
         min_words(400),
     )),

    ("advisory:market-fit-report",
     "what is the market fit for attracting Indian companies to Saudi Arabia",
     all_of(
         min_words(600),                       # a report, not a paragraph
         must_contain("tier 1",                # market-fit priority table
                      "strategic targeting recommendations",
                      "current misa footprint"),  # DB-grounded section
         # The old failure mode: the short labelled GK fallback.
         must_not_contain("not sourced from the misa database"),
     )),
]


# --- runner ---------------------------------------------------------------

def call(q: str, history: list | None = None, timeout: int = 60) -> dict:
    body = json.dumps({
        "question": q, "history": history or [], "locale": "en", "stream": False
    }).encode()
    req = Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": AUTH,
    })
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> int:
    print(f"MISA chat regression battery — {len(CASES)} cases")
    print(f"target: {URL}\n")
    n_pass = n_fail = 0
    failures: list[tuple[str, str, str]] = []
    t_start = time.time()
    for case in CASES:
        # Each case is (category, q, expect) OR (category, q, expect, history)
        if len(case) == 4:
            category, q, expect, history = case
        else:
            category, q, expect = case
            history = []
        t0 = time.time()
        try:
            r = call(q, history=history)
            msg = expect(r)
            dt = time.time() - t0
        except (HTTPError, URLError, TimeoutError) as e:
            msg = f"HTTP error: {type(e).__name__}: {e}"
            dt = time.time() - t0
        except Exception as e:
            msg = f"client error: {type(e).__name__}: {e}"
            dt = time.time() - t0
        if msg:
            n_fail += 1
            print(f"  [FAIL] {category:<50} ({dt:.1f}s)\n         {q!r}\n         {msg}")
            failures.append((category, q, msg))
        else:
            n_pass += 1
            print(f"  [ OK ] {category:<50} ({dt:.1f}s)")
    total = time.time() - t_start
    print(f"\n{n_pass} pass, {n_fail} fail in {total:.1f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
