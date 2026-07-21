"""Comprehensive sweep of EVERY fix from the multi-day session — each
fix class tested with multiple phrasings / countries for robustness."""
import json, os, re, base64, time
from urllib.request import Request, urlopen
AUTH = "Basic " + base64.b64encode(f"{os.environ['U']}:{os.environ['P']}".encode()).decode()

def ask(q):
    body = json.dumps({"question": q, "history": [], "locale": "en", "stream": False}).encode()
    req = Request("http://127.0.0.1:8000/api/v1/chat", data=body,
                  headers={"Content-Type": "application/json", "Authorization": AUTH})
    with urlopen(req, timeout=180) as r:
        return json.load(r)["answer"]

# group, question, checks
C = [
 # ── Advisory: market fit (multiple countries) ──
 ("advisory/market-fit", "what is the market fit for attracting Indian companies to Saudi Arabia",
   dict(minw=700, need=["Investment & Trade Bodies"], reject=["multiple possible matches"])),
 ("advisory/market-fit", "What is the market fit and investment thesis for Canada",
   dict(minw=700, need=["Investment & Trade Bodies"])),
 ("advisory/market-fit", "investment case for German manufacturers in Saudi Arabia",
   dict(minw=600)),
 # ── Advisory: engagement plan (hijack guards) ──
 ("advisory/eng-plan", "Develop an engagement plan for attracting investment from India to Saudi Arabia",
   dict(minw=600, need=["phased"], reject=["multiple possible matches"])),
 ("advisory/eng-plan", "develop an engagement plan with Japan",
   dict(minw=500, reject=["multiple possible matches"])),
 ("advisory/eng-plan", "develop an engagement plan with Japan as a country",
   dict(minw=500, reject=["multiple possible matches"])),
 # ── Advisory: sector priorities ──
 ("advisory/sectors", "what are the top sectors that I should be focusing on for attracting investors from France",
   dict(minw=500, reject=["multiple possible matches"])),
 ("advisory/sectors", "top sectors for Germany",
   dict(minw=400)),
 # ── Target list (verb-suffix hijack guard) ──
 ("advisory/target-list", "give me the list of the best companies to be targeted from China with the investment thesis for each",
   dict(minw=500, reject=["multiple possible matches"])),
 ("advisory/target-list", "which investors should we be attracting from Korea",
   dict(minw=400, reject=["multiple possible matches"])),
 # ── Synthesis (no dead-end) ──
 ("advisory/synthesis", "Develop the dynamic between MNCs with market valuations exceeding $1 trillion and asset managers with AUM exceeding $1 trillion. How will the dynamic be reflected in the Strategic Capital Allocators",
   dict(minw=500, reject=["no record matching"])),
 ("advisory/synthesis", "if MISA wants more money flowing in from Brazil, what would be the smartest move?",
   dict(minw=400, reject=["no record matching", "multiple possible matches"])),
 # ── Macro trends (no EdTech fixation) ──
 ("advisory/trends", "what are the new global trends impacting the investment",
   dict(minw=400, reject=["edtech"])),
 # ── Data / FDI (attribution + no plumbing) ──
 ("data/fdi", "What is the size of outflow FDI from South Korea? How much is it inflow to Saudi Arabia?",
   dict(reject=["i couldn't directly look up", "needs a numeric id", "$501.8b sar"])),
 # ── Licensing counts (focus) ──
 ("count/licenses", "number of active licenses", dict(need=["95,671"])),
 ("count/licenses", "how many companies are licensed by MISA", dict(need=["95,671"])),
 ("count/rhq", "number of active RHQ licenses", dict(need=["727"])),
 ("count/rhq", "how many RHQ licenses do we have", dict(need=["727"])),
 # ── Country-specific licensing / company list ──
 ("list/country", "tell me the indian active companies",
   dict(need=["24 companies headquartered in India"], reject=["multiple possible matches"])),
 ("list/country", "how many total active licenses saudi has from india origin",
   dict(need=["24 companies headquartered in India"])),
 ("list/country", "list the German licensed firms", dict(reject=["multiple possible matches"])),
 # ── Country adjective (not company name) ──
 ("country/adjective", "which Pakistani companies have invested in Saudi Arabia",
   dict(reject=["no record matching \"pakistani\"", "**pakistani**"])),
 # ── Country profile + resolution ──
 ("country/profile", "tell me about Pakistan", dict(minw=200, need=["pakistan"])),
 ("country/profile", "tell me about South Korea", dict(minw=150, reject=["no record matching"])),
 # ── Entity / person / off-topic ──
 ("entity", "tell me about Alphabet", dict(minw=120, need=["alphabet"])),
 ("entity", "What is the China National IC fund (Big Fund)", dict(minw=100)),
 ("person", "tell me something about tim cook", dict(minw=80, need=["background"])),
 ("offtopic", "what is the capital of France", dict(need=["general knowledge"], reject=["greywolf"])),
 # ── No-match honesty ──
 ("nomatch", "tell me about Acme Foo Bar Nonexistent Corp", dict(need=["no record matching"])),
]

def check(a, exp):
    f = []; low = a.lower(); w = len(a.split())
    if w < exp.get("minw", 0): f.append(f"thin:{w}w")
    for n in exp.get("need", []):
        if n.lower() not in low: f.append(f"missing:{n!r}")
    for b in exp.get("reject", []):
        if b.lower() in low: f.append(f"reject:{b!r}")
    if re.search(r"[A-Za-z0-9)]\*\*[A-Z]", a): f.append("glued-bold")
    return w, f

print(f"COMPREHENSIVE FIX SWEEP — {len(C)} cases\n")
allok = True; bygroup = {}
for grp, q, exp in C:
    try:
        t0 = time.time(); a = ask(q); dt = time.time() - t0
        w, f = check(a, exp); ok = not f; allok = allok and ok
        bygroup.setdefault(grp, [0, 0]); bygroup[grp][0] += ok; bygroup[grp][1] += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {grp:22s} {w:5d}w {dt:4.0f}s  {q[:44]:44s} "
              + ("; ".join(f) if f else ""))
    except Exception as e:
        allok = False; print(f"[ERR ] {grp:22s} {type(e).__name__}: {e}  :: {q[:40]}")
print("\n--- by fix class ---")
for g, (p, t) in sorted(bygroup.items()):
    print(f"  {g:22s} {p}/{t}")
print("\n" + ("*** ALL FIXES VERIFIED ***" if allok else "*** SOME FAILED ***"))
