"""
LinkedIn profile resolution for the business card reader.

Given an extracted business card (name, company, title, email, …), find the
person's LinkedIn profile URL. Pipeline:

    1. Direct extraction — a linkedin.com/in/ URL printed on the card wins
       outright (confidence 1.0, no search needed).
    2. Search — a pluggable provider (DuckDuckGo, Serper.dev, or a Playwright
       Bing scraper) runs several expanded queries concurrently and returns
       public SERP results; we keep only real /in/ profile URLs.
    3. Score & rank — rapidfuzz over name (from both the SERP title AND the
       URL slug), company, and title, plus an email-domain ↔ company
       corroboration boost.
    4. Decide — one confident match → "exact" (single URL); otherwise return
       the ranked array of candidate URLs; if NOTHING was found, fall back to
       LinkedIn people-search URL(s) built from the card. `linkedin_urls` is
       NEVER empty/null — there is always at least one usable link.

Providers follow a Strategy pattern (`LinkedInSearchProvider`) so the
extract → score → rank core is identical regardless of backend. The provider
is chosen per request; the network-bound search never breaks card extraction
(all failures degrade to a `match_type="search"` fallback result with an
`error`, never to an empty response).
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from abc import ABC, abstractmethod

from rapidfuzz import fuzz

from app.config import (
    BING_SEARCH_URL,
    LINKEDIN_EXACT_COMPANY_THRESHOLD,
    LINKEDIN_EXACT_NAME_THRESHOLD,
    LINKEDIN_MAX_QUERIES,
    LINKEDIN_MAX_RESULTS,
    LINKEDIN_SEARCH_TIMEOUT_SEC,
    SERPER_API_KEY,
    SERPER_BASE_URL,
)

logger = logging.getLogger(__name__)

# Matches a LinkedIn *profile* URL and captures the slug. Deliberately excludes
# /company/, /school/, /posts/ etc. — the client wants people, not org pages.
LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:[\w\-]+\.)?linkedin\.com/(?:in|pub)/([A-Za-z0-9\-_%\.]+)",
    re.IGNORECASE,
)

# Free/personal mail domains tell us nothing about the person's employer, so we
# must NOT use them for company corroboration (e.g. abdulfadhilm@gmail.com).
_GENERIC_EMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "ymail.com", "hotmail.com",
    "outlook.com", "live.com", "msn.com", "icloud.com", "me.com", "mac.com",
    "aol.com", "protonmail.com", "proton.me", "gmx.com", "zoho.com",
    "yandex.com", "mail.com", "qq.com", "163.com", "126.com",
})

_TOP_N_CANDIDATES = 5


# ===========================================================================
# URL / text helpers
# ===========================================================================

def normalize_linkedin_url(url: str | None) -> str | None:
    """Canonicalize to https://www.linkedin.com/in/<slug>, dropping query
    strings and trailing slashes. Returns None for non-profile URLs."""
    m = LINKEDIN_RE.search(url or "")
    if not m:
        return None
    slug = m.group(1).strip("/").split("?")[0].split("#")[0]
    if not slug:
        return None
    return f"https://www.linkedin.com/in/{slug}"


def _name_from_slug(url: str) -> str:
    """Derive a human name from the profile slug. 'abdul-fadhil-8b2a3c' →
    'abdul fadhil' (id-like tokens containing digits are dropped)."""
    m = LINKEDIN_RE.search(url or "")
    if not m:
        return ""
    slug = m.group(1)
    tokens = re.split(r"[-_]", slug)
    words = [t for t in tokens if t and not any(c.isdigit() for c in t)]
    return " ".join(words)


def _parse_title(title: str, body: str) -> tuple[str, str, str]:
    """Parse a LinkedIn SERP title like
    'Abdul Fadhil - Programmer Analyst - Cognizant | LinkedIn'
    into (name, job_title, company). Splits only on a dash/middot *surrounded
    by spaces* so hyphenated names like 'Jean-Paul' survive."""
    text = (title or "").split("|")[0]
    text = re.sub(r"(?i)\blinkedin\b", "", text)
    parts = [p.strip() for p in re.split(r"\s[-–—·•]\s", text) if p.strip()]
    name = parts[0] if parts else ""
    job = parts[1] if len(parts) > 1 else ""
    company = parts[2] if len(parts) > 2 else ""
    return name, job, company


def _brand_from_email(email: str | None) -> str:
    """Return the employer brand token from a non-generic email domain
    ('john@cognizant.com' → 'cognizant'), or '' for personal/empty emails."""
    if not email or "@" not in email:
        return ""
    domain = email.rsplit("@", 1)[-1].strip().lower()
    if not domain or domain in _GENERIC_EMAIL_DOMAINS:
        return ""
    return domain.split(".")[0]


# ===========================================================================
# Candidate model
# ===========================================================================

class _Candidate:
    __slots__ = ("url", "title", "body", "score", "p_name", "p_title", "p_company")

    def __init__(self, url: str, title: str, body: str):
        self.url = url
        self.title = title
        self.body = body
        self.score = 0.0
        self.p_name = ""
        self.p_title = ""
        self.p_company = ""

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "score": self.score,
            "name": self.p_name,
            "title": self.p_title,
            "company": self.p_company,
        }


# ===========================================================================
# Search providers (Strategy pattern)
# ===========================================================================

def _card_search_terms(card: dict) -> tuple[str, str, str, str, str]:
    """Pull (name, company, title, email-brand, location) from the card."""
    name = (card.get("name") or "").strip()
    company = (card.get("company") or "").strip()
    title = (card.get("title") or "").strip()
    brand = _brand_from_email(card.get("email"))
    addr = card.get("address") or {}
    location = (addr.get("city") or addr.get("country") or "").strip()
    return name, company, title, brand, location


def _dedup_cap(queries: list[str]) -> list[str]:
    """Dedup (case-insensitive) preserving order, drop blanks, cap at
    LINKEDIN_MAX_QUERIES."""
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out[:LINKEDIN_MAX_QUERIES]


def _build_queries(card: dict) -> list[str]:
    """Generic Google/Bing-style query set (used by the Playwright/Bing
    provider and as the base default). DuckDuckGo and Serper each override
    this with their own engine-tuned variants — see their build_queries()."""
    name, company, title, brand, location = _card_search_terms(card)
    if not name:
        return []
    queries: list[str] = []
    if company:
        queries.append(f'site:linkedin.com/in/ "{name}" "{company}"')
        queries.append(f'"{name}" "{company}" LinkedIn')
    if title:
        queries.append(f'site:linkedin.com/in/ "{name}" "{title}"')
    if brand and brand.lower() not in company.lower():
        queries.append(f'site:linkedin.com/in/ "{name}" {brand}')
    if location:
        queries.append(f'"{name}" "{location}" LinkedIn')
    queries.append(f"site:linkedin.com/in/ {name} {company or title}".strip())
    queries.append(f"{name} LinkedIn")
    return _dedup_cap(queries)


def _merge_query_results(results: list, provider_name: str) -> list[dict]:
    """Flatten gathered per-query results; log (don't raise) individual
    failures so one bad query never sinks the whole resolution."""
    hits: list[dict] = []
    for r in results:
        if isinstance(r, BaseException):
            logger.warning("%s query failed: %s", provider_name, r)
        else:
            hits.extend(r)
    return hits


class LinkedInSearchProvider(ABC):
    """Builds its own engine-tuned queries from the card and runs them →
    raw (url, title, body) hits. Each backend is fully self-contained: the
    DuckDuckGo path never touches Serper code and vice versa."""

    name: str = "base"

    def build_queries(self, card: dict) -> list[str]:
        """Engine-tuned query set. Subclasses override; default is generic."""
        return _build_queries(card)

    @abstractmethod
    async def search(self, card: dict, max_results: int) -> list[dict]:
        ...


class DuckDuckGoProvider(LinkedInSearchProvider):
    """Free backend via the `ddgs` library. The library is synchronous, so each
    query runs in a worker thread and all queries run concurrently. Self-
    contained: builds DuckDuckGo-tuned queries and never imports Serper code."""

    name = "ddg"

    def build_queries(self, card: dict) -> list[str]:
        """DuckDuckGo-tuned queries. DDG's relevance favours a mix of one
        site-scoped exact query plus plain `name term linkedin` phrasings
        (heavy operator stacking hurts DDG recall)."""
        name, company, title, brand, location = _card_search_terms(card)
        if not name:
            return []
        queries: list[str] = []
        if company:
            queries.append(f'site:linkedin.com/in/ "{name}" "{company}"')
            queries.append(f"{name} {company} linkedin")
        if title:
            queries.append(f"{name} {title} linkedin")
        if brand and brand.lower() not in company.lower():
            queries.append(f"{name} {brand} linkedin")
        if location:
            queries.append(f"{name} {location} linkedin")
        queries.append(f"{name} linkedin")
        return _dedup_cap(queries)

    async def search(self, card: dict, max_results: int) -> list[dict]:
        queries = self.build_queries(card)
        results = await asyncio.gather(
            *(asyncio.to_thread(self._run_query, q, max_results) for q in queries),
            return_exceptions=True,
        )
        return _merge_query_results(results, self.name)

    def _run_query(self, query: str, max_results: int) -> list[dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            raise RuntimeError(
                "ddgs is not installed — `pip install ddgs` or use another provider."
            )
        hits: list[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                hits.append({
                    "url": r.get("href") or r.get("url") or "",
                    "title": r.get("title", "") or "",
                    "body": r.get("body") or r.get("snippet") or "",
                })
        return hits


class SerperProvider(LinkedInSearchProvider):
    """Paid, reliable backend via Serper.dev's Google Search API. All queries
    fire concurrently over one shared HTTP client. Self-contained: builds
    Google-tuned queries and never touches the DuckDuckGo path."""

    name = "serp"

    def build_queries(self, card: dict) -> list[str]:
        """Google-tuned queries. Google handles stacked `site:` + quoted exact
        operators extremely well, so we lean on precise site-scoped phrasing."""
        name, company, title, brand, location = _card_search_terms(card)
        if not name:
            return []
        queries: list[str] = []
        if company:
            queries.append(f'site:linkedin.com/in/ "{name}" "{company}"')
            queries.append(f'"{name}" "{company}" site:linkedin.com')
        if title:
            queries.append(f'site:linkedin.com/in/ "{name}" "{title}"')
        if brand and brand.lower() not in company.lower():
            queries.append(f'site:linkedin.com/in/ "{name}" {brand}')
        if location:
            queries.append(f'site:linkedin.com/in/ "{name}" "{location}"')
        queries.append(f"site:linkedin.com/in/ {name} {company or title}".strip())
        return _dedup_cap(queries)

    async def search(self, card: dict, max_results: int) -> list[dict]:
        if not SERPER_API_KEY:
            raise RuntimeError("SERPER_API_KEY is not configured (provider=serp).")
        import httpx

        queries = self.build_queries(card)
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=LINKEDIN_SEARCH_TIMEOUT_SEC) as client:
            async def one(query: str) -> list[dict]:
                resp = await client.post(
                    SERPER_BASE_URL, headers=headers,
                    json={"q": query, "num": max_results},
                )
                resp.raise_for_status()
                return [
                    {"url": o.get("link", "") or "", "title": o.get("title", "") or "",
                     "body": o.get("snippet", "") or ""}
                    for o in resp.json().get("organic", [])
                ]

            results = await asyncio.gather(*(one(q) for q in queries), return_exceptions=True)
        return _merge_query_results(results, self.name)


class PlaywrightProvider(LinkedInSearchProvider):
    """Headless-Chromium backend that scrapes Bing search result pages. Yields
    more profiles + richer snippets than the lightweight APIs. One browser +
    context is reused across all queries, which run on concurrent pages.

    IMPORTANT: this scrapes the SEARCH ENGINE only — it never navigates to
    linkedin.com itself (that would breach LinkedIn's ToS and get blocked)."""

    name = "playwright"

    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    async def search(self, card: dict, max_results: int) -> list[dict]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "playwright is not installed — `pip install playwright && "
                "playwright install chromium`, or use provider=ddg/serp."
            )
        queries = self.build_queries(card)
        async with async_playwright() as p:
            # Best-effort anti-automation flags. NOTE: search engines still
            # CAPTCHA/headless-block aggressively; reliable production results
            # come from provider=ddg (ddgs lib) or provider=serp (Serper API).
            browser = await p.chromium.launch(
                headless=True, args=["--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                user_agent=self._UA, locale="en-US",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            await context.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
            )
            try:
                results = await asyncio.gather(
                    *(self._scrape(context, q, max_results) for q in queries),
                    return_exceptions=True,
                )
            finally:
                await context.close()
                await browser.close()
        return _merge_query_results(results, self.name)

    # Extract all results in one in-page pass. Doing it via evaluate() (rather
    # than holding ElementHandles and awaiting inner_text on each) avoids the
    # "execution context was destroyed" error when Bing client-side-navigates,
    # and is much faster (one round-trip instead of N).
    # Extract all results in one in-page pass. Doing it via evaluate() (rather
    # than holding ElementHandles and awaiting inner_text on each) avoids the
    # "execution context was destroyed" error when Bing client-side-navigates,
    # and is much faster. Falls back to scraping any linkedin.com/in/ anchor on
    # the page if Bing's standard result markup (li.b_algo) isn't present.
    _EXTRACT_JS = """() => {
        const out = [];
        const algos = document.querySelectorAll('li.b_algo');
        if (algos.length) {
            algos.forEach(li => {
                const a = li.querySelector('h2 a');
                if (!a) return;
                const cap = li.querySelector('.b_caption p') || li.querySelector('.b_caption');
                out.push({url: a.href || '', title: a.innerText || '',
                          body: cap ? (cap.innerText || '') : ''});
            });
        } else {
            document.querySelectorAll('a[href*="linkedin.com/in"]').forEach(a => {
                out.push({url: a.href || '', title: a.innerText || '', body: ''});
            });
        }
        return out;
    }"""

    async def _scrape(self, context, query: str, max_results: int) -> list[dict]:
        from urllib.parse import quote_plus

        page = await context.new_page()
        try:
            url = f"{BING_SEARCH_URL}?q={quote_plus(query)}&count={max_results}&setlang=en"
            await page.goto(
                url, timeout=int(LINKEDIN_SEARCH_TIMEOUT_SEC * 1000),
                wait_until="domcontentloaded",
            )
            # Wait for either the results list or any LinkedIn anchor; tolerate
            # absence (no results / CAPTCHA / alternate layout) rather than raise.
            try:
                await page.wait_for_selector(
                    'li.b_algo, a[href*="linkedin.com/in"]', timeout=5000
                )
            except Exception:
                return []
            return await page.evaluate(self._EXTRACT_JS)
        finally:
            await page.close()


_PROVIDERS = {
    "ddg": DuckDuckGoProvider,
    "serp": SerperProvider,
    "playwright": PlaywrightProvider,
}


def _get_provider(provider: str) -> LinkedInSearchProvider:
    return _PROVIDERS.get((provider or "").lower(), DuckDuckGoProvider)()


# ===========================================================================
# Fuzzy matching — normalization + field scorers
#
# The quality of LinkedIn resolution lives here. Search engines return noisy,
# inconsistent strings (diacritics, initials, middle names, legal suffixes,
# title abbreviations), so before any rapidfuzz comparison we normalize both
# sides, then blend several scorers per field. Each field scorer returns 0–1.
# ===========================================================================

# Honorifics / name prefixes that carry no identity signal.
_HONORIFICS = frozenset({
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor", "sir", "madam",
    "eng", "engr", "ar", "adv", "hon", "rev", "capt", "col", "lt",
})

# Corporate/legal suffixes & filler stripped before company comparison.
_COMPANY_STOPWORDS = frozenset({
    "inc", "incorporated", "llc", "ltd", "limited", "co", "corp", "corporation",
    "company", "plc", "pvt", "private", "lp", "llp", "gmbh", "ag", "sa", "sas",
    "bv", "nv", "oy", "ab", "spa", "srl", "pte", "pty", "kk", "group", "holdings",
    "holding", "technologies", "technology", "solutions", "systems", "labs",
    "international", "global", "worldwide", "services", "consulting", "ventures",
    "partners", "industries", "the", "and", "&",
})

# Job-title abbreviation expansions, applied token-wise.
_TITLE_SYNONYMS = {
    "vp": "vice president", "svp": "senior vice president",
    "evp": "executive vice president", "avp": "assistant vice president",
    "sr": "senior", "snr": "senior", "jr": "junior", "jnr": "junior",
    "mgr": "manager", "mgmt": "management", "dir": "director", "dept": "department",
    "eng": "engineer", "engr": "engineer", "dev": "developer", "arch": "architect",
    "admin": "administrator", "asst": "assistant", "assoc": "associate",
    "exec": "executive", "ceo": "chief executive officer",
    "cto": "chief technology officer", "cfo": "chief financial officer",
    "coo": "chief operating officer", "cmo": "chief marketing officer",
    "ciso": "chief information security officer", "hr": "human resources",
    "it": "information technology", "pm": "product manager", "sw": "software",
    "qa": "quality assurance", "ux": "user experience",
}


def _strip_diacritics(s: str) -> str:
    """'José' → 'Jose', 'Müller' → 'Muller'."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _normalize_text(s: str | None) -> str:
    """Lowercase, de-accent, punctuation → space, collapse whitespace."""
    s = _strip_diacritics(s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _name_tokens(name: str | None) -> list[str]:
    """Normalized name tokens with honorifics removed."""
    return [t for t in _normalize_text(name).split() if t not in _HONORIFICS]


def _normalize_name(name: str | None) -> str:
    return " ".join(_name_tokens(name))


def _normalize_company(company: str | None) -> str:
    """Drop legal suffixes / filler. Falls back to the plain normalized form
    if stripping would empty the string (e.g. company literally named 'Group')."""
    base = _normalize_text(company)
    tokens = [t for t in base.split() if t not in _COMPANY_STOPWORDS]
    return " ".join(tokens) or base


def _expand_title(title: str | None) -> str:
    return " ".join(_TITLE_SYNONYMS.get(t, t) for t in _normalize_text(title).split())


def _initials_form_score(t1: list[str], t2: list[str]) -> int:
    """Score for initial-vs-full-name forms sharing a surname.
    'S Nadella' ↔ 'Satya Nadella' → 95; exact first names → 100.
    Requires matching last token (surname) to avoid false hits."""
    if len(t1) < 2 or len(t2) < 2 or t1[-1] != t2[-1]:
        return 0
    a, b = t1[0], t2[0]
    if a == b:
        return 100
    if (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b)):
        return 95
    return 0


def _name_score(card_name: str, candidate_names: list[str]) -> float:
    """Best 0–1 name match across candidate name sources (URL slug, SERP
    title). Blends token-sort (order-insensitive), token-set (handles middle
    names / extra tokens), spaceless ratio (separator-less slugs), and an
    initials-aware scorer."""
    n1 = _normalize_name(card_name)
    if not n1:
        return 0.0
    t1 = n1.split()
    n1_ns = n1.replace(" ", "")
    best = 0
    for cand in candidate_names:
        n2 = _normalize_name(cand)
        if not n2:
            continue
        t2 = n2.split()
        best = max(
            best,
            fuzz.token_sort_ratio(n1, n2),
            fuzz.token_set_ratio(n1, n2),
            fuzz.ratio(n1_ns, n2.replace(" ", "")),
            _initials_form_score(t1, t2),
        )
        if best >= 100:
            break
    return best / 100


def _company_score(card_company: str, parsed_company: str, blob: str) -> float:
    """0–1 company match. Compares suffix-stripped forms; also credits a clean
    containment of the company in the result blob (title/snippet/url)."""
    a = _normalize_company(card_company)
    if not a:
        return 0.0
    b = _normalize_company(parsed_company)
    score = fuzz.token_set_ratio(a, b) if b else 0
    # Company often appears in the snippet even when title parsing missed it.
    if a and a in _normalize_text(blob):
        score = max(score, 90)
    return score / 100


def _title_score(card_title: str, parsed_title: str) -> float:
    a = _expand_title(card_title)
    b = _expand_title(parsed_title)
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100


def _brand_in_text(brand: str, text: str) -> bool:
    """Whether the email brand appears in text. Short brands (≤3 chars, e.g.
    'hp', 'ge', 'ibm') require a word boundary to avoid false positives."""
    if not brand:
        return False
    if len(brand) <= 3:
        return re.search(rf"\b{re.escape(brand)}\b", text) is not None
    return brand in text


# ===========================================================================
# Scoring
# ===========================================================================

# Field weights for the blended score (sum to 1.0). Name dominates; title is
# a weak tiebreaker. Bonuses are added on top, then the score is clamped.
_W_NAME = 0.55
_W_COMPANY = 0.30
_W_TITLE = 0.15
_CORROBORATION_BONUS = 0.08
_LOCATION_BONUS = 0.02


def _score_candidate(card: dict, cand: _Candidate) -> tuple[float, float, float]:
    """Returns (overall_score, name_score, company_score), each 0–1, and
    populates the candidate's parsed name/title/company as a side effect."""
    slug_name = _name_from_slug(cand.url)
    p_name, p_title, p_company = _parse_title(cand.title, cand.body)
    cand.p_name, cand.p_title, cand.p_company = p_name, p_title, p_company

    blob = f"{cand.url} {cand.title} {cand.body}"

    name_score = _name_score(card.get("name") or "", [slug_name, p_name])
    company_score = _company_score(card.get("company") or "", p_company, blob)
    title_score = _title_score(card.get("title") or "", p_title)

    score = _W_NAME * name_score + _W_COMPANY * company_score + _W_TITLE * title_score

    # Email-domain ↔ company corroboration: if the employer brand from a
    # corporate email shows up in the slug/title/body, it's a strong signal.
    brand = _brand_from_email(card.get("email"))
    corroborated = _brand_in_text(brand, blob.lower())
    if corroborated:
        score += _CORROBORATION_BONUS
        company_score = max(company_score, 0.9)

    # Location tiebreaker: card city/country appearing in the snippet.
    addr = card.get("address") or {}
    blob_norm = _normalize_text(blob)
    for loc in (addr.get("city"), addr.get("country")):
        loc_n = _normalize_text(loc)
        if loc_n and len(loc_n) > 2 and loc_n in blob_norm:
            score += _LOCATION_BONUS

    cand.score = min(round(score, 4), 1.0)
    return cand.score, name_score, company_score


# ===========================================================================
# Public entry point
# ===========================================================================

def extract_direct_linkedin(card: dict) -> str | None:
    """Find a LinkedIn URL already printed on the card (raw_text, website,
    other lines …). Returns the normalized URL or None."""
    fields = [
        card.get("raw_text", ""),
        card.get("website", ""),
        card.get("email", ""),
        card.get("full_address", ""),
    ]
    fields.extend(card.get("other", []) or [])
    for field in fields:
        if not field:
            continue
        url = normalize_linkedin_url(str(field))
        if url:
            return url
    return None


def _linkedin_search_urls(card: dict) -> list[str]:
    """Build LinkedIn people-search URL(s) from the card. Used as a last-resort
    fallback so the response is NEVER empty — the client always gets at least a
    clickable LinkedIn search to find the person manually."""
    from urllib.parse import quote_plus

    name, company, _title, _brand, _loc = _card_search_terms(card)
    urls: list[str] = []
    combined = " ".join(t for t in (name, company) if t).strip()
    if combined:
        urls.append(
            "https://www.linkedin.com/search/results/people/?keywords="
            + quote_plus(combined)
        )
    if name and company:
        # Broader name-only search as a second option.
        urls.append(
            "https://www.linkedin.com/search/results/people/?keywords="
            + quote_plus(name)
        )
    if not urls:
        urls.append("https://www.linkedin.com/search/results/people/")
    # Dedup preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _search_fallback_result(card: dict, provider: str, error: str | None = None) -> dict:
    """Non-null fallback: when no profile candidate is found, return relevant
    LinkedIn people-search candidates instead of an empty array."""
    urls = _linkedin_search_urls(card)
    entry = {
        "url": urls[0] if urls else "",
        "score": 0.0,
        "name": card.get("name") or "",
        "title": card.get("title") or "",
        "company": card.get("company") or "",
    }
    return {
        "match_type": "search",
        "linkedin_urls": entry,
        "confidence": 0.0,
        "provider": provider,
        "candidates": entry,
        "error": error,
    }


async def resolve_linkedin(card: dict, provider: str = "ddg") -> dict:
    """Resolve LinkedIn profile(s) for an extracted card. Always returns a
    dict matching the LinkedInResult schema; never raises and never returns an
    empty `linkedin_urls` — an exact hit yields one profile candidate, otherwise
    the ranked candidate profiles, otherwise LinkedIn people-search candidates."""
    # Step 1 — direct URL on the card wins outright.
    direct = extract_direct_linkedin(card)
    if direct:
        entry = {
            "url": direct,
            "match_type": "direct",
            "score": 1.0,
            "name": card.get("name") or "",
            "title": card.get("title") or "",
            "company": card.get("company") or "",
        }
        return {
            "match_type": "direct",
            "linkedin_urls": entry,
            "confidence": 1.0,

            "provider": "business_card",
            "candidates": [entry],
            "error": None,
        }


    name = (card.get("name") or "").strip()
    if not name:
        return _search_fallback_result(card, provider or "ddg", "No name on card to search with.")

    # Step 2 — search via the chosen provider. Each provider builds and runs
    # its OWN engine-tuned queries; the DDG and SERP paths are fully separate.
    backend = _get_provider(provider)
    try:
        raw_hits = await backend.search(card, LINKEDIN_MAX_RESULTS)
    except Exception as ex:
        logger.warning("LinkedIn search failed (provider=%s): %s", backend.name, ex)
        return _search_fallback_result(card, backend.name, str(ex))

    # Dedup to real /in/ profile URLs.
    seen: set[str] = set()
    candidates: list[_Candidate] = []
    for hit in raw_hits:
        url = normalize_linkedin_url(hit.get("url"))
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append(_Candidate(url, hit.get("title", ""), hit.get("body", "")))

    if not candidates:
        return _search_fallback_result(card, backend.name)

    # Step 3 — score & rank. Keep each candidate's name/company sub-scores so
    # the exact-match decision below doesn't have to re-score.
    sub_scores: dict[str, tuple[float, float]] = {}
    for cand in candidates:
        _, ns, cs = _score_candidate(card, cand)
        sub_scores[cand.url] = (ns, cs)
    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    best_name_score, best_company_score = sub_scores[best.url]

    # Step 4 — decide exact vs candidate list.
    brand = _brand_from_email(card.get("email"))
    corroborated = bool(brand) and brand in f"{best.url} {best.title} {best.body}".lower()
    is_exact = (
        best_name_score >= LINKEDIN_EXACT_NAME_THRESHOLD
        and (best_company_score >= LINKEDIN_EXACT_COMPANY_THRESHOLD or corroborated)
    )

    top = candidates[:_TOP_N_CANDIDATES]
    top_entries = [t.as_dict() | {"match_type": "exact"} for t in top] if top else [
        {
            "url": "",
            "match_type": "",
            "score": 0.0,
            "name": "",
            "title": "",
            "company": "",
        }
    ]

    top_entry = top_entries[0]

    if is_exact:
        return {
            "match_type": "exact",
            "linkedin_urls": top_entry,
            "confidence": best.score,
            "provider": backend.name,
            "candidates": top_entries,
            "error": None,
        }

    return {
        "match_type": "candidates",
        "linkedin_urls": top_entry,
        "confidence": best.score,
        "provider": backend.name,
        "candidates": top_entries,
        "error": None,
    }


