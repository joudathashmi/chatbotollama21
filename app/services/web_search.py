"""
Web search abstraction — backed by OpenAI's built-in web search.

We use OpenAI's `gpt-4o-search-preview` model (or `gpt-4o-mini-search-preview`
when configured for lower cost) to do live web grounding. This gives us:

  - NO new third-party provider account
  - NO API key beyond the OPENAI_API_KEY we already have
  - Privacy posture unchanged from our existing curation flow:
    the target name is sent to OpenAI, which is already happening
  - Citations come back as structured annotations on the message

Cost: a search-preview call costs more than a normal gpt-4o-mini call
(search overhead is charged separately). Budget rough estimate for
one /profile run: ~$0.01-0.03 with `gpt-4o-mini-search-preview`,
~$0.05-0.15 with `gpt-4o-search-preview`. Both well below what a
Tavily/Brave call costs end-to-end.

Latency: adds ~3-6 seconds to a /profile turn for the search round-trip.

Configurable via env:
  OPENAI_SEARCH_MODEL (default: gpt-4o-mini-search-preview)
    set to gpt-4o-search-preview for higher-quality grounding
"""

from __future__ import annotations

import os

from app.database import get_public_openai_client


# Stable contract: the public function is `search(query, max_results)`.
# Callers should not branch on provider — they get the same shape back
# regardless (or [] if the search call fails or no client is available).
#
# IMPORTANT: web search uses get_public_openai_client() (NOT the main
# Azure-aware client). Reason: Azure OpenAI does not yet host the
# `gpt-4o-mini-search-preview` model. The main client's deployment-name
# patch would silently rewrite the model to the configured Azure
# deployment — which has no web-search capability — so the call would
# succeed but return zero URLs, and the user would see "no reliable
# web sources" even for well-covered topics. Going to the public API
# preserves search functionality. Privacy posture is unchanged: web
# search only ever transmits the user's question / entity name, never
# DB rows.


def _search_model() -> str:
    """The OpenAI model used for web grounding. Mini variant by default
    to keep /profile cost down; override with OPENAI_SEARCH_MODEL=
    gpt-4o-search-preview for richer grounding."""
    return (os.getenv("OPENAI_SEARCH_MODEL")
            or "gpt-4o-mini-search-preview").strip()


def is_configured() -> bool:
    """True if public OpenAI is configured. Web search requires the
    public API even when the main pipeline is routed to Azure, because
    Azure doesn't host search-preview models yet."""
    return get_public_openai_client() is not None


def search(query: str, max_results: int = 5) -> list[dict]:
    """Back-compat list API. Prefer ``search_with_status`` when the
    caller must distinguish SOURCE_UNAVAILABLE from verified empty.
    """
    return list(search_with_status(query, max_results=max_results).get("results") or [])


def search_with_status(query: str, max_results: int = 5) -> dict:
    """Web search with retrieval-status envelope.

    Empty ``results`` + ``do_not_claim_zero`` means failure/unavailable,
    not a verified empty web census.
    """
    q = (query or "").strip()
    if not q:
        return {
            "results": [],
            "retrieval_status": "SUCCESS_EMPTY",
            "do_not_claim_zero": False,
            "source_name": "web_search",
            "error": None,
            "verified_empty": True,
            "record_count": 0,
        }
    client = get_public_openai_client()
    if client is None:
        try:
            rows = _ddg_search_fallback(q, max_results)
        except Exception as exc:
            return {
                "results": [],
                "retrieval_status": "SOURCE_UNAVAILABLE",
                "do_not_claim_zero": True,
                "counts_unavailable": True,
                "source_name": "web_search",
                "error": f"no search client; ddg fallback failed: {exc}"[:400],
                "record_count": 0,
            }
        return {
            "results": rows,
            "retrieval_status": (
                "SUCCESS_WITH_RESULTS" if rows else "SUCCESS_EMPTY"
            ),
            "do_not_claim_zero": False,
            "source_name": "web_search_ddg",
            "error": None,
            "verified_empty": not bool(rows),
            "record_count": len(rows or []),
        }
    try:
        rows = _openai_search(client, q, max_results)
        if not rows:
            # Belt-and-suspenders: empty OpenAI result → try keyless.
            rows = _ddg_search_fallback(q, max_results)
        return {
            "results": rows,
            "retrieval_status": (
                "SUCCESS_WITH_RESULTS" if rows else "SUCCESS_EMPTY"
            ),
            "do_not_claim_zero": False,
            "source_name": "web_search",
            "error": None,
            "verified_empty": not bool(rows),
            "record_count": len(rows or []),
        }
    except Exception as exc:
        # OpenAI search failed (e.g. 429 insufficient_quota) — keyless
        # fallback so time-sensitive answers still get live grounding.
        try:
            rows = _ddg_search_fallback(q, max_results)
            if rows:
                return {
                    "results": rows,
                    "retrieval_status": "SUCCESS_WITH_RESULTS",
                    "do_not_claim_zero": False,
                    "source_name": "web_search_ddg",
                    "error": None,
                    "verified_empty": False,
                    "record_count": len(rows),
                }
        except Exception:
            pass
        return {
            "results": [],
            "retrieval_status": "SOURCE_UNAVAILABLE",
            "do_not_claim_zero": True,
            "counts_unavailable": True,
            "source_name": "web_search",
            "error": str(exc)[:400],
            "record_count": 0,
        }




def _fetch_page_text(url: str, max_chars: int = 2500) -> str:
    """Fetch a public web page and return readable plain text.

    Turns thin search snippets into real article content so the Azure /
    local synthesis model writes from substance. Residency-safe: only a
    public URL is requested; no MISA data leaves the host. Best-effort —
    returns "" on any failure or non-HTML response.
    """
    import re
    import urllib.request

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "html" not in ctype and "text" not in ctype:
                return ""
            raw = resp.read(400_000).decode("utf-8", "replace")
    except Exception:
        return ""
    # Drop script/style/nav, then tags, collapse whitespace.
    raw = re.sub(r"(?is)<(script|style|noscript|header|footer|nav)[^>]*>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    import html as _html
    text = _html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _enrich_with_page_text(results: list, top_n: int = 3) -> None:
    """Attach real article text to the top results, in place."""
    n = 0
    for r in results:
        url = (r.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        body = _fetch_page_text(url)
        if body:
            r["snippet"] = body
            n += 1
        if n >= top_n:
            break


def _ddg_search_fallback(query: str, max_results: int) -> list[dict]:
    """Keyless web search via DuckDuckGo's lite endpoint.

    Fallback when the public OpenAI search client is unavailable or out
    of quota (429 insufficient_quota). Residency-safe: only the QUERY
    TEXT leaves the host — identical egress to the OpenAI search path —
    and the synthesis over these results still runs on the configured
    (Azure) model. No account or API key required.
    """
    import re
    import urllib.request
    import urllib.parse
    import html as _html

    url = (
        "https://lite.duckduckgo.com/lite/?q="
        + urllib.parse.quote((query or "").strip())
    )
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"},
    )
    body = urllib.request.urlopen(req, timeout=15).read().decode(
        "utf-8", "replace",
    )
    rows: list[dict] = []
    seen: set[str] = set()
    for href, title_html in re.findall(
        r'<a[^>]+href="(//duckduckgo\.com/l/\?uddg=[^"]+)"[^>]*>(.*?)</a>',
        body,
    ):
        m = re.search(r"uddg=([^&\"]+)", href)
        real = urllib.parse.unquote(m.group(1)) if m else ""
        title = _html.unescape(re.sub(r"<[^>]+>", "", title_html)).strip()
        if not real or not title or real in seen:
            continue
        seen.add(real)
        rows.append({
            "title": title,
            "url": real,
            "snippet": title,
            "source": "web",
        })
        if len(rows) >= max(1, min(max_results, 10)):
            break
    # Give the (Azure/local) synthesis model real article text, not just
    # titles — this is what closes most of the quality gap vs. the paid
    # search models.
    try:
        _enrich_with_page_text(rows)
    except Exception:
        pass
    return rows


def _openai_search(client, query: str, max_results: int) -> list[dict]:
    """OpenAI implementation using the gpt-4o-search-preview family.

    Returns:
      - Optional lead item with is_synthesis=True carrying the model's
        grounded prose (needed because annotation snippets are often
        bare URLs with no facts).
      - URL citation items for the Sources panel / [web:N] chips.
    """
    model = _search_model()
    n = max(1, min(max_results, 10))
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                f"Search the web for the latest, most authoritative "
                f"sources on the following query. Prioritise official "
                f"corporate / government sources, financial filings, and "
                f"top-tier journalism from the last 12 months when the "
                f"question is about who currently holds an office.\n\n"
                f"In your answer, put the CURRENT holder's full name in "
                f"the first sentence, and note any replacement / end date "
                f"for the previous holder if reported.\n\n"
                f"Cover up to {n} distinct sources.\n\n"
                f"Query: {query}"
            ),
        }],
        web_search_options={},  # required to enable the search tool
    )

    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    annotations = getattr(msg, "annotations", None) or []

    out: list[dict] = []
    seen_urls: set[str] = set()
    if content:
        out.append({
            "title": "Web search synthesis",
            "url": "",
            "snippet": content[:1200],
            "published": "",
            "score": 1.0,
            "is_synthesis": True,
        })
    for ann in annotations:
        ann_d = ann if isinstance(ann, dict) else ann.model_dump()
        if (ann_d.get("type") or "").lower() != "url_citation":
            continue
        cite = ann_d.get("url_citation") or {}
        url = (cite.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = (cite.get("title") or "").strip()
        start = cite.get("start_index") or 0
        end = cite.get("end_index") or 0
        snippet = ""
        if isinstance(start, int) and isinstance(end, int) and end > start:
            snippet = content[start:end].strip()
        if not snippet and content:
            lo = max(0, int(start) - 80) if isinstance(start, int) else 0
            hi = min(len(content), (int(end) + 160) if isinstance(end, int) else 400)
            snippet = content[lo:hi].strip() or content[:400].strip()
        out.append({
            "title":     title or "(untitled)",
            "url":       url,
            "snippet":   snippet[:600],
            "published": "",
            "score":     0.0,
        })
        if len([x for x in out if not x.get("is_synthesis")]) >= n:
            break
    return out


def format_for_prompt(results: list[dict]) -> str:
    """Render web-search results as a compact text block for prompts."""
    if not results:
        return "(no web results — OpenAI search unavailable or query returned nothing)"
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title") or "(untitled)"
        if r.get("is_synthesis"):
            head = f"[{i}] GROUNDED WEB ANSWER (use this for the current holder)"
        else:
            head = f"[{i}] {title}"
        lines.append(head)
        if r.get("url"):
            lines.append(f"    {r['url']}")
        if r.get("snippet"):
            lines.append(f"    {r['snippet']}")
        lines.append("")
    return "\n".join(lines).rstrip()
