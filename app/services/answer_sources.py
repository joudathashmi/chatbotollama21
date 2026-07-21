"""Unified answer sources for world-class provenance in the UI.

Builds a single list the chat UI can render as a Sources panel:
  - type=document → uploaded library (doc://…)
  - type=web → live web URLs
  - type=db → MISA tables from the tool trace
"""

from __future__ import annotations

from typing import Any


def build_unified_sources(
    *,
    doc_sources: list | None = None,
    web_sources: list | None = None,
    tool_calls: list | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for s in doc_sources or []:
        url = (s.get("url") or "").strip()
        title = (s.get("title") or "Document").strip() or "Document"
        key = url or f"doc:{title}"
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "type": "document",
            "title": title,
            "url": url,
            "snippet": (s.get("snippet") or "").strip()[:400],
        })

    for s in web_sources or []:
        url = (s.get("url") or "").strip()
        if not url or url.startswith("doc://"):
            continue
        title = (s.get("title") or url).strip() or "(untitled)"
        key = url
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "type": "web",
            "title": title,
            "url": url,
            "snippet": (s.get("snippet") or "").strip()[:400],
        })

    for tc in tool_calls or []:
        table = (tc.get("table") or "").strip()
        if not table or table.startswith("_"):
            continue
        key = f"db:{table}"
        if key in seen:
            continue
        seen.add(key)
        rows = tc.get("row_count")
        snippet = f"{rows} row(s)" if isinstance(rows, int) else "MISA database"
        out.append({
            "type": "db",
            "title": table,
            "url": f"db://{table}",
            "snippet": snippet,
        })

    return out
