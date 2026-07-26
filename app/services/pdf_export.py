"""
PDF export.

Turns an executive-briefing answer (markdown) into a print-ready PDF.
Pipeline: normalize markdown tables → HTML (cards for wide tables) →
sanitize → xhtml2pdf.

Fixes applied for the India-targeting regression:
  - Pre-parse markdown tables instead of hoping Python-Markdown + nl2br
    leave pipes intact
  - Wide / long-cell tables become profile cards (no overlapping columns)
  - CSS uses word-wrap, header repetition, and row break avoidance
  - Strip leaked raw Markdown (`##`, `|`) after conversion
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from html import escape as _html_escape
from typing import Optional

import markdown
from xhtml2pdf import pisa

from app.utils.text_validation import sanitize_html


_PDF_CSS = """
@page {
  size: A4;
  margin: 1.6cm 1.5cm 2.4cm 1.5cm;
  @frame footer {
    -pdf-frame-content: footer-content;
    bottom: 0.8cm;
    margin-left: 1.5cm;
    margin-right: 1.5cm;
    height: 1cm;
  }
}
@page landscape_table {
  size: A4 landscape;
  margin: 1.2cm 1.2cm 2.0cm 1.2cm;
}
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt;
       color: #1d1d1f; line-height: 1.4; }
.letterhead { border-bottom: 2px solid #006c35; padding-bottom: 6pt;
              margin-bottom: 14pt; }
.letterhead .brand { font-size: 14pt; font-weight: bold; color: #006c35;
                     letter-spacing: 0.4pt; }
.letterhead .meta { font-size: 9pt; color: #6e6e73; margin-top: 2pt; }
.question { font-size: 12pt; font-weight: bold; margin: 0 0 12pt 0;
            color: #1d1d1f; }
.question-sub { font-size: 9pt; color: #6e6e73; font-weight: normal;
            margin: -8pt 0 12pt 0; font-style: italic; }
.answer h1 { font-size: 13pt; color: #006c35; margin: 0 0 10pt 0; }
.answer h2 { font-size: 12pt; color: #006c35; margin: 14pt 0 6pt 0;
             border-bottom: 1px solid #e5e5e7; padding-bottom: 3pt; }
.answer h3 { font-size: 11pt; color: #1d1d1f; margin: 10pt 0 4pt 0; }
.answer p { margin: 0 0 6pt 0; }
.answer ul, .answer ol { margin: 4pt 0 8pt 16pt; padding-left: 0; }
.answer li { margin: 2pt 0; }
.answer table.data-table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  margin: 6pt 0 10pt 0;
  font-size: 8.5pt;
}
.answer table.data-table thead { display: table-header-group; }
.answer table.data-table tr { break-inside: avoid; page-break-inside: avoid; }
.answer table.data-table th,
.answer table.data-table td {
  border: 1px solid #d2d2d7;
  padding: 4pt 5pt;
  text-align: left;
  vertical-align: top;
  overflow-wrap: anywhere;
  word-break: normal;
  word-wrap: break-word;
}
.answer table.data-table th {
  background-color: #f5f5f7;
  font-weight: bold;
  font-size: 8pt;
}
.answer .profile-card {
  border: 1px solid #d2d2d7;
  padding: 8pt 10pt;
  margin: 0 0 8pt 0;
  break-inside: avoid;
  page-break-inside: avoid;
  background: #fafafa;
}
.answer .profile-card .card-title {
  font-weight: bold; color: #006c35; margin: 0 0 4pt 0; font-size: 10pt;
}
.answer .profile-card .card-row { margin: 2pt 0; font-size: 9pt; }
.answer .profile-card .label { color: #6e6e73; font-weight: bold; }
.answer code { font-family: monospace; background: #f5f5f7;
               padding: 1pt 3pt; font-size: 9pt; }
.answer blockquote { border-left: 3pt solid #006c35; padding-left: 8pt;
                     margin: 6pt 0; color: #4a4a4f; font-style: italic; }
.answer strong { color: #1d1d1f; }
.sources { margin-top: 16pt; padding-top: 8pt;
           border-top: 1px solid #e5e5e7; font-size: 9pt; }
.sources .head { font-weight: bold; color: #1d1d1f; margin-bottom: 4pt; }
.sources ol { margin: 0 0 0 18pt; padding-left: 0; }
.sources li { margin: 2pt 0; color: #4a4a4f; }
.sources a { color: #0a73f0; text-decoration: none; }
.footer-content { font-size: 8pt; color: #86868b; text-align: center; }
"""


def _escape(s: str) -> str:
    return _html_escape(s or "", quote=True)


def _render_sources_html(web_sources: Optional[list[dict]]) -> str:
    if not web_sources:
        return ""
    items = []
    for s in web_sources:
        title = (s.get("title") or s.get("url") or "").strip()
        url = (s.get("url") or "").strip()
        if not url:
            continue
        snippet = (s.get("snippet") or "").strip()
        snippet = snippet[:200] + ("…" if len(snippet) > 200 else "")
        items.append(
            f'<li><a href="{_escape(url)}">{_escape(title)}</a>'
            + (f'<br/><span style="color:#6e6e73;">{_escape(snippet)}</span>'
               if snippet else "")
            + "</li>"
        )
    if not items:
        return ""
    return (
        '<div class="sources">'
        '<div class="head">Sources</div>'
        '<ol>' + "".join(items) + '</ol>'
        '</div>'
    )


_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_sep_row(line: str) -> bool:
    return bool(_TABLE_SEP_RE.match(line.strip()))


def _parse_markdown_table(lines: list[str], start: int) -> tuple[list[list[str]], int] | None:
    """Return (rows, next_index) if lines[start:] begins a markdown table."""
    if start >= len(lines) or not _PIPE_ROW_RE.match(lines[start]):
        return None
    header = _split_row(lines[start])
    if len(header) < 2:
        return None
    i = start + 1
    if i >= len(lines) or not _is_sep_row(lines[i]):
        # Tolerate missing separator if next lines look like rows.
        if i >= len(lines) or not _PIPE_ROW_RE.match(lines[i]):
            return None
    else:
        i += 1
    rows = [header]
    while i < len(lines) and _PIPE_ROW_RE.match(lines[i]):
        if _is_sep_row(lines[i]):
            i += 1
            continue
        cells = _split_row(lines[i])
        # Pad / trim to header width
        if len(cells) < len(header):
            cells = cells + [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[: len(header) - 1] + [
                " ".join(cells[len(header) - 1 :])
            ]
        rows.append(cells)
        i += 1
    if len(rows) < 2:
        return None
    return rows, i


def _drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows
    ncols = len(rows[0])
    body = rows[1:] if len(rows) > 1 else []
    keep = []
    for c in range(ncols):
        # Keep column if any body cell has content; if no body, keep
        # header-only columns that look substantive.
        if body:
            if any((r[c] if c < len(r) else "").strip() for r in body):
                keep.append(c)
        elif (rows[0][c] if c < len(rows[0]) else "").strip():
            keep.append(c)
    if not keep:
        return rows
    return [[(r[c] if c < len(r) else "") for c in keep] for r in rows]


def _table_should_use_cards(rows: list[list[str]]) -> bool:
    cols = len(rows[0]) if rows else 0
    if cols >= 7:
        return True
    # Long thesis-like cells in a 5–6 column table still break xhtml2pdf.
    body = rows[1:]
    if cols >= 5 and body:
        avg = sum(len(c) for r in body for c in r) / max(1, len(body) * cols)
        longest = max((len(c) for r in body for c in r), default=0)
        if avg > 55 or longest > 180:
            return True
    return False


def _render_table_html(rows: list[list[str]]) -> str:
    rows = _drop_empty_columns(rows)
    if _table_should_use_cards(rows):
        return _render_cards_html(rows)
    header = rows[0]
    # Allocate widths: first col narrow if Rank-like
    n = len(header)
    widths: list[str] = []
    for i, h in enumerate(header):
        hl = h.lower()
        if i == 0 and ("rank" in hl or h.strip() in {"#", "No", "No."}):
            widths.append("6%")
        elif "company" in hl or "organisation" in hl or "organization" in hl:
            widths.append("16%")
        elif "thesis" in hl or "investment" in hl or "action" in hl:
            widths.append(f"{max(14, int(40 / max(1, n - 3)))}%")
        else:
            widths.append(f"{max(10, int(70 / n))}%")
    # Normalize to ~100%
    # (xhtml2pdf is happier with explicit % on col / th)
    thead = "<thead><tr>" + "".join(
        f'<th width="{widths[i] if i < len(widths) else "10%"}">'
        f"{_escape(h)}</th>"
        for i, h in enumerate(header)
    ) + "</tr></thead>"
    body_rows = []
    for r in rows[1:]:
        tds = "".join(f"<td>{_escape(c)}</td>" for c in r)
        body_rows.append(f"<tr>{tds}</tr>")
    return (
        '<table class="data-table" width="100%">'
        + thead
        + "<tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def _render_cards_html(rows: list[list[str]]) -> str:
    header = rows[0]
    cards = []
    for r in rows[1:]:
        title = r[1] if len(r) > 1 and "company" in header[0].lower() + header[1].lower() else (r[0] if r else "Target")
        # Prefer Company column
        for i, h in enumerate(header):
            if "company" in h.lower() and i < len(r) and r[i].strip():
                title = r[i]
                break
        parts = [f'<div class="card-title">{_escape(title)}</div>']
        for i, h in enumerate(header):
            if i >= len(r):
                continue
            val = r[i].strip()
            if not val:
                continue
            if h.lower() in ("company", "rank") and r[i] == title:
                if h.lower() == "company":
                    continue
            parts.append(
                f'<div class="card-row"><span class="label">{_escape(h)}: '
                f"</span>{_escape(val)}</div>"
            )
        cards.append('<div class="profile-card">' + "".join(parts) + "</div>")
    return "".join(cards)


def normalize_answer_markdown_for_pdf(answer_markdown: str) -> str:
    """Replace markdown pipe-tables with HTML tables/cards; tidy headings."""
    text = answer_markdown or ""
    # Demote leaked raw heading markers that failed markdown parse later
    # are handled after HTML; here ensure ATX headings have a space.
    text = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", text, flags=re.M)

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        parsed = _parse_markdown_table(lines, i)
        if parsed:
            rows, nxt = parsed
            out.append(_render_table_html(rows))
            out.append("")
            i = nxt
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _strip_leaked_markdown(html: str) -> str:
    """Remove obvious raw markdown artefacts that survived conversion."""
    # Headings that were never converted (paragraph starting with ## )
    html = re.sub(
        r"(<(?:p|li|td|th)[^>]*>)\s*#{1,6}\s+",
        r"\1",
        html,
        flags=re.I,
    )
    # Lone pipe-table remnants in paragraphs
    html = re.sub(
        r"<p>\s*\|([^\n|]{0,80}\|){2,}\s*</p>",
        "",
        html,
        flags=re.I,
    )
    return html


def _structured_to_markdown(model) -> str:
    """Render a validated QualityResponse into controlled Markdown."""
    lines: list[str] = []
    if model.title:
        lines += [f"## {model.title}", ""]
    if model.executive_summary:
        lines += ["### Executive Summary", ""]
        for s in model.executive_summary:
            lines.append(f"- {s}")
        lines.append("")
    if model.facts:
        lines += ["### Facts", ""]
        for f in model.facts:
            tag = getattr(f.verification_status, "value", f.verification_status)
            lines.append(f"- {f.statement} _{tag}_")
        lines.append("")
    if model.rankings:
        lines += [
            "### Priority Ranking", "",
            "| Rank | Entity | Motion | Rationale |",
            "|---|---|---|---|",
        ]
        for r in model.rankings:
            lines.append(
                f"| {r.rank} | {r.entity} | {r.investment_motion} | "
                f"{r.rationale[:120]} |"
            )
        lines.append("")
    if model.recommendations:
        lines += ["### Recommendations", ""]
        for rec in model.recommendations:
            lines.append(f"- **{rec.action}**")
            if rec.next_step:
                lines.append(f"  - Next step: {rec.next_step}")
            if rec.justification:
                lines.append(f"  - Why: {rec.justification}")
        lines.append("")
    if model.data_limitations:
        lines += ["### Data Limitations", ""]
        for d in model.data_limitations:
            lines.append(f"- {d}")
        lines.append("")
    if model.sources:
        lines += ["### Sources", ""]
        for s in model.sources:
            lines.append(
                f"- {s.source_name} ({s.retrieval_status or s.source_type})"
            )
    return "\n".join(lines)


def render_pdf(
    question: str,
    answer_markdown: str,
    web_sources: Optional[list[dict]] = None,
    *,
    structured: Optional[dict] = None,
    rtl: bool = False,
) -> bytes:
    """Render the briefing as PDF bytes.

    Prefer ``structured`` (validated QualityResponse dict) when available;
    otherwise normalize Markdown tables → HTML before xhtml2pdf.
    Never pass unvalidated arbitrary model Markdown tables straight through.
    """
    today = datetime.utcnow().strftime("%d %B %Y")
    if structured:
        try:
            from app.schemas.quality_response import (
                validate_quality_response,
            )
            model, errs = validate_quality_response(structured)
            if model and not errs:
                answer_markdown = _structured_to_markdown(model)
        except Exception:
            pass
    prepared = normalize_answer_markdown_for_pdf(answer_markdown or "")
    # Prefer the document H1 as the PDF title — never slap the raw user
    # question above a polished briefing (that was the ugly export look).
    doc_title = ""
    m_title = re.search(r"(?m)^#\s+(.+?)\s*$", answer_markdown or "")
    if m_title:
        doc_title = m_title.group(1).strip()
    display_title = doc_title or (question or "").strip()
    # Never print the raw conversational ask as a subtitle when we already
    # have a polished H1 — that was the ugly "give me the list…" PDF header.
    show_question = False
    # Stash pre-rendered HTML tables/cards so nl2br cannot inject <br>
    # into them (that was a root cause of overlapping/broken PDF tables).
    stashed: list[str] = []

    def _stash(m: re.Match) -> str:
        stashed.append(m.group(0))
        return f"\n\nPDFHTMLBLOCK{len(stashed) - 1}\n\n"

    protected = re.sub(
        r"(?s)(?:<table\b.*?</table>|<div class=\"profile-card\">.*?</div>)",
        _stash,
        prepared,
    )
    answer_html = markdown.markdown(
        protected,
        extensions=["tables", "nl2br", "fenced_code"],
    )
    for i, block in enumerate(stashed):
        answer_html = answer_html.replace(f"PDFHTMLBLOCK{i}", block)
        # markdown may wrap the placeholder in <p>
        answer_html = answer_html.replace(f"<p>PDFHTMLBLOCK{i}</p>", block)
    answer_html = sanitize_html(answer_html)
    answer_html = _strip_leaked_markdown(answer_html)
    # Drop duplicate H1 inside the body when we already show it as title
    if doc_title:
        answer_html = re.sub(
            r"(?is)<h1[^>]*>\s*" + re.escape(doc_title) + r"\s*</h1>\s*",
            "",
            answer_html,
            count=1,
        )
    sources_html = _render_sources_html(web_sources)
    dir_attr = ' dir="rtl"' if rtl else ""
    q_block = (
        f'<div class="question-sub">{_escape(question)}</div>'
        if show_question else ""
    )
    html = f"""<!DOCTYPE html>
<html{dir_attr}><head><meta charset="utf-8"><style>{_PDF_CSS}</style></head>
<body>
<div class="letterhead">
  <div class="brand">MISA Intelligence Briefing</div>
  <div class="meta">Generated {today} &middot; Ministry of Investment</div>
</div>
<div class="question">{_escape(display_title)}</div>
{q_block}
<div class="answer">{answer_html}</div>
{sources_html}
<div id="footer-content" class="footer-content">
  MISA Internal &middot; For executive use &middot; Page <pdf:pagenumber /> of <pdf:pagecount />
</div>
</body></html>
"""
    buf = io.BytesIO()
    status = pisa.CreatePDF(html, dest=buf, encoding="utf-8")
    if status.err:
        raise ValueError(f"PDF rendering failed ({status.err} errors)")
    return buf.getvalue()
