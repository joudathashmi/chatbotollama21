"""
PDF export.

Turns an executive-briefing answer (markdown) into a print-ready PDF
that ministers can email, file, or hand-distribute. The browser UI
exposes a "Download PDF" button under every assistant message.

Implementation choice: `markdown` (pure-Python) → HTML → `xhtml2pdf`
(pure-Python) → PDF. No system libraries (no cairo/pango/wkhtmltopdf
required) so it works on any Python environment.

Layout:
  - Letterhead: "MISA Briefing" + generation date
  - Title: the user's question
  - Body: the markdown-rendered answer (headings, tables, bold,
    bullets all preserved)
  - Sources footer: numbered URL list when web_sources supplied
  - Page footer: confidentiality stamp + page numbers

Caller is the chat router's POST /api/v1/export/pdf endpoint.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import markdown
from xhtml2pdf import pisa

from app.utils.text_validation import sanitize_html


_PDF_CSS = """
@page {
  size: A4;
  margin: 2cm 2cm 2.6cm 2cm;
  @frame footer {
    -pdf-frame-content: footer-content;
    bottom: 1cm;
    margin-left: 2cm;
    margin-right: 2cm;
    height: 1cm;
  }
}
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt;
       color: #1d1d1f; line-height: 1.45; }
.letterhead { border-bottom: 2px solid #006c35; padding-bottom: 6pt;
              margin-bottom: 14pt; }
.letterhead .brand { font-size: 14pt; font-weight: bold; color: #006c35;
                     letter-spacing: 0.4pt; }
.letterhead .meta { font-size: 9pt; color: #6e6e73; margin-top: 2pt; }
.question { font-size: 12pt; font-weight: bold; margin: 0 0 12pt 0;
            color: #1d1d1f; }
.answer h2 { font-size: 12pt; color: #006c35; margin: 16pt 0 6pt 0;
             border-bottom: 1px solid #e5e5e7; padding-bottom: 3pt; }
.answer h3 { font-size: 11pt; color: #1d1d1f; margin: 12pt 0 4pt 0; }
.answer p { margin: 0 0 6pt 0; }
.answer ul, .answer ol { margin: 4pt 0 8pt 18pt; padding-left: 0; }
.answer li { margin: 2pt 0; }
.answer table { border-collapse: collapse; width: 100%;
                margin: 6pt 0 10pt 0; }
.answer th, .answer td { border: 1px solid #d2d2d7; padding: 4pt 6pt;
                          font-size: 9.5pt; text-align: left;
                          vertical-align: top; }
.answer th { background-color: #f5f5f7; font-weight: bold; }
.answer code { font-family: monospace; background: #f5f5f7;
               padding: 1pt 3pt; border-radius: 2pt; font-size: 9pt; }
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


def _render_sources_html(web_sources: Optional[list[dict]]) -> str:
    """Render the numbered sources list. Empty string when none."""
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


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;")
         .replace(">", "&gt;").replace('"', "&quot;")
    )


def render_pdf(
    question: str,
    answer_markdown: str,
    web_sources: Optional[list[dict]] = None,
) -> bytes:
    """Render the briefing as PDF bytes. Returns the raw PDF
    document. The caller wraps it in a StreamingResponse / Response
    with the right Content-Type and Content-Disposition headers.

    Raises ValueError if PDF generation fails (xhtml2pdf returned an
    error). Caller should 500 in that case.
    """
    today = datetime.utcnow().strftime("%d %B %Y")
    # Convert the answer markdown to HTML. We enable the tables and
    # nl2br extensions because the curator emits markdown tables AND
    # the executive-briefing flow uses single-newline line breaks
    # inside bullets.
    answer_html = markdown.markdown(
        answer_markdown or "",
        extensions=["tables", "nl2br", "fenced_code"],
    )
    answer_html = sanitize_html(answer_html)
    sources_html = _render_sources_html(web_sources)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_PDF_CSS}</style></head>
<body>
<div class="letterhead">
  <div class="brand">MISA Intelligence Briefing</div>
  <div class="meta">Generated {today} &middot; Ministry of Investment</div>
</div>
<div class="question">{_escape(question or '')}</div>
<div class="answer">{answer_html}</div>
{sources_html}
<div id="footer-content" class="footer-content">
  MISA Internal &middot; For executive use &middot; Page <pdf:pagenumber /> of <pdf:pagecount />
</div>
</body></html>
"""
    buf = io.BytesIO()
    # xhtml2pdf.pisa returns a pisaStatus object; .err is the count of
    # rendering errors (warnings don't block). We treat ≥1 error as
    # fatal because the output would be visibly broken.
    status = pisa.CreatePDF(html, dest=buf, encoding="utf-8")
    if status.err:
        raise ValueError(f"PDF rendering failed ({status.err} errors)")
    return buf.getvalue()
