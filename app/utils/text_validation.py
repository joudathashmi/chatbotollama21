"""Shared free-text input guards.

Two complementary defenses against the same underlying risk — client-
supplied text that might later be interpreted as HTML — used at two
different points depending on what the destination endpoint actually
does with the text:

  1. `contains_html_markup` / `reject_html_markup` — REJECT outright.
     Used where the text is only ever logged verbatim or fed to an LLM
     prompt (feedback, chat) and never rendered as HTML. Neither layer
     sanitizes on read, so rejecting markup on write is cheaper and
     safer than trusting every future consumer to escape on render.

  2. `sanitize_html` — STRIP dangerous tags/attributes, keep the rest.
     Used where the text is deliberately converted to HTML (the PDF
     export endpoint runs the client's `answer` through
     markdown.markdown()) — outright rejection isn't an option there
     since turning text into HTML is the endpoint's whole job.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# 1. Reject-outright guard (feedback, chat — text is only ever logged/
#    prompted, never rendered)
# ---------------------------------------------------------------------------
# Three narrow, deliberately non-greedy patterns, chosen so ordinary prose
# with bare angle brackets ("a < b and c > d", "vector<int>", "x<5") does
# NOT match, while real markup does:
#   - a genuine closing tag:        </script>, </b>
#   - an opening tag WITH an attribute (the shape every real XSS payload
#     needs — src=, onerror=, href=, etc.):  <img src=x onerror=alert(1)>
#   - a small set of always-dangerous bare tag names, attribute or not:
#     <script>, <iframe>, <object>, <embed>, <svg>, <style>, <link>, <meta>,
#     <base>, <form>
_CLOSE_TAG_RE = re.compile(r"</\s*[a-zA-Z][a-zA-Z0-9-]*\s*>")
_ATTR_TAG_RE = re.compile(r"<\s*[a-zA-Z][a-zA-Z0-9-]*\s+[^<>]*=[^<>]*>")
_DANGEROUS_BARE_TAG_RE = re.compile(
    r"<\s*(script|iframe|object|embed|svg|style|link|meta|base|form)\b[^<>]*>",
    re.IGNORECASE,
)


def contains_html_markup(text: str) -> bool:
    return bool(
        _CLOSE_TAG_RE.search(text)
        or _ATTR_TAG_RE.search(text)
        or _DANGEROUS_BARE_TAG_RE.search(text)
    )


def reject_html_markup(value):
    """Pydantic field_validator body: raise if `value` (a string) contains
    HTML/script-tag-shaped markup. Non-strings pass through untouched so
    this can sit alongside `mode="before"` blank-stripping validators."""
    if isinstance(value, str) and contains_html_markup(value):
        raise ValueError("must not contain HTML/script markup")
    return value


# ---------------------------------------------------------------------------
# 2. Allowlist sanitizer (PDF export — text is deliberately rendered as
#    HTML via markdown.markdown(), so it has to be sanitized, not rejected)
# ---------------------------------------------------------------------------
# Tags Python-Markdown's "tables", "nl2br", "fenced_code" extensions
# legitimately emit for the executive-briefing answer format. Anything
# else (script, style, iframe, object, embed, svg, img, form, input,
# button, link, meta, base, video, audio, ...) is dropped — the tag AND
# its attributes, though any plain text between the tags is kept.
_ALLOWED_TAGS = {
    "p", "br", "hr",
    "strong", "b", "em", "i", "u", "s", "del",
    "ul", "ol", "li",
    "table", "thead", "tbody", "tr", "th", "td",
    "code", "pre",
    "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "a", "span",
}
_VOID_TAGS = {"br", "hr"}
# Per-tag attribute allowlist. Everything not listed here is stripped,
# including all `on*` event handlers and `style`.
_ALLOWED_ATTRS: dict[str, set[str]] = {
    "a": {"href", "title"},
    # PDF export injects class/width on tables & profile cards.
    # `style` is intentionally NOT allowlisted (XSS / css injection).
    "table": {"class", "width"},
    "thead": {"class"},
    "tbody": {"class"},
    "tr": {"class"},
    "th": {"class", "width"},
    "td": {"class", "width"},
    "div": {"class", "id"},
    "span": {"class"},
    "p": {"class"},
}
_SAFE_URL_SCHEMES = {"http", "https", "mailto", ""}
# Elements whose entire text content must be dropped, not just the tag —
# <script>alert(1)</script> and <style>...</style> would otherwise leave
# their inner text as visible-but-inert garbage in the rendered PDF.
_RAW_TEXT_CONTAINERS = {"script", "style", "title", "textarea"}


def _escape_text(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _escape_attr(s: str) -> str:
    return _escape_text(s).replace('"', "&quot;")


def _is_safe_url(value: str) -> bool:
    """Reject javascript:/data:/vbscript: and other script-executing or
    unusual schemes; allow plain http(s)/mailto/relative links."""
    value = (value or "").strip()
    if not value:
        return True
    if ":" not in value.split("/", 1)[0]:
        # No scheme (relative link / fragment) — safe.
        return True
    scheme = value.split(":", 1)[0].strip().lower()
    return scheme in _SAFE_URL_SCHEMES


class _AllowlistHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        # Which raw-text container (script/style/...) we're currently
        # inside, or None. Handles elements whose content must be
        # dropped entirely until the matching close tag.
        self._suppress_tag: "str | None" = None

    def _open(self, tag: str, attrs) -> None:
        if tag in _RAW_TEXT_CONTAINERS:
            self._suppress_tag = tag
            return
        if self._suppress_tag is not None:
            return
        if tag not in _ALLOWED_TAGS:
            return
        allowed = _ALLOWED_ATTRS.get(tag, set())
        safe_attrs = []
        for name, value in attrs:
            if name not in allowed:
                continue
            if name == "href" and not _is_safe_url(value or ""):
                continue
            safe_attrs.append(f' {name}="{_escape_attr(value or "")}"')
        attr_str = "".join(safe_attrs)
        if tag in _VOID_TAGS:
            self._out.append(f"<{tag}{attr_str}/>")
        else:
            self._out.append(f"<{tag}{attr_str}>")

    def handle_starttag(self, tag, attrs):
        self._open(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._open(tag, attrs)

    def handle_endtag(self, tag):
        if tag == self._suppress_tag:
            self._suppress_tag = None
            return
        if self._suppress_tag is not None:
            return
        if tag in _ALLOWED_TAGS and tag not in _VOID_TAGS:
            self._out.append(f"</{tag}>")

    def handle_data(self, data):
        if self._suppress_tag is not None:
            return
        self._out.append(_escape_text(data))

    def handle_entityref(self, name):
        if self._suppress_tag is not None:
            return
        self._out.append(f"&{name};")

    def handle_charref(self, name):
        if self._suppress_tag is not None:
            return
        self._out.append(f"&#{name};")

    def get_html(self) -> str:
        return "".join(self._out)


def sanitize_html(html: str) -> str:
    """Strip every tag/attribute not on the allowlist from `html`, HTML-
    escaping any text content along the way. Malformed input degrades
    gracefully (HTMLParser is lenient) rather than raising."""
    parser = _AllowlistHTMLParser()
    parser.feed(html or "")
    parser.close()
    return parser.get_html()


_CITE_SUP_RE = re.compile(
    r'<sup\b[^>]*\bclass\s*=\s*["\']?cite["\']?[^>]*>(.*?)</sup>',
    re.I | re.S,
)
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


def strip_html_to_plaintext(text: str) -> str:
    """Convert citation HTML / accidental markup in chat history back to
    plain text so follow-up turns are not rejected by the HTML guard.

    `<sup class="cite">1</sup>` → `[web:1]`; other tags are removed.
    """
    if not text or not isinstance(text, str):
        return text
    out = _CITE_SUP_RE.sub(
        lambda m: (
            f"[web:{m.group(1).strip()}]"
            if (m.group(1) or "").strip().isdigit()
            else "[doc]"
        ),
        text,
    )
    out = _TAG_RE.sub("", out)
    return out

