"""Document classification gate and upload consent policy.

Only Public documents may enter the library. Anything declared or marked
Restricted, Secret, or Top Secret is rejected before it is stored or
indexed. Uploads additionally require an explicit consent declaration.
"""

from __future__ import annotations

import re

CLASSIFICATION_PUBLIC = "public"
CLASSIFICATION_LEVELS = ("public", "restricted", "secret", "top_secret")

# Version bump whenever the consent wording changes, so stored/audited
# acceptances can be tied to the exact text the uploader saw.
CONSENT_POLICY_VERSION = "1.0"

CONSENT_POLICY = {
    "version": CONSENT_POLICY_VERSION,
    "title": "Document Upload Consent Declaration",
    "preamble": (
        "By uploading this document you confirm each of the following. "
        "Uploads that breach this declaration may be removed and the "
        "upload logged for review."
    ),
    "terms": [
        {
            "id": "classification",
            "text": (
                "The document is classified Public. It is not Restricted, "
                "Secret, or Top Secret, and it contains no classified "
                "government material or commercially confidential material."
            ),
        },
        {
            "id": "content_standards",
            "text": (
                "The document contains no political, religious, extremist, "
                "defamatory, discriminatory, hateful, or otherwise offensive "
                "content."
            ),
        },
        {
            "id": "personal_data",
            "text": (
                "The document contains no personal data beyond professional "
                "contact details, consistent with the Saudi Personal Data "
                "Protection Law (PDPL)."
            ),
        },
        {
            "id": "rights",
            "text": (
                "You own the document or hold the right to share it, and "
                "uploading it does not breach any third party "
                "confidentiality, contractual, or intellectual property "
                "obligation."
            ),
        },
        {
            "id": "processing",
            "text": (
                "You consent to the system storing, indexing, and processing "
                "the document, including AI assisted analysis, so that it can "
                "be used to answer user queries."
            ),
        },
    ],
}


def normalize_classification(value: str | None) -> str:
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if v in ("", CLASSIFICATION_PUBLIC):
        return CLASSIFICATION_PUBLIC
    if v in ("top_secret", "topsecret"):
        return "top_secret"
    if v in CLASSIFICATION_LEVELS:
        return v
    return v  # unknown labels are rejected by the caller


# Explicit protective-marking patterns. The gate fails closed: a marking
# anywhere in the document blocks ingestion, even mid-text, because page
# headers and footers land mid-string after PDF text extraction.
_MARKING_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("top_secret", re.compile(r"(?i)\btop\s*[- ]?\s*secret\b")),
    ("top_secret", re.compile(r"سري\s*للغاية")),
    (
        "labelled",
        re.compile(
            r"(?i)\b(?:classification|security\s+classification|protective\s+marking)\s*[:\-]\s*"
            r"(restricted|secret|top\s*secret|confidential)\b"
        ),
    ),
    # Bare level words only count in ALL CAPS, the way protective markings
    # are stamped, so ordinary prose such as "trade secret" passes.
    ("secret", re.compile(r"\bSECRET\b")),
    ("restricted", re.compile(r"\bRESTRICTED\b")),
    ("confidential", re.compile(r"\bCONFIDENTIAL\b")),
    ("classified", re.compile(r"\bCLASSIFIED\b")),
    ("secret", re.compile(r"(?<![ء-ي])سري(?![ء-ي])")),
]


# Filenames are short and deliberate, so bare level words match
# case-insensitively there (for example secret-plan.docx).
_FILENAME_PATTERN = re.compile(
    r"(?i)\b(top\s*[- _]?\s*secret|secret|restricted|confidential|classified)\b"
)


def find_classification_marking(text: str, filename: str = "") -> str | None:
    """Return a human-readable description of the first protective marking
    found in the document text or filename, or None when the content shows
    no marking."""
    if filename:
        m = _FILENAME_PATTERN.search(filename)
        if m:
            return f"marking {m.group(0)!r} in filename"
    for level, pattern in _MARKING_PATTERNS:
        m = pattern.search(text or "")
        if m:
            return f"{level} marking {m.group(0)!r} in content"
    return None
