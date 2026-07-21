"""
Business Card extraction engine.

Orchestrates Google Vision API (OCR) → OpenAI (JSON structuring) pipeline.
Provides synchronous functions suitable for use in async endpoints via asyncio.to_thread().
"""

from __future__ import annotations

import json
import logging
import re


from google.api_core import exceptions as google_exceptions
from google.cloud import vision
from openai import (
    AuthenticationError as OAIAuthenticationError,
    PermissionDeniedError as OAIPermissionDeniedError,
    RateLimitError as OAIRateLimitError,
    BadRequestError as OAIBadRequestError,
    NotFoundError as OAINotFoundError,
    APIConnectionError as OAIAPIConnectionError,
    APITimeoutError as OAIAPITimeoutError,
    InternalServerError as OAIInternalServerError,
    UnprocessableEntityError as OAIUnprocessableEntityError,
)

from app.database import get_public_openai_client
from app.services.chat_engine import _chat_completions_create_with_retry
from app.services.country_normalizer import resolve_country_name
from app.config import OPENAI_MODEL

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Error Classes
# ============================================================================

class BusinessCardError(Exception):
    """
    Base exception for business card extraction errors.

    Attributes:
        code: Machine-readable error code (e.g., 'VISION_API_ERROR')
        message: User-friendly error message to return to client
        details: Technical details for server logging (not shown to client)
    """

    def __init__(self, code: str, message: str, details: str | None = None):
        self.code = code
        self.message = message
        self.details = details or message
        super().__init__(self.message)


class VisionAPIError(BusinessCardError):
    """Raised when Google Vision API fails."""

    pass


class OpenAIError(BusinessCardError):
    """Raised when OpenAI API fails."""

    pass


class ExtractionError(BusinessCardError):
    """Raised when extraction logic fails (e.g., invalid JSON, no text found)."""

    pass


class PromptInjectionError(BusinessCardError):
    """Raised when the OCR'd card text carries a disguised prompt-injection
    payload that survived the regex scrub (Risk-20-6).

    A genuine business card never contains a jailbreak. If `_sanitize_ocr_text`
    redacted the literal patterns and `detect_prompt_attack` — which normalizes
    homoglyphs/leetspeak/zero-width tricks first — STILL fires, the text is a
    deliberately obfuscated injection. Redacting that in-place isn't possible
    (the match lives in normalized space, not the raw span), so the safe
    response is to refuse the card rather than forward it to the LLM.
    """

    pass


# ============================================================================
# Google Vision Client
# ============================================================================

# Google Vision client (lazy singleton, similar to OpenAI pattern)


_vision_client_inst: vision.ImageAnnotatorClient | None = None


def _get_vision_client() -> vision.ImageAnnotatorClient | None:
    """
    Lazy-initialize Google Vision API client.

    Returns None if GOOGLE_APPLICATION_CREDENTIALS env var is not set or invalid.
    Google SDK handles authentication automatically via the env var.
    """
    global _vision_client_inst

    if _vision_client_inst is None:
        try:
            _vision_client_inst = vision.ImageAnnotatorClient()
            logger.info("Vision API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vision API client: {e}")
            _vision_client_inst = None
            return None

    return _vision_client_inst


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using Google Cloud Vision API.

    Args:
        image_bytes: Raw image bytes (JPG, PNG, WebP, HEIC)

    Returns:
        Extracted text from the image (full_text_annotation.text)

    Raises:
        VisionAPIError: If Vision API is unavailable or request fails
    """
    client = _get_vision_client()
    if client is None:
        raise VisionAPIError(
            code="VISION_API_AUTH_ERROR",
            message="Service unavailable. Please try again later.",
            details="Vision API client not initialized. Check GOOGLE_APPLICATION_CREDENTIALS env var.",
        )

    try:
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)

        if response.error.message:
            raise VisionAPIError(
                code="VISION_API_ERROR",
                message="Service unavailable. Please try again later.",
                details=f"Vision API error: {response.error.message}",
            )

        raw_text = response.full_text_annotation.text or ""
        if not raw_text or not raw_text.strip():
            raise ExtractionError(
                code="NO_TEXT_DETECTED",
                message="Image contains no readable text. Please use a clearer photo.",
                details="Vision API returned empty text annotation.",
            )

        return raw_text

    except (VisionAPIError, ExtractionError):
        raise
    except google_exceptions.Unauthenticated as e:
        logger.error(f"Vision API auth error: {e}")
        raise VisionAPIError(
            code="VISION_API_AUTH_ERROR",
            message="Service temporarily unavailable.",
            details=f"Google auth: {e}",
        )
    except google_exceptions.PermissionDenied as e:
        logger.error(f"Vision API permission error: {e}")
        raise VisionAPIError(
            code="VISION_API_AUTH_ERROR",
            message="Service temporarily unavailable.",
            details=f"Google permission: {e}",
        )
    except google_exceptions.ResourceExhausted as e:
        logger.error(f"Vision API quota exceeded: {e}")
        raise VisionAPIError(
            code="VISION_API_QUOTA_ERROR",
            message="Service busy. Please try again shortly.",
            details=f"Google quota: {e}",
        )
    except google_exceptions.DeadlineExceeded as e:
        logger.error(f"Vision API timeout: {e}")
        raise VisionAPIError(
            code="VISION_API_TIMEOUT",
            message="Request timed out. Please try again.",
            details=f"Google timeout: {e}",
        )
    except (google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError) as e:
        logger.error(f"Vision API server error: {e}")
        raise VisionAPIError(
            code="VISION_API_ERROR",
            message="Service temporarily unavailable.",
            details=f"Google server: {e}",
        )
    except google_exceptions.InvalidArgument as e:
        logger.error(f"Vision API invalid argument: {e}")
        raise VisionAPIError(
            code="VISION_API_ERROR",
            message="Unable to process this image.",
            details=f"Google invalid arg: {e}",
        )
    except Exception as e:
        logger.error(f"Vision API unexpected error: {e}", exc_info=True)
        raise VisionAPIError(
            code="VISION_API_ERROR",
            message="Service temporarily unavailable.",
            details=f"Unexpected Vision error: {str(e)}",
        )


# ============================================================================
# Field Validation Helpers
# ============================================================================

# Patterns that indicate a value is NOT a title/company
_PHONE_RE = re.compile(r"^[\+\d\s\-\(\)\.]{7,}$")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_ADDRESS_FRAGMENT_RE = re.compile(
    r"^\d+\s+\S+\s*(street|st|road|rd|avenue|ave|blvd|lane|ln|drive|dr|floor|suite|ste)\b",
    re.IGNORECASE,
)

# Max lengths per field type
_MAX_LENGTHS = {"title": 120, "company": 150}


def _validate_text_field(value: str | None, field_type: str) -> str:
    """
    Validate a title or company field extracted from OCR.

    Returns the cleaned value if valid, or "" if the value is noisy/invalid.

    Validation rules:
    - Reject None, empty, or whitespace-only
    - Reject if fewer than 2 alphabetic characters
    - Reject if >80% digits or special characters
    - Reject if matches phone/email/URL patterns
    - Reject if looks like an address fragment
    - Reject if exceeds max length for the field type
    """
    if not value or not value.strip():
        return ""

    value = value.strip()

    # Length check
    max_len = _MAX_LENGTHS.get(field_type, 150)
    if len(value) > max_len:
        return ""

    # Must have at least 2 alphabetic characters
    alpha_count = sum(1 for c in value if c.isalpha())
    if alpha_count < 2:
        return ""

    # Reject if >80% non-alpha characters (digits + special)
    non_alpha = sum(1 for c in value if not c.isalpha() and not c.isspace())
    if len(value) > 0 and non_alpha / len(value) > 0.8:
        return ""

    # Reject phone numbers
    if _PHONE_RE.match(value):
        return ""

    # Reject email addresses
    if _EMAIL_RE.search(value):
        return ""

    # Reject URLs
    if _URL_RE.search(value):
        return ""

    # Reject address fragments
    if _ADDRESS_FRAGMENT_RE.match(value):
        return ""

    return value


# Pincode validation: 4-10 digits with optional dash/space
_PINCODE_RE = re.compile(r"^\d[\d\s\-]{2,8}\d$")


def _validate_pincode(value: str | None) -> str:
    """Validate a postal/ZIP code. Returns "" if invalid."""
    if not value or not value.strip():
        return ""
    value = value.strip()
    if _PINCODE_RE.match(value):
        return value
    return ""


def _process_phone_objects(phone_data: list) -> dict:
    """
    Process phone objects from OpenAI into parallel country_code and mobile_numbers arrays.

    Input formats supported:
    - Structured: [{"country_code": "+91", "number": "7788899721"}, ...]
    - Backward compat (plain strings): ["+91 7788899721", "5550123456"]

    Output: {"country_code": ["+91", ...], "mobile_numbers": ["7788899721", ...]}

    Cleaning:
    - number: remove all non-digit characters (spaces, hyphens, dots, brackets)
    - country_code: strip whitespace, keep only digits and leading "+"

    Deduplicates by (country_code, number) pair.
    Skips entries where the cleaned number is empty.
    """
    if not phone_data:
        return {"country_code": [], "mobile_numbers": []}

    seen: set[tuple[str, str]] = set()
    codes: list[str] = []
    numbers: list[str] = []

    for entry in phone_data:
        if entry is None:
            continue

        if isinstance(entry, dict):
            raw_code = str(entry.get("country_code") or "").strip()
            raw_number = str(entry.get("number") or "").strip()
        elif isinstance(entry, str):
            # Backward compat: plain string, no country code splitting
            raw_code = ""
            raw_number = entry.strip()
        else:
            continue

        # Clean country code: keep only digits and leading "+"
        if raw_code:
            cleaned_code = raw_code[0] if raw_code.startswith("+") else ""
            cleaned_code += re.sub(r"[^\d]", "", raw_code)
            if cleaned_code and not cleaned_code.startswith("+"):
                cleaned_code = "+" + cleaned_code
        else:
            cleaned_code = ""

        # Clean number: digits only
        cleaned_number = re.sub(r"[^\d]", "", raw_number)

        if not cleaned_number:
            continue

        # Deduplicate by (code, number) pair
        key = (cleaned_code, cleaned_number)
        if key in seen:
            continue
        seen.add(key)

        codes.append(cleaned_code)
        numbers.append(cleaned_number)

    return {"country_code": codes, "mobile_numbers": numbers}


def _process_fax_objects(fax_data: list) -> dict:
    """
    Process fax objects from OpenAI into parallel fax_country_code and fax_numbers arrays.

    Input/output/cleaning/dedup rules are identical to _process_phone_objects.
    Output: {"fax_country_code": [...], "fax_numbers": [...]}
    """
    if not fax_data:
        return {"fax_country_code": [], "fax_numbers": []}

    seen: set[tuple[str, str]] = set()
    codes: list[str] = []
    numbers: list[str] = []

    for entry in fax_data:
        if entry is None:
            continue

        if isinstance(entry, dict):
            raw_code = str(entry.get("country_code") or "").strip()
            raw_number = str(entry.get("number") or "").strip()
        elif isinstance(entry, str):
            raw_code = ""
            raw_number = entry.strip()
        else:
            continue

        if raw_code:
            cleaned_code = raw_code[0] if raw_code.startswith("+") else ""
            cleaned_code += re.sub(r"[^\d]", "", raw_code)
            if cleaned_code and not cleaned_code.startswith("+"):
                cleaned_code = "+" + cleaned_code
        else:
            cleaned_code = ""

        cleaned_number = re.sub(r"[^\d]", "", raw_number)

        if not cleaned_number:
            continue

        key = (cleaned_code, cleaned_number)
        if key in seen:
            continue
        seen.add(key)

        codes.append(cleaned_code)
        numbers.append(cleaned_number)

    return {"fax_country_code": codes, "fax_numbers": numbers}


def _build_parsed_address(address_data) -> dict | None:
    """
    Build a ParsedAddress dict from OpenAI response data.

    Handles both:
    - New structured format: {"full_address": ..., "city": ..., ...}
    - Old array format (fallback): ["line1", "line2"]
    """
    if address_data is None:
        return None

    if isinstance(address_data, dict):
        full_address = (address_data.get("full_address") or "").strip()
        city = _validate_text_field(address_data.get("city"), "company")
        district = _validate_text_field(address_data.get("district"), "company")
        state = _validate_text_field(address_data.get("state"), "")
        country = _validate_text_field(address_data.get("country"), "")
        # Map the OCR'd country text to the client's canonical country name
        # (their DB stores a fixed 265-name set). No-op if already canonical
        # or no confident match. See app/services/country_normalizer.py.
        country = resolve_country_name(country)
        pincode = _validate_pincode(address_data.get("pincode"))

        # If nothing was extracted, return None
        if not any([full_address, city, district, state, country, pincode]):
            return None

        return {
            "city": city,
            "district": district,
            "state": state,
            "country": country,
            "pincode": pincode,
        }

    if isinstance(address_data, list):
        # Backward compat with old array format
        joined = ", ".join(str(a) for a in address_data if a)
        if not joined.strip():
            return None
        return {
            "city": "",
            "district": "",
            "state": "",
            "country": "",
            "pincode": "",
        }

    return None

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(ignore|disregard|forget|skip|bypass|override|replace|discard)\b.{0,60}?\b(previous|prior|above|earlier|initial|system|developer)?\b.{0,40}?\b(instruction|instructions|prompt|prompts|message|messages)\b",
        re.I | re.S,
    ),
    re.compile(r"\bfollow\s+(these|my|new)\s+instructions\b", re.I),
    re.compile(r"\bdo\s+not\s+follow\b.{0,50}\b(previous|above|system|developer)\b", re.I | re.S),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bact\s+as\b", re.I),
    re.compile(r"\bpretend\s+to\s+be\b", re.I),
    re.compile(r"\bassume\s+the\s+role\s+of\b", re.I),
    re.compile(r"\bfrom\s+now\s+on\b", re.I),
    re.compile(
        r"\b(show|display|print|reveal|output|repeat|dump|expose|return)\b.{0,60}?\b(system|developer|hidden|secret|internal)\b.{0,40}?\b(prompt|instruction|instructions|message|policy|policies)\b",
        re.I | re.S,
    ),
    re.compile(r"\bwhat\s+(is|are)\s+your\s+(system|developer)\s+prompt\b", re.I),
    re.compile(r"\b(system|developer|assistant|user)\s+prompt\s*:", re.I),
    re.compile(r"\b(new|updated|replacement)\s+instructions?\s*:", re.I),
    re.compile(r"\b(system|developer)\s+message\s*:", re.I),
    re.compile(r"^\s*(system|assistant|developer|user|tool|function)\s*:\s*", re.I | re.M),
    re.compile(r"</?\s*(system|assistant|developer|user|instruction|instructions|prompt|message)\s*>", re.I),
    re.compile(r"```"),
    re.compile(r"~~~"),
    re.compile(
        r"\b(call|invoke|execute|run|use)\b.{0,40}?\b(tool|tools|function|functions|plugin|plugins|api)\b",
        re.I | re.S,
    ),
    re.compile(r"\bfunction\s+call\b", re.I),
    re.compile(r"\b(DAN|jailbreak|developer\s+mode|unrestricted\s+mode)\b", re.I),
    re.compile(r"\bignore\s+safety\b", re.I),
    re.compile(r"\b(highest\s+priority|override\s+all|this\s+instruction\s+takes\s+priority)\b", re.I),
    re.compile(r"\b(end\s+system\s+prompt|begin\s+new\s+prompt|ignore\s+everything\s+above|forget\s+everything)\b", re.I),
    re.compile(r"\b(continue\s+the\s+conversation|respond\s+with|your\s+next\s+response)\b", re.I),
    re.compile(
        r"\b(show|reveal|display|output)\b.{0,40}?\b(chain\s+of\s+thought|reasoning|internal\s+reasoning)\b",
        re.I | re.S,
    ),
]

def _sanitize_ocr_text(raw_text: str) -> str:
    """Neutralize known prompt-injection patterns in OCR'd text before
    it reaches the LLM. Matches are replaced with a visible placeholder
    (not silently dropped) so the substitution is inspectable if this
    text is ever reviewed; a scrub event is logged at WARNING for
    monitoring, without logging the actual matched text.

    NOTE: these are literal regexes over the raw text — they do NOT defeat
    obfuscation (homoglyphs, leetspeak). `_guard_ocr_text` runs after this
    and catches those; see its docstring."""
    sanitized = raw_text
    hit = False
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(sanitized):
            hit = True
            sanitized = pattern.sub("[redacted]", sanitized)
    if hit:
        logger.warning(
            "Prompt-injection pattern(s) detected and scrubbed from "
            "business-card OCR text before LLM extraction."
        )
    return sanitized


def _guard_ocr_text(sanitized_text: str) -> None:
    """Second-layer injection check on already-scrubbed OCR text (Risk-20-6).

    `_sanitize_ocr_text`'s regexes match literal strings, so an attacker who
    prints "Ｉgnore all previous instructions" (homoglyph) or "1gn0r3 4ll
    pr3v10us 1nstruct10ns" (leetspeak) on a card slips straight through them —
    while the same text typed into /chat is caught, because `detect_prompt_attack`
    normalizes homoglyphs/leetspeak/zero-width/whitespace before matching.
    This closes that asymmetry: the card path now gets the same
    normalization-aware detection the chat path has.

    Runs on the SCRUBBED text, so ordinary injections already redacted above
    don't trip it — the card is still processed exactly as before. Only text
    that survives the scrub AND still reads as an attack after normalization
    is refused, which a real business card never is.

    Raises PromptInjectionError when an obfuscated injection is detected.
    """
    try:
        from app.services.prompt_guard import detect_prompt_attack
        is_attack, category = detect_prompt_attack(sanitized_text)
    except Exception:
        # The guard must never take the upload path down; a probe failure
        # falls back to the regex scrub already applied above.
        return
    if not is_attack:
        return
    # Log the category only — never the matched text (it is attacker-supplied
    # and may carry PII from the card).
    logger.warning(
        "Business-card OCR text rejected: obfuscated prompt-injection "
        f"detected after scrubbing (category={category})."
    )
    try:
        from app.services.audit_log import emit_security_event
        emit_security_event({
            "event": "business_card_prompt_injection_rejected",
            # NOT "category" — that key is the audit envelope's own SIEM
            # classification field and would be silently overwritten here.
            "detection_category": category,
        })
    except Exception:
        pass
    raise PromptInjectionError(
        code="PROMPT_INJECTION_DETECTED",
        message="Unable to process this image.",
        details=(
            "OCR text contained an obfuscated prompt-injection payload that "
            f"survived pattern scrubbing (category={category})."
        ),
    )


def structure_extracted_text(raw_text: str) -> dict:
    """
    Use OpenAI to structure OCR'd text into a business card JSON format.

    Args:
        raw_text: Raw text from Vision API

    Returns:
        Dict with keys: name, title, company, email, mobile_numbers, website, address, other

    Raises:
        OpenAIError: If OpenAI call fails or response is invalid JSON
    """
    if not raw_text or not raw_text.strip():
        raise ExtractionError(
            code="EXTRACTION_FAILED",
            message="Unable to extract information from image.",
            details="No text provided to structure.",
        )

    client = get_public_openai_client()
    if client is None:
        raise OpenAIError(
            code="OPENAI_API_AUTH_ERROR",
            message="Service unavailable. Please try again later.",
            details="OpenAI client not configured. Check OPENAI_API_KEY env var.",
        )

    system_prompt = (
    "You are a strict OCR business card parser.\n\n"

    "Your task is to process ONLY business cards. "
    "If the OCR text does not represent a business card, "
    "return the following JSON exactly:\n"
    '{'
    '"is_business_card": false, '
    '"error": "The uploaded document is not a business card."'
    '}\n\n'

    "Business cards typically contain person names, company names, "
    "job titles, phone numbers, emails, websites, or addresses.\n\n"

    "If the OCR text IS a business card, return:\n"
    "{\n"
    '  "is_business_card": true,\n'
    '  "data": {\n'
    '    "name": string or null,\n'
    '    "title": string or null,\n'
    '    "company": string or null,\n'
    '    "email": string or null,\n'
    '    "phone": [\n'
    '      {"country_code": string, "number": string}\n'
    '    ],\n'
    '    "fax": [\n'
    '      {"country_code": string, "number": string}\n'
    '    ],\n'
    '    "website": string or null,\n'
    '    "full_address": string or null,\n'
    '    "address": {\n'
    '      "city": string or null,\n'
    '      "district": string or null,\n'
    '      "state": string or null,\n'
    '      "country": string or null,\n'
    '      "pincode": string or null\n'
    '    },\n'
    '    "other": array\n'
    "  }\n"
    "}\n\n"

    "Strict Rules:\n"
    "- Extract only explicitly present information\n"
    "- Never hallucinate or infer values\n"
    "- Never fabricate phone numbers or emails\n"
    "- Preserve OCR text closely\n"
    "- Return ONLY valid JSON\n"
    "- No markdown or explanations\n\n"

    "Phone and fax number rules:\n"
    "- Return each phone/fax number as {\"country_code\": string, \"number\": string}\n"
    "- country_code: the dialing code with \"+\" prefix (e.g. \"+91\", \"+1\", \"+44\", \"+971\")\n"
    "- number: digits only, no spaces, hyphens, dots, or brackets\n"
    "- If the number already has a country code (like +91 or 0091), extract it\n"
    "- If the number has NO country code, use the address/country on the card "
    "to determine the correct country dialing code using your knowledge\n"
    "- If you truly cannot determine the country code, set country_code to \"\"\n"
    "- Put voice/mobile numbers in \"phone\" and facsimile numbers in \"fax\" — never mix them\n"
    "- Fax numbers are often labelled \"Fax:\", \"F:\", \"Facsimile:\", or shown with a fax icon\n\n"

    "Address field guidance (support ALL countries):\n"
    "- full_address: complete address exactly as it appears on the card\n"
    "- city: city, town, or municipality\n"
    "- district: district, county, region, borough, or local administrative area "
    "(e.g. \"Kings County\" in USA, \"Chennai District\" in India, \"Chiyoda-ku\" in Japan)\n"
    "- state: state, province, prefecture, canton, emirate, or top-level administrative division\n"
    "- country: the FULL, official English country name — never an abbreviation, "
    "code, or short form. Always expand to the complete name "
    "(e.g. \"USA\"/\"U.S.\" → \"United States\", \"UAE\" → \"United Arab Emirates\", "
    "\"UK\" → \"United Kingdom\", \"KSA\" → \"Saudi Arabia\", \"Korea\" → \"South Korea\")\n"
    "- pincode: postal code, ZIP code, PIN code, postcode (any country's format)\n\n"

    "Special rules for address.country and address.state ONLY:\n"
    "- You MAY predict country and state even if not explicitly written, "
    "using strong contextual signals such as: city name, phone country code "
    "(e.g. +44 → United Kingdom), ZIP/postcode format, email/website domain "
    "extension (e.g. .de → Germany), language or script of the text.\n"
    "- Only set the value if your confidence is >= 90%. "
    "If confidence is below 90% or you cannot determine it, set the field to an empty string \"\".\n"
    "- Do NOT predict any other field (name, email, city, pincode, etc.)."
)
    sanitized_text = _sanitize_ocr_text(raw_text)
    # Risk-20-6: normalization-aware second pass — catches homoglyph/leetspeak
    # obfuscation that the literal regexes above cannot. Raises on detection,
    # so nothing reaches the LLM.
    _guard_ocr_text(sanitized_text)
    user_message = f"Extract contact information from this OCR text:\n\n{sanitized_text}"

    try:
        response = _chat_completions_create_with_retry(
            client,
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=800,
            # Business-card text is PII by definition (name/email/phone/
            # address) — opt out of OpenAI retaining this request, same
            # policy the main chat pipeline uses (CHAT_OPENAI_STORE).
            store=False,
        )

        response_text = response.choices[0].message.content or ""

        # Try to parse JSON from the response
        # Sometimes OpenAI wraps it in markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text

        parsed = json.loads(json_str)

        # Check if this is actually a business card
        if parsed.get("is_business_card") is False:
            raise ExtractionError(
                code="NOT_A_BUSINESS_CARD",
                message="This does not appear to be a business card.",
                details=parsed.get("error", "Document does not match business card format"),
            )

        # Normalize response to expected format
        data = parsed.get("data", {})
        phone_data = data.get("phone") or []
        fax_data = data.get("fax") or []
        address_data = data.get("address")

        full_address = None
        if isinstance(address_data, dict):
            full_address = (address_data.get("full_address") or "").strip() or None
        elif isinstance(address_data, list):
            joined = ", ".join(str(a) for a in address_data if a).strip()
            full_address = joined or None

        # Also check top-level full_address from OpenAI response
        if not full_address:
            top_full_addr = (data.get("full_address") or "").strip()
            if top_full_addr:
                full_address = top_full_addr

        # Process phone and fax numbers into parallel arrays
        phone_result = _process_phone_objects(phone_data)
        fax_result = _process_fax_objects(fax_data)

        result = {
            "name": data.get("name"),
            "title": _validate_text_field(data.get("title"), "title"),
            "company": _validate_text_field(data.get("company"), "company"),
            "email": data.get("email"),
            "country_code": phone_result["country_code"],
            "mobile_numbers": phone_result["mobile_numbers"],
            "fax_country_code": fax_result["fax_country_code"],
            "fax_numbers": fax_result["fax_numbers"],
            "website": data.get("website"),
            "full_address": full_address,
            "address": _build_parsed_address(address_data),
            "other": data.get("other") or [],
        }

        return result

    except json.JSONDecodeError as e:
        # response_text is the model's structuring of extracted PII —
        # never write it to the log, just enough to debug the parse
        # failure itself (length + the JSON error).
        logger.error(
            f"Failed to parse OpenAI response as JSON "
            f"(response length={len(response_text)} chars): {e}"
        )
        raise ExtractionError(
            code="EXTRACTION_FAILED",
            message="Unable to extract information from image.",
            details=f"OpenAI response was not valid JSON: {str(e)}",
        )
    except (OpenAIError, ExtractionError):
        raise
    except OAIAuthenticationError as e:
        logger.error(f"OpenAI auth error: {e}")
        raise OpenAIError(
            code="OPENAI_API_AUTH_ERROR",
            message="Service temporarily unavailable.",
            details=f"OpenAI auth: {e}",
        )
    except OAIPermissionDeniedError as e:
        logger.error(f"OpenAI permission error: {e}")
        raise OpenAIError(
            code="OPENAI_API_AUTH_ERROR",
            message="Service temporarily unavailable.",
            details=f"OpenAI permission: {e}",
        )
    except OAIRateLimitError as e:
        logger.error(f"OpenAI rate limit exceeded: {e}")
        raise OpenAIError(
            code="OPENAI_API_QUOTA_ERROR",
            message="Service busy. Please try again shortly.",
            details=f"OpenAI rate limit: {e}",
        )
    except (OAIAPIConnectionError, OAIAPITimeoutError) as e:
        logger.error(f"OpenAI network/timeout error: {e}")
        raise OpenAIError(
            code="OPENAI_API_TIMEOUT",
            message="Request timed out. Please try again.",
            details=f"OpenAI network: {e}",
        )
    except OAIUnprocessableEntityError as e:
        logger.error(f"OpenAI content error: {e}")
        raise OpenAIError(
            code="OPENAI_API_ERROR",
            message="Unable to process this content.",
            details=f"OpenAI content: {e}",
        )
    except (OAIBadRequestError, OAINotFoundError, OAIInternalServerError) as e:
        logger.error(f"OpenAI API error: {e}")
        raise OpenAIError(
            code="OPENAI_API_ERROR",
            message="Service temporarily unavailable.",
            details=f"OpenAI API: {e}",
        )
    except Exception as e:
        logger.error(f"OpenAI unexpected error: {e}", exc_info=True)
        raise OpenAIError(
            code="OPENAI_API_ERROR",
            message="Service temporarily unavailable.",
            details=f"Unexpected OpenAI error: {str(e)}",
        )


def process_business_card(image_bytes: bytes) -> dict:
    """
    Full pipeline: extract text from image → structure via OpenAI.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Dict ready to construct BusinessCardResponse, including:
        - name, title, company, email, country_code, mobile_numbers, website, full_address, address, other
        - raw_text (full OCR output)
        - error (None if successful, error message if failed)

    This function is synchronous and suitable for use in async endpoints via:
        result = await asyncio.to_thread(process_business_card, image_bytes)
    """
    try:
        # Step 1: Extract text from image
        raw_text = extract_text_from_image(image_bytes)

        # Step 2: Structure extracted text
        structured = structure_extracted_text(raw_text)

        # Return with raw_text and no error
        return {
            **structured,
            "raw_text": raw_text,
            "error": None,
        }

    except (BusinessCardError,) as e:
        # Expected errors from Vision/OpenAI APIs
        logger.warning(f"Business card processing failed ({e.code}): {e.details}")
        return {
            "name": None,
            "title": None,
            "company": None,
            "email": None,
            "country_code": [],
            "mobile_numbers": [],
            "fax_country_code": [],
            "fax_numbers": [],
            "website": None,
            "full_address": None,
            "address": None,
            "other": [],
            "raw_text": "",
            "error": e.message,
            "error_code": e.code,
        }
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in business card processing: {e}", exc_info=True)
        return {
            "name": None,
            "title": None,
            "company": None,
            "email": None,
            "country_code": [],
            "mobile_numbers": [],
            "fax_country_code": [],
            "fax_numbers": [],
            "website": None,
            "full_address": None,
            "address": None,
            "other": [],
            "raw_text": "",
            "error": "An unexpected error occurred. Please try again.",
            "error_code": "INTERNAL_ERROR",
        }
