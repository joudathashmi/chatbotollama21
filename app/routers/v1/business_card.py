"""
Business Card Reader API endpoint.

POST /api/v1/business-card/upload — Extract contact info from business card image.
"""

from __future__ import annotations

import asyncio
import logging
import time
import string
import secrets
from typing import Union

from fastapi import APIRouter, Depends, File, UploadFile, Request
from fastapi.responses import JSONResponse

from app.config import BUSINESS_CARD_RATE_LIMIT, MALWARE_SCAN_FAIL_OPEN
from app.rate_limit import rate_limit
from app.schemas.business_card import (
    BusinessCardResponse,
    BusinessCardData,
    ParsedAddress,
    LinkedInResult,
)
from app.services.business_card_engine import (
    process_business_card,
    VisionAPIError,
    OpenAIError,
    ExtractionError,
    PromptInjectionError,
)
from app.services.image_validation import sniff_and_validate
from app.services.malware_scanner import ScanVerdict, check_status as check_malware_scan_status, scan_file
from app.utils.error_handler import (
    create_error_response,
    ERROR_INVALID_MIME_TYPE,
    ERROR_FILE_TOO_LARGE,
    ERROR_INVALID_FILE,
    ERROR_INVALID_FILE_CONTENT,
    ERROR_MALWARE_DETECTED,
    ERROR_SCAN_UNAVAILABLE,
    ERROR_INTERNAL_SERVER,
)

logger = logging.getLogger(__name__)

# Create router with prefix (tags removed to avoid compatibility issues)
router = APIRouter(prefix="/business-card")

_business_card_rl = rate_limit("business_card", *BUSINESS_CARD_RATE_LIMIT)

# Supported MIME types
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


def _generate_request_id() -> str:
    """
    Generate a short unique request ID (8 alphanumeric characters).

    Example: "bx7k9mq2", "a3f7q2x9"
    """
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


class _FileTooLargeDuringRead(Exception):
    """Raised mid-stream by `_read_bounded` when the upload exceeds the
    cap. Distinct from the declared-size check below (`file.size`),
    which is skipped by some clients/proxies that omit it — this is the
    check that actually holds regardless of what the client claims."""

    def __init__(self, bytes_read: int):
        self.bytes_read = bytes_read


async def _read_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in capped chunks instead of one unbounded
    `await file.read()`. `file.size` (checked separately, below) is the
    client-declared Content-Length and can be absent or wrong; this is
    the enforcement that actually bounds memory use no matter what the
    client claims."""
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024  # 1 MiB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise _FileTooLargeDuringRead(total)
        chunks.append(chunk)
    return b"".join(chunks)


@router.get(
    "/scan-status",
    summary="Live malware-scanner status",
    response_description=(
        "Whether malware scanning is enabled and, if so, whether the "
        "configured backend (clamscan / defender) is actually available "
        "right now — not just what MALWARE_SCAN_BACKEND is set to."
    ),
)
async def scan_status():
    """Answers 'is malware scanning actually working' directly: runs
    `clamscan --version` or checks for MpCmdRun.exe right now, instead
    of just echoing config. Use this to verify the AV install before
    trusting it in front of real uploads — on Windows or Ubuntu,
    whichever backend is configured.
    """
    status = await asyncio.to_thread(check_malware_scan_status)
    return {
        "backend": status["backend"],
        "enabled": status["enabled"],
        "available": status["available"],
        "detail": status["detail"],
        "fail_open_on_scan_failure": MALWARE_SCAN_FAIL_OPEN,
    }


@router.post(
    "/upload",
    summary="Extract business card information from image",
    response_description="Structured contact information (name, email, phone, company, etc.)",
    response_model=BusinessCardResponse,
    status_code=200,
    dependencies=[Depends(_business_card_rl)],
)
async def upload_business_card(
    file: UploadFile = File(...),
    request: Request = None,
    resolve_linkedin: bool = True,
    provider: str = "ddg",
) -> Union[BusinessCardResponse, JSONResponse]:
    """
    Upload a business card image and extract contact information.

    **Supported formats:** JPG, PNG, WebP, HEIC

    **File size limit:** 5 MB

    **LinkedIn enrichment (enabled by default):** every successful upload also
    looks up the person's LinkedIn profile and returns it under the top-level
    `linkedin` field as a structured object. The `linkedin_urls` array contains
    candidate objects with `url`, `score`, `name`, `title`, and `company`.
    `?provider=ddg` (default, free DuckDuckGo) or `?provider=serp`
    (Serper.dev, needs SERPER_API_KEY). This adds latency (search is
    network-bound); pass `?resolve_linkedin=false` to skip it for fastest
    extraction-only calls.

    **Response:** Structured JSON with extracted fields:
    - `name`: Person's full name (or null)
    - `title`: Job title/position (or null)
    - `company`: Company name (or null)
    - `email`: Email address (or null)
    - `mobile_numbers`: Validated phone numbers in international format (array)
    - `website`: Website URL (or null)
    - `full_address`: Full address string as it appeared on the card (or null)
    - `address`: Structured address with city, state, country, pincode (or null)
    - `other`: Unclassified text (array)
    - `raw_text`: Full OCR output (for reference)
    - `request_id`: Unique identifier for this request
    - `processing_time_ms`: Time taken to process (in milliseconds)
    - `source_file`: Uploaded filename
    - `error`: Error message if extraction failed (or null)

    If any field cannot be extracted, it will be null or empty.
    If extraction fails completely, check the `error` field.
    """

    # Generate request ID for tracking
    request_id = _generate_request_id()

    # Get request path
    path = str(request.url.path) if request else "/api/v1/business-card/upload"

    # Validate file type
    if file.content_type not in ALLOWED_MIME_TYPES:
        error_response = create_error_response(
            code=ERROR_INVALID_MIME_TYPE["code"],
            message=ERROR_INVALID_MIME_TYPE["message"],
            field="file",
            details=ERROR_INVALID_MIME_TYPE["details"],
            status=400,
            request_id=request_id,
            path=path,
        )
        logger.warning(
            f"[{request_id}] Invalid MIME type: {file.content_type} for {file.filename}"
        )
        return JSONResponse(
            status_code=400,
            content=error_response.model_dump(),
        )

    # Fast-path rejection when the client honestly declared an
    # oversized upload — avoids reading anything for the common case.
    if file.size and file.size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / 1024 / 1024
        actual_mb = file.size / 1024 / 1024
        error_response = create_error_response(
            code=ERROR_FILE_TOO_LARGE["code"],
            message=ERROR_FILE_TOO_LARGE["message"],
            field="file",
            details=f"Maximum: {max_mb:.0f} MB, Uploaded: {actual_mb:.1f} MB",
            status=400,
            request_id=request_id,
            path=path,
        )
        logger.warning(
            f"[{request_id}] File too large: {actual_mb:.1f} MB (max: {max_mb:.0f} MB)"
        )
        return JSONResponse(
            status_code=400,
            content=error_response.model_dump(),
        )

    try:
        # Track processing time
        start_time = time.perf_counter()

        # Read file bytes in capped chunks — the AUTHORITATIVE size
        # enforcement. `file.size` above is the client-declared
        # Content-Length, which can be absent or understated; this
        # bounds actual memory use no matter what the client claims.
        try:
            image_bytes = await _read_bounded(file, MAX_FILE_SIZE_BYTES)
        except _FileTooLargeDuringRead as e:
            max_mb = MAX_FILE_SIZE_BYTES / 1024 / 1024
            actual_mb = e.bytes_read / 1024 / 1024
            error_response = create_error_response(
                code=ERROR_FILE_TOO_LARGE["code"],
                message=ERROR_FILE_TOO_LARGE["message"],
                field="file",
                details=f"Maximum: {max_mb:.0f} MB (exceeded during upload)",
                status=400,
                request_id=request_id,
                path=path,
            )
            logger.warning(
                f"[{request_id}] File too large during streamed read "
                f"(>{actual_mb:.1f} MB, max: {max_mb:.0f} MB)"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.model_dump(),
            )

        # Validate file is not empty
        if not image_bytes:
            error_response = create_error_response(
                code=ERROR_INVALID_FILE["code"],
                message=ERROR_INVALID_FILE["message"],
                field="file",
                details=ERROR_INVALID_FILE["details"],
                status=400,
                request_id=request_id,
                path=path,
            )
            logger.warning(f"[{request_id}] Empty file uploaded: {file.filename}")
            return JSONResponse(
                status_code=400,
                content=error_response.model_dump(),
            )

        # Real-content validation — Content-Type is client-declared and
        # spoofable (a script or executable can claim "image/png").
        # Verify the actual bytes are a real, decodable image matching
        # the declared format before anything reaches Vision/OpenAI.
        content_ok, content_reason = sniff_and_validate(image_bytes, file.content_type)
        if not content_ok:
            error_response = create_error_response(
                code=ERROR_INVALID_FILE_CONTENT["code"],
                message=ERROR_INVALID_FILE_CONTENT["message"],
                field="file",
                details=content_reason,
                status=400,
                request_id=request_id,
                path=path,
            )
            logger.warning(
                f"[{request_id}] File content validation failed for "
                f"{file.filename}: {content_reason}"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.model_dump(),
            )

        # Malware scan — heuristic checks (EICAR, embedded PDF JS) plus
        # a native AV CLI (clamscan on Linux, Defender on Windows); see
        # app/services/malware_scanner.py. Blocking I/O, so it runs off
        # the event loop like the rest of this pipeline.
        scan_result = await asyncio.to_thread(scan_file, image_bytes, file.filename or "upload")
        if scan_result.verdict == ScanVerdict.INFECTED:
            error_response = create_error_response(
                code=ERROR_MALWARE_DETECTED["code"],
                message=ERROR_MALWARE_DETECTED["message"],
                field="file",
                details=ERROR_MALWARE_DETECTED["details"],
                status=400,
                request_id=request_id,
                path=path,
            )
            logger.warning(
                f"[{request_id}] Malware detected in {file.filename} "
                f"(backend={scan_result.backend}, signature={scan_result.detail})"
            )
            return JSONResponse(status_code=400, content=error_response.model_dump())
        elif scan_result.verdict == ScanVerdict.SCAN_FAILED and not MALWARE_SCAN_FAIL_OPEN:
            error_response = create_error_response(
                code=ERROR_SCAN_UNAVAILABLE["code"],
                message=ERROR_SCAN_UNAVAILABLE["message"],
                field="file",
                details=ERROR_SCAN_UNAVAILABLE["details"],
                status=503,
                request_id=request_id,
                path=path,
            )
            logger.error(
                f"[{request_id}] Malware scan unavailable (backend="
                f"{scan_result.backend}): {scan_result.detail} — "
                f"rejecting upload (MALWARE_SCAN_FAIL_OPEN=false)"
            )
            return JSONResponse(status_code=503, content=error_response.model_dump())
        elif scan_result.verdict == ScanVerdict.SCAN_FAILED:
            logger.warning(
                f"[{request_id}] Malware scan unavailable (backend="
                f"{scan_result.backend}): {scan_result.detail} — "
                f"proceeding anyway (MALWARE_SCAN_FAIL_OPEN=true)"
            )

        # Process business card (blocking operation → asyncio.to_thread)
        result = await asyncio.to_thread(process_business_card, image_bytes)

        # Success if no error, failure if error_code is present
        success = result.get("error_code") is None

        # Optional LinkedIn enrichment. Network-bound; only attempted on a
        # successful extraction. Never breaks the response — the resolver
        # returns a match_type="none" result with an error on any failure.
        linkedin_result = None
        if resolve_linkedin and success:
            prov = (provider or "ddg").strip().lower()
            if prov not in ("ddg", "serp"):
                prov = "ddg"
            from app.services.linkedin_resolver import resolve_linkedin as _resolve_li
            li = await _resolve_li(result, prov)
            linkedin_result = LinkedInResult(**li)

        # Calculate processing time in milliseconds (includes LinkedIn lookup)
        elapsed_sec = time.perf_counter() - start_time
        processing_time_ms = elapsed_sec * 1000

        # Get current timestamp
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Build ParsedAddress if address data exists
        address_data = result.get("address")
        parsed_address = None
        if address_data and isinstance(address_data, dict):
            parsed_address = ParsedAddress(**address_data)

        response = BusinessCardResponse(
            success=success,
            data=BusinessCardData(
                name=result.get("name"),
                title=result.get("title"),
                company=result.get("company"),
                email=result.get("email"),
                country_code=result.get("country_code", []),
                mobile_numbers=result.get("mobile_numbers", []),
                fax_country_code=result.get("fax_country_code", []),
                fax_numbers=result.get("fax_numbers", []),
                website=result.get("website"),
                full_address=result.get("full_address"),
                address=parsed_address,
                other=result.get("other", []),
                raw_text=result.get("raw_text", ""),
                source_file=file.filename,
                linkedin=linkedin_result,
            ),
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            timestamp=timestamp,
            error=result.get("error"),
            error_code=result.get("error_code"),
        )

        # Log with appropriate level based on success/failure. PII
        # (extracted name/email/phone/etc.) is deliberately never
        # written to the application log — only presence/absence and
        # timing, so log aggregation/retention doesn't become a second
        # copy of the PII this endpoint exists to extract.
        if response.error:
            logger.info(
                f"[{request_id}] Extraction failed ({response.error_code}): "
                f"{file.filename}"
            )
        else:
            logger.info(
                f"[{request_id}] ✓ Success: {file.filename} "
                f"(name_extracted={bool(response.data.name)}, "
                f"{processing_time_ms:.1f}ms)"
            )

        return response

    except Exception as e:
        # Categorize and log errors
        elapsed_sec = time.perf_counter() - start_time
        processing_time_ms = elapsed_sec * 1000

        if isinstance(e, PromptInjectionError):
            # Risk-20-6: the uploaded card carried an obfuscated prompt-
            # injection payload. That's a bad request from the client, not a
            # server fault — 400, not 500. `message` is deliberately generic
            # ("Unable to process this image."); the specific detection
            # category stays server-side in `details`/the audit log so we
            # don't hand an attacker a probe oracle for tuning payloads.
            error_code = e.code
            error_msg = e.message
            error_details = e.details
            status_code = 400
        elif isinstance(e, (VisionAPIError, OpenAIError)):
            error_code = e.code
            error_msg = e.message
            error_details = e.details
            # Map error codes to proper HTTP status codes
            if "QUOTA" in error_code:
                status_code = 503  # Service Unavailable
            elif "TIMEOUT" in error_code:
                status_code = 504  # Gateway Timeout
            elif "AUTH" in error_code:
                status_code = 500  # Don't reveal auth issues to client
            else:
                status_code = 500
        elif isinstance(e, ExtractionError):
            # Extraction failures return 200 with error field
            # These are already handled in process_business_card
            # This shouldn't reach here
            error_code = e.code
            error_msg = e.message
            error_details = e.details
            status_code = 200
        else:
            # Unexpected error — never expose raw details to client
            error_code = ERROR_INTERNAL_SERVER["code"]
            error_msg = ERROR_INTERNAL_SERVER["message"]
            error_details = ERROR_INTERNAL_SERVER.get("details", "")
            status_code = 500
            logger.error(
                f"[{request_id}] Unexpected error: {str(e)}",
                exc_info=True,
            )

        error_response = create_error_response(
            code=error_code,
            message=error_msg,
            details=error_details,
            status=status_code,
            request_id=request_id,
            path=path,
        )

        logger.error(
            f"[{request_id}] Error ({status_code}) {error_code}: {error_msg}"
        )

        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(),
        )
