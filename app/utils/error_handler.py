"""
Professional error handling utilities for REST APIs.

Provides functions to create consistent, production-grade error responses
across all endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.schemas.error import ApiErrorResponse, ErrorDetail


def create_error_response(
    code: str,
    message: str,
    status: int,
    field: Optional[str] = None,
    details: Optional[str] = None,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
) -> ApiErrorResponse:
    """
    Create a professional API error response.

    Args:
        code: Machine-readable error code (e.g., 'INVALID_MIME_TYPE')
        message: User-friendly error message
        status: HTTP status code (400, 401, 403, 404, 422, 500, etc.)
        field: Optional field name if error is field-specific
        details: Optional additional context or suggestions
        request_id: Optional unique request identifier
        path: Optional API endpoint path

    Returns:
        ApiErrorResponse object ready to be returned as JSON

    Example:
        >>> error = create_error_response(
        ...     code="INVALID_MIME_TYPE",
        ...     message="Unsupported image format",
        ...     status=400,
        ...     field="file",
        ...     details="Supported: JPEG, PNG, WebP, HEIC",
        ...     request_id="a3f7q2x9",
        ...     path="/api/v1/business-card/upload",
        ... )
        >>> # Return as: JSONResponse(status_code=400, content=error.model_dump())
    """

    return ApiErrorResponse(
        success=False,
        error=ErrorDetail(
            code=code,
            message=message,
            field=field,
            details=details,
        ),
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        path=path,
        status=status,
    )


# ============================================================================
# Standard Error Codes & Messages
# ============================================================================

# Client Errors (4xx)
ERROR_INVALID_MIME_TYPE = {
    "code": "INVALID_MIME_TYPE",
    "message": "Unsupported image format",
    "details": "Supported formats: JPEG, PNG, WebP, HEIC",
}

ERROR_FILE_TOO_LARGE = {
    "code": "FILE_TOO_LARGE",
    "message": "File exceeds maximum size limit",
    "details": "Maximum file size: 5 MB",
}

ERROR_INVALID_FILE = {
    "code": "INVALID_FILE",
    "message": "Unable to read file",
    "details": "File may be corrupted or empty",
}

ERROR_INVALID_FILE_CONTENT = {
    "code": "INVALID_FILE_CONTENT",
    "message": "File content does not match a valid image",
    "details": "The uploaded bytes are not a decodable image, or don't match the declared format",
}

ERROR_MALWARE_DETECTED = {
    "code": "MALWARE_DETECTED",
    "message": "The uploaded file was flagged by malware scanning",
    "details": "This file cannot be processed",
}

ERROR_SCAN_UNAVAILABLE = {
    "code": "SCAN_UNAVAILABLE",
    "message": "Malware scanning is temporarily unavailable",
    "details": "Please try again shortly",
}

ERROR_MISSING_FIELD = {
    "code": "MISSING_FIELD",
    "message": "Required field is missing",
}

ERROR_INVALID_REQUEST = {
    "code": "INVALID_REQUEST",
    "message": "Invalid request format or parameters",
}

# Server Errors (5xx)
ERROR_VISION_API_AUTH = {
    "code": "VISION_API_AUTH_ERROR",
    "message": "Unable to process image",
    "details": "Service temporarily unavailable. Please try again later.",
}

ERROR_VISION_API_FAILED = {
    "code": "VISION_API_ERROR",
    "message": "Image processing failed",
    "details": "Please try again with a different image.",
}

ERROR_OPENAI_API_AUTH = {
    "code": "OPENAI_API_AUTH_ERROR",
    "message": "Unable to process request",
    "details": "Service temporarily unavailable. Please try again later.",
}

ERROR_OPENAI_API_FAILED = {
    "code": "OPENAI_API_ERROR",
    "message": "Data extraction failed",
    "details": "Please try again in a few moments.",
}

ERROR_VISION_API_QUOTA = {
    "code": "VISION_API_QUOTA_ERROR",
    "message": "Service busy. Please try again shortly.",
}

ERROR_VISION_API_TIMEOUT = {
    "code": "VISION_API_TIMEOUT",
    "message": "Request timed out. Please try again.",
}

ERROR_OPENAI_API_QUOTA = {
    "code": "OPENAI_API_QUOTA_ERROR",
    "message": "Service busy. Please try again shortly.",
}

ERROR_OPENAI_API_TIMEOUT = {
    "code": "OPENAI_API_TIMEOUT",
    "message": "Request timed out. Please try again.",
}

ERROR_INTERNAL_SERVER = {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "details": "Our team has been notified. Please try again later.",
}

# Extraction Failures (200 + error)
ERROR_NO_TEXT_DETECTED = {
    "code": "NO_TEXT_DETECTED",
    "message": "Image contains no readable text",
    "details": "Please upload a clearer image of the business card.",
}

ERROR_EXTRACTION_FAILED = {
    "code": "EXTRACTION_FAILED",
    "message": "Unable to extract information from image",
    "details": "The image may not be a business card or is unclear.",
}

ERROR_NOT_A_BUSINESS_CARD = {
    "code": "NOT_A_BUSINESS_CARD",
    "message": "This does not appear to be a business card",
    "details": "Please upload an image of a business card (contains name, company, contact info, etc.)",
}

ERROR_EMPTY_INPUT = {
    "code": "EMPTY_INPUT",
    "message": "No content to process",
}
