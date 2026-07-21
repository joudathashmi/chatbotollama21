"""
Professional error response schemas following REST API best practices.

All errors (400, 500, 200+error) use the same professional format.
Based on industry standards: Google API, AWS, Stripe, etc.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Detailed error information for debugging and logging."""

    code: str
    """Machine-readable error code (e.g., 'INVALID_MIME_TYPE', 'VISION_API_ERROR')."""

    message: str
    """User-friendly error message to display to client."""

    field: Optional[str] = None
    """Field name if error is field-specific (e.g., 'file' for upload errors)."""

    details: Optional[str] = None
    """Optional additional context (e.g., 'Supported: JPEG, PNG, WebP, HEIC')."""


class ApiErrorResponse(BaseModel):
    """
    Professional API error response format.

    Used for all error scenarios:
    - 400 Bad Request (client errors)
    - 500 Internal Server Error (server errors)
    - 200 OK with error field (extraction failures that don't prevent response)

    Example:
        {
            "success": false,
            "error": {
                "code": "INVALID_MIME_TYPE",
                "message": "Unsupported image format",
                "field": "file",
                "details": "Supported formats: JPEG, PNG, WebP, HEIC"
            },
            "request_id": "a3f7q2x9",
            "timestamp": "2026-06-03T14:30:45.123Z",
            "path": "/api/v1/business-card/upload",
            "status": 400
        }
    """

    success: bool = False
    """Whether the request succeeded (true) or failed (false)."""

    error: ErrorDetail
    """Error details."""

    request_id: Optional[str] = None
    """Unique request identifier for tracking and debugging."""

    timestamp: Optional[str] = None
    """ISO 8601 timestamp when error occurred."""

    path: Optional[str] = None
    """API endpoint path where error occurred."""

    status: int
    """HTTP status code."""

    trace_id: Optional[str] = None
    """Optional trace ID for distributed tracing (future use)."""

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "example": "Invalid MIME Type (400)",
                    "value": {
                        "error": {
                            "code": "INVALID_MIME_TYPE",
                            "message": "Unsupported image format",
                            "field": "file",
                            "details": "Supported formats: JPEG, PNG, WebP, HEIC",
                        },
                        "request_id": "a3f7q2x9",
                        "timestamp": "2026-06-03T14:30:45.123Z",
                        "path": "/api/v1/business-card/upload",
                        "status": 400,
                    },
                },
                {
                    "example": "File Too Large (400)",
                    "value": {
                        "error": {
                            "code": "FILE_TOO_LARGE",
                            "message": "File exceeds maximum size limit",
                            "field": "file",
                            "details": "Maximum file size: 5 MB. Uploaded: 6.2 MB",
                        },
                        "request_id": "k7m2pq9x",
                        "timestamp": "2026-06-03T14:31:12.456Z",
                        "path": "/api/v1/business-card/upload",
                        "status": 400,
                    },
                },
                {
                    "example": "Vision API Error (500)",
                    "value": {
                        "error": {
                            "code": "VISION_API_ERROR",
                            "message": "Unable to process image. Service temporarily unavailable.",
                            "details": "Please try again in a few moments.",
                        },
                        "request_id": "b4n3xq8w",
                        "timestamp": "2026-06-03T14:32:30.789Z",
                        "path": "/api/v1/business-card/upload",
                        "status": 500,
                    },
                },
                {
                    "example": "No Text Detected (200 with error)",
                    "value": {
                        "error": {
                            "code": "NO_TEXT_DETECTED",
                            "message": "Image contains no readable text",
                            "details": "Please upload a clearer image of the business card",
                        },
                        "request_id": "c5o4yr9v",
                        "timestamp": "2026-06-03T14:33:15.234Z",
                        "path": "/api/v1/business-card/upload",
                        "status": 200,
                    },
                },
            ]
        }
