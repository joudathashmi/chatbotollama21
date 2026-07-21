from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from app.logger import logger
from app.utils.error_handler import create_error_response

# Fallback machine-readable code per HTTP status, used when the raising
# code didn't supply a more specific one (see `http_exception_handler`).
_STATUS_CODE_NAMES: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    408: "REQUEST_TIMEOUT",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "UNPROCESSABLE_ENTITY",
    429: "RATE_LIMIT_EXCEEDED",
    500: "INTERNAL_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


def _status_code_name(status_code: int) -> str:
    return _STATUS_CODE_NAMES.get(status_code, f"HTTP_{status_code}")


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException,
) -> JSONResponse:
    """Catches every `raise HTTPException(...)` in the app — auth
    failures, rate limits, and any ad-hoc 4xx/5xx — and renders it in
    the standard error shape.

    Header pass-through is load-bearing: `verify_credentials` (app/
    auth.py) sets `WWW-Authenticate` on 401s so the browser's native
    Basic Auth prompt fires; rate-limit responses set `Retry-After`.
    Both MUST survive this handler unchanged.
    """
    code = getattr(exc, "error_code", None) or _status_code_name(exc.status_code)
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

    body = create_error_response(
        code=code,
        message=message,
        status=exc.status_code,
        path=str(request.url.path),
    ).model_dump()

    retry_after = getattr(exc, "retry_after_seconds", None)
    if retry_after is not None:
        body["retry_after_seconds"] = retry_after

    if exc.status_code >= 500:
        logger.error(f"HTTP {exc.status_code} on {request.method} {request.url.path}: {message}")
    else:
        logger.warning(f"HTTP {exc.status_code} on {request.method} {request.url.path}: {message}")

    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        headers=dict(exc.headers) if exc.headers else None,
    )


def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle body/query validation errors (422) in the same shape as
    every other error. `details` carries Pydantic's per-field error
    list so clients can still pinpoint the bad field."""
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {exc.errors()}"
    )
    body = create_error_response(
        code="VALIDATION_ERROR",
        message="One or more fields failed validation.",
        status=HTTP_422_UNPROCESSABLE_ENTITY,
        details=str(exc.errors()),
        path=str(request.url.path),
    ).model_dump()
    return JSONResponse(status_code=HTTP_422_UNPROCESSABLE_ENTITY, content=body)


def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that logs the full stack trace and never
    leaks internal details to the client."""
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    body = create_error_response(
        code="INTERNAL_ERROR",
        message="An unexpected error occurred. Please try again later.",
        status=500,
        path=str(request.url.path),
    ).model_dump()
    return JSONResponse(status_code=500, content=body)
