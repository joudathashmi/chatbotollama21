from __future__ import annotations

import time
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming requests and outcomes.

    Captures:
    - method
    - URL path
    - client IP
    - response status code
    - response time
    - exceptions (stack traces)
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]):
        client_ip = self._get_client_ip(request)
        method = request.method
        path = request.url.path

        start = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000.0

            # INFO for successful/normal responses; WARNING+ for 4xx/5xx.
            if response.status_code >= 500:
                logger.error(
                    f"HTTP {method} {path} client_ip={client_ip} status={response.status_code} time_ms={duration_ms:.2f}"
                )
            elif response.status_code >= 400:
                logger.warning(
                    f"HTTP {method} {path} client_ip={client_ip} status={response.status_code} time_ms={duration_ms:.2f}"
                )
            else:
                logger.info(
                    f"HTTP {method} {path} client_ip={client_ip} status={response.status_code} time_ms={duration_ms:.2f}"
                )

            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                f"Unhandled exception in HTTP {method} {path} client_ip={client_ip} time_ms={duration_ms:.2f}: {exc}"
            )
            raise

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        # Honor common reverse-proxy headers first.
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # x-forwarded-for can contain a list: "client, proxy1, proxy2"
            return xff.split(",", 1)[0].strip()

        # Fall back to Starlette's client.
        if request.client and request.client.host:
            return request.client.host

        return "unknown"

