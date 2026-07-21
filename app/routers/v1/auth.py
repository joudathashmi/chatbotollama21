"""
Authentication endpoints (Risk-20-3 — JWT bearer tokens):

  POST /api/v1/auth/login    username + password → access + refresh tokens
  POST /api/v1/auth/refresh  refresh token       → new access + refresh

These two routes are the ONLY unauthenticated /api/v1/* endpoints — they
are mounted in app/main.py without the shared `_auth` dependency, since a
caller cannot present a bearer token before obtaining one. Both are
IP-rate-limited to blunt credential brute-forcing.

Refresh tokens are rotated on every use; reuse of a previously rotated
token revokes the whole token family (theft detection).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth import (
    authenticate_user,
    access_token_ttl_seconds,
    issue_token_pair,
    rotate_refresh_token,
    _unauthorized,
)
from app.config import AUTH_RATE_LIMIT
from app.rate_limit import rate_limit_ip

router = APIRouter(prefix="/auth", tags=["auth"])

_login_rl = rate_limit_ip("auth_login", *AUTH_RATE_LIMIT)
_refresh_rl = rate_limit_ip("auth_refresh", *AUTH_RATE_LIMIT)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=1, max_length=1_000)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1, max_length=4_000)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # access-token lifetime, seconds


@router.post(
    "/login",
    summary="Exchange username + password for JWT access + refresh tokens",
    response_model=TokenResponse,
    dependencies=[Depends(_login_rl)],
    responses={
        401: {"description": "Invalid username or password."},
        429: {"description": "Rate limit exceeded — see Retry-After header."},
        503: {"description": "Auth not configured on the server (JWT_SECRET_KEY unset)."},
    },
)
async def login(req: LoginRequest) -> TokenResponse:
    username = authenticate_user(req.username, req.password)
    if username is None:
        raise _unauthorized("Invalid username or password.")
    access, refresh = issue_token_pair(username)
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=access_token_ttl_seconds(),
    )


@router.post(
    "/refresh",
    summary="Exchange a refresh token for a new access (+ refresh) token",
    response_model=TokenResponse,
    dependencies=[Depends(_refresh_rl)],
    responses={
        401: {"description": "Invalid, expired, or reused refresh token."},
        429: {"description": "Rate limit exceeded — see Retry-After header."},
        503: {"description": "Auth not configured on the server (JWT_SECRET_KEY unset)."},
    },
)
async def refresh(req: RefreshRequest) -> TokenResponse:
    access, new_refresh = rotate_refresh_token(req.refresh_token)
    return TokenResponse(
        access_token=access,
        refresh_token=new_refresh,
        expires_in=access_token_ttl_seconds(),
    )
