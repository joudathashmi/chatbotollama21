"""JWT bearer-token authentication for FastAPI routes (Risk-20-3).

Replaces the previous HTTP Basic Auth. The public surface consumed by the
rest of the app is unchanged: `verify_credentials` is still a FastAPI
dependency returning the authenticated username (`sub`) as a string, so the
router `_auth` list (app/main.py), the rate-limiter key (app/rate_limit.py),
`/health/data`, and every test override keep working untouched.

Flow:
  POST /api/v1/auth/login    username + password  → access + refresh tokens
  POST /api/v1/auth/refresh  refresh token        → new access (+ refresh)
  every other /api/v1/*      Authorization: Bearer <access-token>

Tokens are self-issued HS256 JWTs signed with config.JWT_SECRET_KEY, bound
to JWT_ISSUER / JWT_AUDIENCE. Refresh tokens are tracked server-side
(jti + family id) so rotation invalidates the previous token and reuse of a
stolen refresh revokes the entire family.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import config
from app.services.token_store import get_refresh_token_store

# auto_error=False so a missing/!bearer header yields OUR 401 (+ WWW-
# Authenticate: Bearer) instead of HTTPBearer's default bare 403.
_bearer = HTTPBearer(auto_error=False)

_REALM = "MISA Intelligence API"
_MISSING_CONFIG_DETAIL = "Server auth not configured — set JWT_SECRET_KEY."
_INVALID_TOKEN_DETAIL = "Invalid or expired token."
_FORBIDDEN_DETAIL = "Insufficient privileges for this endpoint."

# JWT `typ` claim values distinguishing the two token kinds.
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _unauthorized(detail: str = _INVALID_TOKEN_DETAIL) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": f'Bearer realm="{_REALM}"'},
    )


def _forbidden(detail: str = _FORBIDDEN_DETAIL) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def _require_secret() -> str:
    secret = (config.JWT_SECRET_KEY or "").strip()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_MISSING_CONFIG_DETAIL,
        )
    return secret


# ---------------------------------------------------------------------------
# Credential verification (username + password → username)
# ---------------------------------------------------------------------------
def authenticate_user(username: str, password: str) -> "str | None":
    """Return the username if the credentials are valid, else None.

    Two credential sources, checked in order:
      1. config.AUTH_USERS — named accounts with bcrypt password hashes
         (provisioned via MISA_AUTH_USERS).
      2. The bootstrap account (config.API_USERNAME / API_PASSWORD), only
         when MISA_ALLOW_PLAINTEXT_BOOTSTRAP is enabled (default: local/dev
         only — disabled in production).
    """
    if not username or not password:
        return None

    stored_hash = config.AUTH_USERS.get(username)
    if stored_hash:
        try:
            if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                return username
        except (ValueError, TypeError):
            return None
        return None

    if not config.ALLOW_PLAINTEXT_BOOTSTRAP:
        return None

    api_user = (config.API_USERNAME or "").strip()
    api_pass = (config.API_PASSWORD or "").strip()
    if api_user and api_pass and username == api_user:
        if secrets.compare_digest(password.encode("utf-8"), api_pass.encode("utf-8")):
            return username

    return None


# ---------------------------------------------------------------------------
# Token creation / decoding
# ---------------------------------------------------------------------------
def _create_token(
    sub: str,
    typ: str,
    ttl: timedelta,
    *,
    family_id: str | None = None,
    jti: str | None = None,
) -> tuple[str, str, float]:
    """Return (token, jti, exp_unix)."""
    secret = _require_secret()
    now = _now()
    exp_dt = now + ttl
    token_jti = jti or secrets.token_urlsafe(16)
    payload: dict = {
        "sub": sub,
        "typ": typ,
        "iat": int(now.timestamp()),
        "exp": int(exp_dt.timestamp()),
        "jti": token_jti,
        "iss": config.JWT_ISSUER,
        "aud": config.JWT_AUDIENCE,
        "role": config.user_role(sub),
    }
    if family_id:
        payload["fid"] = family_id
    token = jwt.encode(payload, secret, algorithm=config.JWT_ALGORITHM)
    return token, token_jti, float(exp_dt.timestamp())


def create_access_token(sub: str) -> str:
    token, _, _ = _create_token(
        sub, TOKEN_TYPE_ACCESS, timedelta(minutes=config.JWT_ACCESS_TTL_MIN)
    )
    return token


def create_refresh_token(sub: str, family_id: str | None = None) -> str:
    """Mint a refresh token and register its jti in the server-side store."""
    fid = family_id or secrets.token_urlsafe(12)
    token, jti, exp = _create_token(
        sub,
        TOKEN_TYPE_REFRESH,
        timedelta(days=config.JWT_REFRESH_TTL_DAYS),
        family_id=fid,
    )
    get_refresh_token_store().register(jti, sub=sub, family_id=fid, exp=exp)
    return token


def issue_token_pair(sub: str) -> tuple[str, str]:
    """Login helper: new access + new refresh family."""
    access = create_access_token(sub)
    refresh = create_refresh_token(sub)
    return access, refresh


def rotate_refresh_token(old_refresh: str) -> tuple[str, str]:
    """Validate + rotate a refresh token. Reuse of an already-rotated jti
    revokes the whole family and raises 401."""
    claims = decode_token(old_refresh, expected_typ=TOKEN_TYPE_REFRESH)
    sub = str(claims["sub"])
    old_jti = str(claims.get("jti") or "")
    family_id = str(claims.get("fid") or "")
    if not old_jti or not family_id:
        raise _unauthorized()

    store = get_refresh_token_store()
    new_access = create_access_token(sub)
    new_refresh, new_jti, exp = _create_token(
        sub,
        TOKEN_TYPE_REFRESH,
        timedelta(days=config.JWT_REFRESH_TTL_DAYS),
        family_id=family_id,
    )
    ok = store.rotate(
        old_jti, new_jti, sub=sub, family_id=family_id, exp=exp
    )
    if not ok:
        store.revoke_family(family_id)
        raise _unauthorized("Refresh token reuse detected — re-authenticate.")
    return new_access, new_refresh


def access_token_ttl_seconds() -> int:
    return int(config.JWT_ACCESS_TTL_MIN) * 60


def decode_token(token: str, expected_typ: str) -> dict:
    """Verify signature + expiry + iss/aud + token type, returning claims."""
    secret = _require_secret()
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[config.JWT_ALGORITHM],
            audience=config.JWT_AUDIENCE,
            issuer=config.JWT_ISSUER,
        )
    except jwt.PyJWTError:
        raise _unauthorized()

    if claims.get("typ") != expected_typ:
        raise _unauthorized()
    if not claims.get("sub"):
        raise _unauthorized()
    return claims


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------
def verify_credentials(
    credentials: "HTTPAuthorizationCredentials | None" = Depends(_bearer),
) -> str:
    """Validate `Authorization: Bearer <access-token>` and return username.

    When ``MISA_AUTH_DISABLED`` is set (local demos only), returns the
    configured open-access username without requiring a token.
    """
    if config.AUTH_DISABLED:
        return config.AUTH_DISABLED_USERNAME
    if credentials is None or (credentials.scheme or "").lower() != "bearer":
        raise _unauthorized("Not authenticated.")
    claims = decode_token(credentials.credentials, expected_typ=TOKEN_TYPE_ACCESS)
    return str(claims["sub"])


def optional_user(
    credentials: "HTTPAuthorizationCredentials | None" = Depends(_bearer),
) -> "str | None":
    """Like verify_credentials, but never raises — None on missing/bad token."""
    if config.AUTH_DISABLED:
        return config.AUTH_DISABLED_USERNAME
    if credentials is None or (credentials.scheme or "").lower() != "bearer":
        return None
    try:
        claims = decode_token(credentials.credentials, expected_typ=TOKEN_TYPE_ACCESS)
    except HTTPException:
        return None
    return str(claims["sub"])


def require_role(minimum: str):
    """FastAPI dependency factory: require `minimum` role or higher."""

    def _dep(user: str = Depends(verify_credentials)) -> str:
        if not config.role_at_least(user, minimum):
            raise _forbidden(
                f"Role '{config.user_role(user)}' cannot access this endpoint "
                f"(requires '{minimum}' or higher)."
            )
        return user

    return _dep
