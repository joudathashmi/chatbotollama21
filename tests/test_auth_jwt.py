"""JWT authentication (Risk-20-3) — login, refresh, and bearer-token
enforcement on protected routes.

These tests exercise REAL token validation, so they:
  - set a test JWT_SECRET_KEY + a bootstrap account + one hashed named user
    on app.config (auth.py reads config.* dynamically), and
  - remove conftest's autouse `verify_credentials` override so the actual
    dependency runs instead of the "test-user" stub.

/api/v1/questions is used as the representative protected route (it needs no
database or OpenAI call).
"""

from __future__ import annotations

import time

import bcrypt
import jwt
import pytest
from fastapi.testclient import TestClient

from app import config
from app.auth import verify_credentials
from app.main import app

client = TestClient(app)

TEST_SECRET = "test-secret-please-ignore-0123456789"
BOOT_USER = "boot-user"
BOOT_PASS = "boot-pass-123"
ALICE_USER = "alice"
ALICE_PASS = "alice-pass-456"

PROTECTED = "/api/v1/questions"


@pytest.fixture(autouse=True)
def _jwt_env(monkeypatch):
    from app.services.token_store import reset_refresh_token_store_for_tests

    monkeypatch.setattr(config, "JWT_SECRET_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(config, "JWT_ISSUER", "misa-intelligence-api")
    monkeypatch.setattr(config, "JWT_AUDIENCE", "misa-intelligence-api")
    monkeypatch.setattr(config, "JWT_ACCESS_TTL_MIN", 30)
    monkeypatch.setattr(config, "JWT_REFRESH_TTL_DAYS", 7)
    monkeypatch.setattr(config, "API_USERNAME", BOOT_USER)
    monkeypatch.setattr(config, "API_PASSWORD", BOOT_PASS)
    monkeypatch.setattr(config, "ALLOW_PLAINTEXT_BOOTSTRAP", True)
    # Real token validation must run even when the local .env demos with
    # MISA_AUTH_DISABLED=true.
    monkeypatch.setattr(config, "AUTH_DISABLED", False)
    monkeypatch.setattr(config, "BOOTSTRAP_ROLE", config.ROLE_ADMIN)
    alice_hash = bcrypt.hashpw(ALICE_PASS.encode(), bcrypt.gensalt()).decode()
    monkeypatch.setattr(config, "AUTH_USERS", {ALICE_USER: alice_hash})
    monkeypatch.setattr(config, "AUTH_USER_ROLES", {ALICE_USER: config.ROLE_ANALYST})
    reset_refresh_token_store_for_tests()
    # Drop conftest's global override so the real dependency runs.
    app.dependency_overrides.pop(verify_credentials, None)
    yield


def _claims(**extra):
    now = int(time.time())
    base = {
        "sub": BOOT_USER,
        "typ": "access",
        "iat": now,
        "exp": now + 999,
        "jti": "x",
        "iss": "misa-intelligence-api",
        "aud": "misa-intelligence-api",
    }
    base.update(extra)
    return base


def _login(username: str, password: str):
    return client.post("/api/v1/auth/login", json={"username": username, "password": password})


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── login ──────────────────────────────────────────────────────────────────

def test_login_success_returns_access_and_refresh():
    r = _login(BOOT_USER, BOOT_PASS)
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"] and body["refresh_token"]
    assert body["expires_in"] == 30 * 60


def test_login_endpoint_is_open_without_auth():
    # No Authorization header at all — login must still be reachable.
    r = client.post("/api/v1/auth/login", json={"username": BOOT_USER, "password": BOOT_PASS})
    assert r.status_code == 200


def test_login_bad_password_401():
    r = _login(BOOT_USER, "wrong")
    assert r.status_code == 401


def test_login_unknown_user_401():
    r = _login("nobody", "whatever")
    assert r.status_code == 401


def test_login_hashed_named_user_succeeds():
    r = _login(ALICE_USER, ALICE_PASS)
    assert r.status_code == 200
    assert r.json()["access_token"]


# ── protected-route enforcement ─────────────────────────────────────────────

def test_access_token_grants_protected_route():
    token = _login(BOOT_USER, BOOT_PASS).json()["access_token"]
    r = client.get(PROTECTED, headers=_bearer(token))
    assert r.status_code == 200


def test_missing_token_401_advertises_bearer():
    r = client.get(PROTECTED)
    assert r.status_code == 401
    assert r.headers.get("www-authenticate", "").lower().startswith("bearer")


def test_garbage_token_401():
    r = client.get(PROTECTED, headers=_bearer("not-a-jwt"))
    assert r.status_code == 401


def test_tampered_token_401():
    token = _login(BOOT_USER, BOOT_PASS).json()["access_token"]
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    r = client.get(PROTECTED, headers=_bearer(tampered))
    assert r.status_code == 401


def test_expired_access_token_401():
    expired = jwt.encode(
        _claims(iat=int(time.time()) - 100, exp=int(time.time()) - 10),
        TEST_SECRET, algorithm="HS256",
    )
    r = client.get(PROTECTED, headers=_bearer(expired))
    assert r.status_code == 401


def test_token_signed_with_wrong_secret_401():
    # >= 32 bytes so PyJWT doesn't emit InsecureKeyLengthWarning — this is
    # a forged-signature test, not a key-strength test, so the wrong
    # secret should still look like a realistic production secret.
    forged = jwt.encode(
        _claims(),
        "some-other-secret-that-is-at-least-32-bytes-long", algorithm="HS256",
    )
    r = client.get(PROTECTED, headers=_bearer(forged))
    assert r.status_code == 401


def test_token_wrong_audience_401():
    bad = jwt.encode(_claims(aud="other-service"), TEST_SECRET, algorithm="HS256")
    r = client.get(PROTECTED, headers=_bearer(bad))
    assert r.status_code == 401


def test_token_wrong_issuer_401():
    bad = jwt.encode(_claims(iss="other-issuer"), TEST_SECRET, algorithm="HS256")
    r = client.get(PROTECTED, headers=_bearer(bad))
    assert r.status_code == 401


def test_access_token_carries_iss_aud():
    token = _login(BOOT_USER, BOOT_PASS).json()["access_token"]
    claims = jwt.decode(
        token, TEST_SECRET, algorithms=["HS256"],
        audience="misa-intelligence-api", issuer="misa-intelligence-api",
    )
    assert claims["iss"] == "misa-intelligence-api"
    assert claims["aud"] == "misa-intelligence-api"


def test_refresh_token_rejected_on_protected_route():
    refresh = _login(BOOT_USER, BOOT_PASS).json()["refresh_token"]
    # A refresh token (typ=refresh) must not authenticate a normal request.
    r = client.get(PROTECTED, headers=_bearer(refresh))
    assert r.status_code == 401


# ── refresh flow ────────────────────────────────────────────────────────────

def test_refresh_issues_working_new_access_token():
    refresh = _login(BOOT_USER, BOOT_PASS).json()["refresh_token"]
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200
    new_access = r.json()["access_token"]
    assert new_access
    # The freshly minted access token works on a protected route.
    assert client.get(PROTECTED, headers=_bearer(new_access)).status_code == 200


def test_refresh_rejects_reuse_of_rotated_token():
    refresh = _login(BOOT_USER, BOOT_PASS).json()["refresh_token"]
    first = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert first.status_code == 200
    # Presenting the OLD refresh again must fail (reuse / theft detection).
    second = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert second.status_code == 401
    # The family is revoked — the NEW refresh from the first rotation is also dead.
    new_refresh = first.json()["refresh_token"]
    third = client.post("/api/v1/auth/refresh", json={"refresh_token": new_refresh})
    assert third.status_code == 401


def test_refresh_rejects_access_token():
    access = _login(BOOT_USER, BOOT_PASS).json()["access_token"]
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": access})
    assert r.status_code == 401


def test_tokens_carry_distinct_subjects_per_user():
    boot_access = _login(BOOT_USER, BOOT_PASS).json()["access_token"]
    alice_access = _login(ALICE_USER, ALICE_PASS).json()["access_token"]
    boot_sub = jwt.decode(
        boot_access, TEST_SECRET, algorithms=["HS256"],
        audience="misa-intelligence-api", issuer="misa-intelligence-api",
    )["sub"]
    alice_sub = jwt.decode(
        alice_access, TEST_SECRET, algorithms=["HS256"],
        audience="misa-intelligence-api", issuer="misa-intelligence-api",
    )["sub"]
    assert boot_sub == BOOT_USER
    assert alice_sub == ALICE_USER
    assert boot_sub != alice_sub


# ── /health diagnostic gating (Risk-20-3) ───────────────────────────────────
# /health must stay reachable unauthenticated (load balancers / uptime probes
# need a 200) but must not hand deployment recon — DB status, model name,
# version — to anyone who curls the URL. Full diagnostics require admin role.

_DIAGNOSTIC_FIELDS = ("postgres", "openai_configured", "openai_model", "malware_scan", "version")


def test_health_unauthenticated_returns_200_for_probes():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_unauthenticated_leaks_no_diagnostics():
    body = client.get("/health").json()
    for field in _DIAGNOSTIC_FIELDS:
        assert field not in body, f"/health leaked `{field}` to an unauthenticated caller"


def test_health_non_admin_gets_no_diagnostics(monkeypatch):
    monkeypatch.setattr(config, "BOOTSTRAP_ROLE", config.ROLE_VIEWER)
    token = _login(BOOT_USER, BOOT_PASS).json()["access_token"]
    body = client.get("/health", headers=_bearer(token)).json()
    for field in _DIAGNOSTIC_FIELDS:
        assert field not in body


def test_health_authenticated_admin_returns_diagnostics():
    token = _login(BOOT_USER, BOOT_PASS).json()["access_token"]
    r = client.get("/health", headers=_bearer(token))
    assert r.status_code == 200
    body = r.json()
    for field in _DIAGNOSTIC_FIELDS:
        assert field in body, f"/health withheld `{field}` from an authenticated caller"


def test_health_with_bad_token_degrades_to_probe_not_401():
    """A stale/garbage credential must not turn a liveness probe into a 401 —
    it's treated as 'no token' and gets the minimal public response."""
    r = client.get("/health", headers=_bearer("garbage.token.value"))
    assert r.status_code == 200
    assert "openai_model" not in r.json()


def test_health_malware_scan_block_reports_live_backend_status():
    """The malware_scan block is a live probe (clamscan --version /
    MpCmdRun.exe presence check), not a config echo — same contract as
    GET /api/v1/business-card/scan-status, surfaced here too so a single
    /health call answers 'is AV scanning actually working' cross-platform
    (clamscan on Linux, Defender on Windows)."""
    from unittest.mock import patch
    token = _login(BOOT_USER, BOOT_PASS).json()["access_token"]
    with patch(
        "app.services.malware_scanner.check_status",
        return_value={
            "backend": "clamscan", "enabled": True, "available": True,
            "detail": "ClamAV 1.4.3/28063/Fri Jul 17 06:24:28 2026",
        },
    ):
        r = client.get("/health", headers=_bearer(token))
    assert r.status_code == 200
    scan = r.json()["malware_scan"]
    assert scan == {
        "backend": "clamscan", "enabled": True, "available": True,
        "detail": "ClamAV 1.4.3/28063/Fri Jul 17 06:24:28 2026",
    }


def test_health_malware_scan_probe_failure_does_not_500_the_endpoint():
    """A crash inside the AV status probe (e.g. a permissions error
    running MpCmdRun.exe) must degrade to 'unavailable', never take the
    whole liveness endpoint down with it."""
    from unittest.mock import patch
    token = _login(BOOT_USER, BOOT_PASS).json()["access_token"]
    with patch(
        "app.services.malware_scanner.check_status",
        side_effect=RuntimeError("boom"),
    ):
        r = client.get("/health", headers=_bearer(token))
    assert r.status_code == 200
    scan = r.json()["malware_scan"]
    assert scan["available"] is False
    assert "boom" in scan["detail"]
