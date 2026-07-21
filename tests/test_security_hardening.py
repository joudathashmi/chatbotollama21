"""RBAC, body-size enforcement, and production config validation."""

from __future__ import annotations

import bcrypt
import pytest
from fastapi.testclient import TestClient

from app import config
from app.auth import verify_credentials
from app.main import app
from app.services.token_store import reset_refresh_token_store_for_tests

client = TestClient(app)

TEST_SECRET = "test-secret-please-ignore-0123456789"
ADMIN_USER, ADMIN_PASS = "admin-u", "admin-pass-1"
VIEWER_USER, VIEWER_PASS = "viewer-u", "viewer-pass-1"
ANALYST_USER, ANALYST_PASS = "analyst-u", "analyst-pass-1"


@pytest.fixture(autouse=True)
def _sec_env(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(config, "JWT_ISSUER", "misa-intelligence-api")
    monkeypatch.setattr(config, "JWT_AUDIENCE", "misa-intelligence-api")
    monkeypatch.setattr(config, "ALLOW_PLAINTEXT_BOOTSTRAP", False)
    monkeypatch.setattr(config, "API_USERNAME", "")
    monkeypatch.setattr(config, "API_PASSWORD", "")

    def _hash(pw: str) -> str:
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

    monkeypatch.setattr(config, "AUTH_USERS", {
        ADMIN_USER: _hash(ADMIN_PASS),
        VIEWER_USER: _hash(VIEWER_PASS),
        ANALYST_USER: _hash(ANALYST_PASS),
    })
    monkeypatch.setattr(config, "AUTH_USER_ROLES", {
        ADMIN_USER: config.ROLE_ADMIN,
        VIEWER_USER: config.ROLE_VIEWER,
        ANALYST_USER: config.ROLE_ANALYST,
    })
    reset_refresh_token_store_for_tests()
    app.dependency_overrides.pop(verify_credentials, None)
    yield


def _token(username: str, password: str) -> str:
    r = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_viewer_can_access_chat_metadata_but_not_engagement():
    tok = _token(VIEWER_USER, VIEWER_PASS)
    assert client.get("/api/v1/questions", headers=_bearer(tok)).status_code == 200
    # engagement requires analyst+
    r = client.get("/api/v1/engagement/modes", headers=_bearer(tok))
    assert r.status_code == 403


def test_analyst_can_access_engagement_modes():
    tok = _token(ANALYST_USER, ANALYST_PASS)
    assert client.get("/api/v1/engagement/modes", headers=_bearer(tok)).status_code == 200


def test_viewer_cannot_hit_health_data():
    tok = _token(VIEWER_USER, VIEWER_PASS)
    r = client.get("/health/data", headers=_bearer(tok))
    assert r.status_code == 403


def test_admin_can_hit_health_data_without_db_host_leak():
    tok = _token(ADMIN_USER, ADMIN_PASS)
    r = client.get("/health/data", headers=_bearer(tok))
    # DB may or may not be up — either ok or error is fine; host must not leak.
    assert r.status_code == 200
    body = r.json()
    assert "database" not in body
    assert "host" not in body
    assert "dbname" not in body


def test_max_body_middleware_rejects_declared_oversize():
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from app.middleware.security import MaxBodySizeMiddleware

    async def ok(request):
        return PlainTextResponse("ok")

    app_m = Starlette(routes=[Route("/", ok, methods=["POST"])])
    app_m.add_middleware(MaxBodySizeMiddleware, max_bytes=100)
    c = TestClient(app_m)
    # Body larger than cap — httpx sets Content-Length to match body length.
    r = c.post("/", content=b"x" * 250)
    assert r.status_code == 413
    assert c.post("/", content=b"x" * 50).status_code == 200


def test_validate_security_config_flags_open_cors_in_production(monkeypatch):
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    monkeypatch.setattr(config, "CORS_ALLOWED_ORIGINS", ["*"])
    monkeypatch.setattr(config, "JWT_SECRET_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "ALLOW_PLAINTEXT_BOOTSTRAP", False)
    monkeypatch.setattr(config, "AUTH_USERS", {"x": "hash"})
    monkeypatch.setattr(config, "MALWARE_SCAN_ENABLED", True)
    monkeypatch.setattr(config, "MALWARE_SCAN_BACKEND", "clamscan")
    monkeypatch.setattr(config, "ENABLE_DOCS", False)
    errs = config.validate_security_config()
    assert any("CORS" in e for e in errs)


def test_chat_history_rejects_html_markup():
    tok = _token(ADMIN_USER, ADMIN_PASS)
    r = client.post(
        "/api/v1/chat",
        headers=_bearer(tok),
        json={
            "question": "hello",
            "stream": False,
            "history": [{"role": "user", "content": "<script>alert(1)</script>"}],
        },
    )
    assert r.status_code == 422


def test_chat_question_max_length_enforced():
    tok = _token(ADMIN_USER, ADMIN_PASS)
    r = client.post(
        "/api/v1/chat",
        headers=_bearer(tok),
        json={"question": "x" * (config.MAX_USER_MESSAGE_CHARS + 1), "stream": False},
    )
    assert r.status_code == 422


def test_safe_markdown_helper_present_in_chat_ui():
    html = client.get("/chat").text
    assert "DOMPurify" in html or "dompurify" in html.lower()
    assert "safeMarkdownHtml" in html
