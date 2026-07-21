"""Chat sessions: create, list, ownership, persist via /chat."""

from __future__ import annotations

from unittest.mock import patch

import bcrypt
import pytest
from fastapi.testclient import TestClient

from app import config
from app.auth import verify_credentials
from app.main import app
from app.services.session_store import (
    get_session_store,
    reset_session_store_for_tests,
    title_from_question,
)
from app.services.token_store import reset_refresh_token_store_for_tests

client = TestClient(app)

TEST_SECRET = "test-secret-please-ignore-0123456789"
ALICE, ALICE_PW = "sess-alice", "alice-pass-xyz"
BOB, BOB_PW = "sess-bob", "bob-pass-xyz"


@pytest.fixture(autouse=True)
def _sess_env(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(config, "JWT_ISSUER", "misa-intelligence-api")
    monkeypatch.setattr(config, "JWT_AUDIENCE", "misa-intelligence-api")
    monkeypatch.setattr(config, "ALLOW_PLAINTEXT_BOOTSTRAP", False)
    monkeypatch.setattr(config, "API_USERNAME", "")
    monkeypatch.setattr(config, "API_PASSWORD", "")
    monkeypatch.setattr(config, "AUTH_DISABLED", False)
    monkeypatch.setattr(config, "SESSIONS_ENABLED", True)
    monkeypatch.setattr(config, "SESSIONS_BACKEND", "memory")
    monkeypatch.setattr(config, "DOCUMENTS_ENABLED", False)

    def _hash(pw: str) -> str:
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

    monkeypatch.setattr(config, "AUTH_USERS", {
        ALICE: _hash(ALICE_PW),
        BOB: _hash(BOB_PW),
    })
    monkeypatch.setattr(config, "AUTH_USER_ROLES", {
        ALICE: config.ROLE_ANALYST,
        BOB: config.ROLE_ANALYST,
    })
    reset_refresh_token_store_for_tests()
    reset_session_store_for_tests()
    app.dependency_overrides.pop(verify_credentials, None)
    yield
    app.dependency_overrides.pop(verify_credentials, None)


def _token(user: str, pw: str) -> str:
    r = client.post("/api/v1/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _bearer(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def test_title_from_question():
    assert title_from_question("Hello world") == "Hello world"
    assert title_from_question("x" * 100).endswith("…")


def test_create_list_get_delete_session():
    tok = _token(ALICE, ALICE_PW)
    created = client.post(
        "/api/v1/sessions",
        headers=_bearer(tok),
        json={"title": "Briefing"},
    )
    assert created.status_code == 200
    sid = created.json()["id"]
    assert created.json()["title"] == "Briefing"

    listed = client.get("/api/v1/sessions", headers=_bearer(tok))
    assert listed.status_code == 200
    assert any(s["id"] == sid for s in listed.json()["sessions"])

    detail = client.get(f"/api/v1/sessions/{sid}", headers=_bearer(tok))
    assert detail.status_code == 200
    assert detail.json()["session"]["id"] == sid
    assert detail.json()["messages"] == []

    renamed = client.patch(
        f"/api/v1/sessions/{sid}",
        headers=_bearer(tok),
        json={"title": "Renamed"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"

    deleted = client.delete(f"/api/v1/sessions/{sid}", headers=_bearer(tok))
    assert deleted.status_code == 200
    assert client.get(f"/api/v1/sessions/{sid}", headers=_bearer(tok)).status_code == 404


def test_session_ownership_isolation():
    alice = _token(ALICE, ALICE_PW)
    bob = _token(BOB, BOB_PW)
    sid = client.post(
        "/api/v1/sessions", headers=_bearer(alice), json={},
    ).json()["id"]
    assert client.get(f"/api/v1/sessions/{sid}", headers=_bearer(bob)).status_code == 404
    assert client.delete(f"/api/v1/sessions/{sid}", headers=_bearer(bob)).status_code == 404
    bob_list = client.get("/api/v1/sessions", headers=_bearer(bob)).json()["sessions"]
    assert all(s["id"] != sid for s in bob_list)


def test_chat_persists_to_session(monkeypatch):
    tok = _token(ALICE, ALICE_PW)
    sid = client.post(
        "/api/v1/sessions", headers=_bearer(tok), json={},
    ).json()["id"]

    fake = {
        "answer": "Alphabet is a holding company.",
        "tool_calls": [],
        "error": None,
        "_answer_source": "fallback",
        "web_sources": [],
        "doc_sources": [],
    }
    with patch("app.routers.v1.chat.chat", return_value=fake):
        r = client.post(
            "/api/v1/chat",
            headers=_bearer(tok),
            json={
                "question": "Tell me about Alphabet",
                "history": [],
                "stream": False,
                "session_id": sid,
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert "Alphabet" in body["answer"]

    detail = client.get(f"/api/v1/sessions/{sid}", headers=_bearer(tok)).json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"]
    assert "Alphabet" in detail["messages"][0]["content"]
    assert detail["session"]["title"] != "New chat"


def test_chat_auto_creates_session(monkeypatch):
    tok = _token(ALICE, ALICE_PW)
    fake = {
        "answer": "ok",
        "tool_calls": [],
        "error": None,
        "_answer_source": "fallback",
    }
    with patch("app.routers.v1.chat.chat", return_value=fake):
        r = client.post(
            "/api/v1/chat",
            headers=_bearer(tok),
            json={"question": "hello there", "history": [], "stream": False},
        )
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid
    store = get_session_store()
    assert store.get(sid, ALICE) is not None


def test_chat_ui_includes_sessions_panel():
    r = client.get("/chat")
    assert r.status_code == 200
    assert "sessPanel" in r.text
    assert "newChatBtn" in r.text
