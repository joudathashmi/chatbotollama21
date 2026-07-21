"""World-class session context: topic-shift, trim, pin/archive."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import bcrypt
import pytest
from fastapi.testclient import TestClient

from app import config
from app.auth import verify_credentials
from app.main import app
from app.services.chat_context import (
    StateCard,
    detect_topic_shift,
    prepare_prompt_history,
    update_state_after_turn,
)
from app.services.session_store import reset_session_store_for_tests
from app.services.token_store import reset_refresh_token_store_for_tests

client = TestClient(app)

TEST_SECRET = "test-secret-please-ignore-0123456789"
ALICE, ALICE_PW = "wc-alice", "alice-pass-xyz"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
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
    monkeypatch.setattr(config, "SESSIONS_PROMPT_HISTORY_TURNS", 2)
    monkeypatch.setattr(config, "SESSIONS_PROMPT_ASSISTANT_CHARS", 80)
    monkeypatch.setattr(config, "SESSIONS_IDLE_ARCHIVE_DAYS", 90)
    monkeypatch.setattr(config, "SESSIONS_HARD_DELETE_DAYS", 365)
    monkeypatch.setattr(config, "DOCUMENTS_ENABLED", False)

    def _hash(pw: str) -> str:
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

    monkeypatch.setattr(config, "AUTH_USERS", {ALICE: _hash(ALICE_PW)})
    monkeypatch.setattr(config, "AUTH_USER_ROLES", {ALICE: config.ROLE_ANALYST})
    reset_refresh_token_store_for_tests()
    reset_session_store_for_tests()
    app.dependency_overrides.pop(verify_credentials, None)
    yield


def _token() -> str:
    r = client.post("/api/v1/auth/login", json={"username": ALICE, "password": ALICE_PW})
    assert r.status_code == 200
    return r.json()["access_token"]


def _bearer(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def test_topic_shift_on_new_entity():
    state = StateCard(active_entity="Helios Energy")
    assert detect_topic_shift("Tell me about Alphabet", state) is True
    assert detect_topic_shift("new topic: what about deals?", state) is True
    assert detect_topic_shift("Tell me more about them", state) is False


def test_prepare_prompt_history_drops_on_shift():
    hist = [
        {"role": "user", "content": "Tell me about Helios Energy"},
        {"role": "assistant", "content": "Helios is a green hydrogen firm. " * 20},
    ]
    state = StateCard(active_entity="Helios Energy")
    effective, new_state, meta = prepare_prompt_history(
        "Tell me about Alphabet", hist, state,
    )
    assert meta["topic_shift"] is True
    assert effective == []
    assert new_state.active_entity == "Alphabet"


def test_prepare_prompt_history_trims_assistant():
    hist = []
    for i in range(5):
        hist.append({"role": "user", "content": f"Question about Helios number {i}"})
        hist.append({"role": "assistant", "content": "A" * 500})
    state = StateCard(active_entity="Helios")
    effective, _, meta = prepare_prompt_history(
        "What else about Helios?", hist, state,
    )
    assert meta["topic_shift"] is False
    # Only last 2 user turns kept (config fixture)
    users = [m for m in effective if m["role"] == "user" and not m["content"].startswith("(Continuing")]
    assert len(users) <= 2
    assts = [m for m in effective if m["role"] == "assistant"]
    assert assts
    assert all(len(m["content"]) <= 81 for m in assts)


def test_update_state_after_turn():
    card = update_state_after_turn(
        StateCard(), "Tell me about Helios Energy",
        answer_source="hybrid", entity="Helios Energy", entity_type="company",
    )
    assert card.active_entity == "Helios Energy"
    assert card.last_answer_source == "hybrid"
    assert "Helios" in card.summary


def test_pin_and_archive_api():
    tok = _token()
    sid = client.post(
        "/api/v1/sessions", headers=_bearer(tok), json={},
    ).json()["id"]
    pinned = client.patch(
        f"/api/v1/sessions/{sid}",
        headers=_bearer(tok),
        json={"pinned": True},
    )
    assert pinned.status_code == 200
    assert pinned.json()["pinned"] is True

    archived = client.patch(
        f"/api/v1/sessions/{sid}",
        headers=_bearer(tok),
        json={"archived": True},
    )
    assert archived.status_code == 200
    assert archived.json()["archived_at"]

    active = client.get("/api/v1/sessions", headers=_bearer(tok)).json()["sessions"]
    assert all(s["id"] != sid for s in active)

    with_arch = client.get(
        "/api/v1/sessions?include_archived=true", headers=_bearer(tok),
    ).json()["sessions"]
    assert any(s["id"] == sid for s in with_arch)


def test_search_sessions():
    tok = _token()
    client.post(
        "/api/v1/sessions", headers=_bearer(tok), json={"title": "Helios briefing"},
    )
    client.post(
        "/api/v1/sessions", headers=_bearer(tok), json={"title": "Pakistan macros"},
    )
    hit = client.get(
        "/api/v1/sessions?q=Helios", headers=_bearer(tok),
    ).json()["sessions"]
    assert len(hit) == 1
    assert "Helios" in hit[0]["title"]


def test_chat_saves_state_card(monkeypatch):
    tok = _token()
    sid = client.post(
        "/api/v1/sessions", headers=_bearer(tok), json={},
    ).json()["id"]
    fake = {
        "answer": "Helios Energy signed an MoU.",
        "tool_calls": [],
        "error": None,
        "_answer_source": "document",
    }
    with patch("app.routers.v1.chat.chat", return_value=fake):
        r = client.post(
            "/api/v1/chat",
            headers=_bearer(tok),
            json={
                "question": "Tell me about Helios Energy",
                "history": [],
                "stream": False,
                "session_id": sid,
            },
        )
    assert r.status_code == 200
    detail = client.get(f"/api/v1/sessions/{sid}", headers=_bearer(tok)).json()
    assert detail["session"]["state"]
    assert detail["session"]["state"].get("active_entity")


def test_idle_archive_ttl(monkeypatch):
    from app.services.session_store import get_session_store
    monkeypatch.setattr(config, "SESSIONS_IDLE_ARCHIVE_DAYS", 1)
    store = get_session_store()
    sess = store.create(ALICE, title="Old")
    # Backdate updated_at
    with store._state.lock:
        store._state.sessions[sess.id].updated_at = (
            datetime.now(timezone.utc) - timedelta(days=5)
        ).isoformat()
    store.maintain_ttl(ALICE)
    got = store.get(sess.id, ALICE)
    assert got and got.archived_at
