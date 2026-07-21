"""Document library: upload, visibility, dedup, ingest, document-first chat."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import bcrypt
import pytest
from docx import Document
from fastapi.testclient import TestClient

from app import config
from app.auth import verify_credentials
from app.main import app
from app.services import document_ingest as di
from app.services.chat_engine import _chat_execute
from app.services.document_store import reset_document_store_for_tests
from app.services.token_store import reset_refresh_token_store_for_tests

client = TestClient(app)

TEST_SECRET = "test-secret-please-ignore-0123456789"
ALICE, ALICE_PW = "doc-alice", "alice-pass-xyz"
BOB, BOB_PW = "doc-bob", "bob-pass-xyz"


@pytest.fixture(autouse=True)
def _docs_env(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JWT_SECRET_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(config, "JWT_ISSUER", "misa-intelligence-api")
    monkeypatch.setattr(config, "JWT_AUDIENCE", "misa-intelligence-api")
    monkeypatch.setattr(config, "ALLOW_PLAINTEXT_BOOTSTRAP", False)
    monkeypatch.setattr(config, "AUTH_DISABLED", False)
    monkeypatch.setattr(config, "API_USERNAME", "")
    monkeypatch.setattr(config, "API_PASSWORD", "")
    monkeypatch.setattr(config, "DOCUMENTS_ENABLED", True)
    monkeypatch.setattr(config, "DOCUMENTS_BACKEND", "memory")
    monkeypatch.setattr(config, "DOCUMENTS_WEB_MODE", "hybrid")
    monkeypatch.setattr(config, "DOCUMENTS_ROOT", str(tmp_path / "docs"))
    monkeypatch.setattr(config, "DOCUMENTS_INGEST_DIR", str(tmp_path / "inbox"))
    monkeypatch.setattr(config, "MALWARE_SCAN_ENABLED", False)
    monkeypatch.setattr(config, "MALWARE_SCAN_BACKEND", "none")

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
    reset_document_store_for_tests()
    app.dependency_overrides.pop(verify_credentials, None)
    # Ensure store singleton picks up memory after reset
    reset_document_store_for_tests()
    yield


def _token(user: str, pw: str) -> str:
    r = client.post("/api/v1/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _bearer(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def _txt_file(text: str = "Acme Corp opened an RHQ in Riyadh in 2024.") -> tuple[str, bytes]:
    return ("briefing.txt", text.encode("utf-8"))


def test_upload_and_list():
    tok = _token(ALICE, ALICE_PW)
    name, data = _txt_file()
    r = client.post(
        "/api/v1/documents/upload",
        headers=_bearer(tok),
        files={"file": (name, data, "text/plain")},
        data={"visibility": "private"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready"
    assert body["filename"] == "briefing.txt"
    assert body["visibility"] == "private"

    listed = client.get("/api/v1/documents", headers=_bearer(tok))
    assert listed.status_code == 200
    assert any(d["id"] == body["id"] for d in listed.json()["documents"])


def test_duplicate_hash_returns_existing():
    tok = _token(ALICE, ALICE_PW)
    name, data = _txt_file("same bytes twice")
    r1 = client.post(
        "/api/v1/documents/upload",
        headers=_bearer(tok),
        files={"file": (name, data, "text/plain")},
        data={"visibility": "private"},
    )
    r2 = client.post(
        "/api/v1/documents/upload",
        headers=_bearer(tok),
        files={"file": (name, data, "text/plain")},
        data={"visibility": "private"},
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]


def test_private_visibility_hidden_from_other_user():
    alice = _token(ALICE, ALICE_PW)
    bob = _token(BOB, BOB_PW)
    name, data = _txt_file("secret alice note about Zephyr Holdings")
    up = client.post(
        "/api/v1/documents/upload",
        headers=_bearer(alice),
        files={"file": (name, data, "text/plain")},
        data={"visibility": "private"},
    )
    assert up.status_code == 200
    doc_id = up.json()["id"]

    bob_list = client.get("/api/v1/documents", headers=_bearer(bob)).json()["documents"]
    assert all(d["id"] != doc_id for d in bob_list)

    bob_get = client.get(f"/api/v1/documents/{doc_id}", headers=_bearer(bob))
    assert bob_get.status_code == 404


def test_org_visibility_visible_to_peers():
    alice = _token(ALICE, ALICE_PW)
    bob = _token(BOB, BOB_PW)
    name, data = _txt_file("shared org memo on Vision 2030 RHQ incentives")
    up = client.post(
        "/api/v1/documents/upload",
        headers=_bearer(alice),
        files={"file": (name, data, "text/plain")},
        data={"visibility": "org"},
    )
    assert up.status_code == 200
    doc_id = up.json()["id"]
    bob_list = client.get("/api/v1/documents", headers=_bearer(bob)).json()["documents"]
    assert any(d["id"] == doc_id for d in bob_list)


def test_ingest_from_inbox(tmp_path, monkeypatch):
    inbox = Path(config.DOCUMENTS_INGEST_DIR)
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "inbox_note.txt").write_text(
        "NovaTech established a regional headquarters in Jeddah.", encoding="utf-8"
    )
    tok = _token(ALICE, ALICE_PW)
    r = client.post(
        "/api/v1/documents/ingest",
        headers=_bearer(tok),
        json={"visibility": "org"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["ingested"]) == 1
    assert body["ingested"][0]["status"] == "ready"
    assert not (inbox / "inbox_note.txt").exists()
    assert (inbox / "processed" / "inbox_note.txt").exists()


def test_malware_reject(monkeypatch):
    from app.services.malware_scanner import ScanResult, ScanVerdict

    def _infected(data, filename):
        return ScanResult(verdict=ScanVerdict.INFECTED, backend="heuristic", detail="eicar")

    monkeypatch.setattr(config, "MALWARE_SCAN_ENABLED", True)
    monkeypatch.setattr("app.services.document_ingest.scan_file", _infected)
    tok = _token(ALICE, ALICE_PW)
    r = client.post(
        "/api/v1/documents/upload",
        headers=_bearer(tok),
        files={"file": ("bad.txt", b"X5O!P%@AP", "text/plain")},
        data={"visibility": "private"},
    )
    assert r.status_code == 422


def test_chat_hybrid_includes_document_and_web(monkeypatch):
    """Hybrid mode: strong doc hit + web results → dual provenance answer."""
    from app.services.audit_log import set_audit_user

    name, data = _txt_file(
        "QuantumLeap Industries announced a major semiconductor investment in NEOM."
    )
    doc = di.ingest_bytes(
        data, filename=name, owner_username=ALICE, visibility="org", source="upload"
    )
    assert doc.status == "ready"

    fake_web = [{
        "title": "QuantumLeap expands in Saudi Arabia",
        "url": "https://example.com/quantumleap",
        "snippet": "Public reporting on the NEOM semiconductor plan.",
    }]

    set_audit_user(ALICE)
    with patch("app.services.chat_engine.get_openai_client", return_value=object()):
        with patch("app.services.prompt_guard.detect_prompt_attack", return_value=(False, None)):
            with patch("app.services.web_search.search", return_value=fake_web):
                out = _chat_execute(
                    "What did QuantumLeap Industries announce about NEOM?",
                    [],
                    "en",
                )
    assert out["_answer_source"] == "hybrid"
    assert "QuantumLeap" in out["answer"]
    assert "From your documents" in out["answer"]
    assert "From the web" in out["answer"]
    assert "[web:1]" in out["answer"]
    assert out.get("web_sources") and out["web_sources"][0]["url"].startswith("https://")
    assert out.get("doc_sources")


def test_chat_docs_only_override_skips_web(monkeypatch):
    """Explicit 'only from the document' keeps answer_source=document."""
    from app.services.audit_log import set_audit_user

    name, data = _txt_file(
        "Helios Energy signed an MoU with MISA covering green hydrogen in 2023."
    )
    di.ingest_bytes(
        data, filename=name, owner_username=ALICE, visibility="org", source="upload"
    )
    set_audit_user(ALICE)
    with patch("app.services.chat_engine.get_openai_client", return_value=object()):
        with patch("app.services.prompt_guard.detect_prompt_attack", return_value=(False, None)):
            with patch("app.services.web_search.search") as web:
                out = _chat_execute(
                    "Only from the document: what did Helios Energy sign with MISA?",
                    [],
                    "en",
                )
                web.assert_not_called()
    assert out["_answer_source"] == "document"
    assert "Helios Energy" in out["answer"]
    assert out.get("web_sources") == []


def test_chat_docs_first_mode_skips_web(monkeypatch):
    """Legacy docs_first mode short-circuits without calling the web."""
    from app.services.audit_log import set_audit_user

    monkeypatch.setattr(config, "DOCUMENTS_WEB_MODE", "docs_first")
    name, data = _txt_file(
        "QuantumLeap Industries announced a major semiconductor investment in NEOM."
    )
    di.ingest_bytes(
        data, filename=name, owner_username=ALICE, visibility="org", source="upload"
    )
    set_audit_user(ALICE)
    with patch("app.services.chat_engine.get_openai_client", return_value=object()):
        with patch("app.services.prompt_guard.detect_prompt_attack", return_value=(False, None)):
            with patch("app.services.web_search.search") as web:
                out = _chat_execute(
                    "What did QuantumLeap Industries announce about NEOM?",
                    [],
                    "en",
                )
                web.assert_not_called()
    assert out["_answer_source"] == "document"
    assert out.get("web_sources") == []


def test_compose_hybrid_helpers():
    assert di.wants_docs_only("answer only from the document please")
    assert di.wants_docs_only("don't use the internet")
    assert not di.wants_docs_only("tell me about Helios")
    assert di.wants_web_preferred("what's the latest news on Helios")
    assert di.should_augment_docs_with_web("tell me about Helios")
    assert not di.should_augment_docs_with_web("documents only: summarise Helios")


def test_extract_docx_roundtrip():
    buf = io.BytesIO()
    d = Document()
    d.add_paragraph("Helios Energy signed an MoU with MISA in 2023.")
    d.save(buf)
    text = di.extract_text(buf.getvalue(), "memo.docx")
    assert "Helios Energy" in text


def test_documents_ui_redirects_to_chat():
    r = client.get("/documents", follow_redirects=False)
    assert r.status_code in (301, 302, 303, 307, 308)
    assert r.headers["location"] == "/chat"
    chat = client.get("/chat")
    assert chat.status_code == 200
    assert "docPanel" in chat.text
    assert "docToggle" in chat.text
    assert "Documents" in chat.text
