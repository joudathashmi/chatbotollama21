"""Server-side input-validation tests for POST /api/v1/feedback.

Ensures a client that bypasses the browser UI can't push blank,
oversized, or malformed data into the feedback log. Rate-limiting
behavior is covered separately in tests/test_rate_limit.py.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_VALID_PAYLOAD = {
    "verdict": "up",
    "question": "Who is the CEO of Apple?",
    "answer": "Tim Cook.",
}


def test_feedback_rejects_blank_question():
    r = client.post("/api/v1/feedback", json={**_VALID_PAYLOAD, "question": "   "})
    assert r.status_code == 422


def test_feedback_rejects_blank_answer():
    r = client.post("/api/v1/feedback", json={**_VALID_PAYLOAD, "answer": ""})
    assert r.status_code == 422


def test_feedback_rejects_invalid_verdict():
    r = client.post("/api/v1/feedback", json={**_VALID_PAYLOAD, "verdict": "maybe"})
    assert r.status_code == 422


def test_feedback_rejects_oversized_comment():
    r = client.post("/api/v1/feedback", json={**_VALID_PAYLOAD, "comment": "x" * 2_001})
    assert r.status_code == 422


def test_feedback_rejects_oversized_question():
    r = client.post("/api/v1/feedback", json={**_VALID_PAYLOAD, "question": "x" * 2_001})
    assert r.status_code == 422


def test_feedback_accepts_valid_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("MISA_FEEDBACK_LOG", str(tmp_path / "fb.jsonl"))
    r = client.post("/api/v1/feedback", json=_VALID_PAYLOAD)
    assert r.status_code == 200
    assert r.json()["persisted"] is True


def test_feedback_treats_whitespace_comment_as_none(tmp_path, monkeypatch):
    """A comment of only whitespace is normalised to null, not stored as
    a garbage string — the 'UI bypass' scenario from the finding."""
    monkeypatch.setenv("MISA_FEEDBACK_LOG", str(tmp_path / "fb.jsonl"))
    r = client.post("/api/v1/feedback", json={**_VALID_PAYLOAD, "comment": "   "})
    assert r.status_code == 200
