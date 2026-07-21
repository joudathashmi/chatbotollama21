"""Tests for the engagement router (previously untested):
  POST /api/v1/engagement/generate  — JSON and SSE paths
  GET  /api/v1/engagement/modes

The OpenAI-backed engine is patched out; these assert the router's contract
(response shape, streaming media type, mode listing, request validation).
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_modes_lists_quick_and_full():
    r = client.get("/api/v1/engagement/modes")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["modes"]}
    assert ids == {"quick", "full"}


def test_generate_json_path_returns_text():
    async def _fake_generate(entity, mode, context):
        assert entity == "Alphabet"
        assert mode == "quick"
        return {"text": "## Brief\nStrategic relevance…", "error": None}

    with patch("app.routers.v1.engagement.engagement_generate", _fake_generate):
        r = client.post(
            "/api/v1/engagement/generate",
            json={"entity": "Alphabet", "mode": "quick", "stream": False},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["text"].startswith("## Brief")
    assert body["error"] is None


def test_generate_json_path_surfaces_engine_error():
    async def _fake_generate(entity, mode, context):
        return {"text": "", "error": "model unavailable"}

    with patch("app.routers.v1.engagement.engagement_generate", _fake_generate):
        r = client.post(
            "/api/v1/engagement/generate",
            json={"entity": "X", "mode": "full", "stream": False},
        )
    assert r.status_code == 200
    assert r.json()["error"] == "model unavailable"


def test_generate_sse_path_streams_deltas():
    async def _fake_stream(entity, mode, context):
        yield 'data: {"meta": {"phase": "opening"}}\n\n'
        yield 'data: {"delta": "Hello "}\n\n'
        yield 'data: {"delta": "world"}\n\n'
        yield "data: [DONE]\n\n"

    with patch("app.routers.v1.engagement.engagement_sse_stream", _fake_stream):
        r = client.post(
            "/api/v1/engagement/generate",
            json={"entity": "Alphabet", "mode": "quick", "stream": True},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert '"delta": "Hello "' in r.text
    assert "[DONE]" in r.text


def test_generate_rejects_invalid_mode():
    r = client.post(
        "/api/v1/engagement/generate",
        json={"entity": "Alphabet", "mode": "deep-dive", "stream": False},
    )
    assert r.status_code == 422


def test_generate_requires_entity():
    r = client.post(
        "/api/v1/engagement/generate",
        json={"mode": "quick", "stream": False},
    )
    assert r.status_code == 422
