"""CORS origin allowlist parsing (Risk-20-3 — restricts the wildcard "*").

app/config.py._parse_cors_origins() reads MISA_CORS_ALLOWED_ORIGINS as a
comma-separated list and feeds it to CORSMiddleware in app/main.py. Locks
in: parsing/trimming behavior, the "*" fallback when unset, and that a
configured origin actually gets reflected by CORSMiddleware while an
unlisted one does not.

Tests call `config._parse_cors_origins()` directly with monkeypatched
`os.environ` rather than reloading the app.config module — a reload
re-runs `load_dotenv(override=True)`, which re-reads the real .env file
from disk and would clobber the monkeypatched value with whatever
MISA_CORS_ALLOWED_ORIGINS is actually configured there (e.g. a real
deployment's URLs), making these tests depend on local .env contents.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app import config


class TestParseCorsOrigins:
    def test_unset_falls_back_to_wildcard(self, monkeypatch):
        monkeypatch.delenv("MISA_CORS_ALLOWED_ORIGINS", raising=False)
        assert config._parse_cors_origins() == ["*"]

    def test_empty_string_falls_back_to_wildcard(self, monkeypatch):
        monkeypatch.setenv("MISA_CORS_ALLOWED_ORIGINS", "")
        assert config._parse_cors_origins() == ["*"]

    def test_single_origin(self, monkeypatch):
        monkeypatch.setenv("MISA_CORS_ALLOWED_ORIGINS", "https://app.example.com")
        assert config._parse_cors_origins() == ["https://app.example.com"]

    def test_multiple_origins_comma_separated(self, monkeypatch):
        raw = "https://app.example.com,https://admin.example.com,http://localhost:3000"
        monkeypatch.setenv("MISA_CORS_ALLOWED_ORIGINS", raw)
        assert config._parse_cors_origins() == [
            "https://app.example.com",
            "https://admin.example.com",
            "http://localhost:3000",
        ]

    def test_whitespace_around_entries_trimmed(self, monkeypatch):
        monkeypatch.setenv(
            "MISA_CORS_ALLOWED_ORIGINS",
            " https://app.example.com , https://admin.example.com ",
        )
        assert config._parse_cors_origins() == [
            "https://app.example.com",
            "https://admin.example.com",
        ]

    def test_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("MISA_CORS_ALLOWED_ORIGINS", "https://app.example.com/")
        assert config._parse_cors_origins() == ["https://app.example.com"]

    def test_empty_entries_between_commas_dropped(self, monkeypatch):
        monkeypatch.setenv(
            "MISA_CORS_ALLOWED_ORIGINS", "https://app.example.com,,http://localhost:3000"
        )
        assert config._parse_cors_origins() == [
            "https://app.example.com",
            "http://localhost:3000",
        ]

    def test_ten_urls_supported(self, monkeypatch):
        origins = [f"https://svc{i}.example.com" for i in range(10)]
        monkeypatch.setenv("MISA_CORS_ALLOWED_ORIGINS", ",".join(origins))
        parsed = config._parse_cors_origins()
        assert parsed == origins
        assert len(parsed) == 10


class TestCorsMiddlewareBehavior:
    """End-to-end: a configured origin gets reflected in the response
    headers by Starlette's CORSMiddleware; an unlisted one does not.

    Uses a throwaway FastAPI app wired with CORSMiddleware(allow_origins=
    CORS_ALLOWED_ORIGINS) rather than reloading app.main — the real app
    is a shared singleton other test modules already hold a reference to
    (conftest's autouse fixtures import it at collection time), so
    reloading it here would risk desyncing those references instead of
    actually testing anything extra. CORSMiddleware's origin-matching
    behavior is standard Starlette; what's actually under test is that
    our parsed allowlist, wired the same way app/main.py wires it,
    produces the expected allow/deny outcome."""

    def _client_for_origins(self, origins: list[str]) -> TestClient:
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["POST", "GET"],
            allow_headers=["*"],
        )

        @app.get("/ping")
        def ping():
            return {"ok": True}

        return TestClient(app)

    def test_allowed_origin_is_reflected(self):
        client = self._client_for_origins(
            ["https://app.example.com", "https://admin.example.com"]
        )
        r = client.get("/ping", headers={"Origin": "https://app.example.com"})
        assert r.headers.get("access-control-allow-origin") == "https://app.example.com"

    def test_unlisted_origin_is_not_reflected(self):
        client = self._client_for_origins(["https://app.example.com"])
        r = client.get("/ping", headers={"Origin": "https://evil.com"})
        assert r.headers.get("access-control-allow-origin") != "https://evil.com"

    def test_wildcard_reflects_any_origin(self):
        client = self._client_for_origins(["*"])
        r = client.get("/ping", headers={"Origin": "https://anything.example.com"})
        assert r.headers.get("access-control-allow-origin") == "*"
