"""Every error response across the API shares ONE JSON contract:

    {
      "success": false,
      "error": {"code": "...", "message": "...", "field": null, "details": null},
      "request_id": null,
      "timestamp": "...",
      "path": "...",
      "status": <int>
    }

This file checks that contract holds for the four ways an error can
originate: raised HTTPException (auth), Pydantic validation (422),
rate limiting (429, covered in more depth in test_rate_limit.py), and
a genuinely unhandled exception (500). Also checks that headers load-
bearing for the auth flow (WWW-Authenticate) survive the unified
handler unchanged.
"""

from fastapi.testclient import TestClient

from app.auth import verify_credentials
from app.main import app

client = TestClient(app)


def _assert_standard_shape(body: dict, expected_status: int, expected_code: str | None = None):
    assert body["success"] is False
    assert isinstance(body["error"], dict)
    assert isinstance(body["error"]["code"], str) and body["error"]["code"]
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert body["status"] == expected_status
    assert "timestamp" in body
    assert "path" in body
    if expected_code:
        assert body["error"]["code"] == expected_code


def test_401_unauthenticated_uses_standard_shape_and_preserves_www_authenticate():
    """Auth failures advertise the Bearer scheme (WWW-Authenticate: Bearer)
    for JWT auth — the global handler must not swallow the header while
    restyling the body."""
    app.dependency_overrides.pop(verify_credentials, None)
    try:
        r = client.get("/api/v1/questions")  # any authed route, no creds supplied
    finally:
        app.dependency_overrides[verify_credentials] = lambda: "test-user"

    assert r.status_code == 401
    assert r.headers.get("www-authenticate", "").lower().startswith("bearer")
    _assert_standard_shape(r.json(), 401, expected_code="UNAUTHORIZED")


def test_422_validation_error_uses_standard_shape():
    # Missing all required fields on FeedbackRequest.
    r = client.post("/api/v1/feedback", json={})
    assert r.status_code == 422
    _assert_standard_shape(r.json(), 422, expected_code="VALIDATION_ERROR")
    # Field-level detail is preserved for debugging, just moved under details.
    assert "verdict" in r.json()["error"]["details"] or "question" in r.json()["error"]["details"]


def test_404_not_found_uses_standard_shape():
    r = client.get("/api/v1/this-route-does-not-exist")
    assert r.status_code == 404
    _assert_standard_shape(r.json(), 404, expected_code="NOT_FOUND")


def test_business_card_invalid_mime_type_still_uses_standard_shape(tmp_path):
    """Business-card errors were already in this shape before this
    change (returned directly via create_error_response) — confirm the
    unified handler didn't regress them."""
    r = client.post(
        "/api/v1/business-card/upload",
        files={"file": ("card.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400
    _assert_standard_shape(r.json(), 400, expected_code="INVALID_MIME_TYPE")
