import pytest

from app import config, rate_limit
from app.auth import verify_credentials
from app.main import app


@pytest.fixture(autouse=True)
def _isolate_local_env(monkeypatch):
    """Keep the unit suite independent of a developer's local `.env`.

    Local demos often set ``MISA_AUTH_DISABLED=true`` and
    ``MISA_RESIDENCY_MODE=strict`` (Ollama). Those flags break auth/RBAC
    assertions and redirect mocked Azure clients to a live Ollama process.
    Residency / auth tests that need the real flags monkeypatch them back.
    """
    monkeypatch.setattr(config, "AUTH_DISABLED", False)
    monkeypatch.setattr(config, "RESIDENCY_STRICT", False)
    monkeypatch.setattr(config, "RESIDENCY_MODE", "standard")
    monkeypatch.setattr(config, "DATA_LLM_BACKEND", "azure")


@pytest.fixture(autouse=True)
def _bypass_auth(monkeypatch):
    """Override auth dependency for all tests — API_USERNAME/PASSWORD are not set in CI.

    test-user is granted admin so router-level RBAC (analyst/admin) does not
    403 the bulk of the suite; dedicated RBAC tests clear this override and
    exercise real roles.
    """
    monkeypatch.setitem(config.AUTH_USER_ROLES, "test-user", config.ROLE_ADMIN)
    app.dependency_overrides[verify_credentials] = lambda: "test-user"
    yield
    app.dependency_overrides.pop(verify_credentials, None)
    config.AUTH_USER_ROLES.pop("test-user", None)


@pytest.fixture(autouse=True)
def _disable_rate_limiting():
    """Rate limiting is off by default in the suite so endpoint tests
    aren't throttled by prior tests hammering the same TestClient IP.
    Tests that specifically verify throttling re-enable it locally (see
    tests/test_rate_limit.py) and reset counters afterward."""
    rate_limit.set_enabled(False)
    rate_limit.reset_all()
    yield
    rate_limit.set_enabled(False)
    rate_limit.reset_all()
