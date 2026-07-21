"""DB query_table hardening (Risk-20-5, Risk-20-1, Risk-20-6).

Consolidated coverage for the retrieval-layer security controls, all of which
are response-transparent for legitimate queries — they only block writes,
unauthorized tables/columns, and bulk/inference extraction:

  - least-privilege table allowlist + strengthened deny-list
  - column-leakage protection at the DB layer AND the user-response boundary
  - single-query row-limit clamp + per-turn aggregate row budget
  - SELECT-only execution guard
  - identity attribution on security events
  - blocked-table / blocked-column audit trail
  - count-only inference/enumeration guard

All offline — exercises the pure predicates/helpers and a synthetic
DataFrame; no live database required.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from app import db_introspect as dbi
from app.database import _assert_select_only, _drop_sensitive_columns
from app.routers.v1.chat import _extract_rows
from app.services import audit_log as al
from app.services import chat_engine as ce
from app.services.curation import cap_rows_for_turn, redact_rows_for_response
from app.services.rate_limiter import RateLimiter


# ═══════════════════════════════════════════════════════════════════
# 1. Table allowlist + deny-list
# ═══════════════════════════════════════════════════════════════════

class TestTableAllowlist:
    def test_investment_tables_are_allowed(self):
        for t in ("company_profiles", "country_profiles", "deals",
                  "opportunities", "fdi_data", "sectors"):
            assert t in dbi._DEFAULT_CHAT_TABLES_ALLOW

    def test_misa_operational_tables_kept_allowed(self):
        """These carry live MISA data (licensing, pipeline, reporting,
        matching, company contacts) and were queryable before the allowlist
        existed. Dropping them silently changes answers, so the allowlist
        must keep admitting them — verified against the real schema."""
        for t in ("bus_data", "leads", "reports", "report_types",
                  "match_outputs", "company_contact_records"):
            assert t in dbi._DEFAULT_CHAT_TABLES_ALLOW

    def test_auth_and_session_tables_denied(self):
        for t in ("users", "sessions", "personal_access_tokens",
                  "password_reset_tokens", "device_tokens", "migrations"):
            assert dbi._is_denied_table(t, set()) is True

    def test_other_app_prefixes_denied(self):
        """Wealth/CRM/IR/document tables from the other apps sharing this
        database must be denied outright."""
        for t in ("wealth_accounts", "family_members", "crm_contacts",
                  "ir_details", "ir_shareholders", "kyc_records",
                  "documents_store", "portfolio_holdings"):
            assert dbi._is_denied_table(t, set()) is True, f"{t} should be denied"

    def test_allowlist_contains_no_denied_table(self):
        for t in dbi._DEFAULT_CHAT_TABLES_ALLOW:
            assert dbi._is_denied_table(t, set()) is False, f"{t} allowed AND denied"


# ═══════════════════════════════════════════════════════════════════
# 2. Column-leakage protection
# ═══════════════════════════════════════════════════════════════════

class TestColumnLeakage:
    def test_credential_and_pii_columns_denied(self):
        for c in ("password", "password_hash", "api_key", "secret_token",
                  "otp_code", "passport_no", "national_id", "iban",
                  "credit_card", "cvv"):
            assert dbi._is_denied_column(c) is True, f"{c} should be denied"

    def test_legitimate_columns_kept(self):
        for c in ("company_name", "revenue_usd", "sector", "rhq_city",
                  "country_id", "stage", "status"):
            assert dbi._is_denied_column(c) is False, f"{c} wrongly denied"

    def test_drop_sensitive_columns_on_dataframe(self):
        df = pd.DataFrame([{"company_name": "X", "password_hash": "abc",
                            "api_key": "k", "revenue_usd": 10}])
        out = _drop_sensitive_columns(df)
        assert "company_name" in out.columns
        assert "revenue_usd" in out.columns
        assert "password_hash" not in out.columns
        assert "api_key" not in out.columns

    def test_redact_rows_for_response_strips_internal_fields(self):
        rows = [{"company_name": "Aramco", "team_comments": "internal note",
                 "review_status": "pending", "revenue_usd": 500}]
        out = redact_rows_for_response(rows)
        assert out and out[0].get("company_name") == "Aramco"
        assert "team_comments" not in out[0]
        assert "review_status" not in out[0]


# ═══════════════════════════════════════════════════════════════════
# 3. SELECT-only execution guard
# ═══════════════════════════════════════════════════════════════════

class TestSelectOnlyGuard:
    @pytest.mark.parametrize("sql", [
        "SELECT * FROM company_profiles LIMIT 10",
        "  select 1",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "(SELECT 1)",
        "SELECT 1;",  # lone trailing semicolon allowed
    ])
    def test_accepts_read_only(self, sql):
        _assert_select_only(sql)  # must not raise

    @pytest.mark.parametrize("sql", [
        "UPDATE t SET x=1", "DELETE FROM t", "DROP TABLE t",
        "TRUNCATE t", "INSERT INTO t VALUES (1)", "ALTER TABLE t ADD c int",
        "GRANT SELECT ON t TO r", "SELECT 1; DROP TABLE t",
    ])
    def test_rejects_writes_and_chaining(self, sql):
        with pytest.raises(ValueError):
            _assert_select_only(sql)


# ═══════════════════════════════════════════════════════════════════
# 4. Per-turn row budget (truncate + audit)
# ═══════════════════════════════════════════════════════════════════

class TestRowBudget:
    def test_under_budget_not_truncated(self, monkeypatch):
        monkeypatch.setattr("app.config.CHAT_MAX_ROWS_PER_TURN", 200, raising=False)
        rows = [{"i": i} for i in range(50)]
        out, truncated = cap_rows_for_turn(rows)
        assert truncated is False
        assert len(out) == 50

    def test_over_budget_truncates_and_audits(self, monkeypatch):
        monkeypatch.setattr("app.config.CHAT_MAX_ROWS_PER_TURN", 10, raising=False)
        captured = {}
        monkeypatch.setattr("app.services.audit_log.emit_security_event",
                            lambda p: captured.update(p))
        rows = [{"i": i} for i in range(25)]
        out, truncated = cap_rows_for_turn(rows, context="unit-test")
        assert truncated is True
        assert len(out) == 10
        assert captured.get("event") == "row_budget_truncated"
        assert captured.get("rows_returned") == 25
        assert captured.get("cap") == 10


# ═══════════════════════════════════════════════════════════════════
# 5. Identity attribution on security events (Risk-20-1)
# ═══════════════════════════════════════════════════════════════════

class TestAuditUserAttribution:
    def test_defaults_to_unknown_outside_a_request(self):
        token = al._audit_user.set("unknown")
        try:
            assert al.get_audit_user() == "unknown"
        finally:
            al._audit_user.reset(token)

    def test_set_then_get_round_trip(self):
        token = al._audit_user.set("unknown")
        try:
            al.set_audit_user("alice")
            assert al.get_audit_user() == "alice"
        finally:
            al._audit_user.reset(token)

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_identity_falls_back_to_unknown(self, blank):
        token = al._audit_user.set("someone")
        try:
            al.set_audit_user(blank)
            assert al.get_audit_user() == "unknown"
        finally:
            al._audit_user.reset(token)

    def test_emit_security_event_stamps_current_user(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(al, "AUDIT_LOG_ENABLED", True)
        monkeypatch.setattr(al._audit_logger, "info",
                            lambda msg: captured.update(json.loads(msg)))
        token = al._audit_user.set("bob")
        try:
            al.emit_security_event({"event": "unit_test_event"})
        finally:
            al._audit_user.reset(token)
        assert captured.get("user") == "bob"
        assert captured.get("event") == "unit_test_event"
        assert captured.get("kind") == "security"


# ═══════════════════════════════════════════════════════════════════
# 6. Blocked-table / blocked-column audit events (Risk-20-1)
# ═══════════════════════════════════════════════════════════════════

class TestBlockedTableAudit:
    def test_error_payload_is_unchanged(self):
        """The model must see exactly what it saw before this event was
        added — otherwise the user-facing answer could change."""
        assert ce._reject_table_and_audit("wealth_accounts") == {
            "error": "table 'wealth_accounts' is not allowed or unknown",
        }

    def test_emits_event_naming_the_table(self, monkeypatch):
        captured = []
        monkeypatch.setattr("app.services.audit_log.emit_security_event",
                            captured.append)
        ce._reject_table_and_audit("shareholder_registry")
        assert captured[0]["event"] == "query_table_blocked"
        assert captured[0]["table"] == "shareholder_registry"

    def test_empty_table_name_is_labelled(self, monkeypatch):
        captured = []
        monkeypatch.setattr("app.services.audit_log.emit_security_event",
                            captured.append)
        assert ce._reject_table_and_audit("") == {
            "error": "table '' is not allowed or unknown",
        }
        assert captured[0]["table"] == "(empty)"


class TestBlockedColumnAudit:
    def test_denied_column_request_is_logged(self, monkeypatch):
        captured = []
        monkeypatch.setattr(ce, "get_table_info",
                            lambda t: {"filterable": ["name", "country_id"]})
        monkeypatch.setattr("app.services.audit_log.emit_security_event",
                            captured.append)
        ce._audit_blocked_columns("company_profiles",
                                  {"name": "Aramco", "password_hash": "x"})
        assert captured[0]["event"] == "query_table_column_blocked"
        assert captured[0]["columns"] == ["password_hash"]

    def test_allowed_columns_emit_nothing(self, monkeypatch):
        captured = []
        monkeypatch.setattr(ce, "get_table_info",
                            lambda t: {"filterable": ["name", "country_id"]})
        monkeypatch.setattr("app.services.audit_log.emit_security_event",
                            captured.append)
        ce._audit_blocked_columns("company_profiles", {"name": "Aramco"})
        assert captured == []

    def test_internal_trace_keys_are_ignored(self, monkeypatch):
        captured = []
        monkeypatch.setattr(ce, "get_table_info", lambda t: {"filterable": ["name"]})
        monkeypatch.setattr("app.services.audit_log.emit_security_event",
                            captured.append)
        ce._audit_blocked_columns("deals", {"name": "x", "_count_only": True})
        assert captured == []


# ═══════════════════════════════════════════════════════════════════
# 7. count_only inference/enumeration guard (Risk-20-6)
# ═══════════════════════════════════════════════════════════════════

class TestCountOnlyInferenceGuard:
    def test_allows_calls_under_the_threshold(self, monkeypatch):
        monkeypatch.setattr(ce, "_count_only_limiter",
                            RateLimiter(max_requests=5, window_seconds=60))
        for _ in range(5):
            assert ce._count_only_guard("deals", {"stage": "won"}) is True

    def test_blocks_and_audits_over_the_threshold(self, monkeypatch):
        captured = []
        monkeypatch.setattr(ce, "_count_only_limiter",
                            RateLimiter(max_requests=2, window_seconds=60))
        monkeypatch.setattr("app.services.audit_log.emit_security_event",
                            captured.append)
        assert ce._count_only_guard("deals", {}) is True
        assert ce._count_only_guard("deals", {}) is True
        assert ce._count_only_guard("deals", {}) is False
        assert captured[-1]["event"] == "count_only_query"
        assert captured[-1]["rate_limited"] is True

    def test_every_call_is_tagged_even_when_allowed(self, monkeypatch):
        captured = []
        monkeypatch.setattr(ce, "_count_only_limiter",
                            RateLimiter(max_requests=10, window_seconds=60))
        monkeypatch.setattr("app.services.audit_log.emit_security_event",
                            captured.append)
        ce._count_only_guard("opportunities", {"country_id": 1, "_internal": True})
        assert captured[0]["event"] == "count_only_query"
        assert captured[0]["filter_cols"] == ["country_id"]
        assert captured[0]["rate_limited"] is False

    def test_fails_open_if_the_limiter_breaks(self, monkeypatch):
        class _Boom:
            def check(self, key):
                raise RuntimeError("boom")

        monkeypatch.setattr(ce, "_count_only_limiter", _Boom())
        assert ce._count_only_guard("deals", {}) is True

    def test_limit_is_keyed_per_identity(self, monkeypatch):
        monkeypatch.setattr(ce, "_count_only_limiter",
                            RateLimiter(max_requests=1, window_seconds=60))
        token = al._audit_user.set("alice")
        try:
            assert ce._count_only_guard("deals", {}) is True
            assert ce._count_only_guard("deals", {}) is False
            al.set_audit_user("bob")
            assert ce._count_only_guard("deals", {}) is True
        finally:
            al._audit_user.reset(token)


# ═══════════════════════════════════════════════════════════════════
# 8. Identity reaches the pipeline end-to-end (Risk-20-1)
#
# The whole design rests on the ContextVar set in chat_endpoint surviving
# asyncio.to_thread — that is where every retrieval-layer security event is
# emitted from. Lock it in for BOTH response paths; the SSE branch runs its
# generator after the endpoint returns, so it is the fragile one.
# ═══════════════════════════════════════════════════════════════════

class TestIdentityReachesPipeline:
    @staticmethod
    def _run(stream: bool) -> str:
        from fastapi.testclient import TestClient
        from app.main import app

        seen = {}

        def _fake_chat(question, history, locale):
            seen["user"] = al.get_audit_user()
            return {"answer": "ok", "tool_calls": [], "error": None,
                    "web_sources": []}

        with patch("app.routers.v1.chat.chat", _fake_chat):
            r = TestClient(app).post(
                "/api/v1/chat", json={"question": "hi", "stream": stream})
            _ = r.text  # drain: the SSE generator only runs on iteration
        assert r.status_code == 200
        return seen.get("user")

    def test_json_path_attributes_the_caller(self):
        # conftest's _bypass_auth fixture overrides verify_credentials to
        # return "test-user" for every request in the suite.
        assert self._run(stream=False) == "test-user"

    def test_sse_path_attributes_the_caller(self):
        assert self._run(stream=True) == "test-user"
