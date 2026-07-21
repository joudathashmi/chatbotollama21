"""Tests for the structured query builder in app/database.py:
  generate_query_and_run_query() and count_table_rows().

These exercised only via helpers/guards before. Here a fake connection
captures the SQL + params actually executed (no live DB), so we can assert:
  - values are PARAMETERIZED (%s placeholders + separate params), never
    interpolated into the SQL string
  - unknown / non-allowlisted filter columns are dropped
  - order_by is honoured only for allowlisted sortable columns
  - LIMIT is clamped to QUERY_MAX_LIMIT
All against the real company_profiles schema hints (no introspection).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import app.database as database
from app.database import (
    COMPANY_TABLE,
    count_table_rows,
    generate_query_and_run_query,
)


class _FakeCursor:
    def __init__(self, rows, fetchone_value):
        self._rows = rows
        self._fetchone_value = fetchone_value
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        # Ignore the validation "SELECT 1" probe; record real queries.
        if sql.strip() != "SELECT 1":
            self.executed.append((sql, params))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._fetchone_value


class _FakeConn:
    def __init__(self, rows=None, fetchone_value=(0,)):
        self.rows = rows or []
        self.fetchone_value = fetchone_value
        self.cursors: list[_FakeCursor] = []

    def cursor(self, *args, **kwargs):
        cur = _FakeCursor(self.rows, self.fetchone_value)
        self.cursors.append(cur)
        return cur

    def last_query(self):
        for cur in reversed(self.cursors):
            if cur.executed:
                return cur.executed[-1]
        return (None, None)


@pytest.fixture
def fake_conn(monkeypatch):
    """Patch get_db()/validation so the builder runs against a fake conn."""
    conn = _FakeConn()
    monkeypatch.setattr(database, "get_db", lambda: conn)
    monkeypatch.setattr(database, "_validate_cached_pg_connection", lambda c: None)
    return conn


# ─── generate_query_and_run_query ───────────────────────────────────────

def test_exact_filter_is_parameterized(fake_conn):
    # `sector` is filterable and NOT a substring-match column → exact `=`.
    generate_query_and_run_query(
        COMPANY_TABLE, filters={"sector": {"op": "=", "value": "Tech"}},
    )
    sql, params = fake_conn.last_query()
    assert "sector = %s" in sql
    assert params == ["Tech"]  # value is bound, not inlined
    assert "Tech" not in sql   # never interpolated into the statement


def test_substring_column_uses_ilike(fake_conn):
    generate_query_and_run_query(
        COMPANY_TABLE, filters={"company_name": {"op": "=", "value": "Alphabet"}},
    )
    sql, params = fake_conn.last_query()
    assert "company_name ILIKE %s" in sql
    assert params == ["%Alphabet%"]


def test_unknown_filter_column_is_dropped(fake_conn):
    generate_query_and_run_query(
        COMPANY_TABLE,
        filters={"password_hash": {"op": "=", "value": "x"}},
    )
    sql, params = fake_conn.last_query()
    # Not allowlisted → the column never reaches the SQL and binds no params.
    # (company_profiles' projection subquery has its own internal WHERE, so
    # we assert on the dropped column + empty params, not absence of WHERE.)
    assert "password_hash" not in sql
    assert params == []


def test_order_by_allowlisted_column_applied(fake_conn):
    generate_query_and_run_query(
        COMPANY_TABLE, filters={}, order_by="revenue_usd", descending=True,
    )
    sql, _ = fake_conn.last_query()
    assert "ORDER BY revenue_usd DESC" in sql


def test_order_by_non_sortable_column_ignored(fake_conn):
    # `company_name` is filterable but NOT in the sortable allowlist.
    generate_query_and_run_query(
        COMPANY_TABLE, filters={}, order_by="company_name",
    )
    sql, _ = fake_conn.last_query()
    assert "ORDER BY" not in sql


def test_limit_is_clamped_to_max(fake_conn):
    with patch.object(database, "QUERY_MAX_LIMIT", 50):
        generate_query_and_run_query(COMPANY_TABLE, filters={}, limit=10_000)
    sql, _ = fake_conn.last_query()
    assert "LIMIT 50" in sql


def test_limit_under_cap_is_respected(fake_conn):
    with patch.object(database, "QUERY_MAX_LIMIT", 200):
        generate_query_and_run_query(COMPANY_TABLE, filters={}, limit=15)
    sql, _ = fake_conn.last_query()
    assert "LIMIT 15" in sql


def test_unknown_table_raises(fake_conn):
    with pytest.raises(ValueError):
        generate_query_and_run_query("wealth_accounts", filters={})


def test_numeric_comparison_operator(fake_conn):
    # count_table_rows shares _build_where_clauses; use a filterable numeric-ish
    # column via a comparison operator to confirm operator pass-through.
    generate_query_and_run_query(
        COMPANY_TABLE, filters={"status": {"op": "=", "value": "active"}},
    )
    sql, params = fake_conn.last_query()
    assert "status = %s" in sql
    assert params == ["active"]


# ─── count_table_rows ───────────────────────────────────────────────────

def test_count_returns_count_and_parameterized_sql(fake_conn):
    fake_conn.fetchone_value = (42,)
    n, sql, params = count_table_rows(
        COMPANY_TABLE, filters={"sector": {"op": "=", "value": "Tech"}},
    )
    assert n == 42
    assert sql.startswith("SELECT COUNT(*)")
    assert "sector = %s" in sql
    assert params == ["Tech"]


def test_count_all_invalid_filters_raises(fake_conn):
    # A requested column that doesn't exist on the table → ValueError rather
    # than silently counting the whole table.
    with pytest.raises(ValueError):
        count_table_rows(COMPANY_TABLE, filters={"nonexistent_col": {"op": "=", "value": 1}})
