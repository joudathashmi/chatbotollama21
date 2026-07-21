"""Tests for POST /api/v1/search (previously untested).

The endpoint streams NDJSON from the structured query builder. These tests
patch the builder so no live database is needed and assert the router's
request-passing, NDJSON serialization, value cleaning, and validation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.database import COMPANY_TABLE
from app.main import app

client = TestClient(app)


def _fake_builder(rows):
    """Return a stand-in for generate_query_and_run_query that records the
    call args and yields a DataFrame built from `rows`."""
    calls = {}

    def _builder(table, filters, order_by, descending, limit):
        calls["table"] = table
        calls["filters"] = filters
        calls["order_by"] = order_by
        calls["descending"] = descending
        calls["limit"] = limit
        return pd.DataFrame(rows), "SELECT * FROM ... LIMIT 25", []

    return _builder, calls


def test_search_returns_ndjson_rows():
    rows = [
        {"id": 1, "company_name": "Alpha Co", "revenue_usd": 100},
        {"id": 2, "company_name": "Beta Co", "revenue_usd": 200},
    ]
    builder, _ = _fake_builder(rows)
    with patch("app.routers.v1.search.generate_query_and_run_query", builder):
        r = client.post("/api/v1/search", json={"filters": {}, "limit": 25})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert len(lines) == 2
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["company_name"] == "Alpha Co"
    assert parsed[1]["id"] == 2


def test_search_passes_filters_order_and_limit_through():
    builder, calls = _fake_builder([])
    body = {
        "filters": {"sector": {"op": "=", "value": "Tech"}},
        "order_by": "revenue_usd",
        "descending": False,
        "limit": 7,
    }
    with patch("app.routers.v1.search.generate_query_and_run_query", builder):
        r = client.post("/api/v1/search", json=body)

    assert r.status_code == 200
    assert calls["table"] == COMPANY_TABLE
    assert calls["filters"] == {"sector": {"op": "=", "value": "Tech"}}
    assert calls["order_by"] == "revenue_usd"
    assert calls["descending"] is False
    assert calls["limit"] == 7


def test_search_cleans_nan_and_inf_floats_to_null():
    rows = [{"id": 1, "revenue_usd": float("nan"), "score": float("inf")}]
    builder, _ = _fake_builder(rows)
    with patch("app.routers.v1.search.generate_query_and_run_query", builder):
        r = client.post("/api/v1/search", json={"filters": {}})

    obj = json.loads(r.text.splitlines()[0])
    # NaN/inf are not valid JSON — the router must coerce them to null.
    assert obj["revenue_usd"] is None
    assert obj["score"] is None


def test_search_empty_result_is_empty_body():
    builder, _ = _fake_builder([])
    with patch("app.routers.v1.search.generate_query_and_run_query", builder):
        r = client.post("/api/v1/search", json={"filters": {}})
    assert r.status_code == 200
    assert r.text.strip() == ""


def test_search_uses_defaults_when_only_filters_given():
    builder, calls = _fake_builder([])
    with patch("app.routers.v1.search.generate_query_and_run_query", builder):
        r = client.post("/api/v1/search", json={"filters": {}})
    assert r.status_code == 200
    # Schema defaults: descending=True, limit=25, order_by=None.
    assert calls["descending"] is True
    assert calls["limit"] == 25
    assert calls["order_by"] is None


def test_search_rejects_non_int_limit():
    r = client.post("/api/v1/search", json={"filters": {}, "limit": "lots"})
    assert r.status_code == 422


def test_search_rejects_non_dict_filters():
    r = client.post("/api/v1/search", json={"filters": ["not", "a", "dict"]})
    assert r.status_code == 422
