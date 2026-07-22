"""In-process quality / retrieval metrics for ops dashboards.

Thread-safe counters. No PII. Reset on process restart (pair with
log aggregators for durable history).
"""

from __future__ import annotations

import threading
import time
from typing import Any


_LOCK = threading.Lock()
_STARTED = time.time()
_COUNTERS: dict[str, int] = {
    "turns_total": 0,
    "retrieval_failures": 0,
    "verified_empty": 0,
    "false_zero_blocked": 0,
    "quality_gate_critical": 0,
    "quality_gate_repairs": 0,
    "schema_validation_failures": 0,
    "pdf_exports": 0,
    "pdf_quality_blocked": 0,
    "docx_exports": 0,
    "truncated_answers": 0,
    "filter_drops": 0,
    "hard_block_answers": 0,
    "eval_fail": 0,
    "eval_pass": 0,
}
_BY_STATUS: dict[str, int] = {}
_BY_INTENT: dict[str, int] = {}


def _inc(d: dict[str, int], key: str, n: int = 1) -> None:
    d[key] = int(d.get(key) or 0) + n


def record_turn(
    *,
    intent: str | None = None,
    retrieval_status: str | None = None,
    quality_gate: dict | None = None,
    quality_eval: dict | None = None,
    truncated: bool = False,
    filter_drop: bool = False,
) -> None:
    with _LOCK:
        _inc(_COUNTERS, "turns_total")
        if intent:
            _inc(_BY_INTENT, str(intent)[:64])
        if retrieval_status:
            st = str(retrieval_status)[:64]
            _inc(_BY_STATUS, st)
            if st in (
                "SOURCE_UNAVAILABLE", "TIMEOUT", "CONNECTION_ERROR",
                "UNKNOWN_ERROR", "AUTHENTICATION_ERROR", "PERMISSION_ERROR",
                "INVALID_QUERY", "error", "ERROR",
            ) or st.endswith("_ERROR"):
                _inc(_COUNTERS, "retrieval_failures")
            if st in ("SUCCESS_EMPTY", "zero_records"):
                _inc(_COUNTERS, "verified_empty")
        qg = quality_gate or {}
        issues = qg.get("issues") or []
        fixes = qg.get("fixes") or []
        if any(
            c in ("false_zero_on_retrieval_failure",
                  "contradicts_internal_licensed_count",
                  "contradicts_internal_rhq_count")
            for c in issues
        ):
            _inc(_COUNTERS, "false_zero_blocked")
        if any("hard_block" in str(f) for f in fixes):
            _inc(_COUNTERS, "hard_block_answers")
        if issues:
            _inc(_COUNTERS, "quality_gate_critical")
        if fixes:
            _inc(_COUNTERS, "quality_gate_repairs", len(fixes))
        ev = quality_eval or {}
        if ev.get("pass") is True:
            _inc(_COUNTERS, "eval_pass")
        elif ev.get("pass") is False:
            _inc(_COUNTERS, "eval_fail")
        if truncated:
            _inc(_COUNTERS, "truncated_answers")
        if filter_drop:
            _inc(_COUNTERS, "filter_drops")


def record_export(*, kind: str, quality_blocked: bool = False) -> None:
    with _LOCK:
        if kind == "pdf":
            _inc(_COUNTERS, "pdf_exports")
            if quality_blocked:
                _inc(_COUNTERS, "pdf_quality_blocked")
        elif kind == "docx":
            _inc(_COUNTERS, "docx_exports")


def snapshot() -> dict[str, Any]:
    with _LOCK:
        return {
            "uptime_sec": round(time.time() - _STARTED, 1),
            "counters": dict(_COUNTERS),
            "by_retrieval_status": dict(_BY_STATUS),
            "by_intent": dict(_BY_INTENT),
        }
