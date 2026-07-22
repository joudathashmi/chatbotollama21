"""Pipeline stage tracing for quality / latency observability.

Complements turn_id logging in chat() with per-stage timings and
retrieval / validation outcomes. Does not log raw answer text.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

log = logging.getLogger("misa.pipeline")


@dataclass
class PipelineTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    intent: dict[str, Any] = field(default_factory=dict)
    retrievals: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def record_stage(
        self,
        name: str,
        *,
        duration_ms: float,
        status: str = "ok",
        detail: dict | None = None,
    ) -> None:
        self.stages[name] = {
            "duration_ms": round(duration_ms, 2),
            "status": status,
            **(detail or {}),
        }

    def record_retrieval(self, envelope: dict[str, Any]) -> None:
        # Keep compact — no row payloads
        self.retrievals.append({
            k: envelope.get(k)
            for k in (
                "source_name", "retrieval_status", "record_count",
                "filters", "confidence", "do_not_claim_zero",
                "counts_unavailable", "error",
            )
            if k in envelope or envelope.get(k) is not None
        })

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "event": "pipeline_trace",
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "stages": self.stages,
            "intent": self.intent,
            "retrievals": self.retrievals[:20],
            "validation": self.validation,
            "quality": self.quality,
            "meta": self.meta,
            "stage_count": len(self.stages),
            "retrieval_failure_n": sum(
                1 for r in self.retrievals
                if r.get("do_not_claim_zero")
                or str(r.get("retrieval_status") or "").endswith("ERROR")
                or r.get("retrieval_status") in (
                    "SOURCE_UNAVAILABLE", "TIMEOUT", "CONNECTION_ERROR",
                    "UNKNOWN_ERROR", "error",
                )
            ),
        }

    def emit(self) -> None:
        try:
            import json
            log.info(json.dumps(self.to_log_dict(), default=str))
        except Exception:
            log.info("pipeline_trace trace_id=%s stages=%s",
                     self.trace_id, list(self.stages))


@contextmanager
def stage(trace: PipelineTrace, name: str) -> Iterator[dict[str, Any]]:
    """Time a pipeline stage; caller may set detail['status'] = 'error'."""
    detail: dict[str, Any] = {}
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield detail
        status = detail.pop("status", "ok")
    except Exception:
        status = "error"
        raise
    finally:
        trace.record_stage(
            name,
            duration_ms=(time.perf_counter() - t0) * 1000,
            status=status,
            detail=detail,
        )


def new_trace() -> PipelineTrace:
    return PipelineTrace()
