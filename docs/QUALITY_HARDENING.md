# Quality Hardening — Deliverables

## 1. Executive summary

Response quality regressed because failures were treated as facts (especially
empty/zero results), prompts conflicted on licensing sources, Markdown was
used as the primary document interface, and validators were fail-open. This
pass adds reusable pipeline contracts: structured intent, retrieval status,
evidence provenance, safe fetch wrappers, hard quality gates, recommendation
filters, structured response schemas, PDF structured rendering, and
observability hooks — without hardcoding country or count facts.

## 2. Root-cause analysis

| Failure class | Root cause | Systemic fix |
|---|---|---|
| False zero | Exception → `[]` / `0` → prose “no companies” | `RetrievalResult` + `safe_fetch` + formatter guards + hard gate |
| Wrong licensing totals | `rhq_licenses` taught as SoR | Canonical `company_profiles.licensed` / `is_rhq` in prompts + short-circuit |
| Truncated rankings | Token limit mid-markdown table | DB-seeded ranking + structured theses; gate flags truncation |
| PDF table breakage | Raw MD tables → nl2br | Pre-parse to HTML/cards; structured PDF path |
| Generic strategy | Unconstrained prose | Recommendation quality rules + eval dimensions |
| Silent errors | Broad `except: pass` | Trace + `_db_error` envelopes; limitations not zeros |
| No intent contract | String labels only | `QueryIntent` object before retrieval |

## 3. Architecture flow

```
User question
  → prompt_guard
  → QueryIntent (structured) + PipelineTrace
  → intent_router / short-circuits (licensing, country list, advisory…)
  → safe retrieval (RetrievalResult / _db_error)
  → evidence_context assembly (usable vs limitations)
  → compose (deterministic briefing | advisory_structured | curation)
  → response_validator + quality_gate (hard_block critical)
  → quality_eval (dimensions + pass/fail)
  → answer_finalize
  → UI Markdown | PDF (normalize tables / structured render)
  → chat_turn + pipeline_trace logs
```

## 4. Systemic issue categories

1. Retrieval failure ≠ empty result  
2. Conflicting source hierarchy  
3. Missing provenance on context  
4. Fail-open validators  
5. Unstructured primary interface  
6. Generic recommendations  
7. Document-rendering from raw Markdown  
8. Weak observability  

## 5–6. Files changed (summary)

**New**

- `app/services/query_intent.py` — structured intent  
- `app/services/evidence_context.py` — provenance + context assembly  
- `app/services/safe_fetch.py` — never convert exceptions to empty facts  
- `app/services/pipeline_trace.py` — stage timings / retrieval telemetry  
- `app/services/recommendation_quality.py` — reject generic recommendations  
- `app/schemas/quality_response.py` — typed response / licensing schemas  
- `tests/test_systemic_quality.py` — failure-class regression suite  
- `docs/QUALITY_HARDENING.md` — this document  

**Updated**

- `engagement_data.py` — sector distribution status; licensing summary try/except  
- `chat_engine.py` — intent/trace wiring; country/licensing false-zero guards  
- `quality_gate.py` — hard block on unrepaired critical issues  
- `quality_eval.py` — multi-dimension scoring + thresholds  
- `answer_finalize.py` — hard gate + eval telemetry  
- `source_policy.py` — recommendation rules addon  
- `pdf_export.py` — structured + RTL-aware render path  
- `prompts/chat_system.py` / `intent_router.py` — canonical licensing source  

## 7–12. New policies (pointers)

- **Source precedence:** `app/services/source_policy.py`  
- **Retrieval status:** `app/services/retrieval_status.py`  
- **Context / provenance:** `app/services/evidence_context.py`  
- **Structured schemas:** `app/schemas/quality_response.py`  
- **Validation / repair:** `quality_gate.py`, `response_validator.py`  
- **Ranking:** `target_ranking.py` + `advisory_structured.py`  

## 13–15. Ranking, rendering, logging

- Ranking criteria remain transparent in `target_ranking` / advisory structured path.  
- PDF: markdown tables normalized to HTML/cards; optional validated structured input.  
- Logging: `chat_turn` now includes `query_intent`, `quality_eval`, `retrieval_failures_n`; `pipeline_trace` emits stage metrics.

## 16–17. Tests & evaluation

```bash
cd /Users/joudathashmi/Documents/NIMs_Chatbot_ollama21
./venv/bin/pytest tests/test_systemic_quality.py tests/test_quality_hardening.py \
  tests/test_india_company_targeting.py tests/test_advisory.py -q
```

Evaluator: `evaluate_answer(..., min_pass_score=70)` — critical issues always fail.

## 18. Before / after (behavioural)

| Scenario | Before | After |
|---|---|---|
| DB column missing on India footprint | “0 licensed / 0 RHQ” | “data unavailable — not a verified zero” |
| “Active MISA licenses” | RHQ-led title; sometimes `rhq_licenses` | Licensing Snapshot; `company_profiles` counts |
| Targeting mid-table cut | Truncated pipe row | Structured ranking + gate |
| Generic “engage stakeholders” | Shipped | Scored/rejected by recommendation quality |

## 19. Remaining limitations

**Closed in this pass** (was previously deferred):

- ChatResponse now exposes `trace_id`, `intent`, `retrieval_status`, `quality`, `data_limitations`
- In-process metrics at `GET /api/v1/chat/quality/metrics`
- First-paragraph validator fail-closed for advisory intents; soft contracts hard-fail false-zeros
- Truncation / partial-result banners; filter-drop refused when all filters invalid
- Quality-gated DOCX export at `POST /api/v1/chat/export/docx` (needs `python-docx`)

**Still out of scope for this codebase revision:**

- Full hybrid RAG re-ranker / parent-document rewrite (existing hybrid_briefing + document_store remain)
- Dedicated metrics UI / Grafana dashboards (API snapshot is the ops surface)
- PowerPoint (PPTX) structured exporter (DOCX + PDF covered)

## 21. Audit P0 follow-up (post [Audit quality failure patterns](9a740bf0-bfee-4599-882a-2c8177818bfe))

Closed from the remaining backlog:

- Smart-search failures marked `-- RETRIEVAL_FAILED` + propagated (`row_count=None`)
- Country licensing exceptions no longer fall through to global aggregates
- Status vocabulary unified toward `SUCCESS_*` / `SOURCE_UNAVAILABLE` (legacy aliases still accepted)
- Web `search_with_status` envelope; correlator injects `_section_errors` on partial failure
- PDF export runs `quality_gate` before render

Still deferred: full hybrid RAG rewrite; dedicated Grafana UI; PPTX exporter.

```bash
# Tests
cd /Users/joudathashmi/Documents/NIMs_Chatbot_ollama21
./venv/bin/pytest tests/test_systemic_quality.py tests/test_quality_hardening.py \
  tests/test_india_company_targeting.py tests/test_advisory.py -q

# Server
export MISA_AUTH_DISABLED=true MISA_NARRATIVE_CLOUD=true
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# UI: http://localhost:8000/chat
```

Smoke:

1. “Tell me the active MISA licenses” → Licensing Snapshot + live totals  
2. Origin licensed count (any country) → positive live totals (not 0/0)  
3. Company targeting for any origin → full ranking, no mid-row truncation  
4. Force DB error path (tests) → never claims verified zero  

## 13. Jul21 platform lock (all origins)

**Licensing SoR:** `company_profiles.licensed` / `is_rhq` via
`engagement_data._licensing_predicates` (`LICENSING_SOR`). Role+lifecycle
is fallback only when boolean columns are absent — never the preferred
corridor count for modern schemas.

**Advisory repair:** after compose, `enrich_advisory_deliverable` runs,
then `advisory_deliverable_violations` re-checks required sections and
re-enriches once if gaps remain (Phases / KPIs / deep-dives / trade bodies).

**Trade bodies:** `_default_trade_bodies` returns ≥5 rows per origin
(catalog IPA/chambers + sector association pads).

**Narrative cloud:** `MISA_NARRATIVE_CLOUD=true` is required for full Jul21
memo depth. When false, keep `MISA_DB_BRIEFING_MODE=deterministic` so
company/person templates still ship Snapshot of Operations + Strategic Read.

**SSE stream repair:** after fast-stream assemble, `repair_company_answer_if_thin`
runs soft_check; ops-less drafts fall back to deterministic templates.
Streaming curation uses the advisory token budget (8k), not 3072.

**One exit:** docs / GK / deep profile / officeholder-no-DB / advisory /
PDF / DOCX all pass `finalize_answer` (cite-preserving when web/docs present).

**Hard-rec eval:** `quality_eval` fails soft-rec / non-world-class rec sections
and missing Snapshot of Operations on company briefs.

**Regression suite:** `tests/test_jul21_multi_origin_goldens.py` covers
multi-origin IPA, enrich completeness, and SoR teaching;
`tests/test_stream_repair.py` + `tests/test_world_class_quality_battery.py`
cover SSE repair and world-class rec bar.
