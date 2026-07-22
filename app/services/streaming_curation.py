"""
Streaming variant of the curation pipeline.

The non-streaming `curate_company_insights` blocks until the LLM
produces its full ~700-1500 output tokens — on Azure Sweden Central
at ~50 tok/s that's 14-30 seconds before the user sees anything.

This module exposes `stream_company_insights_chunks()` which:
  - Builds the SAME prompt as the non-streaming variant
  - Opens an OpenAI client with stream=True
  - Yields content deltas one-by-one as they arrive from the model

The user sees text appearing within ~1-2 seconds (TTFT) and the
full answer accumulates progressively. Total wall-clock is similar
to the non-streaming call, but perceived latency is dramatically
better.

Trade-offs (the streaming path skips these vs. the full curation):
  - Output validator regen (style violations) — caller can run on
    the assembled text after the stream completes if needed
  - Executive quality check (Rule 10) — same; caller can re-curate
    non-streamingly if the score is too low
  - Anti-noise scrubber — applied per chunk inline (best-effort)

When streaming is off (the JSON-response path), callers continue to
use `curate_company_insights` and get the full quality pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from app.config import (
    CHAT_OPENAI_STORE,
    openai_advisory_max_tokens_kw,
)
from app.prompts.chat_system import curation_system_prompt
from app.services.curation import (
    safe_rows_for_curation,
    slim_rows_for_person_brief,
    classify_match,
)


def _build_curation_messages(
    rows: list[dict],
    user_question: str,
    *,
    locale: str = "en",
    entity_candidate: str | None = None,
    entity_matched: str | None = None,
    table: str | None = None,
    intent: str | None = None,
    depth: str | None = None,
) -> list[dict] | None:
    """Build the same [system, user] message pair the non-streaming
    curator uses. Returns None when there's no useful payload to
    curate (caller should fall back to non-streaming path)."""
    safe = safe_rows_for_curation(rows)
    if not safe:
        return None

    _PEOPLE_TABLES = (
        "executives", "company_executives", "rhq_topexecutives",
        "board_positions", "contacts", "company_contact_records",
        "related_people", "profiles", "personal_informations",
        "misa_contact_details",
    )
    if (
        table in _PEOPLE_TABLES
        or (intent or "") in ("executive_lookup", "person_lookup", "executive_succession")
    ):
        if table not in _PEOPLE_TABLES:
            table = "company_executives"
        safe = slim_rows_for_person_brief(safe) or safe

    # Skip the no-match / fuzzy / broad classification logic for the
    # streaming path — those produce special messages that are simpler
    # to emit via the non-streaming path. Streaming is for the happy
    # path: we have rows, we have an entity, compose a real answer.
    classification, _ = classify_match(entity_candidate, entity_matched, rows)
    if classification == "none":
        return None  # caller falls back to non-streaming for graceful handling

    # Build depth + intent + market-intel + missing-data notes the
    # same way the non-streaming curator does.
    depth_note = ""
    if depth:
        try:
            from app.services.depth_detector import depth_note_for_curation
            depth_note = depth_note_for_curation(depth)
            if depth_note:
                depth_note += "\n"
        except Exception:
            pass
    intent_note = ""
    market_intel_note = ""
    missing_data_note = ""
    if intent:
        try:
            from app.services.intent_router import (
                intent_note_for_curation, ANTI_HALLUCINATION_NOTE,
                market_intel_note_for, missing_data_note_for,
            )
            d = intent_note_for_curation(intent)
            if d:
                intent_note = d + "\n" + ANTI_HALLUCINATION_NOTE + "\n"
            m = market_intel_note_for(intent, depth or "")
            if m:
                market_intel_note = m + "\n"
            md = missing_data_note_for(depth or "", intent)
            if md:
                missing_data_note = md + "\n"
        except Exception:
            pass

    payload = json.dumps(safe, ensure_ascii=False, default=str)
    table_label = f"`{table}`" if table else "the database"
    anti_loop_note = ""
    try:
        from app.services.llm_residency import is_local_data_backend
        if is_local_data_backend():
            anti_loop_note = (
                "HARD STOP (local model): Write EACH section at most ONCE. "
                "End after ONE ## 🇸🇦 Strategic Read and one _Sources_ line. "
                "Never invent From your documents / From the web / "
                "What's Reported / Supporting reporting.\n\n"
            )
    except Exception:
        pass
    user_content = (
        f"User question:\n{user_question}\n\n"
        f"{anti_loop_note}"
        f"{depth_note}"
        f"{intent_note}"
        f"{market_intel_note}"
        f"{missing_data_note}"
        f"Retrieved records from {table_label} (privacy-filtered JSON):\n{payload}"
    )
    return [
        {"role": "system", "content": curation_system_prompt(locale, table)},
        {"role": "user", "content": user_content},
    ]


def stream_company_insights_chunks(
    rows: list[dict],
    user_question: str,
    *,
    locale: str = "en",
    entity_candidate: str | None = None,
    entity_matched: str | None = None,
    table: str | None = None,
    client: Any,
    model: str,
    intent: str | None = None,
    depth: str | None = None,
) -> Iterator[str]:
    """Yields content deltas from the curation LLM call as they arrive.

    The caller (the SSE generator) forwards each chunk to the client
    immediately. The user sees text appearing within ~1-2 seconds
    instead of waiting 15-20s for the full response.

    Yields:
      - str chunks: token / phrase fragments from the model
      - StopIteration when the stream is exhausted

    Returns None implicitly (Iterator protocol) if anything fails —
    caller must guard. Falls back gracefully:
      - If build_curation_messages returns None (no usable rows, true
        no-match, etc.) — yields nothing.
      - If the OpenAI stream raises mid-flight — yields what it had
        and stops; the caller's "done" event is still emitted.
    """
    msgs = _build_curation_messages(
        rows, user_question,
        locale=locale,
        entity_candidate=entity_candidate,
        entity_matched=entity_matched,
        table=table,
        intent=intent,
        depth=depth,
    )
    if msgs is None:
        return
    # Same model tiering as the non-streaming curation path: quick
    # facts on the chat model, analytical depths on the advisory tier.
    from app.config import curation_model_for_depth, openai_determinism_kw
    model = curation_model_for_depth(depth, model)
    try:
        from app.services.llm_residency import resolve_narrative_completion_client
        client, model = resolve_narrative_completion_client(
            client, preferred_model=model,
        )
    except Exception:
        return
    try:
        from app.services.prompt_masking import mask_messages_for_llm
        msgs = mask_messages_for_llm(msgs)
    except Exception:
        pass
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=msgs,
            store=CHAT_OPENAI_STORE,
            stream=True,
            **openai_determinism_kw(),
            **openai_advisory_max_tokens_kw(),
        )
    except Exception:
        return
    try:
        assembled = ""
        for event in stream:
            try:
                delta = event.choices[0].delta
                content = getattr(delta, "content", None)
                if not content:
                    continue
                assembled += content
                low = assembled.lower()
                # Abort early if the local model starts looping sections.
                if low.count("strategic read") >= 3:
                    break
                if "from your documents" in low and low.find("from your documents") > 200:
                    break
                if "what's reported" in low and low.count("what's reported") >= 2:
                    break
                if len(assembled) > 400:
                    tail = assembled[-120:]
                    if assembled[:-120].find(tail) != -1:
                        break
                yield content
            except (IndexError, AttributeError):
                continue
    except Exception:
        # Stream interrupted — caller already got whatever arrived.
        return
