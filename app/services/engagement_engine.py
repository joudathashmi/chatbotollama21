"""
Engagement dossier engine: uses OpenAI Responses API with web_search_preview
to generate strategic MISA investor dossiers.

Uses AsyncOpenAI for native async streaming — avoids thread-pool pressure for
long-running 2–5 minute model calls.
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import HTTPException
from openai import AsyncOpenAI

from app.config import ENGAGEMENT_MODEL, ENGAGEMENT_OPENAI_KEY
from app.prompts.engagement_system import build_system_prompt, detect_response_language


def _get_async_client() -> AsyncOpenAI:
    from app.services.llm_residency import public_engagement_allowed
    if not public_engagement_allowed():
        raise HTTPException(
            status_code=503,
            detail=(
                "Engagement dossiers are disabled under "
                "MISA_RESIDENCY_MODE=strict (public OpenAI egress blocked). "
                "Set MISA_RESIDENCY_BLOCK_PUBLIC_ENGAGEMENT=false only if "
                "policy allows sending entity/context text to api.openai.com."
            ),
        )
    if not ENGAGEMENT_OPENAI_KEY or ENGAGEMENT_OPENAI_KEY.startswith("sk-REPLACE"):
        raise HTTPException(
            status_code=503,
            detail="MISA_ENGAGEMENT_OPENAI_API_KEY (or OPENAI_API_KEY) is not configured.",
        )
    return AsyncOpenAI(api_key=ENGAGEMENT_OPENAI_KEY)


def _build_input(entity: str, mode: str, context: str) -> str:
    return "\n".join([
        f"Entity: {entity.strip()}",
        f"Mode: {mode}",
        f"Additional context: {context.strip() or 'none'}",
    ])


_CHUNK_MIN_CHARS = 80   # don't emit until buffer reaches this size …
_CHUNK_MAX_CHARS = 300  # … unless it exceeds this (hard flush)


def _should_flush(buf: str) -> bool:
    """True when the buffer ends at a natural prose boundary."""
    if len(buf) >= _CHUNK_MAX_CHARS:
        return True
    # Paragraph break (markdown heading, blank line, list item boundary)
    if buf.endswith("\n\n") or buf.endswith("\n---\n"):
        return True
    # Sentence end followed by space, but only once the buffer is big enough
    if len(buf) >= _CHUNK_MIN_CHARS and buf[-1] == " ":
        stripped = buf.rstrip()
        if stripped and stripped[-1] in ".!?:":
            return True
    return False


async def engagement_sse_stream(
    entity: str,
    mode: str,
    context: str,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE-formatted strings.

    Deltas from the Responses API are tiny (word/character level). This
    generator buffers them and only emits at natural prose boundaries
    (paragraph breaks, sentence endings, or after ~300 chars) so clients
    receive readable chunks rather than single words.

    Event contract:
      {"meta": {"phase": "opening"}}   — before API call
      {"meta": {"phase": "research"}}  — after API call starts (web_search running)
      {"delta": "...text..."}          — buffered prose chunk
      {"error": "...message..."}       — any error (followed by [DONE])
      [DONE]                           — always last
    """
    import json

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield sse({"meta": {"phase": "opening"}})

    try:
        client = _get_async_client()
    except HTTPException as e:
        yield sse({"error": e.detail})
        yield "data: [DONE]\n\n"
        return

    user_input = _build_input(entity, mode, context)
    # Detect the dossier's response language from the entity/context text
    # itself — Arabic input → Arabic dossier, otherwise English. Never mixed.
    instructions = build_system_prompt(detect_response_language(entity, context))
    had_delta = False
    buf = ""

    try:
        response = await client.responses.create(
            model=ENGAGEMENT_MODEL,
            instructions=instructions,
            input=user_input,
            tools=[{"type": "web_search_preview", "search_context_size": "high"}],
            tool_choice="auto",
            stream=True,
            max_output_tokens=12288,
        )
        yield sse({"meta": {"phase": "research"}})

        async for event in response:
            if event.type == "response.output_text.delta":
                had_delta = True
                buf += event.delta
                if _should_flush(buf):
                    yield sse({"delta": buf})
                    buf = ""
            elif event.type in ("response.incomplete", "response.failed"):
                reason = getattr(getattr(event, "response", None), "incomplete_details", None)
                if buf.strip():
                    yield sse({"delta": buf})
                    buf = ""
                yield sse({"error": f"Response ended: {reason}"})
                break
            elif event.type == "error":
                if buf.strip():
                    yield sse({"delta": buf})
                    buf = ""
                yield sse({"error": event.message})
                break

        # Flush any remaining buffered text
        if buf.strip():
            yield sse({"delta": buf})
            buf = ""

        if not had_delta:
            # Fallback: non-streaming call when stream yields no deltas
            fb = await client.responses.create(
                model=ENGAGEMENT_MODEL,
                instructions=instructions,
                input=user_input,
                tools=[{"type": "web_search_preview", "search_context_size": "high"}],
                tool_choice="auto",
                max_output_tokens=12288,
            )
            text = getattr(fb, "output_text", "") or ""
            if text.strip():
                yield sse({"delta": text})
            else:
                yield sse({"error": "Model returned no text."})

    except Exception as e:
        if buf.strip():
            yield sse({"delta": buf})
        yield sse({"error": str(e)})

    yield "data: [DONE]\n\n"


async def engagement_generate(entity: str, mode: str, context: str) -> dict:
    """Non-streaming path: collects all deltas and returns complete text."""
    text_parts: list[str] = []
    error: str | None = None

    async for raw in engagement_sse_stream(entity, mode, context):
        import json
        line = raw.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "delta" in evt:
            text_parts.append(evt["delta"])
        elif "error" in evt:
            error = evt["error"]

    return {"text": "".join(text_parts), "error": error}
