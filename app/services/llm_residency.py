"""Saudi / enterprise data-residency controls for LLM calls.

Policy (strict mode):
  - Question-only prompts  → Azure / OpenAI allowed (no Postgres rows).
  - Data-grounded prompts  → local Ollama by default (raw / unfiltered).
  - Narrative compose       → when MISA_NARRATIVE_CLOUD=true, Azure/OpenAI
    may receive *privacy-filtered fact cards* only (Jul21 quality path).
    Templates remain the offline fallback.
"""

from __future__ import annotations

from typing import Any, Literal

from app import config
from app.logger import logger

PayloadClass = Literal["question_only", "data_grounded"]


class ResidencyViolation(RuntimeError):
    """Raised when a call would send DB/doc data to a non-local LLM."""


def residency_strict() -> bool:
    return bool(getattr(config, "RESIDENCY_STRICT", False))


def narrative_cloud_enabled() -> bool:
    """Jul21-quality narrative over privacy-filtered fact cards via Azure."""
    return bool(getattr(config, "NARRATIVE_CLOUD_ENABLED", True))


def data_backend() -> str:
    return (getattr(config, "DATA_LLM_BACKEND", "ollama") or "ollama").strip().lower()


def data_model_name(fallback: str | None = None) -> str:
    if data_backend() == "ollama":
        return (getattr(config, "DATA_LLM_MODEL", None) or "llama3.1").strip()
    return (fallback or getattr(config, "OPENAI_MODEL", None) or "gpt-4o-mini").strip()


def is_local_data_backend() -> bool:
    return data_backend() == "ollama"


def assert_can_send_payload(payload_class: PayloadClass, *, path: str) -> None:
    if payload_class != "data_grounded":
        return
    if not residency_strict():
        return
    if is_local_data_backend():
        logger.info(
            "residency: data_grounded path=%r → local backend=%r model=%r",
            path, data_backend(), data_model_name(),
        )
        return
    raise ResidencyViolation(
        f"Refusing data-grounded LLM call on path={path!r}: "
        f"MISA_RESIDENCY_MODE=strict requires MISA_DATA_LLM_BACKEND=ollama "
        f"(got {data_backend()!r}). Postgres/document rows must not leave "
        f"the machine. For Jul21 narrative quality over filtered fact cards, "
        f"use resolve_narrative_completion_client (MISA_NARRATIVE_CLOUD=true)."
    )


def public_web_allowed() -> bool:
    if not residency_strict():
        return True
    return bool(getattr(config, "RESIDENCY_ALLOW_PUBLIC_WEB", False))


def public_engagement_allowed() -> bool:
    if not residency_strict():
        return True
    return not bool(getattr(config, "RESIDENCY_BLOCK_PUBLIC_ENGAGEMENT", True))


def audit_llm_call(
    *,
    path: str,
    payload_class: PayloadClass,
    backend: str,
    model: str,
) -> None:
    logger.info(
        "llm_audit path=%s payload=%s backend=%s model=%s residency=%s",
        path,
        payload_class,
        backend,
        model,
        getattr(config, "RESIDENCY_MODE", "standard"),
    )


def resolve_data_completion_client(
    fallback_client: Any = None,
    *,
    preferred_model: str | None = None,
):
    """Return (client, model) for hard data-grounded completions.

    Under local Ollama, always use the data LLM client (never cloud).
    Prefer ``resolve_narrative_completion_client`` for company/person
    briefings when Jul21 narrative quality is required.
    """
    from app.database import get_data_llm_client

    assert_can_send_payload("data_grounded", path="resolve_data_completion_client")
    local = is_local_data_backend()
    data_client = get_data_llm_client()

    if local:
        client = data_client
        if client is None:
            raise ResidencyViolation(
                "No data LLM client available. Start Ollama "
                f"({getattr(config, 'DATA_LLM_BASE_URL', '')}) and pull "
                f"{data_model_name()!r}."
            )
        model = data_model_name()
        audit_llm_call(
            path="resolve_data_completion_client",
            payload_class="data_grounded",
            backend=data_backend(),
            model=model,
        )
        return client, model

    client = fallback_client if fallback_client is not None else data_client
    if client is None:
        raise ResidencyViolation(
            "No data LLM client available. Start Ollama "
            f"({getattr(config, 'DATA_LLM_BASE_URL', '')}) and pull "
            f"{data_model_name()!r}."
        )
    model = preferred_model or data_model_name()
    audit_llm_call(
        path="resolve_data_completion_client",
        payload_class="data_grounded",
        backend=data_backend(),
        model=model,
    )
    return client, model


def resolve_narrative_completion_client(
    fallback_client: Any = None,
    *,
    preferred_model: str | None = None,
):
    """Client for Jul21-style narrative over privacy-filtered fact cards.

    When ``MISA_NARRATIVE_CLOUD`` is on (default), use Azure/OpenAI so
    answer quality matches the pre-Ollama compose path. Ollama remains
    available as a fallback if the cloud client is missing, or when
    narrative cloud is explicitly disabled.
    """
    if narrative_cloud_enabled():
        from app.database import get_openai_client

        client = fallback_client if fallback_client is not None else get_openai_client()
        if client is not None:
            model = (
                preferred_model
                or getattr(config, "OPENAI_MODEL", None)
                or "gpt-4o-mini"
            )
            audit_llm_call(
                path="resolve_narrative_completion_client",
                payload_class="data_grounded",
                backend="azure",
                model=str(model),
            )
            logger.info(
                "residency: narrative cloud path → Azure/OpenAI model=%r "
                "(privacy-filtered fact cards only)",
                model,
            )
            return client, str(model)
        logger.warning(
            "narrative cloud enabled but no Azure/OpenAI client; "
            "falling back to local data LLM"
        )

    return resolve_data_completion_client(
        fallback_client, preferred_model=preferred_model,
    )
