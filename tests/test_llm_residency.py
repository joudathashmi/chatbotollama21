"""Data-residency gate: Postgres/doc rows must not hit cloud LLMs in strict."""

from __future__ import annotations

import pytest


def test_assert_blocks_cloud_data_backend_in_strict(monkeypatch):
    from app.services import llm_residency as lr

    monkeypatch.setattr(lr.config, "RESIDENCY_STRICT", True)
    monkeypatch.setattr(lr.config, "DATA_LLM_BACKEND", "azure")
    with pytest.raises(lr.ResidencyViolation):
        lr.assert_can_send_payload("data_grounded", path="unit_test")


def test_assert_allows_ollama_data_backend_in_strict(monkeypatch):
    from app.services import llm_residency as lr

    monkeypatch.setattr(lr.config, "RESIDENCY_STRICT", True)
    monkeypatch.setattr(lr.config, "DATA_LLM_BACKEND", "ollama")
    lr.assert_can_send_payload("data_grounded", path="unit_test")  # no raise


def test_question_only_never_blocked(monkeypatch):
    from app.services import llm_residency as lr

    monkeypatch.setattr(lr.config, "RESIDENCY_STRICT", True)
    monkeypatch.setattr(lr.config, "DATA_LLM_BACKEND", "azure")
    lr.assert_can_send_payload("question_only", path="unit_test")


def test_public_web_sealed_by_default_in_strict(monkeypatch):
    from app.services import llm_residency as lr

    monkeypatch.setattr(lr.config, "RESIDENCY_STRICT", True)
    monkeypatch.setattr(lr.config, "RESIDENCY_ALLOW_PUBLIC_WEB", False)
    assert lr.public_web_allowed() is False


def test_public_engagement_blocked_in_strict(monkeypatch):
    from app.services import llm_residency as lr

    monkeypatch.setattr(lr.config, "RESIDENCY_STRICT", True)
    monkeypatch.setattr(lr.config, "RESIDENCY_BLOCK_PUBLIC_ENGAGEMENT", True)
    assert lr.public_engagement_allowed() is False


def test_resolve_prefers_local_model_name(monkeypatch):
    from app.services import llm_residency as lr

    class _Fake:
        pass

    monkeypatch.setattr(lr.config, "RESIDENCY_STRICT", True)
    monkeypatch.setattr(lr.config, "DATA_LLM_BACKEND", "ollama")
    monkeypatch.setattr(lr.config, "DATA_LLM_MODEL", "llama3.1")
    monkeypatch.setattr(
        "app.database.get_data_llm_client", lambda: _Fake(),
    )
    client, model = lr.resolve_data_completion_client(
        preferred_model="gpt-4o",
    )
    assert isinstance(client, _Fake)
    assert model == "llama3.1"


def test_narrative_cloud_uses_azure_not_ollama(monkeypatch):
    from app.services import llm_residency as lr

    class _Azure:
        pass

    class _Ollama:
        pass

    monkeypatch.setattr(lr.config, "RESIDENCY_STRICT", True)
    monkeypatch.setattr(lr.config, "NARRATIVE_CLOUD_ENABLED", True)
    monkeypatch.setattr(lr.config, "DATA_LLM_BACKEND", "ollama")
    monkeypatch.setattr(lr.config, "DATA_LLM_MODEL", "llama3.1")
    monkeypatch.setattr(lr.config, "OPENAI_MODEL", "gpt-4o")
    monkeypatch.setattr(
        "app.database.get_openai_client", lambda: _Azure(),
    )
    monkeypatch.setattr(
        "app.database.get_data_llm_client", lambda: _Ollama(),
    )
    client, model = lr.resolve_narrative_completion_client(
        preferred_model="gpt-4o",
    )
    assert isinstance(client, _Azure)
    assert model == "gpt-4o"


def test_narrative_cloud_off_falls_back_to_ollama(monkeypatch):
    from app.services import llm_residency as lr

    class _Ollama:
        pass

    monkeypatch.setattr(lr.config, "RESIDENCY_STRICT", True)
    monkeypatch.setattr(lr.config, "NARRATIVE_CLOUD_ENABLED", False)
    monkeypatch.setattr(lr.config, "DATA_LLM_BACKEND", "ollama")
    monkeypatch.setattr(lr.config, "DATA_LLM_MODEL", "llama3.1")
    monkeypatch.setattr(
        "app.database.get_data_llm_client", lambda: _Ollama(),
    )
    client, model = lr.resolve_narrative_completion_client(
        preferred_model="gpt-4o",
    )
    assert isinstance(client, _Ollama)
    assert model == "llama3.1"
