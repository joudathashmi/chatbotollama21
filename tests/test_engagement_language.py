"""Tests for engagement dossier language detection (English vs Arabic)."""

from __future__ import annotations

from app.prompts.engagement_system import build_system_prompt, detect_response_language


class TestDetectResponseLanguage:
    def test_english_entity_and_context(self):
        assert detect_response_language("Apple Inc", "technology sector") == "en"

    def test_empty_input_defaults_english(self):
        assert detect_response_language("", "") == "en"
        assert detect_response_language(None, None) == "en"

    def test_arabic_entity(self):
        assert detect_response_language("شركة أرامكو السعودية", "") == "ar"

    def test_arabic_context_only(self):
        assert detect_response_language("Aramco", "هل لديهم تواجد في السعودية؟") == "ar"

    def test_mixed_input_with_any_arabic_is_arabic(self):
        assert detect_response_language("Aramco شركة", "context") == "ar"


class TestBuildSystemPrompt:
    def test_english_prompt_has_no_arabic_directive_language(self):
        prompt = build_system_prompt("en")
        assert "Generate all content in English only." in prompt
        assert "Generate the entire dossier in Modern Standard Arabic only." not in prompt

    def test_arabic_prompt_has_arabic_directive(self):
        prompt = build_system_prompt("ar")
        assert "Generate the entire dossier in Modern Standard Arabic only." in prompt
        assert "Generate all content in English only." not in prompt

    def test_default_is_english(self):
        assert build_system_prompt() == build_system_prompt("en")

    def test_unknown_language_falls_back_to_english(self):
        assert build_system_prompt("fr") == build_system_prompt("en")

    def test_structure_sections_present_regardless_of_language(self):
        for lang in ("en", "ar"):
            prompt = build_system_prompt(lang)
            assert "FULL DOSSIER — SECTIONS" in prompt
            assert "19. Intelligence Gaps" in prompt
            assert "VISUAL STRUCTURE (MANDATORY)" in prompt
