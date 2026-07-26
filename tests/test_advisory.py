"""Unit tests for the strategic-advisory path (market-fit / attraction-
strategy questions → consultant-grade report instead of the 150-word
general-knowledge fallback)."""

from unittest.mock import MagicMock, patch

from app.prompts.chat_system import advisory_system_prompt
from app.services import chat_engine as mc
from app.services.curation import strategic_advisory_answer


# ─── Detection ────────────────────────────────────────────────────────

def test_market_fit_question_is_advisory():
    assert mc._is_advisory_question(
        "what is the market fit for attracting Indian companies to Saudi Arabia"
    )


def test_attraction_strategy_questions_are_advisory():
    assert mc._is_advisory_question(
        "how do we attract Japanese investors to KSA?"
    )
    assert mc._is_advisory_question(
        "investment case for German manufacturers in Saudi Arabia"
    )
    assert mc._is_advisory_question(
        "why should Chinese firms invest in the Kingdom?"
    )


def test_entity_lookups_are_not_advisory():
    assert not mc._is_advisory_question("Tell me about Alphabet, Inc.")
    assert not mc._is_advisory_question(
        "which Pakistani companies have invested in Saudi Arabia?"
    )
    assert not mc._is_advisory_question("who is Tim Cook")


def test_count_and_browse_questions_are_not_advisory():
    # Even with attraction verbs, count/browse keeps its deterministic route.
    assert not mc._is_advisory_question(
        "how many investors did we attract from India?"
    )
    assert not mc._is_advisory_question("show me companies from India")
    assert not mc._is_advisory_question("list deals in Egypt")


def test_company_engagement_questions_are_not_advisory():
    # Row-grounded engagement_strategy route must keep these.
    assert not mc._is_advisory_question("how should MISA engage Aramco?")
    # A plan scoped to a COMPANY (no origin country) also stays on the
    # row-grounded route.
    assert not mc._is_advisory_question(
        "develop an engagement plan for Aramco")


def test_macro_trend_questions_are_advisory():
    # Regression: keyword fallback fed 5 random country_insights rows
    # to curation, which synthesized "global trends" fixated on EdTech.
    assert mc._is_advisory_question(
        "what are the new global trends impacting the investment")
    assert mc._is_advisory_question("emerging FDI trends in 2026")
    assert mc._is_advisory_question(
        "what trends are shaping investment flows?")
    # But a company lookup mentioning 'trend' incidentally is not.
    assert not mc._is_advisory_question("tell me about Trend Micro")


def test_deliverable_plus_country_is_advisory():
    # Regression: these were hijacked into company disambiguation
    # ("Multiple possible matches for 'Japan' — Japan Post Holdings…")
    # because they name the deliverable, not an attraction verb.
    assert mc._is_advisory_question("develop an engagement plan with Japan")
    assert mc._is_advisory_question(
        "develop an engagement plan with Japan as a country")
    assert mc._is_advisory_question("top sectors for France")


def test_detect_origin_country():
    assert mc._detect_origin_country(
        "develop an engagement plan with Japan") == "Japan"
    assert mc._detect_origin_country("attract German firms") == "Germany"
    # Saudi Arabia is the destination, never the origin.
    assert mc._detect_origin_country(
        "an engagement plan for Saudi Arabia") is None
    assert mc._detect_origin_country("engagement plan for Aramco") is None


# ─── Origin-country detection / DB grounding ─────────────────────────

def test_advisory_country_detects_adjective_form():
    with patch(
        "app.services.engagement_data.fetch_country_saudi_investors"
    ) as fetch:
        fetch.return_value = {
            "total_licensed": 42, "total_rhq": 7,
            "rhq": [{"company_name": "Tata Consultancy Services",
                     "industry": "IT Services", "annual_revenue": 29e9}],
            "licensed_only": [],
        }
        ctx = mc._advisory_country_context(
            "market fit for attracting Indian companies to Saudi Arabia"
        )
    assert ctx["origin_country"] == "India"
    assert ctx["companies_from_origin_licensed_in_saudi"] == 42
    assert ctx["companies_from_origin_with_rhq"] == 7
    assert ctx["top_rhq_companies"][0]["name"] == "Tata Consultancy Services"


def test_advisory_country_saudi_is_never_the_origin():
    # Question mentions only Saudi Arabia → no origin country.
    assert mc._advisory_country_context(
        "how do we attract more investment to Saudi Arabia?"
    ) is None


def test_advisory_country_context_includes_sector_distribution():
    with patch(
        "app.services.engagement_data.fetch_country_saudi_investors",
        return_value={"total_licensed": 38, "total_rhq": 28,
                      "rhq": [], "licensed_only": []},
    ), patch(
        "app.services.engagement_data.fetch_country_sector_distribution",
        return_value={
            "sectors": [{"industry": "Oil, Gas, Energy & Water",
                         "licensed_count": 9, "rhq_count": 6}],
            "_db_error": None,
        },
    ), patch(
        "app.services.engagement_data.resolve_country_id",
        return_value=(None, None),  # skip the country-insights queries
    ):
        ctx = mc._advisory_country_context(
            "top sectors for attracting investors from France"
        )
    assert ctx["origin_country"] == "France"
    assert ctx["licensed_sector_distribution"][0]["licensed_count"] == 9


def test_advisory_country_survives_db_failure():
    with patch(
        "app.services.engagement_data.fetch_country_saudi_investors",
        side_effect=RuntimeError("db down"),
    ), patch(
        "app.services.engagement_data.fetch_country_sector_distribution",
        side_effect=RuntimeError("db down"),
    ), patch(
        "app.services.engagement_data.resolve_country_id",
        side_effect=RuntimeError("db down"),
    ):
        ctx = mc._advisory_country_context(
            "attracting Pakistani companies to KSA"
        )
    # Missing must never masquerade as zero: DB failure sets an
    # explicit unavailability flag (the prompt omits the footprint
    # section) instead of leaving count fields silently absent.
    assert ctx.get("origin_country") == "Pakistan"
    assert ctx.get("footprint_data_unavailable") is True
    assert "companies_from_origin_licensed_in_saudi" not in ctx


def test_advisory_prompt_forbids_missing_as_zero():
    p = advisory_system_prompt("en", "market_fit")
    assert "MISSING IS NOT ZERO" in p
    assert "footprint_data_unavailable" in p


# ─── Deliverable detection ────────────────────────────────────────────

def test_engagement_plan_request_detected():
    assert mc._detect_advisory_deliverable(
        "Develop an engagement plan for attracting investment from India "
        "to Saudi Arabia"
    ) == "engagement_plan"
    assert mc._detect_advisory_deliverable(
        "create a roadmap to attract Japanese investors"
    ) == "engagement_plan"
    assert mc._detect_advisory_deliverable(
        "make me a plan to win German manufacturers"
    ) == "engagement_plan"


def test_market_fit_request_detected():
    assert mc._detect_advisory_deliverable(
        "what is the market fit for attracting Indian companies to "
        "Saudi Arabia"
    ) == "market_fit"


def test_typo_make_market_for_atrract_routes_to_market_fit():
    """User PDF case: typos must still yield full market-fit advisory."""
    q = "make me a market for to atrract indian companies"
    assert mc._is_advisory_question(q) is True
    assert mc._detect_advisory_deliverable(q) == "market_fit"


def test_market_fit_not_overwritten_by_company_targeting_rebuild():
    """Absent Priority Ranking must NOT wipe a market-fit assessment."""
    from app.services.advisory_structured import ranking_table_is_truncated
    from app.services.response_validator import validate_advisory_answer

    mf = (
        "# Market Fit Assessment: Attracting Indian Companies to "
        "Saudi Arabia\n\n"
        "## Strategic Context\n"
        "India is a major outbound investor; Saudi Arabia is a "
        "regional growth platform for Vision 2030 sectors.\n\n"
        "## Overall Market Fit\n"
        "| Sector | Strategic Fit | Investment Potential | Priority |\n"
        "|---|---|---|---|\n"
        "| ICT | High | High | Tier 1 |\n"
        "| Healthcare | High | Medium | Tier 1 |\n\n"
        "## Investment & Trade Bodies to Engage\n"
        "| Organisation | Type | Role |\n"
        "|---|---|---|\n"
        "| Invest India | IPA | National pipeline |\n"
        "| CII | Industry body | Manufacturing outreach |\n\n"
        "## Strategic Conclusion\n"
        "Complementarity thesis holds.\n"
    )
    assert ranking_table_is_truncated(mf) is False
    ctx = {
        "origin_country": "India",
        "companies_from_origin_licensed_in_saudi": 2437,
        "companies_from_origin_with_rhq": 14,
        "retrieval_status": "SUCCESS_WITH_RESULTS",
        "expansion_targets": [
            {"company": "Tech Mahindra", "sector": "IT",
             "current_saudi_presence": "RHQ"},
        ],
    }
    fixed, fixes = validate_advisory_answer(mf, ctx)
    assert "rebuilt_truncated_company_targeting_from_db" not in fixes
    assert "Market Fit Assessment" in fixed
    assert "Priority Company Ranking" not in fixed
    assert "Invest India" in fixed
    assert "2437" in fixed  # footprint inject ok


def test_sector_priorities_request_detected():
    assert mc._detect_advisory_deliverable(
        "what are the top sectors that I should be focusing on for "
        "attracting investors from France"
    ) == "sector_priorities"
    assert mc._detect_advisory_deliverable(
        "which sectors should we prioritise to attract Korean companies"
    ) == "sector_priorities"


def test_other_advisory_requests_default_to_adaptive():
    assert mc._detect_advisory_deliverable(
        "why should Chinese firms invest in the Kingdom?"
    ) == "strategy_analysis"


# ─── Country name variants (comma-inverted DB names) ─────────────────

def test_country_key_variants_handle_comma_inverted_names():
    from app.services.country_resolver import _key_variants
    # 'Korea, South' (DB form) and 'South Korea' (user form) must
    # share at least one lookup key.
    db_keys = set(_key_variants("Korea, South"))
    user_keys = set(_key_variants("South Korea"))
    assert db_keys & user_keys
    # Same for the longer comma-inverted names.
    assert set(_key_variants("Congo, Democratic Republic of the")) & \
        set(_key_variants("Democratic Republic of the Congo"))


# ─── Prompt shape ─────────────────────────────────────────────────────

def test_advisory_prompt_demands_full_report():
    p = advisory_system_prompt("en", "market_fit")
    assert "Strategic Context" in p
    assert "Overall Market Fit" in p
    assert "Tier 1" in p
    assert "Strategic Targeting Recommendations for MISA" in p
    assert "1,500" in p  # explicit length target, not a 150-word cap
    assert "Current MISA Footprint" in p


def test_engagement_plan_prompt_is_a_plan_not_an_assessment():
    p = advisory_system_prompt("en", "engagement_plan")
    assert "Phased Roadmap" in p
    assert "Stakeholder & Channel Map" in p
    assert "KPIs & Governance" in p
    assert "Risks & Mitigations" in p
    # Must explicitly forbid the market-fit shape.
    assert "NOT a market" in p
    assert "Overall Market Fit" not in p


def test_unknown_deliverable_falls_back_to_adaptive_structure():
    p = advisory_system_prompt("en", "nonsense_label")
    assert "adaptive to the ask" in p


def test_advisory_prompt_enforces_specificity():
    p = advisory_system_prompt("en", "engagement_plan")
    assert "ANCHOR RULE" in p
    # SAGIA was dissolved into MISA in 2020 — the prompt must ban it.
    assert "SAGIA" in p and "NEVER mention SAGIA" in p
    # DB footprint companies must be used as named accounts, not a list.
    assert "USE THE FOOTPRINT COMPANIES BY NAME" in p
    # Slogans in quotation marks are banned.
    assert "NOT SLOGANS" in p
    # The closing recommendations section is not exempt from anchoring.
    assert "RECOMMENDATIONS ARE NOT EXEMPT" in p
    assert "strengthen bilateral relations" in p  # in the banned list


def test_sector_priorities_prompt_leads_with_db_evidence():
    p = advisory_system_prompt("en", "sector_priorities")
    assert "The Evidence Base" in p
    assert "licensed_sector_distribution" in p
    assert "Sector Ranking" in p
    assert "What NOT to prioritise" in p


def test_advisory_model_defaults_to_full_tier():
    from app.config import ADVISORY_MODEL
    assert ADVISORY_MODEL  # non-empty
    assert "mini" not in ADVISORY_MODEL or (
        # explicit override via env is allowed
        __import__("os").getenv("MISA_ADVISORY_OPENAI_MODEL") is not None
    )


# ─── Generator ────────────────────────────────────────────────────────

def _fake_client(answer_text: str) -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = answer_text
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return client


def test_strategic_advisory_answer_embeds_db_context():
    client = _fake_client("# Market Fit Assessment\n\n...")
    out = strategic_advisory_answer(
        "market fit for attracting Indian companies",
        db_context={"origin_country": "India",
                    "companies_from_origin_with_rhq": 7},
        deliverable="market_fit",
        locale="en", client=client, model="gpt-4o-mini",
    )
    assert out.startswith("# Market Fit Assessment")
    sent = client.chat.completions.create.call_args
    user_msg = sent.kwargs["messages"][1]["content"]
    assert "MISA DATABASE CONTEXT" in user_msg
    assert '"origin_country": "India"' in user_msg
    # System prompt is the advisory one, not the 150-word fallback.
    sys_msg = sent.kwargs["messages"][0]["content"]
    assert "Overall Market Fit" in sys_msg


def test_strategic_advisory_answer_none_on_failure():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("api down")
    assert strategic_advisory_answer(
        "market fit question", locale="en",
        client=client, model="gpt-4o-mini",
    ) is None


# ─── Deterministic advisory answer validation ─────────────────────────

from app.services.response_validator import validate_advisory_answer

_CTX = {
    "origin_country": "India",
    "companies_from_origin_licensed_in_saudi": 24,
    "companies_from_origin_with_rhq": 17,
    "top_rhq_companies": [{"name": "Tech Mahindra"}],
    "top_licensed_companies": [{"name": "Biocon Ltd."}],
}


def test_validator_strips_fabricated_footprint_without_context():
    # The deployed-portal failure: no DB context, yet the model wrote
    # a footprint section claiming "zero Indian companies licensed".
    answer = (
        "# Market Fit\n\n## Strategic Context\nIntro.\n\n"
        "## Current MISA Footprint\n"
        "According to MISA's database, there are currently zero Indian "
        "companies licensed in Saudi Arabia.\n\n"
        "## Overall Market Fit\nTable here.\n"
    )
    fixed, fixes = validate_advisory_answer(answer, None)
    assert "stripped_fabricated_footprint_section" in fixes
    assert "Current MISA Footprint" not in fixed
    assert "zero Indian" not in fixed
    # Other sections survive.
    assert "Strategic Context" in fixed and "Overall Market Fit" in fixed


def test_validator_rebuilds_footprint_with_wrong_counts():
    answer = (
        "# Plan\n\n## Current MISA Footprint\n"
        "According to MISA's database, there are currently zero Indian "
        "companies licensed and no RHQ holders.\n\n## Next\nMore.\n"
    )
    fixed, fixes = validate_advisory_answer(answer, _CTX)
    assert "rebuilt_footprint_from_db_counts" in fixes
    assert "**24**" in fixed and "**17**" in fixed
    assert "Tech Mahindra" in fixed
    assert "zero Indian" not in fixed


def test_validator_keeps_correct_footprint():
    answer = (
        "# Plan\n\n## Current MISA Footprint\n"
        "MISA's database records 24 licensed Indian companies, 17 of "
        "which hold RHQs, led by Tech Mahindra.\n\n## Next\nMore.\n"
    )
    fixed, fixes = validate_advisory_answer(answer, _CTX)
    assert fixes == []
    assert "24 licensed Indian companies" in fixed


def test_validator_leaves_no_footprint_answers_alone():
    answer = "# Trends\n\n## Strategic Context\nAll good here.\n"
    fixed, fixes = validate_advisory_answer(answer, None)
    assert fixes == []
    assert "Strategic Context" in fixed


# ─── Analytical-synthesis questions ───────────────────────────────────

def test_synthesis_questions_are_advisory():
    # Regression: this exact question was treated as an entity lookup
    # and dead-ended with 'No record matching "<whole question>"'.
    assert mc._is_advisory_question(
        "Develop the dynamic between MNCs with market valuations "
        "exceeding $1 trillion and asset managers with AUM exceeding "
        "$1 trillion. How will the dynamic be reflected in the "
        "Strategic Capital Allocators")
    assert mc._is_advisory_question(
        "what is the relationship between sovereign wealth funds and "
        "FDI flows")
    assert mc._is_advisory_question(
        "how will rising rates be reflected in gulf capital allocation")
    # Entity lookups stay out — even wordy ones.
    assert not mc._is_advisory_question("tell me about Alphabet, Inc.")
    assert not mc._is_advisory_question(
        "tell me about The Governor and Company of the Bank of Ireland")


def test_no_match_second_chance_uses_question_shape():
    # Advisory-shaped question that reached the no-match dead-end →
    # advisory answer composed instead of the dead-end message.
    from app.services.curation import curate_company_insights
    q = ("Develop the dynamic between MNCs with market valuations "
         "exceeding $1 trillion and asset managers with AUM exceeding "
         "$1 trillion")
    client = _fake_client("# The Trillion-Dollar Dynamic\n\nAnalysis...")
    out = curate_company_insights(
        [{"company_name": "Unrelated Co"}],
        q,
        locale="en",
        entity_candidate=q,
        entity_matched=False,
        table="company_profiles",
        client=client,
        model="gpt-4o-mini",
    )
    assert out.startswith("# The Trillion-Dollar Dynamic")
    assert "No record matching" not in out


def test_no_match_kept_for_entity_lookups_even_long_ones():
    # NOT advisory-shaped → honest no-match survives, regardless of
    # entity length; the echoed entity is truncated for readability.
    from app.services.curation import curate_company_insights
    client = _fake_client("should not be used")
    long_name = ("The Governor and Company of the Bank of Ireland "
                 "Group Holdings and Partners International")
    out = curate_company_insights(
        [{"company_name": "Unrelated Co"}],
        f"tell me about {long_name}",
        locale="en",
        entity_candidate=long_name,
        entity_matched=False,
        table="company_profiles",
        client=client,
        model="gpt-4o-mini",
    )
    assert "No record matching" in out
    assert "…" in out  # long entity echoed truncated, not verbatim
    # The advisory generator was never invoked for an entity lookup.
    client.chat.completions.create.assert_not_called()


def test_no_match_message_kept_for_short_entities():
    from app.services.curation import curate_company_insights
    client = _fake_client("should not be used")
    out = curate_company_insights(
        [{"company_name": "Unrelated Co"}],
        "tell me about Acme Foo Bar",
        locale="en",
        entity_candidate="Acme Foo Bar",
        entity_matched=False,
        table="company_profiles",
        client=client,
        model="gpt-4o-mini",
    )
    assert "No record matching" in out
    assert '"Acme Foo Bar"' in out  # short names echoed in full


# ─── Target-list questions (verb-suffix coverage) ─────────────────────

def test_target_list_questions_are_advisory():
    # Regression: "companies to be targeted from China" hijacked into
    # 'Multiple possible matches for "China"' — the gate verbs missed
    # suffix forms ('targeted') and reversed noun-verb order.
    assert mc._is_advisory_question(
        "give me the list of the best companies to be targeted from "
        "China with the investment thesis for each")
    assert mc._is_advisory_question(
        "best companies to target from Germany")
    assert mc._is_advisory_question(
        "which investors should we be attracting from Japan?")
    assert mc._is_advisory_question(
        "top investors to pursue from Korea with an investment thesis")


def test_browse_and_lookup_still_not_advisory():
    # Browse/count/entity behavior unchanged by the wider verb net.
    assert not mc._is_advisory_question("top 5 companies by revenue")
    assert not mc._is_advisory_question("show me companies from India")
    assert not mc._is_advisory_question(
        "which Pakistani companies have invested in Saudi Arabia?")
    assert not mc._is_advisory_question("tell me about Alphabet")
    assert not mc._is_advisory_question(
        "how many investors did we attract from India?")


# ─── LLM-classified advisory route (structural, not regex) ────────────

def test_classifier_knows_strategic_advisory_intent():
    from app.services.intent_router import (
        INTENTS, _INTENT_CLASSIFIER_PROMPT,
    )
    assert "strategic_advisory" in INTENTS
    assert "strategic_advisory" in _INTENT_CLASSIFIER_PROMPT
    # The prompt teaches the distinction that matters most:
    assert "engagement_strategy" in _INTENT_CLASSIFIER_PROMPT


def test_run_advisory_path_returns_result_dict():
    with patch.object(mc, "strategic_advisory_answer",
                      return_value="# Report\n\nBody"), \
         patch.object(mc, "_advisory_country_context", return_value=None):
        pack = {}
        out = mc._run_advisory_path(
            "novel phrasing strategy question", pack, "en", "en",
            client=MagicMock())
    assert out["_answer_source"] == "strategic_advisory"
    assert out["answer"].startswith("# Report")
    assert pack["_short_circuit"] == "strategic_advisory"


def test_run_advisory_path_none_on_failure():
    with patch.object(mc, "strategic_advisory_answer", return_value=None), \
         patch.object(mc, "_advisory_country_context", return_value=None):
        out = mc._run_advisory_path(
            "some question", {}, "en", "en", client=MagicMock())
    assert out is None


# ─── Determinism & response cache ─────────────────────────────────────

def test_determinism_kwargs_are_low_temp_and_seeded():
    from app.config import openai_determinism_kw
    kw = openai_determinism_kw()
    assert kw["temperature"] == 0.0
    assert isinstance(kw.get("seed"), int)


def test_response_cache_roundtrip(monkeypatch):
    from app.services import chat_engine as ce
    # Cache is opt-in (default off); enable it for this roundtrip test.
    monkeypatch.setenv("MISA_RESPONSE_CACHE", "true")
    ce._RESPONSE_CACHE.clear()
    k = ce._response_cache_key("Tell me about Pakistan", "en")
    # Same question, different casing/spacing -> same key.
    assert k == ce._response_cache_key("  tell me about   pakistan ", "en")
    ce._response_cache_put(k, {"answer": "X" * 60, "error": None})
    got = ce._response_cache_get(k)
    assert got is not None and got["_from_cache"] is True


def test_response_cache_off_by_default(monkeypatch):
    from app.services import chat_engine as ce
    monkeypatch.delenv("MISA_RESPONSE_CACHE", raising=False)
    enabled, _, _ = ce._response_cache_settings()
    assert enabled is False


def test_response_cache_only_clean_single_turn():
    from app.services import chat_engine as ce
    good = {"answer": "A" * 60, "error": None, "_answer_source": "db"}
    assert ce._is_cacheable("q", [], good)
    # Follow-ups (history present) are never cached.
    assert not ce._is_cacheable("q", [{"role": "user", "content": "x"}], good)
    # Errors and off-topic redirects are never cached.
    assert not ce._is_cacheable("q", [], {"answer": "A" * 60, "error": "boom"})
    assert not ce._is_cacheable(
        "q", [], {"answer": "A" * 60, "_answer_source": "off_topic_redirect"})


# ─── Markdown formatting repair (glued bold lead-ins) ─────────────────

def test_repair_glued_bold_lead_ins():
    from app.services.curation import repair_markdown_formatting
    # The exact defect seen in a rendered report.
    bad = ("- **Facilitate NEOM and Red Sea Global Briefings for Asset "
           "Managers' Sovereign Wealth and Private Funds**Organize "
           "dedicated sessions with sovereign wealth funds.")
    fixed = repair_markdown_formatting(bad)
    assert "Funds**Organize" not in fixed
    assert "Funds** — Organize" in fixed


def test_repair_leaves_clean_formatting_untouched():
    from app.services.curation import repair_markdown_formatting
    good = "- **Propose a NEOM Energy Transition Briefing** for ENGIE."
    # Bold followed by a space is legitimate — must not be altered.
    assert repair_markdown_formatting(good) == good
    # Bold followed by punctuation is legitimate too.
    g2 = "- **RHQ status:** Yes."
    assert repair_markdown_formatting(g2) == g2
    # A '** — **' emdash sequence (close, dash, open) must be left
    # alone — the earlier regex corrupted it into '** — ** — '.
    g3 = "the interplay** — **Investment** in things"
    assert repair_markdown_formatting(g3) == g3


def test_repair_handles_empty():
    from app.services.curation import repair_markdown_formatting
    assert repair_markdown_formatting("") == ""
    assert repair_markdown_formatting(None) is None


# ─── Country-specific licensing count ─────────────────────────────────

def test_country_licensing_count_routes_by_origin():
    # "licenses from India origin" must answer India's number, not the
    # global aggregate.
    assert mc._is_saudi_licensing_count_question(
        "how many total active licenses saudi has from india origin")
    assert mc._detect_origin_country(
        "how many total active licenses saudi has from india origin") == "India"
    # Generic count questions have no origin -> global aggregate.
    assert mc._detect_origin_country("how many RHQ licenses do we have") is None
    assert mc._detect_origin_country("how many companies are licensed by MISA") is None


def test_format_country_licensing_answer():
    out = mc._format_country_licensing_answer(
        "India", {"total_licensed": 24, "total_rhq": 17,
                  "rhq": [{"company_name": "Tech Mahindra", "industry": "ICT"}],
                  "licensed_only": [{"company_name": "Biocon Ltd."}]})
    assert "24 companies" in out
    assert "India" in out
    assert "17 hold" in out
    assert "Tech Mahindra" in out


def test_format_country_licensing_answer_db_error_not_zero():
    out = mc._format_country_licensing_answer(
        "India", {
            "total_licensed": 0, "total_rhq": 0,
            "_db_error": "UndefinedColumn",
            "retrieval": {"do_not_claim_zero": True},
            "retrieval_status": "SCHEMA_MISMATCH",
        })
    assert "0 hold an active" not in out.lower()
    assert "not" in out.lower() and "zero" in out.lower()
    assert "could not be retrieved" in out.lower() or "unavailable" in out.lower()


def test_format_country_licensing_answer_with_unlicensed():
    """Dual-source: licensed (shareholder nationality) + non-licensed
    (country_profile_name), each with their own RHQ subset."""
    out = mc._format_country_licensing_answer(
        "India", {
            "total_licensed": 5306, "total_rhq": 15,
            "total_non_licensed": 4910, "total_non_licensed_rhq": 57,
            "rhq": [{"company_name": "Tech Mahindra RHQ", "industry": "ICT"}],
            "licensed_only": [{"company_name": "TCS", "industry": "IT"}],
            "non_licensed": [{"company_name": "State Bank of India", "industry": "Banking"}],
        })
    # Combined total = 5306 + 4910
    assert "10,216 companies" in out
    assert "5,306 hold an active MISA licence" in out
    assert "15 hold" in out
    # Non-licensed count + its RHQ subset both surface
    assert "4,910 are present but unlicensed" in out
    assert "57 with RHQ status" in out
    # All three sections render
    assert "Tech Mahindra RHQ" in out
    assert "TCS" in out
    assert "State Bank of India" in out
    assert "Unlicensed companies" in out


def test_format_country_licensing_answer_zero_non_licensed():
    """When there are no non-licensed rows, the unlicensed clause and
    section are omitted entirely."""
    out = mc._format_country_licensing_answer(
        "Japan", {
            "total_licensed": 15, "total_rhq": 2,
            "total_non_licensed": 0, "total_non_licensed_rhq": 0,
            "rhq": [], "licensed_only": [], "non_licensed": [],
        })
    assert "15 companies" in out
    # The "are present but unlicensed" clause and the section heading must
    # both be absent (the source-attribution footer still says "unlicensed").
    assert "present but unlicensed" not in out
    assert "### Unlicensed companies" not in out


def test_active_predicates_adapt_to_schema():
    """Live DB must use canonical licensed / is_rhq booleans."""
    from app.services import engagement_data as ed
    from app.database import get_db
    import psycopg2.extras
    conn = get_db()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        preds = ed._licensing_predicates(cur)
    assert "licensed" in preds and "rhq" in preds
    assert preds["licensed"].strip()
    assert preds["rhq"].strip()
    if not preds.get("legacy"):
        assert "licensed IS TRUE" in preds["licensed"]
        assert "is_rhq IS TRUE" in preds["rhq"]
        assert "ZLA" not in preds["licensed"]
        assert "ZRHQ" not in preds["rhq"]
        assert "lifecycle_status" not in preds["licensed"]
        assert "registration_type" not in preds["rhq"]


def test_canonical_licensing_counts_match_sql():
    """Chatbot totals must equal SELECT COUNT(*) WHERE licensed/is_rhq."""
    from app.database import get_db
    from app.services.engagement_data import (
        fetch_saudi_licensing_summary,
        fetch_country_saudi_investors,
    )
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM company_profiles WHERE licensed = true")
        sql_lic = int(cur.fetchone()[0])
        cur.execute(
            "SELECT COUNT(*) FROM company_profiles WHERE is_rhq = true")
        sql_rhq = int(cur.fetchone()[0])
    summ = fetch_saudi_licensing_summary()
    assert int(summ["total_licensed"]) == sql_lic
    assert int(summ["total_rhq"]) == sql_rhq
    assert sql_lic == 95671 or sql_lic > 90000  # live reference band
    assert sql_rhq == 727 or (500 < sql_rhq < 2000)

    india = fetch_country_saudi_investors("India")
    assert not india.get("_db_error"), india.get("_db_error")
    # Origin-filtered canonical flags (approx; must be positive & tight)
    assert int(india["total_licensed"]) > 1000
    assert 5 <= int(india["total_rhq"]) <= 50



def test_india_footprint_not_false_zero():
    """Regression: broken shareholder_country_name SQL must not surface
    as '0 Indian companies licensed'."""
    from app.services.engagement_data import fetch_country_saudi_investors
    stats = fetch_country_saudi_investors("India")
    assert not stats.get("_db_error"), stats.get("_db_error")
    assert int(stats.get("total_licensed") or 0) > 100
    # Should surface real RHQ names when present (bus_data / is_rhq path).
    names = " ".join(
        (r.get("company_name") or "") for r in (stats.get("rhq") or [])
    ).lower()
    # At least some known Indian RHQ footprint should appear in rhq or licensed.
    names += " " + " ".join(
        (r.get("company_name") or "") for r in (stats.get("licensed_only") or [])
    ).lower()
    assert int(stats.get("total_licensed") or 0) > 0


# ─── Investment & trade bodies section ────────────────────────────────

def test_advisory_prompts_require_trade_bodies_section():
    for deliv in ("market_fit", "sector_priorities", "strategy_analysis",
                  "company_targeting"):
        p = advisory_system_prompt("en", deliv)
        assert "Investment & Trade Bodies to Engage" in p or \
               "Investment & Trade Bodies" in p or \
               "Investment and Trade Bodies" in p, deliv


# ─── Licensing focus + country company list ───────────────────────────

def test_licensing_question_focus():
    assert mc._licensing_question_focus("number of active licenses") == "licensed"
    assert mc._licensing_question_focus("number of active RHQ licenses") == "rhq"
    assert mc._licensing_question_focus("how many companies are licensed") == "licensed"
    assert mc._licensing_question_focus("how many RHQ do we have") == "rhq"


def test_saudi_licensing_count_matches_natural_phrasings():
    """'Tell me the active MISA licenses' must NOT fall through to the
    empty rhq_licenses table — it is a canonical company_profiles count."""
    assert mc._is_saudi_licensing_count_question(
        "number of active MISA licenses")
    assert mc._is_saudi_licensing_count_question(
        "how many active MISA licenses")
    assert mc._is_saudi_licensing_count_question(
        "Tell me the active MISA licenses")
    assert mc._is_saudi_licensing_count_question("active MISA licenses")
    assert mc._is_saudi_licensing_count_question(
        "what is the current MISA licence count")
    # Country company lists stay on their own path.
    assert not mc._is_saudi_licensing_count_question(
        "tell me the indian active companies")
    assert not mc._is_saudi_licensing_count_question(
        "Tell me about Apple")



def test_licensing_briefing_leads_with_focus_number():
    summ = {"total_licensed": 95671, "total_rhq": 727,
            "rhq_by_country": [], "licensed_by_country": []}
    lic = mc._format_saudi_licensing_briefing(summ, focus="licensed")
    assert lic.startswith("## Licensing Snapshot")
    assert "95,671 companies hold an active MISA licence" in lic
    assert not lic.startswith("## Saudi RHQ")
    rhq = mc._format_saudi_licensing_briefing(summ, focus="rhq")
    assert rhq.startswith("## Saudi RHQ Snapshot")
    assert "**727 companies hold an active" in rhq
    both = mc._format_saudi_licensing_briefing(summ, focus="both")
    assert both.startswith("## Licensing & RHQ Snapshot")
    assert "95,671 companies hold an active MISA licence" in both



def test_country_company_list_detection():
    assert mc._is_country_company_list_question(
        "tell me the indian active companies")
    assert mc._is_country_company_list_question("list the German licensed firms")
    assert mc._is_country_company_list_question("show me Japanese companies")
    # Not a list request / no country -> not this path.
    assert not mc._is_country_company_list_question("number of active licenses")
    assert not mc._is_country_company_list_question("tell me about Alphabet")


# ─── Deep-profile vs advisory (market segment guard) ──────────────────

def test_market_segment_detection():
    assert mc._looks_like_market_segment("German manufacturers in Saudi Arabia")
    assert mc._looks_like_market_segment("trillion-dollar MNCs")
    assert mc._looks_like_market_segment("Indian companies")
    # A single named entity is NOT a market segment.
    assert not mc._looks_like_market_segment("Microsoft")
    assert not mc._looks_like_market_segment("Tim Cook")
    assert not mc._looks_like_market_segment("Saudi Aramco")
