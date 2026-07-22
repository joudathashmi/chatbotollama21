"""Platform origin parity — examples are symptoms; behaviour must be uniform.

These tests parametrize the SAME ask shapes across many origin markets.
If India works and Pakistan does not, that is a platform bug — not an
acceptable India-only patch.
"""

from __future__ import annotations

import pytest

from app.services.advisory_enrichment import enrich_advisory_deliverable
from app.services.advisory_structured import (
    _default_trade_bodies,
    primary_trade_body_name,
)
from app.services.chat_engine import (
    _detect_advisory_deliverable,
    _detect_origin_country,
    _format_country_licensing_answer,
    _is_advisory_question,
)


# Representative FDI source markets — NOT an India special list.
_ORIGINS = [
    ("Indian", "India", "Invest India"),
    ("Pakistani", "Pakistan", "Board of Investment"),
    ("German", "Germany", "GTAI"),
    ("Japanese", "Japan", "JETRO"),
    ("Korean", "South Korea", "KOTRA"),
    ("Brazilian", "Brazil", "ApexBrasil"),
    ("French", "France", "Business France"),
    ("Mexican", "Mexico", "Mexico"),  # catalog or national IPA label
    ("Egyptian", "Egypt", "GAFI"),
    ("Swedish", "Sweden", "Business Sweden"),
    ("Nigerian", "Nigeria", "NIPC"),
    ("Indonesian", "Indonesia", "BKPM"),
    ("Polish", "Poland", "PAIH"),
    ("Turkish", "Turkey", "Investment Office"),
    ("American", "United States", "SelectUSA"),
]


_ASK_SHAPES = [
    ("market_fit", "make me a market fit to attract {adj} companies"),
    ("engagement_plan",
     "give me an engagement plan to attract {noun} companies"),
    ("sector_priorities",
     "which sectors should we prioritise to attract {adj} companies"),
    ("company_targeting",
     "best companies to target from {noun} with investment thesis"),
    ("strategy_analysis",
     "investment opportunities from {noun} in Saudi Arabia"),
]


@pytest.mark.parametrize("adj,noun,ipa_frag", _ORIGINS)
@pytest.mark.parametrize("deliverable,shape", _ASK_SHAPES)
def test_same_ask_shape_routes_identically_for_every_origin(
    adj, noun, ipa_frag, deliverable, shape,
):
    q = shape.format(adj=adj, noun=noun)
    assert _detect_origin_country(q) in {noun, "South Korea", "United States"} or (
        noun == "South Korea" and _detect_origin_country(q) == "South Korea"
    )
    assert _is_advisory_question(q), q
    assert _detect_advisory_deliverable(q) == deliverable, q


@pytest.mark.parametrize("adj,noun,ipa_frag", _ORIGINS)
def test_trade_bodies_never_bleed_india_into_other_origins(adj, noun, ipa_frag):
    bodies = _default_trade_bodies(noun)
    blob = " ".join(
        f"{b.get('organisation')} {b.get('role')}" for b in bodies
    ).casefold()
    assert ipa_frag.casefold() in blob or noun.casefold() in blob
    if noun != "India":
        assert "invest india" not in blob
        assert "nasscom" not in blob


@pytest.mark.parametrize("adj,noun,ipa_frag", _ORIGINS)
def test_enrichment_injects_origin_ipa_not_a_fixed_country(adj, noun, ipa_frag):
    thin = (
        f"# Market Fit Assessment: Attracting {noun} Companies\n\n"
        "## Overall Market Fit\n"
        "| Sector | Priority |\n|---|---|\n| ICT | Tier 1 |\n"
    )
    out, _ = enrich_advisory_deliverable(
        thin,
        deliverable="market_fit",
        db_context={
            "origin_country": noun,
            "companies_from_origin_licensed_in_saudi": 10,
            "companies_from_origin_with_rhq": 1,
            "expansion_targets": [
                {"company": "ExampleCo", "sector": "ICT",
                 "current_saudi_presence": "RHQ"},
            ],
        },
    )
    assert "Strategic Context" in out
    assert "Investment & Trade Bodies" in out
    primary = primary_trade_body_name(noun)
    assert primary
    # Primary IPA (or a distinctive fragment) must appear.
    assert (
        primary.split("(")[0].strip()[:12].casefold() in out.casefold()
        or ipa_frag.casefold() in out.casefold()
    )
    if noun != "India":
        assert "Invest India" not in out


@pytest.mark.parametrize("adj,noun,ipa_frag", [
    ("German", "Germany", "GTAI"),
    ("Pakistani", "Pakistan", "Board of Investment"),
    ("Japanese", "Japan", "JETRO"),
    ("Korean", "South Korea", "KOTRA"),
])
def test_licensing_formatter_is_origin_generic(adj, noun, ipa_frag):
    stats = {
        "total_licensed": 50,
        "total_rhq": 3,
        "total_non_licensed": 0,
        "total_non_licensed_rhq": 0,
        "retrieval_status": "ok",
        "rhq": [{"company_name": f"{noun} Anchor Co", "industry": "ICT"}],
        "licensed_only": [],
    }
    ans = _format_country_licensing_answer(noun, stats)
    assert noun in ans
    assert "50" in ans
    assert "Recommended Next Moves" in ans
    assert ipa_frag in ans or primary_trade_body_name(noun).split()[0] in ans
    assert "Invest India" not in ans


def test_korean_adjective_resolves_to_south_korea_not_bare_korea():
    assert _detect_origin_country("attract Korean companies") == "South Korea"
    assert "KOTRA" in primary_trade_body_name("Korea")
    assert "KOTRA" in primary_trade_body_name("South Korea")


def test_no_adjective_map_origin_left_on_generic_ipa_fallback():
    """Every nationality adjective MISA can detect should resolve to a
    real IPA catalog entry — not '{Country} national investment…'."""
    from app.services.chat_engine import _COUNTRY_ADJECTIVE_TO_NOUN

    leftover = []
    for _adj, noun in _COUNTRY_ADJECTIVE_TO_NOUN.items():
        if noun == "Saudi Arabia":
            continue
        org = (_default_trade_bodies(noun)[0].get("organisation") or "")
        if "national investment promotion agency" in org.casefold():
            leftover.append(noun)
    assert leftover == [], f"Still on fallback IPA: {leftover}"


def test_sector_opportunity_briefing_has_jul21_sections():
    from app.services.chat_engine import _format_sector_opportunity_briefing
    rows = [
        {"sector_name": "ICT", "opportunity_count": 120},
        {"sector_name": "Healthcare", "opportunity_count": 80},
        {"sector_name": "Energy", "opportunity_count": 60},
    ]
    ans = _format_sector_opportunity_briefing(rows)
    assert "Strategic Context" in ans
    assert "Sector Ranking" in ans
    assert "Tier-1 Sector Deep-Dives" in ans
    assert "### 1. ICT" in ans
    assert "SDAIA" in ans
    assert "Recommended Next Actions" in ans
    assert "120" in ans
