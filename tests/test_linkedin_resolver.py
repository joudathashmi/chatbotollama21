"""Tests for LinkedIn profile resolution (business card enrichment).

All search backends are mocked — no live network calls.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import linkedin_resolver as L

client = TestClient(app)

_AUTH = {"Authorization": "Basic U21hcnRDaGF0Ym90Om5SdFEyNEhUOG1vaUxwdWlBTDQ0U1ozR2plUnM3V0FWWlc="}


def _real_jpeg_bytes() -> bytes:
    """A genuine, minimal (1x1 pixel) JPEG — the upload endpoint now
    validates actual file content, not just the declared Content-Type,
    so tests must upload a real image."""
    buf = BytesIO()
    Image.new("RGB", (1, 1), color="white").save(buf, format="JPEG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

class TestUrlHelpers:
    @pytest.mark.parametrize("raw,expected", [
        ("https://www.linkedin.com/in/abdul-fadhil-8b2a", "https://www.linkedin.com/in/abdul-fadhil-8b2a"),
        ("linkedin.com/in/john-doe?trk=public", "https://www.linkedin.com/in/john-doe"),
        ("http://in.linkedin.com/in/jane/", "https://www.linkedin.com/in/jane"),
        ("https://www.linkedin.com/company/acme", None),   # company page, not a profile
        ("https://example.com/about", None),
        ("", None),
        (None, None),
    ])
    def test_normalize(self, raw, expected):
        assert L.normalize_linkedin_url(raw) == expected

    def test_name_from_slug_drops_id_tokens(self):
        assert L._name_from_slug("https://www.linkedin.com/in/abdul-fadhil-8b2a3c") == "abdul fadhil"
        assert L._name_from_slug("https://www.linkedin.com/in/tim-cook") == "tim cook"

    def test_parse_title_keeps_hyphenated_names(self):
        name, title, company = L._parse_title(
            "Jean-Paul Sartre - Senior Engineer - Acme Corp | LinkedIn", ""
        )
        assert name == "Jean-Paul Sartre"
        assert title == "Senior Engineer"
        assert company == "Acme Corp"

    def test_brand_from_email(self):
        assert L._brand_from_email("john@cognizant.com") == "cognizant"
        assert L._brand_from_email("abdulfadhilm@gmail.com") == ""   # generic → no signal
        assert L._brand_from_email("") == ""
        assert L._brand_from_email("noatsign") == ""


# --------------------------------------------------------------------------
# Direct extraction
# --------------------------------------------------------------------------

class TestFuzzyNormalization:
    def test_strip_diacritics(self):
        assert L._strip_diacritics("José Müller") == "Jose Muller"

    def test_normalize_text(self):
        assert L._normalize_text("  O'Brien-Smith, Jr.! ") == "o brien smith jr"

    def test_name_tokens_drop_honorifics(self):
        assert L._name_tokens("Dr. Tim Cook") == ["tim", "cook"]
        assert L._name_tokens("Eng. José Mourinho") == ["jose", "mourinho"]

    def test_normalize_company_strips_suffixes(self):
        assert L._normalize_company("Acme Corp LLC") == "acme"
        assert L._normalize_company("Cognizant Technology Solutions") == "cognizant"

    def test_normalize_company_fallback_when_all_stripped(self):
        # 'Group Holdings' is all stopwords → fall back to plain normalized form.
        assert L._normalize_company("Group Holdings") == "group holdings"

    def test_expand_title(self):
        assert L._expand_title("Sr VP") == "senior vice president"
        assert L._expand_title("CTO") == "chief technology officer"


class TestFuzzyNameScore:
    @pytest.mark.parametrize("card,cands,min_score", [
        ("José Mourinho", ["Jose Mourinho"], 0.95),       # diacritics
        ("Satya Nadella", ["S Nadella"], 0.90),            # initial → full
        ("Abdul Fadhil", ["Abdul Fadhil M"], 0.95),        # extra middle token
        ("Tim Cook", ["Cook Tim"], 0.95),                  # order swap
        ("Abdul Fadhil", ["abdulfadhil"], 0.95),           # separator-less slug
        ("Dr. Tim Cook", ["Tim Cook"], 0.95),              # honorific stripped
    ])
    def test_strong_matches(self, card, cands, min_score):
        assert L._name_score(card, cands) >= min_score

    @pytest.mark.parametrize("card,cands", [
        ("Tim Cook", ["John Doe"]),
        ("Satya Nadella", ["Sundar Pichai"]),
        ("Abdul Fadhil", [""]),
    ])
    def test_weak_matches_stay_low(self, card, cands):
        assert L._name_score(card, cands) < 0.6

    def test_empty_card_name_is_zero(self):
        assert L._name_score("", ["Tim Cook"]) == 0.0

    def test_initials_require_matching_surname(self):
        # 'S Nadella' must NOT match 'S Pichai' just because both start with S.
        assert L._initials_form_score(["s", "nadella"], ["s", "pichai"]) == 0


class TestFuzzyCompanyTitleScore:
    def test_company_suffix_insensitive(self):
        assert L._company_score("Acme", "Acme LLC", "") >= 0.95
        assert L._company_score("Cognizant", "Cognizant Technology Solutions", "") >= 0.95

    def test_company_found_in_blob(self):
        # Parsed company empty, but it appears in the snippet blob.
        assert L._company_score("Microsoft", "", "Satya - CEO at Microsoft") >= 0.85

    def test_company_mismatch_low(self):
        assert L._company_score("Acme", "Globex", "") < 0.5

    def test_title_abbreviation_match(self):
        assert L._title_score("VP Sales", "Vice President of Sales") >= 0.9
        assert L._title_score("Sr Engineer", "Senior Software Engineer") >= 0.8


class TestBrandInText:
    def test_long_brand_substring(self):
        assert L._brand_in_text("cognizant", "works at cognizant india") is True

    def test_short_brand_needs_word_boundary(self):
        assert L._brand_in_text("ge", "ge healthcare division") is True
        assert L._brand_in_text("ge", "storage solutions") is False   # 'ge' in 'storage'

    def test_empty_brand_false(self):
        assert L._brand_in_text("", "anything") is False


class TestQueryExpansion:
    def test_expands_multiple_angles(self):
        card = {"name": "Abdul Fadhil", "company": "Cognizant", "title": "Analyst",
                "email": "abdul@cognizant.com", "address": {"city": "Chennai"}}
        qs = L._build_queries(card)
        assert len(qs) >= 4
        assert any("Cognizant" in q for q in qs)
        assert any("Chennai" in q for q in qs)
        assert any(q.startswith("site:linkedin.com/in/") for q in qs)

    def test_capped_at_max(self):
        from app.config import LINKEDIN_MAX_QUERIES
        card = {"name": "A B", "company": "C", "title": "T",
                "email": "a@corp.com", "address": {"city": "X City"}}
        assert len(L._build_queries(card)) <= LINKEDIN_MAX_QUERIES

    def test_no_name_no_queries(self):
        assert L._build_queries({"name": ""}) == []


class TestProviderSeparation:
    """DuckDuckGo and Serper must build their own queries independently."""

    def test_ddg_and_serp_build_different_queries(self):
        card = {"name": "Abdul Fadhil", "company": "Cognizant", "title": "Analyst",
                "email": "abdul@cognizant.com", "address": {"city": "Chennai"}}
        ddg_q = L.DuckDuckGoProvider().build_queries(card)
        serp_q = L.SerperProvider().build_queries(card)
        assert ddg_q != serp_q
        # DDG leans on plain phrasings; Serper leans on stacked site: operators.
        assert any("linkedin" in q and "site:" not in q for q in ddg_q)
        assert all("linkedin.com" in q for q in serp_q)

    def test_ddg_path_does_not_touch_serper_key(self):
        """Resolving via DDG must not require/raise on a missing SERPER_API_KEY."""
        card = {"name": "Jane Doe", "company": "Acme", "raw_text": "x"}
        hits = [{"url": "https://www.linkedin.com/in/jane-doe",
                 "title": "Jane Doe - VP - Acme | LinkedIn", "body": "Acme"}]
        with patch.object(L.DuckDuckGoProvider, "search", new=AsyncMock(return_value=hits)):
            out = asyncio.run(L.resolve_linkedin(card, "ddg"))
        assert out["provider"] == "ddg"
        assert out["error"] is None


class TestProviderSelection:
    @pytest.mark.parametrize("prov,cls", [
        ("ddg", L.DuckDuckGoProvider),
        ("serp", L.SerperProvider),
        ("playwright", L.PlaywrightProvider),
        ("unknown", L.DuckDuckGoProvider),   # falls back to ddg
        ("", L.DuckDuckGoProvider),
    ])
    def test_get_provider(self, prov, cls):
        assert isinstance(L._get_provider(prov), cls)


class TestScoringImprovements:
    def test_separatorless_slug_scores_high(self):
        """'abdulfadhil' (no separators) must match 'Abdul Fadhil' at full
        strength via the spaces-stripped comparison."""
        card = {"name": "Abdul Fadhil", "company": "Cognizant", "email": "abdul@cognizant.com"}
        cand = L._Candidate(
            "https://www.linkedin.com/in/abdulfadhil",
            "Abdul Fadhil - Programmer Analyst - Cognizant | LinkedIn", "Cognizant",
        )
        score, name_score, _ = L._score_candidate(card, cand)
        assert name_score >= 0.95   # the slug fix — this is what gates "exact"
        assert score >= 0.90


class TestPlaywrightProvider:
    def test_missing_playwright_raises_clean_error(self):
        """If playwright import fails, the provider surfaces a clear message
        (resolve_linkedin turns this into match_type='none' + error)."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith("playwright"):
                raise ImportError("no playwright")
            return real_import(name, *a, **k)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="playwright is not installed"):
                asyncio.run(L.PlaywrightProvider().search(["q"], 10))

    def test_resolve_with_playwright_provider(self):
        """provider='playwright' routes through the resolver; search is mocked
        so no real browser launches."""
        card = {"name": "Jane Doe", "company": "Acme", "email": "jane@acme.com", "raw_text": "x"}
        hits = [{"url": "https://www.linkedin.com/in/jane-doe",
                 "title": "Jane Doe - VP - Acme | LinkedIn", "body": "Acme"}]
        with patch.object(L.PlaywrightProvider, "search", new=AsyncMock(return_value=hits)):
            out = asyncio.run(L.resolve_linkedin(card, "playwright"))
        assert out["provider"] == "playwright"
        assert out["linkedin_urls"]["url"] == "https://www.linkedin.com/in/jane-doe"
        assert out["linkedin_urls"]["name"] == "Jane Doe"



class TestDirectExtraction:
    def test_url_in_raw_text(self):
        card = {"raw_text": "John Doe\nlinkedin.com/in/john-doe-123\nAcme"}
        assert L.extract_direct_linkedin(card) == "https://www.linkedin.com/in/john-doe-123"

    def test_url_in_other_lines(self):
        card = {"raw_text": "no url here", "other": ["LinkedIn: https://linkedin.com/in/jane-x"]}
        assert L.extract_direct_linkedin(card) == "https://www.linkedin.com/in/jane-x"

    def test_no_url(self):
        assert L.extract_direct_linkedin({"raw_text": "nothing", "other": []}) is None

    def test_resolve_direct_short_circuits_search(self):
        card = {"name": "John Doe", "raw_text": "linkedin.com/in/john-doe"}
        out = asyncio.run(L.resolve_linkedin(card, "ddg"))
        assert out["match_type"] == "direct"
        assert out["linkedin_urls"]["url"] == "https://www.linkedin.com/in/john-doe"
        assert out["linkedin_urls"]["score"] == 1.0
        assert out["confidence"] == 1.0
        assert out["provider"] == "business_card"


# --------------------------------------------------------------------------
# Search + scoring (mocked provider)
# --------------------------------------------------------------------------

def _fake_provider(hits):
    """Patch _get_provider to return a provider whose .search returns `hits`."""
    prov = L.DuckDuckGoProvider()
    prov.search = AsyncMock(return_value=hits)
    return prov


class TestNeverNullFallback:
    """linkedin_urls must NEVER be empty — exact match -> 1 URL, otherwise
    relevant candidates, otherwise a LinkedIn people-search URL."""

    def test_search_urls_built_from_name_and_company(self):
        urls = L._linkedin_search_urls({"name": "Tim Cook", "company": "Apple"})
        assert urls
        assert all(u.startswith("https://www.linkedin.com/search/results/people/") for u in urls)
        assert any("Tim" in u or "Cook" in u for u in urls)

    def test_search_urls_never_empty_even_with_blank_card(self):
        urls = L._linkedin_search_urls({})
        assert urls
        assert urls[0] == "https://www.linkedin.com/search/results/people/"

    def test_search_fallback_result_shape(self):
        out = L._search_fallback_result({"name": "Jane Doe"}, "ddg", "boom")
        assert out["match_type"] == "search"
        assert out["linkedin_urls"]
        assert out["linkedin_urls"]["url"].startswith("https://www.linkedin.com/search/results/people/")
        assert out["provider"] == "ddg"
        assert out["error"] == "boom"
        assert out["candidates"]["url"] == out["linkedin_urls"]["url"]

    def test_weak_match_still_returns_candidate_urls_not_empty(self):
        """Even the existing 'weak match' path must never end up empty."""
        card = {"name": "Abdul Fadhil", "company": "Cognizant", "title": "Analyst",
                "email": "abdulfadhilm@gmail.com", "raw_text": "x"}
        hits = [
            {"url": "https://www.linkedin.com/in/abdul-something",
             "title": "Abdul S - Designer - Unrelated | LinkedIn", "body": ""},
        ]
        with patch.object(L, "_get_provider", return_value=_fake_provider(hits)):
            out = asyncio.run(L.resolve_linkedin(card, "ddg"))
        assert out["linkedin_urls"]
        assert out["match_type"] in ("exact", "candidates")


class TestResolveWithSearch:
    def test_exact_match_returns_single_url(self):
        card = {"name": "Abdul Fadhil", "company": "Cognizant", "title": "Programmer Analyst",
                "email": "abdul@cognizant.com", "raw_text": "Abdul Fadhil Cognizant"}
        hits = [
            {"url": "https://www.linkedin.com/in/abdul-fadhil-8b2a",
             "title": "Abdul Fadhil - Programmer Analyst - Cognizant | LinkedIn", "body": "Cognizant"},
            {"url": "https://www.linkedin.com/in/some-other-person",
             "title": "Other Person - Manager - Different Co | LinkedIn", "body": ""},
        ]
        with patch.object(L, "_get_provider", return_value=_fake_provider(hits)):
            out = asyncio.run(L.resolve_linkedin(card, "ddg"))
        assert out["match_type"] == "exact"
        assert out["linkedin_urls"]["url"] == "https://www.linkedin.com/in/abdul-fadhil-8b2a"
        assert out["linkedin_urls"]["name"] == "Abdul Fadhil"
        assert out["confidence"] >= 0.9

    def test_weak_match_returns_candidate_array(self):
        card = {"name": "Abdul Fadhil", "company": "Cognizant", "title": "Analyst",
                "email": "abdulfadhilm@gmail.com", "raw_text": "x"}
        hits = [
            {"url": "https://www.linkedin.com/in/abdul-something",
             "title": "Abdul S - Designer - Unrelated | LinkedIn", "body": ""},
            {"url": "https://www.linkedin.com/in/another-abdul",
             "title": "Abdul K - Teacher - School | LinkedIn", "body": ""},
        ]
        with patch.object(L, "_get_provider", return_value=_fake_provider(hits)):
            out = asyncio.run(L.resolve_linkedin(card, "ddg"))
        assert out["match_type"] == "candidates"
        assert out["linkedin_urls"]["url"].startswith("https://www.linkedin.com/in/")

    def test_dedup_and_profile_filter(self):
        card = {"name": "Jane Doe", "company": "Acme", "raw_text": "x"}
        hits = [
            {"url": "https://www.linkedin.com/in/jane-doe?trk=1", "title": "Jane Doe - Acme | LinkedIn", "body": ""},
            {"url": "https://www.linkedin.com/in/jane-doe", "title": "dup", "body": ""},        # dup
            {"url": "https://www.linkedin.com/company/acme", "title": "Acme", "body": ""},      # not a profile
        ]
        with patch.object(L, "_get_provider", return_value=_fake_provider(hits)):
            out = asyncio.run(L.resolve_linkedin(card, "ddg"))
        # Only one unique profile URL survives dedup + profile filter.
        assert isinstance(out["candidates"], list)
        assert len(out["candidates"]) == 1
        assert out["candidates"][0]["url"].startswith("https://www.linkedin.com/in/")



    def test_no_name_returns_search_fallback_not_null(self):
        """No name on card → still returns a non-empty LinkedIn search URL,
        never an empty/null linkedin_urls."""
        out = asyncio.run(L.resolve_linkedin({"name": "", "raw_text": "x"}, "ddg"))
        assert out["match_type"] == "search"
        assert out["error"]
        assert out["linkedin_urls"]
        assert out["linkedin_urls"]["url"]

    def test_provider_failure_degrades_to_search_fallback(self):
        """A provider error must NOT produce an empty result — it falls back
        to a LinkedIn search URL built from the card."""
        prov = L.DuckDuckGoProvider()
        prov.search = AsyncMock(side_effect=RuntimeError("rate limited"))
        card = {"name": "John Doe", "company": "Acme", "raw_text": "x"}
        with patch.object(L, "_get_provider", return_value=prov):
            out = asyncio.run(L.resolve_linkedin(card, "ddg"))
        assert out["match_type"] == "search"
        assert out["error"] == "rate limited"
        assert out["linkedin_urls"]
        assert "John" in out["linkedin_urls"]["url"] or "Acme" in out["linkedin_urls"]["url"]

    def test_no_candidates_found_falls_back_to_search_url(self):
        """Zero search results must still return a usable LinkedIn search URL,
        not an empty array."""
        card = {"name": "Ghost Person", "company": "Nowhere", "raw_text": "x"}
        with patch.object(L, "_get_provider", return_value=_fake_provider([])):
            out = asyncio.run(L.resolve_linkedin(card, "ddg"))
        assert out["match_type"] == "search"
        assert out["error"] is None
        assert out["linkedin_urls"]
        assert out["linkedin_urls"]["url"].startswith(
            "https://www.linkedin.com/search/results/people/"
        )


# --------------------------------------------------------------------------
# Endpoint integration
# --------------------------------------------------------------------------

class TestEndpointLinkedInFlag:
    @patch("app.services.linkedin_resolver.resolve_linkedin", new_callable=AsyncMock)
    @patch("app.routers.v1.business_card.process_business_card")
    def test_default_call_resolves_linkedin_automatically(self, mock_proc, mock_resolve):
        """resolve_linkedin now defaults to True — a plain /upload call (no
        query params) must run LinkedIn resolution without being asked."""
        mock_proc.return_value = {
            "name": "Bob", "title": None, "company": None, "email": None,
            "country_code": [], "mobile_numbers": [], "website": None,
            "full_address": None, "address": None, "other": [], "raw_text": "Bob",
            "error": None,
        }
        mock_resolve.return_value = {
            "match_type": "search",
            "linkedin_urls": {"url": "https://www.linkedin.com/search/results/people/?keywords=Bob", "score": 0.0, "name": "Bob", "title": "", "company": ""},
            "confidence": 0.0, "provider": "ddg",
            "candidates": [{"url": "https://www.linkedin.com/search/results/people/?keywords=Bob",
                            "score": 0.0, "name": "Bob", "title": "", "company": ""}],

            "error": None,
        }

        files = {"file": ("c.jpg", BytesIO(_real_jpeg_bytes()), "image/jpeg")}
        resp = client.post("/api/v1/business-card/upload", files=files, headers=_AUTH)
        assert resp.status_code == 200
        li = resp.json()["data"]["linkedin"]
        assert li is not None
        assert li["linkedin_urls"]
        mock_resolve.assert_awaited_once()

    @patch("app.services.linkedin_resolver.resolve_linkedin", new_callable=AsyncMock)
    @patch("app.routers.v1.business_card.process_business_card")
    def test_explicit_opt_out_skips_linkedin(self, mock_proc, mock_resolve):
        """?resolve_linkedin=false must still be honoured to skip the lookup."""
        mock_proc.return_value = {
            "name": "Bob", "title": None, "company": None, "email": None,
            "country_code": [], "mobile_numbers": [], "website": None,
            "full_address": None, "address": None, "other": [], "raw_text": "Bob",
            "error": None,
        }
        files = {"file": ("c.jpg", BytesIO(_real_jpeg_bytes()), "image/jpeg")}
        resp = client.post(
            "/api/v1/business-card/upload?resolve_linkedin=false", files=files, headers=_AUTH
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["linkedin"] is None
        mock_resolve.assert_not_awaited()

    @patch("app.services.linkedin_resolver.resolve_linkedin", new_callable=AsyncMock)
    @patch("app.routers.v1.business_card.process_business_card")
    def test_flag_on_attaches_linkedin(self, mock_proc, mock_resolve):
        mock_proc.return_value = {
            "name": "Bob Smith", "title": "CEO", "company": "Acme", "email": None,
            "country_code": [], "mobile_numbers": [], "website": None,
            "full_address": None, "address": None, "other": [], "raw_text": "Bob Smith Acme",
            "error": None,
        }
        mock_resolve.return_value = {
            "match_type": "exact",
            "linkedin_urls": {"url": "https://www.linkedin.com/in/bob-smith", "score": 0.97,
                               "name": "Bob Smith", "title": "CEO", "company": "Acme"},
            "confidence": 0.97, "provider": "ddg",
            "candidates": [{"url": "https://www.linkedin.com/in/bob-smith", "score": 0.97,
                            "name": "Bob Smith", "title": "CEO", "company": "Acme"}],

            "error": None,
        }

        files = {"file": ("c.jpg", BytesIO(_real_jpeg_bytes()), "image/jpeg")}
        resp = client.post(
            "/api/v1/business-card/upload?resolve_linkedin=true&provider=ddg",
            files=files, headers=_AUTH,
        )
        assert resp.status_code == 200
        li = resp.json()["data"]["linkedin"]
        assert li["match_type"] == "exact"
        assert li["linkedin_urls"]["url"] == "https://www.linkedin.com/in/bob-smith"
        assert li["linkedin_urls"]["name"] == "Bob Smith"
        mock_resolve.assert_awaited_once()
