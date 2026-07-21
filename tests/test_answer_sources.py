"""Tests for unified answer provenance packing."""

from app.services.answer_sources import build_unified_sources


def test_build_unified_sources_merges_and_dedupes():
    out = build_unified_sources(
        doc_sources=[{
            "title": "brief.pdf",
            "url": "doc://abc#0",
            "snippet": "Helios MoU",
        }],
        web_sources=[{
            "title": "Reuters",
            "url": "https://example.com/a",
            "snippet": "news",
        }],
        tool_calls=[
            {"table": "company_profiles", "row_count": 1},
            {"table": "_internal", "row_count": 0},
            {"table": "company_profiles", "row_count": 2},
        ],
    )
    types = [s["type"] for s in out]
    assert types == ["document", "web", "db"]
    assert out[0]["title"] == "brief.pdf"
    assert out[1]["url"] == "https://example.com/a"
    assert out[2]["title"] == "company_profiles"
    assert out[2]["url"] == "db://company_profiles"


def test_build_unified_sources_skips_doc_urls_in_web_list():
    out = build_unified_sources(
        web_sources=[
            {"title": "brief", "url": "doc://x#1", "snippet": ""},
            {"title": "News", "url": "https://example.com/n", "snippet": ""},
        ],
    )
    assert len(out) == 1
    assert out[0]["type"] == "web"
