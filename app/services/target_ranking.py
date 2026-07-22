"""Rank MISA footprint companies for investment-attraction targeting.

Pure helpers — no LLM. Used by the advisory path to inject
expansion-target candidates from live DB rows so the model cannot
replace the installed base with generic famous brands.
"""

from __future__ import annotations

from typing import Any


_NOISE_NAME = (
    "restaurant", "cafe", "café", "chicken", "corner", "grocery",
    "supermarket", "bakery", "salon", "laundry", "tailor",
)


def _is_substantive(row: dict) -> bool:
    name = (row.get("company_name") or row.get("name") or "").strip()
    if len(name) < 3:
        return False
    nl = name.lower()
    if any(tok in nl for tok in _NOISE_NAME):
        return False
    rev = row.get("annual_revenue")
    try:
        if rev is not None and float(rev) >= 1_000_000:
            return True
    except (TypeError, ValueError):
        pass
    industry = (row.get("industry") or "").strip()
    if industry and industry.lower() not in ("n/a", "unclassified", "none", "-"):
        return True
    markers = (
        "limited", "ltd", "inc", "corp", "regional", "hq", "headquarter",
        "technologies", "systems", "pharma", "bank", "group", "services",
    )
    return any(m in nl for m in markers) or len(name) >= 18


def rank_expansion_targets(
    stats: dict[str, Any],
    *,
    max_rhq: int = 8,
    max_licensed: int = 6,
) -> list[dict[str, Any]]:
    """Build expansion-target cards from fetch_country_saudi_investors output."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(row: dict, presence: str) -> None:
        name = (row.get("company_name") or row.get("name") or "").strip()
        if not name:
            return
        key = name.casefold()
        if key in seen:
            return
        if not _is_substantive(row):
            return
        seen.add(key)
        out.append({
            "company": name,
            "sector": (row.get("industry") or "Unclassified").strip() or "Unclassified",
            "current_saudi_presence": presence,
            "target_type": "expansion",
            "annual_revenue": row.get("annual_revenue"),
            "evidence_strength": "high" if presence == "RHQ" else "medium",
        })

    for r in (stats.get("rhq") or []):
        if len([t for t in out if t["current_saudi_presence"] == "RHQ"]) >= max_rhq:
            break
        _add(r, "RHQ")

    for r in (stats.get("licensed_only") or []):
        if len([t for t in out if t["current_saudi_presence"] == "Licensed"]) >= max_licensed:
            break
        _add(r, "Licensed")

    # Rank: RHQ first, then by revenue desc
    def _rev(t: dict) -> float:
        try:
            return float(t.get("annual_revenue") or 0)
        except (TypeError, ValueError):
            return 0.0

    rhq = [t for t in out if t["current_saudi_presence"] == "RHQ"]
    lic = [t for t in out if t["current_saudi_presence"] == "Licensed"]
    rhq.sort(key=_rev, reverse=True)
    lic.sort(key=_rev, reverse=True)
    ranked = rhq + lic
    for i, t in enumerate(ranked, 1):
        t["rank"] = i
    return ranked
