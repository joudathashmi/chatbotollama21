"""
Business Card extraction request/response schemas.

Pydantic models for the Business Card Reader API endpoint.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ParsedAddress(BaseModel):
    """Structured address parsed from a business card."""

    city: str = ""
    """City or locality name."""

    district: str = ""
    """District, county, region, or administrative area."""

    state: str = ""
    """State, province, or region."""

    country: str = ""
    """Country name."""

    pincode: str = ""
    """Postal code or ZIP code."""


class LinkedInCandidate(BaseModel):
    """A single scored LinkedIn profile candidate from search."""

    url: str
    """Canonical LinkedIn profile URL (https://www.linkedin.com/in/<slug>)."""

    match_type: str = ""
    """Resolution type associated with this candidate (e.g. 'exact', 'candidates', 'search')."""

    score: float = 0.0
    """Match confidence 0.0–1.0 (name + company + title + email-domain signals)."""

    name: str = ""
    """Name parsed from the search result / profile slug."""

    title: str = ""
    """Job title parsed from the search result."""

    company: str = ""
    """Company parsed from the search result."""



class LinkedInResult(BaseModel):
    """Outcome of LinkedIn profile resolution for an extracted card."""

    match_type: str = "search"
    """One of: 'direct' (URL was on the card), 'exact' (one confident profile
    match), 'candidates' (single best-ranked profile candidate, no exact
    match), 'search' (no profile found — linkedin_urls holds a LinkedIn
    people-search candidate instead). linkedin_urls is NEVER empty."""

    linkedin_urls: LinkedInCandidate = Field(default_factory=LinkedInCandidate)
    """A single candidate object containing the resolved profile details.
    For 'direct'/'exact': the matched profile candidate. For 'candidates': the
    best-ranked candidate. For 'search': a people-search fallback candidate.
    Never empty."""

    confidence: float = 0.0
    """Confidence of the best match (1.0 for a direct on-card URL)."""

    provider: str = ""
    """Search backend used: 'ddg', 'serp', or 'business_card' for direct hits."""

    candidates: list[LinkedInCandidate] = Field(default_factory=list)
    """Top-ranked candidate list (top N) for client-side review."""


    error: Optional[str] = None
    """Populated if resolution failed; the card extraction still succeeds."""


class BusinessCardData(BaseModel):
    """
    Extracted business card data fields.

    All fields are optional to accommodate partial extractions or failures.
    """

    linkedin: Optional[LinkedInResult] = None
    """LinkedIn profile resolution result nested inside the data payload."""

    name: Optional[str] = None
    """Person's full name extracted from the card."""

    title: Optional[str] = None
    """Job title or position."""

    company: Optional[str] = None
    """Company name."""

    email: Optional[str] = None
    """Email address."""

    country_code: list[str] = Field(default_factory=list)
    """Country dialing codes for each mobile number (parallel array, same index)."""

    mobile_numbers: list[str] = Field(default_factory=list)
    """Phone numbers extracted from the card (digits only, no formatting)."""

    fax_country_code: list[str] = Field(default_factory=list)
    """Country dialing codes for each fax number (parallel array, same index)."""

    fax_numbers: list[str] = Field(default_factory=list)
    """Fax numbers extracted from the card (digits only, no formatting)."""

    website: Optional[str] = None
    """Company website or personal website URL."""

    full_address: Optional[str] = None
    """Complete address string as it appeared on the card."""

    address: Optional[ParsedAddress] = None
    """Structured address parsed from the card."""

    other: list[str] = Field(default_factory=list)
    """Any other text from the card that wasn't categorized."""

    raw_text: str = ""
    """Full OCR text extracted from the image (for reference/debugging)."""

    source_file: Optional[str] = None
    """Filename of the uploaded image."""


class BusinessCardResponse(BaseModel):
    """
    Professional API response for business card extraction.

    Response structure follows REST API best practices:
    - success: Boolean indicating extraction success/failure
    - data: Extracted contact information (nested object)
    - error: Error message (if success=false)
    - error_code: Machine-readable error code
    - request_id: Unique request identifier for tracking
    - processing_time_ms: Execution time in milliseconds
    - timestamp: ISO 8601 UTC timestamp

    All data fields are optional to accommodate partial extractions or failures.
    """

    success: bool
    """Whether extraction succeeded (true) or failed (false)."""

    data: BusinessCardData
    """Extracted business card information including nested LinkedIn data when
    requested."""

    request_id: str = ""
    """Unique request identifier for tracking and debugging."""

    processing_time_ms: float = 0.0
    """Time taken to process the business card (in milliseconds)."""

    timestamp: Optional[str] = None
    """ISO 8601 UTC timestamp when processing completed."""

    error: Optional[str] = None
    """User-friendly error message if extraction failed."""

    error_code: Optional[str] = None
    """Machine-readable error code (e.g., 'NO_TEXT_DETECTED', 'VISION_API_ERROR')."""

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "name": "John Doe",
                    "title": "Senior Software Engineer",
                    "company": "Acme Corporation",
                    "email": "john.doe@acme.com",
                    "country_code": ["+1", "+1"],
                    "mobile_numbers": ["5550123456", "5550456789"],
                    "fax_country_code": ["+1"],
                    "fax_numbers": ["5559876543"],
                    "website": "www.acme.com",
                    "full_address": "123 Main Street, San Francisco, CA 94105, USA",
                    "address": {
                        "city": "San Francisco",
                        "district": "San Francisco County",
                        "state": "California",
                        "country": "USA",
                        "pincode": "94105",
                    },
                    "other": ["LinkedIn: linkedin.com/in/johndoe"],
                    "raw_text": "[full OCR output...]",
                    "source_file": "card.jpg",
                    "linkedin": {
                        "match_type": "exact",
                        "linkedin_urls": {
                            "url": "https://www.linkedin.com/in/jane-doe-tech",
                            "score": 0.9123,
                            "name": "Jane Doe",
                            "title": "Principal Software Engineer",
                            "company": "Acme Corporation"
                        },
                        "confidence": 0.9123,
                        "provider": "ddg",
                        "candidates": [
                            {
                                "url": "https://www.linkedin.com/in/jane-doe-tech",
                                "score": 0.9123,
                                "name": "Jane Doe",
                                "title": "Principal Software Engineer",
                                "company": "Acme Corporation"
                            }
                        ],

                        "error": None
                    }
                },
                "request_id": "k7m2pq9x",
                "processing_time_ms": 2450.5,
                "timestamp": "2026-06-01T14:30:45.123Z",
                "error": None,
                "error_code": None
            }
        }

