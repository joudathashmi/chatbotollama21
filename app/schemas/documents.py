from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    owner_username: str
    visibility: Literal["private", "org"]
    filename: str
    content_type: str
    sha256: str
    byte_size: int
    status: Literal["pending", "ready", "failed"]
    source: Literal["upload", "ingest"]
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]


class IngestRequest(BaseModel):
    visibility: Literal["private", "org"] = "org"


class IngestResponse(BaseModel):
    ingested: list[DocumentOut] = Field(default_factory=list)
    duplicates: list[dict] = Field(default_factory=list)
    failed: list[dict] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
