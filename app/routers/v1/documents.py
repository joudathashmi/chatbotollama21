"""Document library API.

  POST /api/v1/documents/upload  — multipart file upload (analyst+)
  POST /api/v1/documents/ingest  — scan server ingest folder (analyst+)
  GET  /api/v1/documents         — list visible docs (any auth)
  GET  /api/v1/documents/{id}    — metadata
  GET  /api/v1/documents/{id}/status
  DELETE /api/v1/documents/{id}
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app import config
from app.auth import require_role, verify_credentials
from app.config import DOCUMENTS_RATE_LIMIT, ROLE_ADMIN, ROLE_ANALYST
from app.rate_limit import rate_limit
from app.schemas.documents import (
    DocumentListResponse,
    DocumentOut,
    IngestRequest,
    IngestResponse,
)
from app.services.document_classification import CONSENT_POLICY
from app.services.document_ingest import DocumentIngestError, ingest_bytes, ingest_inbox
from app.services.document_store import get_document_store
from app.utils.error_handler import create_error_response

router = APIRouter(prefix="/documents", tags=["documents"])

_docs_rl = rate_limit("documents", *DOCUMENTS_RATE_LIMIT)


def _to_out(d) -> DocumentOut:
    return DocumentOut(**d.to_dict())


def _err(code: str, message: str, status: int, path: str) -> JSONResponse:
    body = create_error_response(
        code=code, message=message, status=status, path=path
    ).model_dump()
    return JSONResponse(status_code=status, content=body)


async def _read_bounded(upload: UploadFile, max_bytes: int) -> bytes:
    buf = bytearray()
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise DocumentIngestError("FILE_TOO_LARGE", "File exceeds maximum allowed size.")
    return bytes(buf)


@router.post(
    "/upload",
    summary="Upload a document into the library",
    response_model=DocumentOut,
    dependencies=[Depends(_docs_rl), Depends(require_role(ROLE_ANALYST))],
)
async def upload_document(
    file: UploadFile = File(...),
    visibility: Literal["private", "org"] = Form("private"),
    classification: str = Form("public"),
    consent: bool = Form(False),
    user: str = Depends(verify_credentials),
):
    if not config.DOCUMENTS_ENABLED:
        return _err("DISABLED", "Document library is disabled.", 503, "/api/v1/documents/upload")
    try:
        data = await _read_bounded(file, config.DOCUMENTS_MAX_BYTES)
        doc = await asyncio.to_thread(
            ingest_bytes,
            data,
            filename=file.filename or "upload.bin",
            owner_username=user,
            visibility=visibility,
            source="upload",
            content_type=file.content_type,
            classification=classification,
            consent=consent,
        )
        return _to_out(doc)
    except DocumentIngestError as e:
        status = {
            "FILE_TOO_LARGE": 413,
            "MALWARE_DETECTED": 422,
            "SCAN_UNAVAILABLE": 503,
            "UNSUPPORTED_TYPE": 415,
            "DISABLED": 503,
            "CLASSIFIED_DOCUMENT": 403,
            "CLASSIFIED_CONTENT": 403,
            "CONSENT_REQUIRED": 428,
        }.get(e.code, 400)
        return _err(e.code, e.message, status, "/api/v1/documents/upload")


@router.get(
    "/consent-policy",
    summary="Upload consent declaration shown before any document upload",
)
async def consent_policy():
    return CONSENT_POLICY


@router.post(
    "/ingest",
    summary="Ingest files from the server inbox directory",
    response_model=IngestResponse,
    dependencies=[Depends(_docs_rl), Depends(require_role(ROLE_ANALYST))],
)
async def ingest_directory(
    req: IngestRequest | None = None,
    user: str = Depends(verify_credentials),
):
    if not config.DOCUMENTS_ENABLED:
        return _err("DISABLED", "Document library is disabled.", 503, "/api/v1/documents/ingest")
    visibility = (req.visibility if req else "org")
    result = await asyncio.to_thread(
        ingest_inbox, owner_username=user, visibility=visibility
    )
    return IngestResponse(
        ingested=[DocumentOut(**d) for d in result["ingested"]],
        duplicates=result["duplicates"],
        failed=result["failed"],
        skipped=result["skipped"],
    )


@router.get(
    "",
    summary="List documents visible to the caller",
    response_model=DocumentListResponse,
    dependencies=[Depends(_docs_rl)],
)
async def list_documents(user: str = Depends(verify_credentials)):
    store = get_document_store()
    docs = await asyncio.to_thread(store.list_visible, user)
    return DocumentListResponse(documents=[_to_out(d) for d in docs])


@router.get(
    "/{doc_id}",
    summary="Get document metadata",
    response_model=DocumentOut,
    dependencies=[Depends(_docs_rl)],
)
async def get_document(doc_id: str, user: str = Depends(verify_credentials)):
    store = get_document_store()
    doc = await asyncio.to_thread(store.get, doc_id, user)
    if doc is None:
        return _err("NOT_FOUND", "Document not found.", 404, f"/api/v1/documents/{doc_id}")
    return _to_out(doc)


@router.get(
    "/{doc_id}/status",
    summary="Get document processing status",
    dependencies=[Depends(_docs_rl)],
)
async def document_status(doc_id: str, user: str = Depends(verify_credentials)):
    store = get_document_store()
    doc = await asyncio.to_thread(store.get, doc_id, user)
    if doc is None:
        return _err("NOT_FOUND", "Document not found.", 404, f"/api/v1/documents/{doc_id}/status")
    return {"id": doc.id, "status": doc.status, "error": doc.error, "filename": doc.filename}


@router.delete(
    "/{doc_id}",
    summary="Delete a document",
    dependencies=[Depends(_docs_rl), Depends(require_role(ROLE_ANALYST))],
)
async def delete_document(doc_id: str, user: str = Depends(verify_credentials)):
    store = get_document_store()
    is_admin = config.role_at_least(user, ROLE_ADMIN)
    ok = await asyncio.to_thread(store.delete, doc_id, user, is_admin=is_admin)
    if not ok:
        return _err(
            "NOT_FOUND",
            "Document not found or not permitted.",
            404,
            f"/api/v1/documents/{doc_id}",
        )
    return {"deleted": True, "id": doc_id}
