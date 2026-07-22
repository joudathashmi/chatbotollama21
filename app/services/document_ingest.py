"""Document ingest: validate, malware-scan, extract text, chunk, index."""

from __future__ import annotations

import io
import re
import shutil
from pathlib import Path

from app import config
from app.logger import logger
from app.services.document_store import (
    DocumentRecord,
    ensure_dirs,
    get_document_store,
    safe_filename,
    sha256_bytes,
    storage_path_for,
)
from app.services.malware_scanner import ScanVerdict, scan_file

ALLOWED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

ALLOWED_MIME_TYPES = set(ALLOWED_EXTENSIONS.values()) | {
    "application/octet-stream",  # some browsers
}


class DocumentIngestError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def guess_content_type(filename: str, declared: str | None = None) -> str:
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_EXTENSIONS:
        return ALLOWED_EXTENSIONS[ext]
    if declared and declared in ALLOWED_MIME_TYPES:
        return declared
    raise DocumentIngestError("UNSUPPORTED_TYPE", f"Unsupported file type: {ext or 'unknown'}")


def extract_text(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in (".txt", ".md", ".markdown"):
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(parts).strip()
        if not text:
            raise DocumentIngestError("EMPTY_TEXT", "No extractable text in PDF.")
        return text
    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs if p.text).strip()
        if not text:
            raise DocumentIngestError("EMPTY_TEXT", "No extractable text in DOCX.")
        return text
    raise DocumentIngestError("UNSUPPORTED_TYPE", f"Unsupported file type: {ext}")


def chunk_text(text: str, *, size: int = 1000, overlap: int = 120) -> list[str]:
    cleaned = "\n".join(line.rstrip() for line in (text or "").splitlines()).strip()
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + size, n)
        # Prefer breaking on paragraph/sentence boundaries.
        if end < n:
            window = cleaned[start:end]
            br = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if br > size // 3:
                end = start + br + 1
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def _malware_check(data: bytes, filename: str) -> None:
    result = scan_file(data, filename=filename)
    if result.verdict == ScanVerdict.INFECTED:
        raise DocumentIngestError("MALWARE_DETECTED", "Upload rejected by malware scanner.")
    if result.verdict == ScanVerdict.SCAN_FAILED and not config.MALWARE_SCAN_FAIL_OPEN:
        raise DocumentIngestError("SCAN_UNAVAILABLE", "Malware scanner unavailable.")


def ingest_bytes(
    data: bytes,
    *,
    filename: str,
    owner_username: str,
    visibility: str = "private",
    source: str = "upload",
    content_type: str | None = None,
) -> DocumentRecord:
    """Validate, store, extract, chunk, and index a document. Returns the record."""
    if not config.DOCUMENTS_ENABLED:
        raise DocumentIngestError("DISABLED", "Document library is disabled.")
    if visibility not in ("private", "org"):
        raise DocumentIngestError("BAD_VISIBILITY", "visibility must be private or org.")
    if len(data) > config.DOCUMENTS_MAX_BYTES:
        raise DocumentIngestError("FILE_TOO_LARGE", "File exceeds maximum allowed size.")
    if not data:
        raise DocumentIngestError("EMPTY_FILE", "Empty file.")

    fname = safe_filename(filename)
    ctype = guess_content_type(fname, content_type)
    _malware_check(data, fname)

    digest = sha256_bytes(data)
    store = get_document_store()
    store.ensure()

    dup = store.find_duplicate(
        sha256=digest, visibility=visibility, owner_username=owner_username
    )
    if dup is not None:
        return dup

    import uuid
    doc_id = str(uuid.uuid4())
    dest = storage_path_for(doc_id, fname)
    dest.write_bytes(data)

    doc = store.create_pending(
        owner_username=owner_username,
        visibility=visibility,
        filename=fname,
        content_type=ctype,
        sha256=digest,
        byte_size=len(data),
        storage_path=str(dest),
        source=source,
        doc_id=doc_id,
    )

    try:
        text = extract_text(data, fname)
        chunks = chunk_text(text)
        if not chunks:
            raise DocumentIngestError("EMPTY_TEXT", "No text content to index.")
        store.replace_chunks(doc.id, chunks)
        store.set_status(doc.id, "ready")
        doc.status = "ready"
        doc.error = None
    except DocumentIngestError as e:
        store.set_status(doc.id, "failed", e.message)
        doc.status = "failed"
        doc.error = e.message
        raise
    except Exception as e:
        msg = "Document processing failed."
        logger.exception("document ingest failed")
        store.set_status(doc.id, "failed", msg)
        doc.status = "failed"
        doc.error = msg
        raise DocumentIngestError("PROCESSING_FAILED", msg) from e
    return doc


def ingest_inbox(
    *,
    owner_username: str,
    visibility: str = "org",
) -> dict:
    """Scan MISA_DOCUMENTS_INGEST_DIR for new files and ingest them."""
    ensure_dirs()
    inbox = Path(config.DOCUMENTS_INGEST_DIR).resolve()
    processed = inbox / "processed"
    failed = inbox / "failed"
    results = {"ingested": [], "duplicates": [], "failed": [], "skipped": []}

    for path in sorted(inbox.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            results["skipped"].append(path.name)
            continue
        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results["skipped"].append(path.name)
            continue
        try:
            data = path.read_bytes()
            digest = sha256_bytes(data)
            store = get_document_store()
            dup = store.find_duplicate(
                sha256=digest, visibility=visibility, owner_username=owner_username
            )
            if dup is not None:
                results["duplicates"].append({"filename": path.name, "id": dup.id})
                shutil.move(str(path), str(processed / path.name))
                continue
            doc = ingest_bytes(
                data,
                filename=path.name,
                owner_username=owner_username,
                visibility=visibility,
                source="ingest",
            )
            results["ingested"].append(doc.to_dict())
            shutil.move(str(path), str(processed / path.name))
        except DocumentIngestError as e:
            results["failed"].append({"filename": path.name, "error": e.message})
            try:
                shutil.move(str(path), str(failed / path.name))
            except Exception:
                pass
        except Exception as e:
            results["failed"].append({"filename": path.name, "error": str(e)[:200]})
            try:
                shutil.move(str(path), str(failed / path.name))
            except Exception:
                pass
    return results


def compose_document_answer(
    question: str,
    hits,
    *,
    section_heading: str | None = None,
    footer: str | None = None,
) -> dict:
    """Build a document-sourced answer with provenance.

    Prefers an LLM briefing (STYLE_GUIDE + [doc:] citations). Falls back to
    a deterministic chunk listing when no model client is available.
    """
    if not hits:
        return {"answer": "", "doc_sources": [], "enough": False}

    min_score = config.DOCUMENTS_RETRIEVAL_MIN_SCORE
    strong = [h for h in hits if h.score >= min_score]
    if not strong:
        # Memory backend scores are fractional token overlap — accept any hit.
        strong = list(hits)
    if not strong:
        return {"answer": "", "doc_sources": [], "enough": False}

    enough = any(len(h.text) > 40 for h in strong) or sum(h.score for h in strong) >= min_score
    sources = []
    evidence_blocks = []
    for h in strong:
        cite = f"[doc:{h.filename}#{h.chunk_index}]"
        evidence_blocks.append(f"{cite}\n{h.text.strip()}")
        sources.append({
            "title": h.filename,
            "url": f"doc://{h.document_id}#{h.chunk_index}",
            "snippet": h.text[:240],
            "document_id": h.document_id,
            "chunk_index": h.chunk_index,
            "score": h.score,
            "type": "document",
        })

    synthesized = _synthesize_document_briefing(question, evidence_blocks)
    if synthesized:
        return {
            "answer": synthesized,
            "doc_sources": sources,
            "enough": enough,
        }

    if section_heading is None:
        section_heading = "## From your documents"
    if footer is None:
        footer = (
            "_Sources: document library. "
            "Uploaded documents take priority over the database and the internet._"
        )
    lines = [section_heading, ""]
    for block in evidence_blocks:
        lines.append(block)
        lines.append("")
    if footer:
        lines.append(footer)
    return {
        "answer": "\n".join(lines).strip(),
        "doc_sources": sources,
        "enough": enough,
    }


def _synthesize_document_briefing(question: str, evidence_blocks: list[str]) -> str:
    """Compose a STYLE_GUIDE briefing from document excerpts. Empty on failure."""
    try:
        from app.config import openai_max_completion_tokens_kw
        from app.services.style_guide import STYLE_GUIDE_PROMPT, HEADERS
        from app.services.llm_residency import resolve_data_completion_client
        client, model = resolve_data_completion_client()
        if client is None or not evidence_blocks:
            return ""
        evidence = "\n\n---\n\n".join(evidence_blocks[:8])
        prompt = (
            STYLE_GUIDE_PROMPT
            + "\n\nYou are composing an answer from UPLOADED DOCUMENT excerpts only.\n"
            f"Use the heading `{HEADERS['from_documents']}` first.\n"
            "Rules:\n"
            "- Answer the user question directly in executive prose (not a paste of chunks).\n"
            "- Cite every factual claim with the exact [doc:filename#N] markers from the excerpts.\n"
            "- If excerpts are insufficient, say what is missing — do not invent.\n"
            "- Do not cite the web or MISA database here.\n"
            "- End with `_Sources: document library._`\n\n"
            f"QUESTION:\n{question.strip()}\n\n"
            f"EXCERPTS:\n{evidence}\n"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            **openai_max_completion_tokens_kw(),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"Document briefing synthesis failed: {e}")
        return ""


def _curate_web_section(question: str, web_results: list) -> tuple[str, list]:
    """Turn raw web hits into a curated ## From the web section with [web:N]."""
    web_sources = []
    for i, r in enumerate(web_results, start=1):
        title = (r.get("title") or "(untitled)").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        if url:
            web_sources.append({
                "title": title,
                "url": url,
                "snippet": snippet[:600],
                "type": "web",
            })
    if not web_sources:
        return "", []

    try:
        from app.database import get_openai_client
        from app.config import OPENAI_MODEL, openai_max_completion_tokens_kw
        from app.services.style_guide import STYLE_GUIDE_PROMPT, HEADERS
        from app.services import web_search
        client = get_openai_client()
        if client is None:
            raise RuntimeError("no client")
        evidence = web_search.format_for_prompt(web_results)
        prompt = (
            STYLE_GUIDE_PROMPT
            + f"\nWrite ONLY a `{HEADERS['from_web']}` section.\n"
            "3–6 bullets. Every claim ends with [web:N] matching the evidence.\n"
            "No invented names/dates. Prefer recent public reporting.\n"
            "Documents remain the org source of truth — do not override them.\n\n"
            f"QUESTION:\n{question.strip()}\n\nWEB EVIDENCE:\n{evidence}\n"
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            **openai_max_completion_tokens_kw(),
        )
        section = (resp.choices[0].message.content or "").strip()
        if section and not section.startswith("##"):
            section = f"{HEADERS['from_web']}\n\n{section}"
        if section:
            return section, web_sources
    except Exception as e:
        logger.warning(f"Web section curation failed: {e}")

    # Deterministic fallback
    from app.services.style_guide import HEADERS
    lines = [HEADERS["from_web"], ""]
    for i, s in enumerate(web_sources, start=1):
        lines.append(f"- **{s['title']}** [web:{i}]")
        if s.get("snippet"):
            lines.append(f"  {s['snippet'][:240]}")
        lines.append("")
    lines.append("_Web sources complement the document library; they do not override it._")
    return "\n".join(lines).strip(), web_sources


_DOCS_ONLY_RE = re.compile(
    r"(?i)\b("
    r"only\s+from\s+(the\s+|my\s+|this\s+)?(document|doc|file|upload|pdf|library)"
    r"|from\s+(the\s+|my\s+)?(uploaded\s+)?(document|file|pdf)s?\s+only"
    r"|don'?t\s+(use|search|consult)\s+(the\s+)?(internet|web|online)"
    r"|do\s+not\s+(use|search|consult)\s+(the\s+)?(internet|web|online)"
    r"|without\s+(using\s+)?(the\s+)?(internet|web)"
    r"|documents?\s+only"
    r")\b"
)

_WEB_PREFERRED_RE = re.compile(
    r"(?i)\b("
    r"latest\s+news|recent\s+news|current\s+(news|reporting|status)"
    r"|what'?s\s+(reported|public|online)|from\s+the\s+(internet|web)"
    r"|search\s+the\s+(internet|web)|live\s+web"
    r")\b"
)


def wants_docs_only(question: str) -> bool:
    """True when the user explicitly asks to stay on uploaded documents."""
    return bool(question and _DOCS_ONLY_RE.search(question))


def wants_web_preferred(question: str) -> bool:
    """True when the user explicitly asks for live/public web context."""
    return bool(question and _WEB_PREFERRED_RE.search(question))


def should_augment_docs_with_web(question: str) -> bool:
    """Whether a strong document hit should also consult the internet."""
    mode = getattr(config, "DOCUMENTS_WEB_MODE", "hybrid")
    if mode == "docs_only":
        return False
    if mode == "docs_first":
        return wants_web_preferred(question)
    if wants_docs_only(question):
        return False
    return True


def compose_hybrid_document_web_answer(
    question: str,
    hits,
    web_results: list | None = None,
) -> dict:
    """Document-primary briefing optionally complemented with curated web.

    Returns doc_sources, web_sources, answer_source document|hybrid.
    Preserves full session/answer sequence — only the compose quality improves.
    """
    web_results = list(web_results or [])
    doc = compose_document_answer(question, hits)
    if not doc.get("enough"):
        return {
            "answer": "",
            "doc_sources": [],
            "web_sources": [],
            "enough": False,
            "answer_source": "document",
        }

    if not web_results:
        return {
            **doc,
            "web_sources": [],
            "answer_source": "document",
        }

    web_section, web_sources = _curate_web_section(question, web_results)
    conflict = (
        "\n\n_Uploaded documents are the organisation's source of truth. "
        "Where the web disagrees, prefer the document; treat web material as "
        "recent public reporting._"
    )
    answer = doc["answer"].rstrip()
    if web_section:
        answer = f"{answer}\n\n{web_section}{conflict}".strip()
    return {
        "answer": answer,
        "doc_sources": doc.get("doc_sources") or [],
        "web_sources": web_sources,
        "enough": True,
        "answer_source": "hybrid",
    }
