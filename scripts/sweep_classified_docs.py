"""Retroactive classification sweep of the document library.

Re-screens every stored document with the same protective-marking detector
used at upload time and deletes any document that carries a Restricted,
Secret, Top Secret, Confidential, or Classified marking. Run after the
upload gate was introduced, or any time the marking patterns change.

    python scripts/sweep_classified_docs.py            # report only
    python scripts/sweep_classified_docs.py --delete   # delete flagged docs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.document_classification import find_classification_marking
from app.services.document_ingest import _deep_scan_text, extract_text
from app.services.document_store import get_document_store, _pg_conn, _row_to_doc


def _all_documents():
    store = get_document_store()
    store.ensure()
    if hasattr(store, "_state"):  # memory backend
        return list(getattr(store, "_state").docs.values())
    import psycopg2.extras
    with _pg_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM documents ORDER BY created_at")
        return [_row_to_doc(r) for r in cur.fetchall()]


def main() -> int:
    delete = "--delete" in sys.argv
    store = get_document_store()
    flagged, clean, unreadable = [], 0, []

    for doc in _all_documents():
        path = Path(doc.storage_path)
        if not path.is_file():
            unreadable.append((doc, "stored file missing"))
            continue
        data = path.read_bytes()
        try:
            text = extract_text(data, doc.filename)
        except Exception:
            text = ""
        text += _deep_scan_text(data, doc.filename)
        marking = find_classification_marking(text, doc.filename)
        if marking:
            flagged.append((doc, marking))
        else:
            clean += 1

    print(f"Scanned: {clean + len(flagged) + len(unreadable)} documents")
    print(f"Clean: {clean}")
    for doc, reason in unreadable:
        print(f"UNREADABLE {doc.id} {doc.filename}: {reason}")
    for doc, marking in flagged:
        action = "DELETED" if delete else "FLAGGED"
        if delete:
            store.delete(doc.id, doc.owner_username, is_admin=True)
        print(f"{action} {doc.id} {doc.filename} ({doc.owner_username}): {marking}")
    if flagged and not delete:
        print("\nRe-run with --delete to remove the flagged documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
