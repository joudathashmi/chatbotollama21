-- Document library schema (also auto-bootstrapped by app/services/document_store.py).
-- Safe to run manually against the MISA Postgres if preferred.

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    owner_username TEXT NOT NULL,
    visibility TEXT NOT NULL CHECK (visibility IN ('private', 'org')),
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
    sha256 TEXT NOT NULL,
    byte_size BIGINT NOT NULL DEFAULT 0,
    storage_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'ready', 'failed')),
    source TEXT NOT NULL CHECK (source IN ('upload', 'ingest')),
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_sha_private_uidx
    ON documents (sha256, owner_username) WHERE visibility = 'private';
CREATE UNIQUE INDEX IF NOT EXISTS documents_sha_org_uidx
    ON documents (sha256) WHERE visibility = 'org';
CREATE INDEX IF NOT EXISTS documents_owner_idx ON documents (owner_username);
CREATE INDEX IF NOT EXISTS documents_visibility_idx ON documents (visibility);

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx
    ON document_chunks USING GIN (tsv);
