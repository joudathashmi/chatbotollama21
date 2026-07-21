-- Chat session upgrades: pin, archive, state card, rolling summary.

ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;
ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS state_json JSONB;
ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS chat_sessions_owner_pinned_idx
    ON chat_sessions (owner_username, pinned DESC, updated_at DESC);
CREATE INDEX IF NOT EXISTS chat_sessions_owner_archived_idx
    ON chat_sessions (owner_username, archived_at);
