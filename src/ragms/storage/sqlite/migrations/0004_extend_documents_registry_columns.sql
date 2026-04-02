ALTER TABLE documents ADD COLUMN failure_reason TEXT;
ALTER TABLE documents ADD COLUMN last_ingested_at TEXT;
ALTER TABLE documents ADD COLUMN version INTEGER NOT NULL DEFAULT 1;
