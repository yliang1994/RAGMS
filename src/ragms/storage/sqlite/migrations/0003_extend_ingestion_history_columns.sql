ALTER TABLE ingestion_history ADD COLUMN last_error TEXT;
ALTER TABLE ingestion_history ADD COLUMN started_at TEXT;
ALTER TABLE ingestion_history ADD COLUMN completed_at TEXT;
ALTER TABLE ingestion_history ADD COLUMN config_version TEXT;
