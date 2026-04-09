CREATE TABLE IF NOT EXISTS evaluations (
    run_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    dataset_name TEXT,
    dataset_version TEXT,
    backend_set TEXT NOT NULL DEFAULT '[]',
    baseline_scope TEXT,
    config_snapshot TEXT NOT NULL DEFAULT '{}',
    aggregate_metrics TEXT NOT NULL DEFAULT '{}',
    quality_gate_status TEXT,
    sample_count INTEGER NOT NULL DEFAULT 0,
    failed_sample_count INTEGER NOT NULL DEFAULT 0,
    report_path TEXT,
    artifacts TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_evaluations_collection
ON evaluations(collection);

CREATE INDEX IF NOT EXISTS idx_evaluations_dataset_version
ON evaluations(dataset_version);
