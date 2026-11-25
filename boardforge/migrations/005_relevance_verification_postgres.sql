CREATE TABLE IF NOT EXISTS relevance_verifications (
  cache_key TEXT PRIMARY KEY,
  judgment JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_relevance_verifications_created
  ON relevance_verifications(created_at);
