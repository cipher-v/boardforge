CREATE TABLE IF NOT EXISTS query_embeddings (
  cache_key TEXT PRIMARY KEY,
  normalized_query TEXT NOT NULL,
  model TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  vector VECTOR(768) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  last_used_at TIMESTAMPTZ NOT NULL,
  hit_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_query_embeddings_model_used
  ON query_embeddings(model,last_used_at DESC);
