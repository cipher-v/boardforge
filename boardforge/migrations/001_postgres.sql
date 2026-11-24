CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
  stable_id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL,
  source_key TEXT NOT NULL,
  external_id TEXT NOT NULL,
  payload JSONB NOT NULL,
  search_document TEXT NOT NULL,
  search_vector TSVECTOR GENERATED ALWAYS AS
    (to_tsvector('english', search_document)) STORED,
  content_hash TEXT NOT NULL,
  first_seen TIMESTAMPTZ NOT NULL,
  last_seen TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  missing_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source_kind, source_key);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_search_vector ON jobs USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_jobs_last_seen ON jobs(last_seen DESC);

CREATE TABLE IF NOT EXISTS job_events (
  id BIGSERIAL PRIMARY KEY,
  stable_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  observed_at TIMESTAMPTZ NOT NULL,
  previous_payload JSONB,
  current_payload JSONB,
  sync_run_id BIGINT
);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(stable_id, observed_at);

CREATE TABLE IF NOT EXISTS sync_runs (
  id BIGSERIAL PRIMARY KEY,
  source_kind TEXT NOT NULL,
  source_key TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ,
  status TEXT NOT NULL,
  fetched INTEGER NOT NULL DEFAULT 0,
  inserted INTEGER NOT NULL DEFAULT 0,
  updated INTEGER NOT NULL DEFAULT 0,
  unchanged INTEGER NOT NULL DEFAULT 0,
  missing INTEGER NOT NULL DEFAULT 0,
  closed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS job_embeddings (
  stable_id TEXT NOT NULL,
  model TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  vector VECTOR(1536) NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY(stable_id, model)
);
CREATE INDEX IF NOT EXISTS idx_job_embeddings_hnsw
  ON job_embeddings USING hnsw (vector vector_cosine_ops);

CREATE TABLE IF NOT EXISTS profiles (
  profile_id TEXT PRIMARY KEY,
  preferences JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_profiles (
  profile_id TEXT PRIMARY KEY,
  profile JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
  id BIGSERIAL PRIMARY KEY,
  profile_id TEXT NOT NULL,
  job_id TEXT NOT NULL,
  verdict TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_profile ON feedback(profile_id, created_at);

CREATE TABLE IF NOT EXISTS discovery_requests (
  id BIGSERIAL PRIMARY KEY,
  profile_id TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  query TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL,
  report JSONB,
  requested_at TIMESTAMPTZ NOT NULL,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_discovery_fingerprint
  ON discovery_requests(fingerprint, requested_at DESC);
CREATE INDEX IF NOT EXISTS idx_discovery_queue
  ON discovery_requests(status, id);

CREATE TABLE IF NOT EXISTS discovered_sources (
  source_kind TEXT NOT NULL,
  source_key TEXT NOT NULL,
  board_url TEXT NOT NULL,
  discovery_query TEXT NOT NULL DEFAULT '',
  first_discovered TIMESTAMPTZ NOT NULL,
  last_discovered TIMESTAMPTZ NOT NULL,
  last_synced TIMESTAMPTZ,
  last_status TEXT NOT NULL DEFAULT 'discovered',
  PRIMARY KEY(source_kind, source_key)
);
