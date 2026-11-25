CREATE TABLE IF NOT EXISTS llm_usage_reservations (
  reservation_id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  reserved_tokens INTEGER NOT NULL,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  thinking_tokens INTEGER NOT NULL DEFAULT 0,
  total_tokens INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL CHECK (status IN ('reserved', 'completed', 'failed')),
  estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_provider_model_created
  ON llm_usage_reservations(provider, model, created_at);
