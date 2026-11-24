-- Vertex Gemini embeddings use an explicit 768-dimensional output.
-- Existing 1536-dimensional vectors are intentionally invalidated because
-- embeddings from different model spaces/dimensions must never be mixed.

DROP INDEX IF EXISTS idx_job_embeddings_hnsw;
TRUNCATE TABLE job_embeddings;
ALTER TABLE job_embeddings
  ALTER COLUMN vector TYPE VECTOR(768);
CREATE INDEX idx_job_embeddings_hnsw
  ON job_embeddings USING hnsw (vector vector_cosine_ops);
