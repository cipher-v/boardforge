CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS embedding_hash TEXT;
UPDATE jobs
SET embedding_hash = encode(digest(substring(search_document FROM 1 FOR 6000), 'sha256'), 'hex')
WHERE embedding_hash IS NULL;
ALTER TABLE jobs ALTER COLUMN embedding_hash SET NOT NULL;
