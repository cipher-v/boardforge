CREATE TABLE IF NOT EXISTS beta_daily_usage (
    day date NOT NULL,
    profile_id text NOT NULL,
    operation text NOT NULL,
    count integer NOT NULL CHECK (count >= 0),
    PRIMARY KEY (day, profile_id, operation)
);
