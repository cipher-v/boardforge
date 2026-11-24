from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from boardforge.models import CandidateProfile, FeedbackRequest, Job, Preferences, SyncSummary


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS jobs (
  stable_id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL,
  source_key TEXT NOT NULL,
  external_id TEXT NOT NULL,
  payload TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  missing_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source_kind, source_key);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  stable_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  previous_payload TEXT,
  current_payload TEXT,
  sync_run_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(stable_id, observed_at);

CREATE TABLE IF NOT EXISTS sync_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_kind TEXT NOT NULL,
  source_key TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
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
  vector TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(stable_id, model)
);

CREATE TABLE IF NOT EXISTS profiles (
  profile_id TEXT PRIMARY KEY,
  preferences TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_profiles (
  profile_id TEXT PRIMARY KEY,
  profile TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id TEXT NOT NULL,
  job_id TEXT NOT NULL,
  verdict TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_profile ON feedback(profile_id, created_at);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(job: Job) -> str:
    raw = job.model_dump_json(exclude_none=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("BOARDFORGE_DB_PATH", "./boardforge.db")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "missing_count" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN missing_count INTEGER NOT NULL DEFAULT 0"
                )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _record_event(
        connection: sqlite3.Connection,
        stable_id: str,
        event_type: str,
        observed_at: str,
        run_id: int,
        previous_payload: str | None = None,
        current_payload: str | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO job_events
               (stable_id,event_type,observed_at,previous_payload,current_payload,sync_run_id)
               VALUES(?,?,?,?,?,?)""",
            (stable_id, event_type, observed_at, previous_payload, current_payload, run_id),
        )

    def upsert_source_jobs(self, source_kind: str, source_key: str, jobs: list[Job]) -> SyncSummary:
        now = _utcnow()
        inserted = updated = unchanged = missing = closed = 0
        seen = {job.stable_id for job in jobs}

        with self.connect() as connection:
            existing = {
                row["stable_id"]: row
                for row in connection.execute(
                    """SELECT stable_id,content_hash,status,payload,missing_count
                       FROM jobs WHERE source_kind=? AND source_key=?""",
                    (source_kind, source_key),
                )
            }
            active_count = sum(row["status"] == "active" for row in existing.values())
            if active_count and not jobs:
                raise ValueError("refusing to treat an empty fetch as a successful source sync")
            if active_count >= 10 and len(jobs) < max(1, int(active_count * 0.25)):
                raise ValueError(
                    f"refusing abnormal source shrink: fetched {len(jobs)} from {active_count} active jobs"
                )

            cursor = connection.execute(
                """INSERT INTO sync_runs(source_kind,source_key,started_at,status,fetched)
                   VALUES(?,?,?,'running',?)""",
                (source_kind, source_key, now, len(jobs)),
            )
            run_id = int(cursor.lastrowid)

            for job in jobs:
                digest = _digest(job)
                payload = job.model_dump_json(exclude_none=True)
                row = existing.get(job.stable_id)
                if row is None:
                    connection.execute(
                        """INSERT INTO jobs
                           (stable_id,source_kind,source_key,external_id,payload,content_hash,
                            first_seen,last_seen,status,missing_count)
                           VALUES(?,?,?,?,?,?,?,?, 'active',0)""",
                        (
                            job.stable_id, source_kind, source_key, job.external_id,
                            payload, digest, now, now,
                        ),
                    )
                    inserted += 1
                    self._record_event(connection, job.stable_id, "discovered", now, run_id, current_payload=payload)
                elif row["content_hash"] != digest or row["status"] != "active":
                    event_type = "reopened" if row["status"] != "active" else "updated"
                    connection.execute(
                        """UPDATE jobs SET payload=?,content_hash=?,last_seen=?,status='active',missing_count=0
                           WHERE stable_id=?""",
                        (payload, digest, now, job.stable_id),
                    )
                    updated += 1
                    self._record_event(
                        connection, job.stable_id, event_type, now, run_id,
                        previous_payload=row["payload"], current_payload=payload,
                    )
                else:
                    connection.execute(
                        "UPDATE jobs SET last_seen=?,missing_count=0 WHERE stable_id=?",
                        (now, job.stable_id),
                    )
                    unchanged += 1

            for stable_id, row in existing.items():
                if stable_id in seen or row["status"] != "active":
                    continue
                missing += 1
                new_missing_count = int(row["missing_count"]) + 1
                if new_missing_count >= 2:
                    connection.execute(
                        "UPDATE jobs SET status='closed',missing_count=? WHERE stable_id=?",
                        (new_missing_count, stable_id),
                    )
                    closed += 1
                    self._record_event(
                        connection, stable_id, "closed", now, run_id,
                        previous_payload=row["payload"],
                    )
                else:
                    connection.execute(
                        "UPDATE jobs SET missing_count=? WHERE stable_id=?",
                        (new_missing_count, stable_id),
                    )

            connection.execute(
                """UPDATE sync_runs SET completed_at=?,status='succeeded',inserted=?,updated=?,
                   unchanged=?,missing=?,closed=? WHERE id=?""",
                (now, inserted, updated, unchanged, missing, closed, run_id),
            )

        return SyncSummary(
            source_kind=source_kind,
            source_key=source_key,
            fetched=len(jobs),
            inserted=inserted,
            updated=updated,
            unchanged=unchanged,
            missing=missing,
            closed=closed,
            run_id=run_id,
        )

    def active_jobs(self) -> list[Job]:
        with self.connect() as connection:
            rows = connection.execute("SELECT payload FROM jobs WHERE status='active'").fetchall()
        return [Job.model_validate_json(row["payload"]) for row in rows]

    def job_status(self, stable_id: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE stable_id=?", (stable_id,)).fetchone()
        return row["status"] if row else None

    def job_events(self, stable_id: str) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT event_type FROM job_events WHERE stable_id=? ORDER BY id",
                (stable_id,),
            ).fetchall()
        return [row["event_type"] for row in rows]

    def load_embedding(self, stable_id: str, model: str, content_hash: str) -> list[float] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT vector FROM job_embeddings
                   WHERE stable_id=? AND model=? AND content_hash=?""",
                (stable_id, model, content_hash),
            ).fetchone()
        if row is None:
            return None
        import json
        return [float(value) for value in json.loads(row["vector"])]

    def save_embedding(
        self,
        stable_id: str,
        model: str,
        content_hash: str,
        vector: list[float],
    ) -> None:
        import json
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO job_embeddings(stable_id,model,content_hash,vector,updated_at)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(stable_id,model) DO UPDATE SET
                     content_hash=excluded.content_hash,
                     vector=excluded.vector,
                     updated_at=excluded.updated_at""",
                (stable_id, model, content_hash, json.dumps(vector), _utcnow()),
            )

    def save_preferences(self, profile_id: str, preferences: Preferences) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO profiles(profile_id,preferences,updated_at) VALUES(?,?,?)
                   ON CONFLICT(profile_id) DO UPDATE SET preferences=excluded.preferences,
                   updated_at=excluded.updated_at""",
                (profile_id, preferences.model_dump_json(), _utcnow()),
            )

    def load_preferences(self, profile_id: str) -> Preferences:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT preferences FROM profiles WHERE profile_id=?", (profile_id,)
            ).fetchone()
        return Preferences.model_validate_json(row["preferences"]) if row else Preferences()

    def save_candidate_profile(self, profile_id: str, profile: CandidateProfile) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO candidate_profiles(profile_id,profile,updated_at) VALUES(?,?,?)
                   ON CONFLICT(profile_id) DO UPDATE SET profile=excluded.profile,
                   updated_at=excluded.updated_at""",
                (profile_id, profile.model_dump_json(), _utcnow()),
            )

    def load_candidate_profile(self, profile_id: str) -> CandidateProfile | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT profile FROM candidate_profiles WHERE profile_id=?", (profile_id,)
            ).fetchone()
        return CandidateProfile.model_validate_json(row["profile"]) if row else None

    def record_feedback(self, request: FeedbackRequest) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO feedback(profile_id,job_id,verdict,note,created_at)
                   VALUES(?,?,?,?,?)""",
                (request.profile_id, request.job_id, request.verdict, request.note, _utcnow()),
            )

    def rejected_job_ids(self, profile_id: str) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT job_id FROM feedback
                   WHERE profile_id=? AND verdict IN ('reject','ignore')""",
                (profile_id,),
            ).fetchall()
        return {row["job_id"] for row in rows}
