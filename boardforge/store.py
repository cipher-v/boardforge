from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from boardforge.models import FeedbackRequest, Job, Preferences, SyncSummary


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
  status TEXT NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source_kind, source_key);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS profiles (
  profile_id TEXT PRIMARY KEY,
  preferences TEXT NOT NULL,
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
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def upsert_source_jobs(self, source_kind: str, source_key: str, jobs: list[Job]) -> SyncSummary:
        now = _utcnow()
        inserted = updated = unchanged = 0
        seen = {j.stable_id for j in jobs}

        with self.connect() as con:
            existing = {
                row["stable_id"]: row
                for row in con.execute(
                    "SELECT stable_id, content_hash, status FROM jobs WHERE source_kind=? AND source_key=?",
                    (source_kind, source_key),
                )
            }

            for job in jobs:
                digest = _digest(job)
                payload = job.model_dump_json(exclude_none=True)
                row = existing.get(job.stable_id)
                if row is None:
                    con.execute(
                        """INSERT INTO jobs
                           (stable_id,source_kind,source_key,external_id,payload,content_hash,first_seen,last_seen,status)
                           VALUES(?,?,?,?,?,?,?,?, 'active')""",
                        (job.stable_id, source_kind, source_key, job.external_id, payload, digest, now, now),
                    )
                    inserted += 1
                elif row["content_hash"] != digest or row["status"] != "active":
                    con.execute(
                        "UPDATE jobs SET payload=?, content_hash=?, last_seen=?, status='active' WHERE stable_id=?",
                        (payload, digest, now, job.stable_id),
                    )
                    updated += 1
                else:
                    con.execute("UPDATE jobs SET last_seen=? WHERE stable_id=?", (now, job.stable_id))
                    unchanged += 1

            closed = 0
            for stable_id, row in existing.items():
                if stable_id not in seen and row["status"] == "active":
                    con.execute("UPDATE jobs SET status='closed', last_seen=? WHERE stable_id=?", (now, stable_id))
                    closed += 1

        return SyncSummary(
            source_kind=source_kind,
            source_key=source_key,
            fetched=len(jobs),
            inserted=inserted,
            updated=updated,
            unchanged=unchanged,
            closed=closed,
        )

    def active_jobs(self) -> list[Job]:
        with self.connect() as con:
            rows = con.execute("SELECT payload FROM jobs WHERE status='active'").fetchall()
        return [Job.model_validate_json(r["payload"]) for r in rows]

    def job_status(self, stable_id: str) -> str | None:
        with self.connect() as con:
            row = con.execute("SELECT status FROM jobs WHERE stable_id=?", (stable_id,)).fetchone()
        return row["status"] if row else None

    def save_preferences(self, profile_id: str, preferences: Preferences) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO profiles(profile_id,preferences,updated_at) VALUES(?,?,?)
                   ON CONFLICT(profile_id) DO UPDATE SET preferences=excluded.preferences, updated_at=excluded.updated_at""",
                (profile_id, preferences.model_dump_json(), _utcnow()),
            )

    def load_preferences(self, profile_id: str) -> Preferences:
        with self.connect() as con:
            row = con.execute("SELECT preferences FROM profiles WHERE profile_id=?", (profile_id,)).fetchone()
        return Preferences.model_validate_json(row["preferences"]) if row else Preferences()

    def record_feedback(self, req: FeedbackRequest) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO feedback(profile_id,job_id,verdict,note,created_at) VALUES(?,?,?,?,?)",
                (req.profile_id, req.job_id, req.verdict, req.note, _utcnow()),
            )

    def rejected_job_ids(self, profile_id: str) -> set[str]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT DISTINCT job_id FROM feedback WHERE profile_id=? AND verdict IN ('reject','ignore')",
                (profile_id,),
            ).fetchall()
        return {r["job_id"] for r in rows}
