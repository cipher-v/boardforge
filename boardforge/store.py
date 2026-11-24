from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from boardforge.matching import job_text
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

CREATE TABLE IF NOT EXISTS discovery_requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  query TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL,
  report TEXT,
  requested_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  next_attempt_at TEXT,
  last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_discovery_fingerprint
  ON discovery_requests(fingerprint, requested_at);

CREATE TABLE IF NOT EXISTS discovered_sources (
  source_kind TEXT NOT NULL,
  source_key TEXT NOT NULL,
  board_url TEXT NOT NULL,
  discovery_query TEXT NOT NULL DEFAULT '',
  first_discovered TEXT NOT NULL,
  last_discovered TEXT NOT NULL,
  last_synced TEXT,
  last_status TEXT NOT NULL DEFAULT 'discovered',
  PRIMARY KEY(source_kind, source_key)
);
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
            discovery_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(discovery_requests)")
            }
            if "started_at" not in discovery_columns:
                connection.execute(
                    "ALTER TABLE discovery_requests ADD COLUMN started_at TEXT"
                )
            for statement in (
                "ALTER TABLE discovery_requests ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE discovery_requests ADD COLUMN next_attempt_at TEXT",
                "ALTER TABLE discovery_requests ADD COLUMN last_error TEXT",
            ):
                column = statement.split("ADD COLUMN ", 1)[1].split()[0]
                if column not in discovery_columns:
                    connection.execute(statement)

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

    def ping(self) -> bool:
        with self.connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

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
            rows = connection.execute(
                "SELECT payload,first_seen,last_seen FROM jobs WHERE status='active'"
            ).fetchall()
        return [
            Job.model_validate_json(row["payload"]).model_copy(update={
                "first_seen_at": datetime.fromisoformat(row["first_seen"]),
                "last_seen_at": datetime.fromisoformat(row["last_seen"]),
            })
            for row in rows
        ]

    def candidate_jobs(
        self,
        query: str,
        *,
        limit: int = 250,
        query_vector: list[float] | None = None,
        embedding_model: str = "",
        preferences: Preferences | None = None,
    ) -> list[Job]:
        """SQLite fallback keeps exact behavior for local demos and tests.

        PostgreSQL overrides this with indexed full-text/vector retrieval.
        """
        del query, query_vector, embedding_model, preferences
        return self.active_jobs()[:max(1, limit)]

    def embedding_candidates(
        self,
        model: str,
        *,
        limit: int = 100,
        countries: list[str] | None = None,
    ) -> list[Job]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT j.payload,j.first_seen,j.last_seen,e.content_hash AS embedded_hash
                   FROM jobs j LEFT JOIN job_embeddings e
                     ON e.stable_id=j.stable_id AND e.model=?
                   WHERE j.status='active' ORDER BY j.last_seen DESC""",
                (model,),
            ).fetchall()
        jobs: list[Job] = []
        for row in rows:
            job = Job.model_validate_json(row["payload"]).model_copy(update={
                "first_seen_at": datetime.fromisoformat(row["first_seen"]),
                "last_seen_at": datetime.fromisoformat(row["last_seen"]),
            })
            digest = hashlib.sha256(job_text(job)[:6000].encode("utf-8")).hexdigest()
            if row["embedded_hash"] != digest:
                jobs.append(job)
        if countries:
            allowed = {country.casefold() for country in countries}
            jobs = [job for job in jobs if job.country.casefold() in allowed]
        return jobs[:max(1, limit)]

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

    def embedding_status(self, model: str) -> dict:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT j.payload,e.content_hash AS embedded_hash
                   FROM jobs j LEFT JOIN job_embeddings e
                     ON e.stable_id=j.stable_id AND e.model=?
                   WHERE j.status='active'""",
                (model,),
            ).fetchall()
        total = len(rows)
        embedded = sum(
            row["embedded_hash"]
            == hashlib.sha256(
                job_text(Job.model_validate_json(row["payload"]))[:6000].encode("utf-8")
            ).hexdigest()
            for row in rows
        )
        return {
            "model": model,
            "active_jobs": total,
            "embedded_jobs": embedded,
            "backlog": max(0, total - embedded),
            "coverage": round(embedded / total, 4) if total else 1.0,
        }

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

    def delete_profile(self, profile_id: str) -> None:
        """Delete candidate-owned state. The shared public job index is retained."""
        with self.connect() as connection:
            connection.execute("DELETE FROM candidate_profiles WHERE profile_id=?", (profile_id,))
            connection.execute("DELETE FROM profiles WHERE profile_id=?", (profile_id,))
            connection.execute("DELETE FROM feedback WHERE profile_id=?", (profile_id,))
            connection.execute("DELETE FROM discovery_requests WHERE profile_id=?", (profile_id,))

    def queue_discovery(
        self,
        profile_id: str,
        fingerprint: str,
        query: str,
        *,
        ttl_seconds: int = 900,
    ) -> dict:
        now = datetime.now(timezone.utc)
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM discovery_requests WHERE fingerprint=?
                   ORDER BY id DESC LIMIT 1""",
                (fingerprint,),
            ).fetchone()
            if row is not None:
                age = (now - datetime.fromisoformat(row["requested_at"])).total_seconds()
                if row["status"] in {"queued", "running"} or (
                    row["status"] == "completed" and age < ttl_seconds
                ):
                    return self._discovery_row(row, cached=True)
            cursor = connection.execute(
                """INSERT INTO discovery_requests
                   (profile_id,fingerprint,query,status,requested_at)
                   VALUES(?,?,?,'queued',?)""",
                (profile_id, fingerprint, query, now.isoformat()),
            )
            request_id = int(cursor.lastrowid)
        return {
            "request_id": request_id,
            "status": "queued",
            "cached": False,
            "report": None,
        }

    @staticmethod
    def _discovery_row(row: sqlite3.Row, *, cached: bool = False) -> dict:
        return {
            "request_id": int(row["id"]),
            "status": row["status"],
            "cached": cached,
            "report": json.loads(row["report"]) if row["report"] else None,
            "attempt_count": int(row["attempt_count"]),
            "last_error": row["last_error"],
        }

    def discovery_request(self, request_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM discovery_requests WHERE id=?", (request_id,)
            ).fetchone()
        return self._discovery_row(row) if row else None

    def start_discovery(self, request_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE discovery_requests SET status='running',started_at=?,
                   attempt_count=attempt_count+1,next_attempt_at=NULL
                   WHERE id=? AND status='queued'""",
                (_utcnow(), request_id),
            )
        return cursor.rowcount == 1

    def pending_discoveries(self, limit: int = 10) -> list[dict]:
        now = _utcnow()
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id,profile_id,query FROM discovery_requests
                   WHERE status='queued' AND (next_attempt_at IS NULL OR next_attempt_at<=?)
                   ORDER BY id LIMIT ?""",
                (now, max(1, min(limit, 100))),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_discoveries(self, limit: int = 10) -> list[dict]:
        now = _utcnow()
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id,profile_id,query FROM discovery_requests
                   WHERE status='queued' AND (next_attempt_at IS NULL OR next_attempt_at<=?)
                   ORDER BY id LIMIT ?""",
                (now, max(1, min(limit, 100))),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """UPDATE discovery_requests SET status='running',started_at=?,
                       attempt_count=attempt_count+1,next_attempt_at=NULL
                       WHERE id=? AND status='queued'""",
                    (now, row["id"]),
                )
        return [dict(row) for row in rows]

    def requeue_stale_discoveries(self, stale_seconds: int = 900) -> int:
        cutoff = datetime.now(timezone.utc).timestamp() - max(60, stale_seconds)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id,started_at FROM discovery_requests WHERE status='running'"
            ).fetchall()
            stale_ids = [
                row["id"] for row in rows
                if row["started_at"] is not None
                and datetime.fromisoformat(row["started_at"]).timestamp() < cutoff
            ]
            if stale_ids:
                connection.executemany(
                    """UPDATE discovery_requests SET status='queued',started_at=NULL
                       WHERE id=?""",
                    [(request_id,) for request_id in stale_ids],
                )
        return len(stale_ids)

    def finish_discovery(self, request_id: int, report: dict, *, failed: bool = False) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE discovery_requests SET status=?,report=?,completed_at=? WHERE id=?""",
                ("failed" if failed else "completed", json.dumps(report), _utcnow(), request_id),
            )

    def retry_discovery(self, request_id: int, error: str, *, max_attempts: int = 3) -> str:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT attempt_count FROM discovery_requests WHERE id=?", (request_id,)
            ).fetchone()
            if row is None:
                return "missing"
            attempts = int(row["attempt_count"])
            if attempts >= max_attempts:
                connection.execute(
                    """UPDATE discovery_requests SET status='failed',last_error=?,completed_at=?
                       WHERE id=?""",
                    (error[:1000], _utcnow(), request_id),
                )
                return "failed"
            delay = min(300, 5 * (2 ** max(0, attempts - 1)))
            next_attempt = datetime.now(timezone.utc) + timedelta(seconds=delay)
            connection.execute(
                """UPDATE discovery_requests SET status='queued',started_at=NULL,
                   next_attempt_at=?,last_error=? WHERE id=?""",
                (next_attempt.isoformat(), error[:1000], request_id),
            )
        return "queued"

    def record_discovered_source(
        self,
        source_kind: str,
        source_key: str,
        board_url: str,
        query: str,
        *,
        synced: bool = False,
        status: str = "discovered",
    ) -> None:
        now = _utcnow()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO discovered_sources
                   (source_kind,source_key,board_url,discovery_query,first_discovered,
                    last_discovered,last_synced,last_status)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(source_kind,source_key) DO UPDATE SET
                     board_url=excluded.board_url,
                     discovery_query=excluded.discovery_query,
                     last_discovered=excluded.last_discovered,
                     last_synced=COALESCE(excluded.last_synced,discovered_sources.last_synced),
                     last_status=excluded.last_status""",
                (
                    source_kind, source_key, board_url, query, now, now,
                    now if synced else None, status,
                ),
            )

    def discovered_sources(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT source_kind,source_key,board_url,discovery_query,
                          first_discovered,last_discovered,last_synced,last_status
                   FROM discovered_sources ORDER BY source_kind,source_key"""
            ).fetchall()
        return [dict(row) for row in rows]

    def sources_due(self, *, interval_seconds: int = 3600, limit: int = 25) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(60, interval_seconds))
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT source_kind,source_key,board_url,last_synced,last_status
                   FROM discovered_sources
                   WHERE last_synced IS NULL OR last_synced<=?
                   ORDER BY COALESCE(last_synced,first_discovered) LIMIT ?""",
                (cutoff.isoformat(), max(1, min(limit, 500))),
            ).fetchall()
        return [dict(row) for row in rows]

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
