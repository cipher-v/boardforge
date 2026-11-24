from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pgvector import Vector
from pgvector.psycopg import register_vector
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from boardforge.matching import job_text
from boardforge.semantic import POSTGRES_EMBEDDING_DIMENSIONS
from boardforge.models import CandidateProfile, FeedbackRequest, Job, Preferences, SyncSummary


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _digest(job: Job) -> str:
    return hashlib.sha256(job.model_dump_json(exclude_none=True).encode("utf-8")).hexdigest()


def _embedding_digest(job: Job) -> str:
    return hashlib.sha256(job_text(job)[:6000].encode("utf-8")).hexdigest()


class PostgresStore:
    """Production store with FTS/pgvector retrieval and durable worker state."""

    def __init__(self, database_url: str, *, min_size: int = 1, max_size: int = 10):
        self.database_url = database_url
        # pgvector adapters can only be registered after the extension exists.
        with psycopg.connect(database_url, autocommit=True) as bootstrap:
            bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.pool = ConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            configure=self._configure,
            open=True,
        )
        self.migrate()

    @staticmethod
    def _configure(connection) -> None:
        register_vector(connection)

    @contextmanager
    def connect(self):
        with self.pool.connection() as connection:
            with connection.transaction():
                yield connection

    def close(self) -> None:
        self.pool.close()

    def ping(self) -> bool:
        with self.connect() as connection:
            return connection.execute("SELECT 1 AS ok").fetchone()["ok"] == 1

    def migrate(self) -> None:
        migrations = Path(__file__).resolve().parent / "migrations"
        with self.connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                     version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL)"""
            )
            applied = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
            }
            for path in sorted(migrations.glob("*_postgres.sql")):
                if path.name in applied:
                    continue
                connection.execute(path.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT INTO schema_migrations(version,applied_at) VALUES(%s,%s)",
                    (path.name, _utcnow()),
                )

    @staticmethod
    def _record_event(
        connection,
        stable_id: str,
        event_type: str,
        observed_at: datetime,
        run_id: int,
        previous_payload: dict | None = None,
        current_payload: dict | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO job_events
               (stable_id,event_type,observed_at,previous_payload,current_payload,sync_run_id)
               VALUES(%s,%s,%s,%s,%s,%s)""",
            (
                stable_id, event_type, observed_at,
                Jsonb(previous_payload) if previous_payload is not None else None,
                Jsonb(current_payload) if current_payload is not None else None,
                run_id,
            ),
        )

    def upsert_source_jobs(self, source_kind: str, source_key: str, jobs: list[Job]) -> SyncSummary:
        now = _utcnow()
        inserted = updated = unchanged = missing = closed = 0
        seen = {job.stable_id for job in jobs}
        with self.connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"{source_kind}:{source_key}",),
            )
            rows = connection.execute(
                """SELECT stable_id,content_hash,status,payload,missing_count
                   FROM jobs WHERE source_kind=%s AND source_key=%s FOR UPDATE""",
                (source_kind, source_key),
            ).fetchall()
            existing = {row["stable_id"]: row for row in rows}
            active_count = sum(row["status"] == "active" for row in rows)
            if active_count and not jobs:
                raise ValueError("refusing to treat an empty fetch as a successful source sync")
            if active_count >= 10 and len(jobs) < max(1, int(active_count * 0.25)):
                raise ValueError(
                    f"refusing abnormal source shrink: fetched {len(jobs)} from {active_count} active jobs"
                )
            run_id = connection.execute(
                """INSERT INTO sync_runs(source_kind,source_key,started_at,status,fetched)
                   VALUES(%s,%s,%s,'running',%s) RETURNING id""",
                (source_kind, source_key, now, len(jobs)),
            ).fetchone()["id"]

            for job in jobs:
                digest = _digest(job)
                embedding_digest = _embedding_digest(job)
                payload = job.model_dump(mode="json", exclude_none=True)
                document = job_text(job)[:12000]
                row = existing.get(job.stable_id)
                if row is None:
                    connection.execute(
                        """INSERT INTO jobs
                           (stable_id,source_kind,source_key,external_id,payload,search_document,
                            content_hash,embedding_hash,first_seen,last_seen,status,missing_count)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'active',0)""",
                        (
                            job.stable_id, source_kind, source_key, job.external_id,
                            Jsonb(payload), document, digest, embedding_digest, now, now,
                        ),
                    )
                    inserted += 1
                    self._record_event(connection, job.stable_id, "discovered", now, run_id, current_payload=payload)
                elif row["content_hash"] != digest or row["status"] != "active":
                    event_type = "reopened" if row["status"] != "active" else "updated"
                    connection.execute(
                        """UPDATE jobs SET payload=%s,search_document=%s,content_hash=%s,
                           embedding_hash=%s,last_seen=%s,
                           status='active',missing_count=0 WHERE stable_id=%s""",
                        (Jsonb(payload), document, digest, embedding_digest, now, job.stable_id),
                    )
                    updated += 1
                    self._record_event(
                        connection, job.stable_id, event_type, now, run_id,
                        previous_payload=row["payload"], current_payload=payload,
                    )
                else:
                    connection.execute(
                        "UPDATE jobs SET last_seen=%s,missing_count=0 WHERE stable_id=%s",
                        (now, job.stable_id),
                    )
                    unchanged += 1

            for stable_id, row in existing.items():
                if stable_id in seen or row["status"] != "active":
                    continue
                missing += 1
                count = int(row["missing_count"]) + 1
                if count >= 2:
                    connection.execute(
                        "UPDATE jobs SET status='closed',missing_count=%s WHERE stable_id=%s",
                        (count, stable_id),
                    )
                    closed += 1
                    self._record_event(
                        connection, stable_id, "closed", now, run_id,
                        previous_payload=row["payload"],
                    )
                else:
                    connection.execute(
                        "UPDATE jobs SET missing_count=%s WHERE stable_id=%s", (count, stable_id)
                    )
            connection.execute(
                """UPDATE sync_runs SET completed_at=%s,status='succeeded',inserted=%s,updated=%s,
                   unchanged=%s,missing=%s,closed=%s WHERE id=%s""",
                (now, inserted, updated, unchanged, missing, closed, run_id),
            )
        return SyncSummary(
            source_kind=source_kind, source_key=source_key, fetched=len(jobs),
            inserted=inserted, updated=updated, unchanged=unchanged,
            missing=missing, closed=closed, run_id=run_id,
        )

    @staticmethod
    def _jobs(rows: list[dict]) -> list[Job]:
        return [
            Job.model_validate(row["payload"]).model_copy(update={
                "first_seen_at": row["first_seen"], "last_seen_at": row["last_seen"],
            })
            for row in rows
        ]

    def active_jobs(self) -> list[Job]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload,first_seen,last_seen FROM jobs WHERE status='active'"
            ).fetchall()
        return self._jobs(rows)

    def embedding_candidates(
        self,
        model: str,
        *,
        limit: int = 100,
        countries: list[str] | None = None,
    ) -> list[Job]:
        country_clause = ""
        params: list = [model]
        if countries:
            country_clause = (
                " AND lower(COALESCE(j.payload->>'country','')) = ANY(%s)"
            )
            params.append([country.casefold() for country in countries])
        params.append(max(1, min(limit, 1000)))

        with self.connect() as connection:
            rows = connection.execute(
                """SELECT j.payload,j.first_seen,j.last_seen
                   FROM jobs j LEFT JOIN job_embeddings e
                     ON e.stable_id=j.stable_id AND e.model=%s
                   WHERE j.status='active'
                     AND (e.stable_id IS NULL OR e.content_hash<>j.embedding_hash)"""
                + country_clause
                + " ORDER BY j.last_seen DESC LIMIT %s",
                tuple(params),
            ).fetchall()
        return self._jobs(rows)

    @staticmethod
    def _candidate_filter_clause(
        preferences: Preferences | None,
        *,
        alias: str = "j",
    ) -> tuple[str, list]:
        if preferences is None:
            return "", []

        clauses: list[str] = []
        params: list = []

        if preferences.countries:
            clauses.append(
                f"lower(COALESCE({alias}.payload->>'country','')) = ANY(%s)"
            )
            params.append([country.casefold() for country in preferences.countries])

        if preferences.locations:
            location_clause = (
                f"EXISTS (SELECT 1 FROM unnest(%s::text[]) AS wanted(location) "
                f"WHERE lower(COALESCE({alias}.payload->>'location','')) "
                f"LIKE '%%' || lower(wanted.location) || '%%')"
            )
            params.append(preferences.locations)
            if preferences.remote_ok:
                location_clause = (
                    f"({location_clause} OR "
                    f"COALESCE({alias}.payload->>'remote_policy','unknown')='remote')"
                )
            clauses.append(location_clause)

        if preferences.max_required_experience_years is not None:
            clauses.append(
                f"(({alias}.payload->>'min_experience_years') IS NULL OR "
                f"({alias}.payload->>'min_experience_years')::double precision <= %s)"
            )
            params.append(preferences.max_required_experience_years)

        if not preferences.remote_ok:
            clauses.append(
                f"COALESCE({alias}.payload->>'remote_policy','unknown') <> 'remote'"
            )

        if not preferences.onsite_ok:
            clauses.append(
                f"COALESCE({alias}.payload->>'remote_policy','unknown') "
                f"NOT IN ('onsite','unknown')"
            )

        if preferences.paid_only:
            clauses.append(
                f"lower({alias}.search_document) NOT LIKE ALL(%s)"
            )
            params.append(["%unpaid%", "%no stipend%", "%without stipend%"])

        if preferences.min_salary_lpa is not None:
            # Unknown/non-INR salaries remain eligible for verification, matching
            # the Python hard-constraint semantics. Only known comparable INR
            # salaries below the requested floor are removed here.
            clauses.append(
                f"((({alias}.payload->>'salary_min') IS NULL) OR "
                f"upper(COALESCE({alias}.payload->>'salary_currency','')) <> 'INR' OR "
                f"COALESCE({alias}.payload->>'salary_period','unknown') NOT IN ('year','month') OR "
                f"(CASE COALESCE({alias}.payload->>'salary_period','unknown') "
                f"WHEN 'year' THEN ({alias}.payload->>'salary_min')::double precision / 100000.0 "
                f"WHEN 'month' THEN ({alias}.payload->>'salary_min')::double precision * 12.0 / 100000.0 "
                f"ELSE NULL END) >= %s)"
            )
            params.append(preferences.min_salary_lpa)

        if not clauses:
            return "", []
        return " AND " + " AND ".join(f"({clause})" for clause in clauses), params

    def candidate_jobs(
        self,
        query: str,
        *,
        limit: int = 250,
        query_vector: list[float] | None = None,
        embedding_model: str = "",
        preferences: Preferences | None = None,
    ) -> list[Job]:
        limit = max(1, min(limit, 2000))
        if query_vector is not None and len(query_vector) != POSTGRES_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"PostgreSQL vector index expects {POSTGRES_EMBEDDING_DIMENSIONS} dimensions; "
                f"got {len(query_vector)}"
            )

        normalized = " ".join(query.split())
        filters, filter_params = self._candidate_filter_clause(preferences)

        with self.connect() as connection:
            if not normalized and query_vector is None:
                sql = (
                    "SELECT j.payload,j.first_seen,j.last_seen FROM jobs j "
                    "WHERE j.status='active'"
                    + filters
                    + " ORDER BY j.last_seen DESC LIMIT %s"
                )
                rows = connection.execute(
                    sql, (*filter_params, limit)
                ).fetchall()
            elif query_vector is None:
                sql = (
                    "SELECT j.payload,j.first_seen,j.last_seen,"
                    " ts_rank_cd(j.search_vector,websearch_to_tsquery('english',%s)) AS lexical"
                    " FROM jobs j WHERE j.status='active'"
                    + filters
                    + " ORDER BY lexical DESC,j.last_seen DESC LIMIT %s"
                )
                rows = connection.execute(
                    sql, (normalized, *filter_params, limit)
                ).fetchall()
            else:
                vector_param = Vector(query_vector)
                # Generate bounded lexical and vector candidate sets separately.
                # Keeping the vector CTE ordered directly by distance allows
                # pgvector's HNSW index to participate; the previous LEFT JOIN
                # combined-score scan had to score every active job first.
                sql = (
                    "WITH lexical AS ("
                    " SELECT j.stable_id,"
                    " ts_rank_cd(j.search_vector,websearch_to_tsquery('english',%s)) AS score"
                    " FROM jobs j WHERE j.status='active'"
                    + filters
                    + " ORDER BY score DESC,j.last_seen DESC LIMIT %s"
                    "), semantic AS ("
                    " SELECT j.stable_id,1-(e.vector <=> %s) AS score"
                    " FROM job_embeddings e JOIN jobs j ON j.stable_id=e.stable_id"
                    " WHERE e.model=%s AND j.status='active'"
                    + filters
                    + " ORDER BY e.vector <=> %s LIMIT %s"
                    "), candidates AS ("
                    " SELECT stable_id FROM lexical UNION SELECT stable_id FROM semantic"
                    ")"
                    " SELECT j.payload,j.first_seen,j.last_seen,"
                    " COALESCE(l.score,0) AS lexical,COALESCE(s.score,0) AS semantic"
                    " FROM candidates c JOIN jobs j ON j.stable_id=c.stable_id"
                    " LEFT JOIN lexical l ON l.stable_id=c.stable_id"
                    " LEFT JOIN semantic s ON s.stable_id=c.stable_id"
                    " ORDER BY (0.65*COALESCE(l.score,0)+0.35*COALESCE(s.score,0)) DESC,"
                    " j.last_seen DESC LIMIT %s"
                )
                rows = connection.execute(
                    sql,
                    (
                        normalized,
                        *filter_params,
                        limit,
                        vector_param,
                        embedding_model,
                        *filter_params,
                        vector_param,
                        limit,
                        limit,
                    ),
                ).fetchall()
        return self._jobs(rows)

    def job_status(self, stable_id: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE stable_id=%s", (stable_id,)).fetchone()
        return row["status"] if row else None

    def job_events(self, stable_id: str) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT event_type FROM job_events WHERE stable_id=%s ORDER BY id", (stable_id,)
            ).fetchall()
        return [row["event_type"] for row in rows]

    def load_embedding(self, stable_id: str, model: str, content_hash: str) -> list[float] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT vector FROM job_embeddings
                   WHERE stable_id=%s AND model=%s AND content_hash=%s""",
                (stable_id, model, content_hash),
            ).fetchone()
        return row["vector"].to_list() if row else None

    def save_embedding(self, stable_id: str, model: str, content_hash: str, vector: list[float]) -> None:
        if len(vector) != POSTGRES_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"PostgreSQL vector index expects {POSTGRES_EMBEDDING_DIMENSIONS} dimensions; "
                f"got {len(vector)}"
            )
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO job_embeddings(stable_id,model,content_hash,vector,updated_at)
                   VALUES(%s,%s,%s,%s,%s)
                   ON CONFLICT(stable_id,model) DO UPDATE SET
                     content_hash=excluded.content_hash,vector=excluded.vector,
                     updated_at=excluded.updated_at""",
                (stable_id, model, content_hash, vector, _utcnow()),
            )

    def embedding_status(self, model: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) FILTER (WHERE j.status='active') AS active_jobs,
                          COUNT(*) FILTER (
                            WHERE j.status='active' AND e.stable_id IS NOT NULL
                              AND e.content_hash=j.embedding_hash
                          ) AS embedded_jobs
                   FROM jobs j LEFT JOIN job_embeddings e
                     ON e.stable_id=j.stable_id AND e.model=%s""",
                (model,),
            ).fetchone()
        total = row["active_jobs"]
        embedded = row["embedded_jobs"]
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
                """INSERT INTO profiles(profile_id,preferences,updated_at) VALUES(%s,%s,%s)
                   ON CONFLICT(profile_id) DO UPDATE SET preferences=excluded.preferences,
                   updated_at=excluded.updated_at""",
                (profile_id, Jsonb(preferences.model_dump(mode="json")), _utcnow()),
            )

    def load_preferences(self, profile_id: str) -> Preferences:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT preferences FROM profiles WHERE profile_id=%s", (profile_id,)
            ).fetchone()
        return Preferences.model_validate(row["preferences"]) if row else Preferences()

    def save_candidate_profile(self, profile_id: str, profile: CandidateProfile) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO candidate_profiles(profile_id,profile,updated_at) VALUES(%s,%s,%s)
                   ON CONFLICT(profile_id) DO UPDATE SET profile=excluded.profile,
                   updated_at=excluded.updated_at""",
                (profile_id, Jsonb(profile.model_dump(mode="json")), _utcnow()),
            )

    def load_candidate_profile(self, profile_id: str) -> CandidateProfile | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT profile FROM candidate_profiles WHERE profile_id=%s", (profile_id,)
            ).fetchone()
        return CandidateProfile.model_validate(row["profile"]) if row else None

    def delete_profile(self, profile_id: str) -> None:
        with self.connect() as connection:
            for table in ("candidate_profiles", "profiles", "feedback", "discovery_requests"):
                connection.execute(f"DELETE FROM {table} WHERE profile_id=%s", (profile_id,))

    def record_feedback(self, request: FeedbackRequest) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO feedback(profile_id,job_id,verdict,note,created_at)
                   VALUES(%s,%s,%s,%s,%s)""",
                (request.profile_id, request.job_id, request.verdict, request.note, _utcnow()),
            )

    def rejected_job_ids(self, profile_id: str) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT job_id FROM feedback
                   WHERE profile_id=%s AND verdict IN ('reject','ignore')""",
                (profile_id,),
            ).fetchall()
        return {row["job_id"] for row in rows}

    @staticmethod
    def _discovery_row(row: dict, *, cached: bool = False) -> dict:
        return {
            "request_id": int(row["id"]), "status": row["status"],
            "cached": cached, "report": row["report"],
            "attempt_count": int(row["attempt_count"]),
            "last_error": row["last_error"],
        }

    def queue_discovery(
        self, profile_id: str, fingerprint: str, query: str, *, ttl_seconds: int = 900
    ) -> dict:
        now = _utcnow()
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM discovery_requests WHERE fingerprint=%s
                   ORDER BY id DESC LIMIT 1""",
                (fingerprint,),
            ).fetchone()
            if row:
                age = (now - row["requested_at"]).total_seconds()
                if row["status"] in {"queued", "running"} or (
                    row["status"] == "completed" and age < ttl_seconds
                ):
                    return self._discovery_row(row, cached=True)
            request_id = connection.execute(
                """INSERT INTO discovery_requests
                   (profile_id,fingerprint,query,status,requested_at)
                   VALUES(%s,%s,%s,'queued',%s) RETURNING id""",
                (profile_id, fingerprint, query, now),
            ).fetchone()["id"]
        return {"request_id": request_id, "status": "queued", "cached": False, "report": None}

    def discovery_request(self, request_id: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM discovery_requests WHERE id=%s", (request_id,)
            ).fetchone()
        return self._discovery_row(row) if row else None

    def start_discovery(self, request_id: int) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """UPDATE discovery_requests SET status='running',started_at=%s,
                   attempt_count=attempt_count+1,next_attempt_at=NULL
                   WHERE id=%s AND status='queued' RETURNING id""",
                (_utcnow(), request_id),
            ).fetchone()
        return row is not None

    def pending_discoveries(self, limit: int = 10) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id,profile_id,query FROM discovery_requests
                   WHERE status='queued' ORDER BY id LIMIT %s""",
                (max(1, min(limit, 100)),),
            ).fetchall()
        return rows

    def claim_discoveries(self, limit: int = 10) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """WITH pending AS (
                     SELECT id FROM discovery_requests
                     WHERE status='queued'
                       AND (next_attempt_at IS NULL OR next_attempt_at<=%s)
                     ORDER BY id
                     FOR UPDATE SKIP LOCKED LIMIT %s
                   )
                   UPDATE discovery_requests d
                   SET status='running',started_at=%s,attempt_count=d.attempt_count+1,
                       next_attempt_at=NULL
                   FROM pending WHERE d.id=pending.id
                   RETURNING d.id,d.profile_id,d.query""",
                (_utcnow(), max(1, min(limit, 100)), _utcnow()),
            ).fetchall()
        return rows

    def requeue_stale_discoveries(self, stale_seconds: int = 900) -> int:
        cutoff = _utcnow() - timedelta(seconds=max(60, stale_seconds))
        with self.connect() as connection:
            rows = connection.execute(
                """UPDATE discovery_requests SET status='queued',started_at=NULL
                   WHERE status='running' AND started_at<%s RETURNING id""",
                (cutoff,),
            ).fetchall()
        return len(rows)

    def finish_discovery(self, request_id: int, report: dict, *, failed: bool = False) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE discovery_requests SET status=%s,report=%s,completed_at=%s WHERE id=%s""",
                ("failed" if failed else "completed", Jsonb(report), _utcnow(), request_id),
            )

    def retry_discovery(self, request_id: int, error: str, *, max_attempts: int = 3) -> str:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT attempt_count FROM discovery_requests WHERE id=%s FOR UPDATE",
                (request_id,),
            ).fetchone()
            if row is None:
                return "missing"
            attempts = int(row["attempt_count"])
            if attempts >= max_attempts:
                connection.execute(
                    """UPDATE discovery_requests SET status='failed',last_error=%s,completed_at=%s
                       WHERE id=%s""",
                    (error[:1000], _utcnow(), request_id),
                )
                return "failed"
            delay = min(300, 5 * (2 ** max(0, attempts - 1)))
            connection.execute(
                """UPDATE discovery_requests SET status='queued',started_at=NULL,
                   next_attempt_at=%s,last_error=%s WHERE id=%s""",
                (_utcnow() + timedelta(seconds=delay), error[:1000], request_id),
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
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(source_kind,source_key) DO UPDATE SET
                     board_url=excluded.board_url,discovery_query=excluded.discovery_query,
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
            return connection.execute(
                """SELECT source_kind,source_key,board_url,discovery_query,
                          first_discovered,last_discovered,last_synced,last_status
                   FROM discovered_sources ORDER BY source_kind,source_key"""
            ).fetchall()

    def sources_due(self, *, interval_seconds: int = 3600, limit: int = 25) -> list[dict]:
        cutoff = _utcnow() - timedelta(seconds=max(60, interval_seconds))
        with self.connect() as connection:
            return connection.execute(
                """SELECT source_kind,source_key,board_url,last_synced,last_status
                   FROM discovered_sources
                   WHERE last_synced IS NULL OR last_synced<=%s
                   ORDER BY COALESCE(last_synced,first_discovered) LIMIT %s""",
                (cutoff, max(1, min(limit, 500))),
            ).fetchall()
