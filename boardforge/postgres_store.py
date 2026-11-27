from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pgvector import Vector
from pgvector.psycopg import register_vector
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from boardforge.enrich import infer_country
from boardforge.matching import job_text
from boardforge.quarantine import quarantined_sources
from boardforge.semantic import POSTGRES_EMBEDDING_DIMENSIONS
from boardforge.models import CandidateProfile, FeedbackRequest, Job, Preferences, SyncSummary
from boardforge.store import stale_job_cutoff


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

    def upsert_source_jobs(
        self,
        source_kind: str,
        source_key: str,
        jobs: list[Job],
        *,
        allow_verified_empty: bool = False,
    ) -> SyncSummary:
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
            if active_count and not jobs and not allow_verified_empty:
                raise ValueError("refusing to treat an empty fetch as a successful source sync")
            if active_count >= 10 and 0 < len(jobs) < max(1, int(active_count * 0.25)):
                raise ValueError(
                    f"refusing abnormal source shrink: fetched {len(jobs)} from {active_count} active jobs"
                )
            close_threshold = 3 if (not jobs and active_count >= 10) else 2
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
                if count >= close_threshold:
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

    @staticmethod
    def _quarantine_filter_clause(*, alias: str = "j") -> tuple[str, list]:
        configured = sorted(
            quarantined_sources(),
            key=lambda item: (item[0] or "", item[1]),
        )
        if not configured:
            return "", []
        prefix = f"{alias}." if alias else ""
        parts: list[str] = []
        params: list = []
        for kind, key in configured:
            if kind:
                parts.append(f"(lower({prefix}source_kind)=%s AND lower({prefix}source_key)=%s)")
                params.extend([kind, key])
            else:
                parts.append(f"(lower({prefix}source_key)=%s)")
                params.append(key)
        return f" AND NOT ({' OR '.join(parts)})", params

    def active_jobs(
        self,
        *,
        include_stale: bool = False,
        include_quarantined: bool = False,
    ) -> list[Job]:
        cutoff = None if include_stale else stale_job_cutoff()
        q_clause, q_params = (
            ("", []) if include_quarantined else self._quarantine_filter_clause(alias="")
        )
        with self.connect() as connection:
            if cutoff is None:
                rows = connection.execute(
                    "SELECT payload,first_seen,last_seen FROM jobs WHERE status='active'"
                    + q_clause,
                    tuple(q_params),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT payload,first_seen,last_seen FROM jobs WHERE status='active' AND last_seen>=%s"
                    + q_clause,
                    (cutoff, *q_params),
                ).fetchall()
        return self._jobs(rows)

    def embedding_candidates(
        self,
        model: str,
        *,
        limit: int = 100,
        countries: list[str] | None = None,
    ) -> list[Job]:
        cutoff = stale_job_cutoff()
        q_clause, q_params = self._quarantine_filter_clause(alias="j")
        extra_clause = ""
        params: list = [model]
        if q_clause:
            extra_clause += q_clause
            params.extend(q_params)
        if cutoff is not None:
            extra_clause += " AND j.last_seen >= %s"
            params.append(cutoff)
        if countries:
            extra_clause += (
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
                + extra_clause
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
                inferred_countries = sorted({
                    c.casefold()
                    for c in (infer_country(loc) for loc in preferences.locations)
                    if c
                })
                if inferred_countries:
                    remote_branch = (
                        f"(COALESCE({alias}.payload->>'remote_policy','unknown')='remote' "
                        f"AND lower(COALESCE({alias}.payload->>'country','')) = ANY(%s))"
                    )
                    params.append(inferred_countries)
                else:
                    remote_branch = f"COALESCE({alias}.payload->>'remote_policy','unknown')='remote'"
                location_clause = f"({location_clause} OR {remote_branch})"
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
        if "\x00" in query:
            raise ValueError("search query must not contain NUL bytes")
        limit = max(1, min(limit, 2000))
        if query_vector is not None and len(query_vector) != POSTGRES_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"PostgreSQL vector index expects {POSTGRES_EMBEDDING_DIMENSIONS} dimensions; "
                f"got {len(query_vector)}"
            )

        normalized = " ".join(query.split())
        filters, filter_params = self._candidate_filter_clause(preferences)
        cutoff = stale_job_cutoff()
        if cutoff is not None:
            filters = " AND (j.last_seen >= %s)" + filters
            filter_params = [cutoff, *filter_params]
        q_clause, q_params = self._quarantine_filter_clause(alias="j")
        if q_clause:
            filters = q_clause + filters
            filter_params = [*q_params, *filter_params]

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

    def explain_candidate_jobs(
        self,
        query: str,
        *,
        limit: int = 250,
        query_vector: list[float] | None = None,
        embedding_model: str = "",
        preferences: Preferences | None = None,
    ) -> dict:
        """Run the actual candidate query under EXPLAIN ANALYZE.

        Indexes are reported only when an executed plan node names them. This
        intentionally does not infer utilization from pg_indexes.
        """
        if "\x00" in query:
            raise ValueError("search query must not contain NUL bytes")
        limit = max(1, min(limit, 2000))
        if query_vector is not None and len(query_vector) != POSTGRES_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"PostgreSQL vector index expects {POSTGRES_EMBEDDING_DIMENSIONS} dimensions; "
                f"got {len(query_vector)}"
            )
        normalized = " ".join(query.split())
        filters, filter_params = self._candidate_filter_clause(preferences)
        cutoff = stale_job_cutoff()
        if cutoff is not None:
            filters = " AND (j.last_seen >= %s)" + filters
            filter_params = [cutoff, *filter_params]
        q_clause, q_params = self._quarantine_filter_clause(alias="j")
        if q_clause:
            filters = q_clause + filters
            filter_params = [*q_params, *filter_params]
        if not normalized and query_vector is None:
            sql = (
                "SELECT j.payload,j.first_seen,j.last_seen FROM jobs j "
                "WHERE j.status='active'" + filters
                + " ORDER BY j.last_seen DESC LIMIT %s"
            )
            params = (*filter_params, limit)
        elif query_vector is None:
            sql = (
                "SELECT j.payload,j.first_seen,j.last_seen,"
                " ts_rank_cd(j.search_vector,websearch_to_tsquery('english',%s)) AS lexical"
                " FROM jobs j WHERE j.status='active'" + filters
                + " ORDER BY lexical DESC,j.last_seen DESC LIMIT %s"
            )
            params = (normalized, *filter_params, limit)
        else:
            vector_param = Vector(query_vector)
            sql = (
                "WITH lexical AS ("
                " SELECT j.stable_id,"
                " ts_rank_cd(j.search_vector,websearch_to_tsquery('english',%s)) AS score"
                " FROM jobs j WHERE j.status='active'" + filters
                + " ORDER BY score DESC,j.last_seen DESC LIMIT %s"
                "), semantic AS ("
                " SELECT j.stable_id,1-(e.vector <=> %s) AS score"
                " FROM job_embeddings e JOIN jobs j ON j.stable_id=e.stable_id"
                " WHERE e.model=%s AND j.status='active'" + filters
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
            params = (
                normalized, *filter_params, limit, vector_param, embedding_model,
                *filter_params, vector_param, limit, limit,
            )
        with self.connect() as connection:
            raw = connection.execute(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params
            ).fetchone()["QUERY PLAN"]
        plan = raw[0] if isinstance(raw, list) else raw
        indexes: list[str] = []

        def visit(node: dict) -> None:
            if node.get("Index Name"):
                indexes.append(node["Index Name"])
            for child in node.get("Plans", []):
                visit(child)

        visit(plan["Plan"])
        return {
            "actual_indexes": sorted(set(indexes)),
            "gin_used": "idx_jobs_search_vector" in indexes,
            "hnsw_used": "idx_job_embeddings_hnsw" in indexes,
            "execution_time_ms": plan.get("Execution Time"),
            "plan": plan,
        }

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

    def load_query_embedding(self, cache_key: str) -> list[float] | None:
        with self.connect() as connection:
            row = connection.execute(
                """UPDATE query_embeddings SET last_used_at=%s,hit_count=hit_count+1
                   WHERE cache_key=%s RETURNING vector""",
                (_utcnow(), cache_key),
            ).fetchone()
        return row["vector"].to_list() if row else None

    def save_query_embedding(
        self, cache_key: str, normalized_query: str, model: str,
        config_hash: str, vector: list[float],
    ) -> None:
        if len(vector) != POSTGRES_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"PostgreSQL query cache expects {POSTGRES_EMBEDDING_DIMENSIONS} dimensions; "
                f"got {len(vector)}"
            )
        now = _utcnow()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO query_embeddings
                   (cache_key,normalized_query,model,config_hash,vector,created_at,last_used_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(cache_key) DO UPDATE SET vector=excluded.vector,
                   last_used_at=excluded.last_used_at""",
                (cache_key, normalized_query, model, config_hash, vector, now, now),
            )

    def load_relevance_verification(self, cache_key: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT judgment FROM relevance_verifications WHERE cache_key=%s",
                (cache_key,),
            ).fetchone()
        return dict(row["judgment"]) if row else None

    def save_relevance_verification(self, cache_key: str, judgment: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO relevance_verifications(cache_key,judgment,created_at)
                   VALUES(%s,%s,%s) ON CONFLICT(cache_key) DO UPDATE SET
                   judgment=excluded.judgment,created_at=excluded.created_at""",
                (cache_key, Jsonb(judgment), _utcnow()),
            )

    def reserve_llm_usage(
        self, provider: str, model: str, reservation_id: str,
        reserved_tokens: int, *, tpm_limit: int, tpd_limit: int,
        reserved_cost_usd: float = 0.0,
        cumulative_cost_limit_usd: float = 1000000.0,
    ) -> dict:
        now = _utcnow()
        with self.connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                ("boardforge-provider-usage-global",),
            )
            row = connection.execute(
                """SELECT
                     COALESCE(SUM(CASE WHEN status='completed' THEN total_tokens
                       ELSE reserved_tokens END)
                       FILTER (WHERE created_at >= %s), 0) AS minute_tokens,
                     COALESCE(SUM(CASE WHEN status='completed' THEN total_tokens
                       ELSE reserved_tokens END), 0) AS day_tokens,
                     MIN(created_at) FILTER (WHERE created_at >= %s) AS minute_oldest
                     ,(SELECT COALESCE(SUM(CASE WHEN status='completed'
                         THEN estimated_cost_usd ELSE reserved_cost_usd END), 0)
                       FROM llm_usage_reservations) AS cumulative_cost
                   FROM llm_usage_reservations
                   WHERE provider=%s AND model=%s AND created_at >= %s""",
                (now - timedelta(seconds=60), now - timedelta(seconds=60),
                 provider, model, now - timedelta(hours=24)),
            ).fetchone()
            daily = int(row["day_tokens"])
            minute = int(row["minute_tokens"])
            cumulative_cost = float(row["cumulative_cost"])
            if daily + reserved_tokens > tpd_limit:
                return {"allowed": False, "reason": "daily_token_limit",
                        "used_tokens": daily, "retry_after_seconds": 86400}
            if cumulative_cost + reserved_cost_usd > cumulative_cost_limit_usd:
                return {"allowed": False, "reason": "cumulative_cost_limit",
                        "used_tokens": daily, "used_cost_usd": cumulative_cost,
                        "retry_after_seconds": 86400}
            if minute + reserved_tokens > tpm_limit:
                oldest = row["minute_oldest"]
                retry_after = max(1, math.ceil(60 - (now - oldest).total_seconds())) if oldest else 60
                return {"allowed": False, "reason": "minute_token_limit",
                        "used_tokens": minute, "retry_after_seconds": retry_after}
            connection.execute(
                """INSERT INTO llm_usage_reservations(
                     reservation_id,provider,model,reserved_tokens,reserved_cost_usd,
                     status,created_at) VALUES(%s,%s,%s,%s,%s,'reserved',%s)""",
                (reservation_id, provider, model, reserved_tokens, reserved_cost_usd, now),
            )
        return {"allowed": True, "used_tokens": minute, "daily_tokens": daily,
                "cumulative_cost_usd": cumulative_cost}

    def finalize_llm_usage(
        self, reservation_id: str, *, status: str, input_tokens: int = 0,
        output_tokens: int = 0, thinking_tokens: int = 0, total_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE llm_usage_reservations SET status=%s,input_tokens=%s,
                     output_tokens=%s,thinking_tokens=%s,total_tokens=%s,
                     estimated_cost_usd=%s,completed_at=%s WHERE reservation_id=%s""",
                (status, input_tokens, output_tokens, thinking_tokens, total_tokens,
                 estimated_cost_usd, _utcnow(), reservation_id),
            )

    def embedding_status(self, model: str) -> dict:
        q_clause, q_params = self._quarantine_filter_clause(alias="j")
        with self.connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) FILTER (WHERE j.status='active') AS active_jobs,
                          COUNT(*) FILTER (
                            WHERE j.status='active' AND e.stable_id IS NOT NULL
                              AND e.content_hash=j.embedding_hash
                          ) AS embedded_jobs
                   FROM jobs j LEFT JOIN job_embeddings e
                     ON e.stable_id=j.stable_id AND e.model=%s
                   WHERE TRUE"""
                + q_clause,
                (model, *q_params),
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

    def reset_preferences(self, profile_id: str) -> None:
        """Clear saved preferences for a profile without deleting its resume or feedback."""
        with self.connect() as connection:
            connection.execute("DELETE FROM profiles WHERE profile_id=%s", (profile_id,))

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

    def feedback_history(self, profile_id: str, *, limit: int = 20) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT job_id,verdict,note,created_at FROM feedback
                   WHERE profile_id=%s ORDER BY id DESC LIMIT %s""",
                (profile_id, max(1, min(limit, 100))),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _discovery_row(row: dict, *, cached: bool = False) -> dict:
        return {
            "request_id": int(row["id"]),
            "profile_id": row["profile_id"],
            "status": row["status"],
            "cached": cached,
            "report": row["report"],
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
        return {
            "request_id": request_id,
            "profile_id": profile_id,
            "status": "queued",
            "cached": False,
            "report": None,
        }

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
        synced_at: datetime | None = None,
        discovered_at: datetime | None = None,
    ) -> None:
        def _as_utc(val: datetime) -> datetime:
            dt = val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        now = _utcnow()
        disc_ts = _as_utc(discovered_at) if discovered_at is not None else now
        sync_ts = (
            _as_utc(synced_at)
            if synced_at is not None
            else (now if synced else None)
        )
        last_disc_ts = (
            sync_ts
            if (discovered_at is not None and sync_ts is not None and sync_ts > disc_ts)
            else disc_ts
        )
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO discovered_sources
                   (source_kind,source_key,board_url,discovery_query,first_discovered,
                    last_discovered,last_synced,last_status)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(source_kind,source_key) DO UPDATE SET
                     board_url=excluded.board_url,discovery_query=excluded.discovery_query,
                     first_discovered=LEAST(excluded.first_discovered,discovered_sources.first_discovered),
                     last_discovered=GREATEST(excluded.last_discovered,discovered_sources.last_discovered),
                     last_synced=CASE
                       WHEN excluded.last_synced IS NOT NULL AND discovered_sources.last_synced IS NOT NULL
                         THEN GREATEST(excluded.last_synced,discovered_sources.last_synced)
                       ELSE COALESCE(excluded.last_synced,discovered_sources.last_synced)
                     END,
                     last_status=excluded.last_status""",
                (
                    source_kind, source_key, board_url, query, disc_ts, last_disc_ts,
                    sync_ts, status,
                ),
            )

    def consume_beta_usage(self, profile_id: str, operation: str, limit: int,
                           global_limit: int | None = None) -> bool:
        # Serialize admission for each operation/day across instances, then
        # update the user counter in the same transaction. No global sentinel row.
        with self.connect() as connection:
            day = connection.execute("SELECT (now() AT TIME ZONE 'UTC')::date AS day").fetchone()["day"]
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                               (str(day), "beta:" + operation))
            if global_limit is not None:
                total = connection.execute(
                    "SELECT COALESCE(SUM(count),0) AS total FROM beta_daily_usage WHERE day=%s AND operation=%s",
                    (day, operation),
                ).fetchone()["total"]
                if total >= global_limit:
                    return False
            row = connection.execute(
                """INSERT INTO beta_daily_usage(day,profile_id,operation,count)
                   VALUES(%s,%s,%s,1)
                   ON CONFLICT(day,profile_id,operation) DO UPDATE
                   SET count=beta_daily_usage.count+1
                   WHERE beta_daily_usage.count < %s RETURNING count""",
                (day, profile_id, operation, limit),
            ).fetchone()
            return row is not None

    def usage_cost_summary(self) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS reservations,
                          COALESCE(SUM(CASE WHEN status='completed'
                            THEN estimated_cost_usd ELSE reserved_cost_usd END),0) AS cost,
                          COALESCE(SUM(total_tokens),0) AS tokens
                   FROM llm_usage_reservations"""
            ).fetchone()
        return {
            "reservations": int(row["reservations"]),
            "estimated_cost_usd": float(row["cost"]),
            "total_tokens": int(row["tokens"]),
        }

    def discovered_sources(self) -> list[dict]:
        with self.connect() as connection:
            return connection.execute(
                """SELECT source_kind,source_key,board_url,discovery_query,
                          first_discovered,last_discovered,last_synced,last_status
                   FROM discovered_sources ORDER BY source_kind,source_key"""
            ).fetchall()

    def sources_due(self, *, interval_seconds: int = 3600, limit: int = 25) -> list[dict]:
        cutoff = _utcnow() - timedelta(seconds=max(60, interval_seconds))
        q_clause, q_params = self._quarantine_filter_clause(alias="")
        with self.connect() as connection:
            return connection.execute(
                """SELECT source_kind,source_key,board_url,last_synced,last_status
                   FROM discovered_sources
                   WHERE ((last_status != 'failed' AND (last_synced IS NULL OR last_synced<=%s))
                      OR (last_status = 'failed' AND last_discovered<=%s))"""
                + q_clause
                + """
                   ORDER BY CASE
                     WHEN last_status = 'failed'
                       THEN GREATEST(COALESCE(last_synced,first_discovered), last_discovered)
                     ELSE COALESCE(last_synced,first_discovered)
                   END ASC,
                   source_kind ASC, source_key ASC
                   LIMIT %s""",
                (cutoff, cutoff, *q_params, max(1, min(limit, 500))),
            ).fetchall()

    def record_telemetry_event(self, event_kind: str, payload: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO telemetry_events(created_at,event_kind,payload)
                   VALUES(%s,%s,%s)""",
                (_utcnow(), event_kind, Jsonb(payload)),
            )
            connection.execute(
                """DELETE FROM telemetry_events
                   WHERE id < (
                     SELECT COALESCE(MIN(id), 0)
                     FROM (SELECT id FROM telemetry_events ORDER BY id DESC LIMIT 500) recent
                   )"""
            )
