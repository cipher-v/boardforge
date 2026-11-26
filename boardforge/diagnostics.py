"""Read-only operational diagnostics without exposing private candidate or query data."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

from boardforge.quarantine import is_quarantined_source, quarantined_sources_list
from boardforge.store import stale_job_cutoff

_SAFE_DISCOVERY_REPORT_KEYS = (
    "enabled",
    "provider",
    "searched_queries",
    "candidate_results",
    "discovered_boards",
    "skipped_existing_boards",
    "skipped_low_relevance_boards",
    "synced_boards",
    "failed_boards",
    "jobs_fetched",
    "jobs_inserted",
    "jobs_updated",
    "new_matching_jobs",
    "new_recommended_jobs",
    "failures",
    "reason",
    "retryable",
    "error_kind",
    "estimated_cost_usd",
    "latency_ms",
)


def _sanitize_discovery_report(raw_report: Any) -> dict[str, Any] | None:
    if raw_report is None:
        return None
    if isinstance(raw_report, str):
        if not raw_report.strip():
            return None
        try:
            raw_report = json.loads(raw_report)
        except ValueError:
            return None
    if not isinstance(raw_report, dict):
        return None
    return {k: raw_report[k] for k in _SAFE_DISCOVERY_REPORT_KEYS if k in raw_report}


def _parse_iso(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str) and val.strip():
        try:
            dt = datetime.fromisoformat(val.strip())
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _duration_ms(start: Any, end: Any) -> float | None:
    s_dt = _parse_iso(start)
    e_dt = _parse_iso(end)
    if s_dt is None or e_dt is None:
        return None
    return round(max(0.0, (e_dt - s_dt).total_seconds() * 1000.0), 2)


def _sanitize_telemetry_payload(raw_payload: Any) -> dict[str, Any]:
    if isinstance(raw_payload, str):
        try:
            raw_payload = json.loads(raw_payload)
        except ValueError:
            return {}
    if not isinstance(raw_payload, dict):
        return {}
    allowed_top = {
        "endpoint",
        "orchestration_mode",
        "stopped_reason",
        "provider",
        "model",
        "iterations",
        "provider_calls",
        "tools_executed",
        "match_count",
        "recommended_count",
        "alternative_count",
        "needs_verification_count",
        "discovery_requested",
        "discovery_status",
        "discovery_request_id",
        "cache_hits",
        "latency_ms",
        "cost_usd",
    }
    cleaned: dict[str, Any] = {}
    for key in allowed_top:
        if key not in raw_payload:
            continue
        if key == "tools_executed" and isinstance(raw_payload[key], list):
            cleaned[key] = [
                {
                    "tool": str(t.get("tool", "")),
                    "ok": bool(t.get("ok", False)),
                    "duration_ms": t.get("duration_ms"),
                    "error": t.get("error"),
                }
                for t in raw_payload[key]
                if isinstance(t, dict)
            ]
        else:
            cleaned[key] = raw_payload[key]
    return cleaned


def _inspect_postgres_readonly(database_url: str, *, limit: int) -> dict[str, Any]:
    import psycopg
    from psycopg.rows import dict_row

    bounded = max(1, min(25, int(limit)))
    cutoff = stale_job_cutoff()
    with psycopg.connect(database_url, row_factory=dict_row, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on")
            db_name = cur.execute("SELECT current_database() AS db").fetchone()["db"]

            job_rows = cur.execute(
                "SELECT source_kind, source_key, last_seen FROM jobs WHERE status='active'"
            ).fetchall()
            quarantined_jobs = sum(
                1 for r in job_rows if is_quarantined_source(r["source_kind"], r["source_key"])
            )
            stale_jobs = sum(
                1
                for r in job_rows
                if not is_quarantined_source(r["source_kind"], r["source_key"])
                and cutoff is not None
                and _parse_iso(r["last_seen"]) is not None
                and _parse_iso(r["last_seen"]) < cutoff
            )
            visible_jobs = len(job_rows) - quarantined_jobs - stale_jobs

            has_telemetry = bool(
                cur.execute("SELECT to_regclass('public.telemetry_events') AS reg").fetchone()["reg"]
            )
            telemetry_rows: list[dict[str, Any]] = []
            if has_telemetry:
                raw_tel = cur.execute(
                    """SELECT id, created_at, event_kind, payload
                       FROM telemetry_events ORDER BY id DESC LIMIT %s""",
                    (bounded,),
                ).fetchall()
                for r in raw_tel:
                    telemetry_rows.append({
                        "id": int(r["id"]),
                        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                        "event_kind": r["event_kind"],
                        **_sanitize_telemetry_payload(r["payload"]),
                    })

            disc_rows = cur.execute(
                """SELECT id, status, attempt_count, requested_at, started_at, completed_at,
                          last_error, report
                   FROM discovery_requests ORDER BY id DESC LIMIT %s""",
                (bounded,),
            ).fetchall()
            discoveries = [
                {
                    "request_id": int(r["id"]),
                    "status": r["status"],
                    "attempt_count": int(r["attempt_count"] or 0),
                    "requested_at": r["requested_at"].isoformat() if r["requested_at"] else None,
                    "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
                    "duration_ms": _duration_ms(r["started_at"] or r["requested_at"], r["completed_at"]),
                    "last_error": r["last_error"],
                    "report": _sanitize_discovery_report(r["report"]),
                }
                for r in disc_rows
            ]

            qe = cur.execute(
                """SELECT COUNT(*) AS cached_queries,
                          COALESCE(SUM(hit_count), 0) AS total_hits,
                          MAX(last_used_at) AS last_used_at
                   FROM query_embeddings"""
            ).fetchone()
            rv = cur.execute(
                "SELECT COUNT(*) AS cached_judgments, MAX(created_at) AS latest_at FROM relevance_verifications"
            ).fetchone()

            by_provider = cur.execute(
                """SELECT provider, model, status,
                          COUNT(*) AS calls,
                          COALESCE(SUM(total_tokens), 0) AS total_tokens,
                          COALESCE(SUM(CASE WHEN status='completed'
                            THEN estimated_cost_usd ELSE reserved_cost_usd END), 0) AS cost_usd
                   FROM llm_usage_reservations
                   GROUP BY provider, model, status
                   ORDER BY provider, model, status"""
            ).fetchall()
            recent_usage = cur.execute(
                """SELECT provider, model, status, input_tokens, output_tokens, thinking_tokens,
                          total_tokens, estimated_cost_usd, reserved_cost_usd, created_at, completed_at
                   FROM llm_usage_reservations
                   ORDER BY created_at DESC LIMIT %s""",
                (bounded,),
            ).fetchall()
            ledger_totals = cur.execute(
                """SELECT COUNT(*) AS reservations,
                          COALESCE(SUM(total_tokens), 0) AS total_tokens,
                          COALESCE(SUM(CASE WHEN status='completed'
                            THEN estimated_cost_usd ELSE reserved_cost_usd END), 0) AS total_cost_usd
                   FROM llm_usage_reservations"""
            ).fetchone()

    return {
        "mode": "read_only",
        "database": f"postgres:{db_name}",
        "pii_redacted": True,
        "index_visibility": {
            "active_stored_jobs": len(job_rows),
            "visible_active_jobs": visible_jobs,
            "quarantined_active_jobs": quarantined_jobs,
            "stale_hidden_jobs": stale_jobs,
            "configured_quarantined_sources": quarantined_sources_list(),
        },
        "telemetry_table_initialized": has_telemetry,
        "latest_agent_and_search_turns": telemetry_rows,
        "latest_discovery_requests": discoveries,
        "caches": {
            "query_embeddings": {
                "cached_entries": int(qe["cached_queries"]),
                "total_cache_hits": int(qe["total_hits"]),
                "last_used_at": qe["last_used_at"].isoformat() if qe["last_used_at"] else None,
            },
            "relevance_verifications": {
                "cached_judgments": int(rv["cached_judgments"]),
                "latest_created_at": rv["latest_at"].isoformat() if rv["latest_at"] else None,
            },
        },
        "expenditure_and_usage": {
            "totals": {
                "reservations": int(ledger_totals["reservations"]),
                "total_tokens": int(ledger_totals["total_tokens"]),
                "total_cost_usd": round(float(ledger_totals["total_cost_usd"]), 8),
            },
            "by_provider": [
                {
                    "provider": r["provider"],
                    "model": r["model"],
                    "status": r["status"],
                    "calls": int(r["calls"]),
                    "total_tokens": int(r["total_tokens"]),
                    "cost_usd": round(float(r["cost_usd"]), 8),
                }
                for r in by_provider
            ],
            "recent_calls": [
                {
                    "provider": r["provider"],
                    "model": r["model"],
                    "status": r["status"],
                    "input_tokens": int(r["input_tokens"] or 0),
                    "output_tokens": int(r["output_tokens"] or 0),
                    "thinking_tokens": int(r["thinking_tokens"] or 0),
                    "total_tokens": int(r["total_tokens"] or 0),
                    "cost_usd": round(
                        float(
                            r["estimated_cost_usd"]
                            if r["status"] == "completed"
                            else r["reserved_cost_usd"]
                        ),
                        8,
                    ),
                    "duration_ms": _duration_ms(r["created_at"], r["completed_at"]),
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in recent_usage
            ],
        },
    }


def _inspect_sqlite_readonly(db_path: Path, *, limit: int) -> dict[str, Any]:
    resolved = db_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")
    bounded = max(1, min(25, int(limit)))
    cutoff = stale_job_cutoff()
    uri = f"file:{quote(resolved.as_posix())}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        job_rows = (
            conn.execute("SELECT source_kind, source_key, last_seen FROM jobs WHERE status='active'").fetchall()
            if "jobs" in tables
            else []
        )
        quarantined_jobs = sum(
            1 for r in job_rows if is_quarantined_source(r["source_kind"], r["source_key"])
        )
        stale_jobs = sum(
            1
            for r in job_rows
            if not is_quarantined_source(r["source_kind"], r["source_key"])
            and cutoff is not None
            and _parse_iso(r["last_seen"]) is not None
            and _parse_iso(r["last_seen"]) < cutoff
        )
        visible_jobs = len(job_rows) - quarantined_jobs - stale_jobs

        has_telemetry = "telemetry_events" in tables
        telemetry_rows: list[dict[str, Any]] = []
        if has_telemetry:
            raw_tel = conn.execute(
                "SELECT id, created_at, event_kind, payload FROM telemetry_events ORDER BY id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
            for r in raw_tel:
                telemetry_rows.append({
                    "id": int(r["id"]),
                    "created_at": r["created_at"],
                    "event_kind": r["event_kind"],
                    **_sanitize_telemetry_payload(r["payload"]),
                })

        discoveries: list[dict[str, Any]] = []
        if "discovery_requests" in tables:
            disc_cols = {
                c["name"] for c in conn.execute("PRAGMA table_info(discovery_requests)").fetchall()
            }
            attempt_expr = "attempt_count" if "attempt_count" in disc_cols else "0 AS attempt_count"
            started_expr = "started_at" if "started_at" in disc_cols else "NULL AS started_at"
            err_expr = "last_error" if "last_error" in disc_cols else "NULL AS last_error"
            disc_rows = conn.execute(
                f"""SELECT id, status, {attempt_expr}, requested_at, {started_expr}, completed_at,
                           {err_expr}, report
                    FROM discovery_requests ORDER BY id DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
            for r in disc_rows:
                discoveries.append({
                    "request_id": int(r["id"]),
                    "status": r["status"],
                    "attempt_count": int(r["attempt_count"] or 0),
                    "requested_at": r["requested_at"],
                    "completed_at": r["completed_at"],
                    "duration_ms": _duration_ms(r["started_at"] or r["requested_at"], r["completed_at"]),
                    "last_error": r["last_error"],
                    "report": _sanitize_discovery_report(r["report"]),
                })

        qe = (
            conn.execute(
                "SELECT COUNT(*) AS cached_queries, COALESCE(SUM(hit_count),0) AS total_hits, MAX(last_used_at) AS last_used_at FROM query_embeddings"
            ).fetchone()
            if "query_embeddings" in tables
            else {"cached_queries": 0, "total_hits": 0, "last_used_at": None}
        )
        rv = (
            conn.execute(
                "SELECT COUNT(*) AS cached_judgments, MAX(created_at) AS latest_at FROM relevance_verifications"
            ).fetchone()
            if "relevance_verifications" in tables
            else {"cached_judgments": 0, "latest_at": None}
        )

        by_provider: list[dict[str, Any]] = []
        recent_calls: list[dict[str, Any]] = []
        totals = {"reservations": 0, "total_tokens": 0, "total_cost_usd": 0.0}
        if "llm_usage_reservations" in tables:
            lt = conn.execute(
                """SELECT COUNT(*) AS reservations,
                          COALESCE(SUM(total_tokens), 0) AS total_tokens,
                          COALESCE(SUM(CASE WHEN status='completed'
                            THEN estimated_cost_usd ELSE reserved_cost_usd END), 0) AS total_cost_usd
                   FROM llm_usage_reservations"""
            ).fetchone()
            totals = {
                "reservations": int(lt["reservations"]),
                "total_tokens": int(lt["total_tokens"]),
                "total_cost_usd": round(float(lt["total_cost_usd"]), 8),
            }
            for r in conn.execute(
                """SELECT provider, model, status, COUNT(*) AS calls,
                          COALESCE(SUM(total_tokens), 0) AS total_tokens,
                          COALESCE(SUM(CASE WHEN status='completed'
                            THEN estimated_cost_usd ELSE reserved_cost_usd END), 0) AS cost_usd
                   FROM llm_usage_reservations
                   GROUP BY provider, model, status
                   ORDER BY provider, model, status"""
            ).fetchall():
                by_provider.append({
                    "provider": r["provider"],
                    "model": r["model"],
                    "status": r["status"],
                    "calls": int(r["calls"]),
                    "total_tokens": int(r["total_tokens"]),
                    "cost_usd": round(float(r["cost_usd"]), 8),
                })
            for r in conn.execute(
                """SELECT provider, model, status, input_tokens, output_tokens, thinking_tokens,
                          total_tokens, estimated_cost_usd, reserved_cost_usd, created_at, completed_at
                   FROM llm_usage_reservations ORDER BY created_at DESC LIMIT ?""",
                (bounded,),
            ).fetchall():
                recent_calls.append({
                    "provider": r["provider"],
                    "model": r["model"],
                    "status": r["status"],
                    "input_tokens": int(r["input_tokens"] or 0),
                    "output_tokens": int(r["output_tokens"] or 0),
                    "thinking_tokens": int(r["thinking_tokens"] or 0),
                    "total_tokens": int(r["total_tokens"] or 0),
                    "cost_usd": round(
                        float(
                            r["estimated_cost_usd"]
                            if r["status"] == "completed"
                            else r["reserved_cost_usd"]
                        ),
                        8,
                    ),
                    "duration_ms": _duration_ms(r["created_at"], r["completed_at"]),
                    "created_at": r["created_at"],
                })
    finally:
        conn.close()

    return {
        "mode": "read_only",
        "database": f"sqlite:{resolved}",
        "pii_redacted": True,
        "index_visibility": {
            "active_stored_jobs": len(job_rows),
            "visible_active_jobs": visible_jobs,
            "quarantined_active_jobs": quarantined_jobs,
            "stale_hidden_jobs": stale_jobs,
            "configured_quarantined_sources": quarantined_sources_list(),
        },
        "telemetry_table_initialized": has_telemetry,
        "latest_agent_and_search_turns": telemetry_rows,
        "latest_discovery_requests": discoveries,
        "caches": {
            "query_embeddings": {
                "cached_entries": int(qe["cached_queries"]),
                "total_cache_hits": int(qe["total_hits"]),
                "last_used_at": qe["last_used_at"],
            },
            "relevance_verifications": {
                "cached_judgments": int(rv["cached_judgments"]),
                "latest_created_at": rv["latest_at"],
            },
        },
        "expenditure_and_usage": {
            "totals": totals,
            "by_provider": by_provider,
            "recent_calls": recent_calls,
        },
    }


def run_readonly_diagnostics(*, limit: int = 5) -> dict[str, Any]:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url.startswith(("postgres://", "postgresql://")):
        return _inspect_postgres_readonly(database_url, limit=limit)
    db_path = Path(os.getenv("BOARDFORGE_DB_PATH", "./boardforge.db"))
    return _inspect_sqlite_readonly(db_path, limit=limit)
