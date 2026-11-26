"""Curated ATS source seeding and safe registration of already-indexed trusted boards."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from boardforge.database import StoreProtocol
from boardforge.discovery.web import _valid_supported_ats_source
from boardforge.quarantine import is_quarantined_source
from boardforge.security import has_sensitive_query
from boardforge.sync import sync_url

SEED_PROVENANCE = "curated-seed"
INDEXED_BACKFILL_PROVENANCE = "indexed-backfill"
MAX_SEED_SOURCES = 200

CANONICAL_ATS_BOARD_TEMPLATES: dict[str, str] = {
    "greenhouse": "https://job-boards.greenhouse.io/{key}",
    "lever": "https://jobs.lever.co/{key}",
    "ashby": "https://jobs.ashbyhq.com/{key}",
}


def canonical_ats_board_url(source_kind: str, source_key: str) -> str | None:
    """Construct and validate the canonical official ATS board URL for (kind, key)."""
    template = CANONICAL_ATS_BOARD_TEMPLATES.get(source_kind.strip().casefold())
    slug = source_key.strip()
    if not template or not slug:
        return None
    candidate_url = template.format(key=slug)
    verified = _valid_supported_ats_source(candidate_url)
    return verified.board_url if verified is not None else None


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip())
            dt = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def load_seed_file(path: str | Path) -> list[str]:
    """Accept a JSON list of URLs or objects with a 'url' / 'board_url' field."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("sources", [])
    if not isinstance(raw, list):
        raise ValueError("seed file must be a JSON list or an object with a 'sources' list")
    urls: list[str] = []
    for item in raw:
        if isinstance(item, str):
            urls.append(item.strip())
        elif isinstance(item, dict):
            urls.append(str(item.get("url") or item.get("board_url") or "").strip())
    return [url for url in urls if url]


def validate_seed_sources(urls: list[str]) -> dict[str, Any]:
    """Offline validation against the Greenhouse/Lever/Ashby allowlist (no network)."""
    if len(urls) > MAX_SEED_SOURCES:
        raise ValueError(f"at most {MAX_SEED_SOURCES} sources per seed run")
    valid: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for url in urls:
        if has_sensitive_query(url):
            rejected.append({"url": url, "reason": "sensitive query parameter"})
            continue
        source = _valid_supported_ats_source(url)
        if source is None:
            rejected.append({"url": url, "reason": "not an official Greenhouse/Lever/Ashby board"})
            continue
        if is_quarantined_source(source.kind, source.key):
            rejected.append({"url": url, "reason": "source is quarantined"})
            continue
        marker = (source.kind, source.key.casefold())
        if marker in seen:
            rejected.append({"url": url, "reason": "duplicate board"})
            continue
        seen.add(marker)
        valid.append({"kind": source.kind, "key": source.key, "board_url": source.board_url})
    return {"valid": valid, "rejected": rejected}


async def apply_seed_sources(
    store: StoreProtocol, sources: list[dict[str, str]], *, concurrency: int = 3,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, min(concurrency, 4)))

    async def one(source: dict[str, str]) -> dict[str, Any]:
        async with semaphore:
            try:
                summary = await sync_url(source["board_url"], store)
                store.record_discovered_source(
                    source["kind"], source["key"], source["board_url"], SEED_PROVENANCE,
                    synced=True, status="synced",
                )
                return {**source, "status": "synced", "fetched": summary.fetched,
                        "inserted": summary.inserted}
            except Exception as exc:
                store.record_discovered_source(
                    source["kind"], source["key"], source["board_url"], SEED_PROVENANCE,
                    status="failed",
                )
                return {**source, "status": "failed", "error": str(exc)[:300]}

    return list(await asyncio.gather(*(one(source) for source in sources)))


def database_identity(store: StoreProtocol) -> str:
    if type(store).__name__ == "PostgresStore":
        with store.connect() as connection:
            row = connection.execute("SELECT current_database() AS db").fetchone()
        return f"postgres:{row['db']}"
    raw_path = getattr(store, "path", getattr(store, "db_path", None))
    if raw_path is not None:
        return f"sqlite:{Path(str(raw_path)).resolve()}"
    return "sqlite:unknown"


def matches_expected_database(identity: str, expected: str) -> bool:
    """Require exact database name/path match so 'boardforge' never matches 'boardforge_smoke'."""
    exp = expected.strip()
    if not exp:
        return False
    if identity == exp:
        return True
    if identity.startswith("postgres:"):
        actual_db = identity.removeprefix("postgres:")
        exp_db = exp.removeprefix("postgres:")
        return actual_db == exp_db
    if identity.startswith("sqlite:"):
        actual_raw = identity.removeprefix("sqlite:")
        if actual_raw == "unknown":
            return False
        actual_path = Path(actual_raw).resolve()
        exp_clean = exp.removeprefix("sqlite:")
        if actual_path.name == exp_clean:
            return True
        try:
            return actual_path == Path(exp_clean).resolve()
        except OSError:
            return False
    return False


def assert_expected_database(store: StoreProtocol, expected: str | None) -> str:
    if not expected or not expected.strip():
        raise RuntimeError("explicit --expect-database is required to confirm the target database")
    identity = database_identity(store)
    if not matches_expected_database(identity, expected):
        raise RuntimeError(
            f"connected database {identity!r} does not match --expect-database {expected!r}"
        )
    return identity


def _observed_indexed_board_stats(store: StoreProtocol) -> list[dict[str, Any]]:
    if hasattr(store, "connect"):
        with store.connect() as connection:
            job_rows = connection.execute(
                """SELECT source_kind, source_key, COUNT(*) AS active_jobs,
                          MIN(first_seen) AS first_seen, MAX(last_seen) AS last_seen
                   FROM jobs
                   WHERE status = 'active'
                   GROUP BY source_kind, source_key
                   ORDER BY source_kind, source_key"""
            ).fetchall()
            sync_rows = connection.execute(
                """SELECT source_kind, source_key,
                          MIN(started_at) AS first_started,
                          MAX(completed_at) AS last_completed
                   FROM sync_runs
                   WHERE status = 'succeeded' AND completed_at IS NOT NULL
                   GROUP BY source_kind, source_key"""
            ).fetchall()
        sync_by_marker = {
            (str(r["source_kind"]), str(r["source_key"]).casefold()): dict(r)
            for r in sync_rows
        }
        stats: list[dict[str, Any]] = []
        for row in job_rows:
            r = dict(row)
            kind = str(r["source_kind"])
            key = str(r["source_key"])
            sr = sync_by_marker.get((kind, key.casefold()), {})
            first_candidates = [
                ts for ts in (
                    _parse_utc_timestamp(r.get("first_seen")),
                    _parse_utc_timestamp(sr.get("first_started")),
                )
                if ts is not None
            ]
            last_candidates = [
                ts for ts in (
                    _parse_utc_timestamp(r.get("last_seen")),
                    _parse_utc_timestamp(sr.get("last_completed")),
                )
                if ts is not None
            ]
            stats.append({
                "source_kind": kind,
                "source_key": key,
                "active_jobs": int(r["active_jobs"]),
                "first_discovered": min(first_candidates) if first_candidates else None,
                "last_synced": max(last_candidates) if last_candidates else None,
            })
        return stats

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for job in store.active_jobs():
        marker = (job.source_kind, job.source_key.casefold())
        entry = grouped.setdefault(marker, {
            "source_kind": job.source_kind,
            "source_key": job.source_key,
            "active_jobs": 0,
            "first_discovered": None,
            "last_synced": None,
        })
        entry["active_jobs"] += 1
        fs = _parse_utc_timestamp(job.first_seen_at)
        ls = _parse_utc_timestamp(job.last_seen_at)
        if fs and (entry["first_discovered"] is None or fs < entry["first_discovered"]):
            entry["first_discovered"] = fs
        if ls and (entry["last_synced"] is None or ls > entry["last_synced"]):
            entry["last_synced"] = ls
    return sorted(grouped.values(), key=lambda x: (x["source_kind"], x["source_key"]))


def register_indexed_sources(
    store: StoreProtocol,
    *,
    apply: bool = False,
    interval_seconds: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Inspect or register already-indexed trusted ATS boards in discovered_sources.

    Preserves each board's historical first_discovered / last_synced timestamps from
    jobs and sync_runs so stale boards become immediately due in sources_due() while
    recently synchronized boards retain their remaining freshness window.
    Makes zero external HTTP calls in both dry-run and apply modes.
    """
    sync_interval = max(
        60,
        int(
            interval_seconds
            if interval_seconds is not None
            else os.getenv("BOARDFORGE_SOURCE_SYNC_SECONDS", "3600")
        ),
    )
    ref_now = (
        now.astimezone(timezone.utc)
        if (now is not None and now.tzinfo is not None)
        else (now.replace(tzinfo=timezone.utc) if now is not None else datetime.now(timezone.utc))
    )
    identity = database_identity(store)
    existing_by_marker: dict[tuple[str, str], dict[str, Any]] = {}
    for row in store.discovered_sources():
        kind = str(row.get("source_kind") or "")
        key = str(row.get("source_key") or "").casefold()
        if kind and key:
            existing_by_marker[(kind, key)] = dict(row)

    to_register: list[dict[str, Any]] = []
    already_registered: list[dict[str, Any]] = []
    skipped_unsupported: list[dict[str, Any]] = []

    for item in _observed_indexed_board_stats(store):
        kind = item["source_kind"]
        key = item["source_key"]
        board_url = canonical_ats_board_url(kind, key)
        if board_url is None:
            skipped_unsupported.append({
                "source_kind": kind,
                "source_key": key,
                "active_jobs": item["active_jobs"],
                "reason": "not an official Greenhouse/Lever/Ashby board",
            })
            continue
        if is_quarantined_source(kind, key):
            skipped_unsupported.append({
                "source_kind": kind,
                "source_key": key,
                "active_jobs": item["active_jobs"],
                "reason": "source is quarantined",
            })
            continue

        first_disc: datetime | None = item["first_discovered"]
        last_sync: datetime | None = item["last_synced"]
        marker = (kind, key.casefold())
        existing = existing_by_marker.get(marker)
        existing_sync = _parse_utc_timestamp(existing.get("last_synced")) if existing else None

        effective_sync = (
            max(ts for ts in (last_sync, existing_sync) if ts is not None)
            if (last_sync is not None or existing_sync is not None)
            else None
        )
        age_seconds = (
            round(max(0.0, (ref_now - effective_sync).total_seconds()), 1)
            if effective_sync is not None
            else None
        )
        due_now = effective_sync is None or (ref_now - effective_sync).total_seconds() >= sync_interval

        entry = {
            "source_kind": kind,
            "source_key": key,
            "board_url": board_url,
            "active_jobs": item["active_jobs"],
            "first_discovered": first_disc.isoformat() if first_disc else None,
            "last_synced": effective_sync.isoformat() if effective_sync else None,
            "age_seconds": age_seconds,
            "due_for_maintenance": due_now,
        }

        needs_backfill = (
            existing is not None
            and existing_sync is None
            and last_sync is not None
            and str(existing.get("last_status") or "") != "failed"
        )
        if existing is not None and not needs_backfill:
            already_registered.append({
                **entry,
                "action": "already_registered",
                "last_status": existing.get("last_status"),
            })
            continue

        action = "backfill_sync_timestamp" if needs_backfill else "register"
        to_register.append({
            **entry,
            "action": action,
            "_first_dt": first_disc,
            "_sync_dt": last_sync,
            "_provenance": (
                str(existing.get("discovery_query") or INDEXED_BACKFILL_PROVENANCE)
                if existing is not None
                else INDEXED_BACKFILL_PROVENANCE
            ),
        })

    registered_count = 0
    if apply:
        for entry in to_register:
            sync_dt: datetime | None = entry["_sync_dt"]
            first_dt: datetime | None = entry["_first_dt"]
            store.record_discovered_source(
                entry["source_kind"],
                entry["source_key"],
                entry["board_url"],
                entry["_provenance"],
                synced=sync_dt is not None,
                status="synced" if sync_dt is not None else "discovered",
                synced_at=sync_dt,
                discovered_at=first_dt or sync_dt,
            )
            registered_count += 1

    public_to_register = [
        {k: v for k, v in entry.items() if not k.startswith("_")}
        for entry in to_register
    ]
    all_trusted = [*public_to_register, *already_registered]
    return {
        "mode": "applied" if apply else "dry_run",
        "database": identity,
        "network_calls": 0,
        "interval_seconds": sync_interval,
        "indexed_trusted_boards": len(all_trusted),
        "registered_count": registered_count,
        "to_register_count": len(public_to_register),
        "already_registered_count": len(already_registered),
        "due_for_maintenance_count": sum(1 for x in all_trusted if x["due_for_maintenance"]),
        "fresh_within_interval_count": sum(1 for x in all_trusted if not x["due_for_maintenance"]),
        "skipped_unsupported_count": len(skipped_unsupported),
        "to_register": public_to_register,
        "already_registered": already_registered,
        "skipped_unsupported": skipped_unsupported,
    }
