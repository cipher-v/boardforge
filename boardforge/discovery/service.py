from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any

import httpx

from boardforge.database import StoreProtocol
from boardforge.discovery.web import SearchProvider, create_search_provider, discover_and_sync
from boardforge.models import CandidateProfile, Preferences
from boardforge.preferences import effective_search_preferences


_LOCAL_DRAIN_RUNNING: set[int] = set()
_LOCAL_DRAIN_PENDING: set[int] = set()
_LOCAL_DRAIN_TASKS: set[asyncio.Task[Any]] = set()
_LAST_CLOUD_DISPATCH_MONOTONIC: float = 0.0


def discovery_fingerprint(
    profile: CandidateProfile,
    preferences: Preferences,
    query: str,
    *,
    profile_id: str = "",
) -> str:
    effective = (
        effective_search_preferences(preferences, query)
        if query.strip()
        else preferences
    )
    payload = {
        "query": " ".join(query.casefold().split()),
        "roles": effective.roles or profile.target_roles or profile.role_signals,
        "locations": effective.locations,
        "countries": effective.countries,
        "max_exp": effective.max_required_experience_years,
        "profile_id": profile_id,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def queue_discovery(
    store: StoreProtocol,
    profile_id: str,
    profile: CandidateProfile,
    preferences: Preferences,
    *,
    query: str = "",
    client: SearchProvider | None = None,
) -> dict:
    client = client or create_search_provider(store)
    if not client.enabled:
        return {
            "request_id": None,
            "status": "disabled",
            "cached": False,
            "report": None,
            "reason": (
                "web discovery is disabled; search uses the existing local index"
            ),
        }
    return store.queue_discovery(
        profile_id,
        discovery_fingerprint(profile, preferences, query, profile_id=profile_id),
        query,
        ttl_seconds=max(60, int(os.getenv("BOARDFORGE_DISCOVERY_TTL_SECONDS", "900"))),
    )


async def run_discovery_request(
    store: StoreProtocol,
    request_id: int,
    profile: CandidateProfile,
    preferences: Preferences,
    *,
    query: str = "",
    profile_id: str = "",
    client: SearchProvider | None = None,
    claimed: bool = False,
) -> None:
    if not claimed and not store.start_discovery(request_id):
        return
    try:
        report = await discover_and_sync(
            store,
            profile,
            preferences,
            query=query,
            profile_id=profile_id,
            client=client,
        )
        if report.failures and report.synced_boards == 0:
            if report.retryable:
                store.retry_discovery(
                    request_id,
                    report.reason or (
                        f"discovery had {report.failures} failures and no successful board sync"
                    ),
                )
            else:
                store.finish_discovery(request_id, report.as_dict(), failed=True)
        else:
            store.finish_discovery(request_id, report.as_dict())
    except Exception as exc:  # keep failures inspectable without breaking search
        store.retry_discovery(request_id, str(exc))


async def run_pending_discoveries(
    store: StoreProtocol,
    *,
    limit: int = 10,
    client: SearchProvider | None = None,
) -> dict:
    bounded_limit = max(1, min(25, int(limit)))
    requeued = store.requeue_stale_discoveries()
    processed = 0
    for item in store.claim_discoveries(bounded_limit):
        profile = store.load_candidate_profile(item["profile_id"]) or CandidateProfile()
        preferences = store.load_preferences(item["profile_id"])
        await run_discovery_request(
            store,
            item["id"],
            profile,
            preferences,
            query=item["query"],
            profile_id=item["profile_id"],
            client=client,
            claimed=True,
        )
        processed += 1
    return {"processed": processed, "requeued_stale": requeued}


async def _drain_local_store(
    store: StoreProtocol,
    *,
    client: SearchProvider | None = None,
) -> None:
    from boardforge.semantic import EmbeddingClient, warm_job_embeddings

    key = id(store)
    try:
        while True:
            _LOCAL_DRAIN_PENDING.discard(key)
            cycle_limit = max(
                1, min(10, int(os.getenv("BOARDFORGE_DISCOVERY_REQUESTS_PER_CYCLE", "5")))
            )
            outcome = await run_pending_discoveries(
                store, limit=cycle_limit, client=client
            )
            if outcome["processed"] > 0 and client is None:
                embeddings = EmbeddingClient(budget_store=store)
                if embeddings.enabled:
                    try:
                        await warm_job_embeddings(
                            store,
                            embeddings,
                            batch_size=max(
                                1,
                                min(
                                    64,
                                    int(os.getenv("BOARDFORGE_EMBEDDING_BATCH_SIZE", "32")),
                                ),
                            ),
                            max_jobs=max(
                                1,
                                min(
                                    250,
                                    int(
                                        os.getenv(
                                            "BOARDFORGE_DISCOVERY_EMBED_BACKFILL_LIMIT",
                                            "32",
                                        )
                                    ),
                                ),
                            ),
                        )
                    except Exception:
                        pass
            if key not in _LOCAL_DRAIN_PENDING:
                break
    finally:
        _LOCAL_DRAIN_RUNNING.discard(key)
        _LOCAL_DRAIN_PENDING.discard(key)


def _cloud_run_access_token() -> str:
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    if not credentials.valid or not credentials.token:
        credentials.refresh(GoogleAuthRequest())
    if not credentials.token:
        raise RuntimeError("failed to acquire Cloud Run ADC access token")
    return str(credentials.token)


async def trigger_cloud_run_maintenance_job() -> dict[str, Any]:
    """Dispatch the bounded one-shot Cloud Run maintenance job on demand."""
    global _LAST_CLOUD_DISPATCH_MONOTONIC

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    region = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
    job_name = os.getenv("BOARDFORGE_MAINTENANCE_JOB", "boardforge-maintenance").strip()
    if not project or not job_name:
        return {
            "mode": "cloud_run_job",
            "dispatched": False,
            "reason": "missing_project_or_job_configuration",
        }

    cooldown = max(
        0.0,
        min(300.0, float(os.getenv("BOARDFORGE_JOB_DISPATCH_COOLDOWN_SECONDS", "15"))),
    )
    now = time.monotonic()
    if _LAST_CLOUD_DISPATCH_MONOTONIC and (now - _LAST_CLOUD_DISPATCH_MONOTONIC) < cooldown:
        return {
            "mode": "cloud_run_job",
            "dispatched": False,
            "coalesced": True,
            "job": job_name,
        }

    url = (
        f"https://run.googleapis.com/v2/projects/{project}/locations/"
        f"{region}/jobs/{job_name}:run"
    )
    try:
        token = await asyncio.to_thread(_cloud_run_access_token)
        async with httpx.AsyncClient(timeout=5.0) as http:
            response = await http.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={},
            )
        if response.status_code >= 400:
            return {
                "mode": "cloud_run_job",
                "dispatched": False,
                "status_code": response.status_code,
                "error": f"cloud_run_job_http_{response.status_code}",
            }
        _LAST_CLOUD_DISPATCH_MONOTONIC = time.monotonic()
        return {
            "mode": "cloud_run_job",
            "dispatched": True,
            "job": job_name,
        }
    except Exception as exc:
        return {
            "mode": "cloud_run_job",
            "dispatched": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def dispatch_discovery_execution(
    store: StoreProtocol,
    *,
    client: SearchProvider | None = None,
) -> dict[str, Any]:
    """Schedule bounded worker execution for queued discovery rows."""
    configured = os.getenv("BOARDFORGE_DISCOVERY_DISPATCH_MODE", "auto").strip().casefold()
    if configured == "auto":
        mode = "cloud_run_job" if os.getenv("K_SERVICE") else "local"
    elif configured in {"local", "cloud_run_job", "none"}:
        mode = configured
    else:
        raise ValueError(
            "BOARDFORGE_DISCOVERY_DISPATCH_MODE must be auto, local, cloud_run_job, or none"
        )

    if mode == "none":
        return {"mode": "none", "dispatched": False}
    if mode == "cloud_run_job":
        return await trigger_cloud_run_maintenance_job()

    if client is not None and not hasattr(client, "search_urls"):
        return {"mode": "local", "dispatched": False, "reason": "non_executable_client"}

    key = id(store)
    if key in _LOCAL_DRAIN_RUNNING:
        _LOCAL_DRAIN_PENDING.add(key)
        return {"mode": "local", "dispatched": True, "coalesced": True}

    _LOCAL_DRAIN_RUNNING.add(key)
    task = asyncio.create_task(_drain_local_store(store, client=client))
    _LOCAL_DRAIN_TASKS.add(task)
    task.add_done_callback(_LOCAL_DRAIN_TASKS.discard)
    return {"mode": "local", "dispatched": True}


async def queue_and_dispatch_discovery(
    store: StoreProtocol,
    profile_id: str,
    profile: CandidateProfile,
    preferences: Preferences,
    *,
    query: str = "",
    client: SearchProvider | None = None,
) -> dict[str, Any]:
    queued = queue_discovery(
        store,
        profile_id,
        profile,
        preferences,
        query=query,
        client=client,
    )
    if queued.get("status") == "queued":
        queued["dispatch"] = await dispatch_discovery_execution(store, client=client)
    return queued
