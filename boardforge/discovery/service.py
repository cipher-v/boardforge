from __future__ import annotations

import hashlib
import json
import os

from boardforge.discovery.web import SearchProvider, create_search_provider, discover_and_sync
from boardforge.models import CandidateProfile, Preferences
from boardforge.database import StoreProtocol


def discovery_fingerprint(
    profile: CandidateProfile,
    preferences: Preferences,
    query: str,
    *,
    profile_id: str = "",
) -> str:
    payload = {
        "query": " ".join(query.casefold().split()),
        "roles": preferences.roles or profile.target_roles or profile.role_signals,
        "locations": preferences.locations,
        "countries": preferences.countries,
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
    client = client or create_search_provider()
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
            client=client,
        )
        if report.failures and report.synced_boards == 0:
            store.retry_discovery(
                request_id,
                f"discovery had {report.failures} failures and no successful board sync",
            )
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
    requeued = store.requeue_stale_discoveries()
    processed = 0
    for item in store.claim_discoveries(limit):
        profile = store.load_candidate_profile(item["profile_id"]) or CandidateProfile()
        preferences = store.load_preferences(item["profile_id"])
        await run_discovery_request(
            store,
            item["id"],
            profile,
            preferences,
            query=item["query"],
            client=client,
            claimed=True,
        )
        processed += 1
    return {"processed": processed, "requeued_stale": requeued}
