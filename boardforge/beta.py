"""Persistent production admission and pseudonymous aggregate usage counters."""
from contextvars import ContextVar
import os

from fastapi import HTTPException

current_profile: ContextVar[str | None] = ContextVar("beta_profile", default=None)
LIMITS = {"search": 20, "verification": 3, "reasoning": 3, "resume": 3,
          "discovery": 1, "preferences": 20, "profile_update": 20,
          "feedback": 50, "apply_click": 50}
GLOBAL_LIMITS = {"search": 220, "verification": 33, "reasoning": 33,
                 "resume": 33, "discovery": 5}


def enabled():
    return os.getenv("BOARDFORGE_BETA_LIMITS", "0") == "1"


def admit(store, operation, profile_id=None):
    if not enabled():
        return True
    profile_id = profile_id or current_profile.get()
    if not profile_id:
        return False
    limit = LIMITS[operation]
    return store.consume_beta_usage(profile_id, operation, limit, GLOBAL_LIMITS.get(operation))


def require(store, operation, profile_id=None):
    if not admit(store, operation, profile_id):
        raise HTTPException(429, detail=f"Daily {operation.replace('_', ' ')} limit reached. Try again tomorrow (UTC).")


def record(store, operation, profile_id=None):
    if not enabled():
        return
    profile_id = profile_id or current_profile.get()
    if profile_id:
        # Telemetry failure must not turn a completed user action into a failure.
        try:
            store.consume_beta_usage(profile_id, operation, 2147483647, None)
        except Exception:
            pass
