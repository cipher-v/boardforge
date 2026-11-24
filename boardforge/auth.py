from __future__ import annotations

import json
import os
import secrets

from fastapi import HTTPException, Request


def _bearer_token(request: Request) -> str:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token if scheme.casefold() == "bearer" else ""


def _profile_tokens() -> dict[str, str]:
    raw = os.getenv("BOARDFORGE_PROFILE_TOKENS", "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise RuntimeError("BOARDFORGE_PROFILE_TOKENS must be a JSON token-to-profile object")
    return value


def authorize_profile(request: Request, profile_id: str) -> None:
    tokens = _profile_tokens()
    if not tokens:
        return
    owner = tokens.get(_bearer_token(request))
    if owner is None:
        raise HTTPException(status_code=401, detail="valid bearer token required")
    if not secrets.compare_digest(owner, profile_id):
        raise HTTPException(status_code=403, detail="profile access denied")


def authorize_any_profile(request: Request) -> None:
    tokens = _profile_tokens()
    if tokens and _bearer_token(request) not in tokens:
        raise HTTPException(status_code=401, detail="valid bearer token required")


def authorize_admin(request: Request) -> None:
    expected = os.getenv("BOARDFORGE_ADMIN_TOKEN", "")
    if not expected:
        return
    supplied = _bearer_token(request)
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="admin bearer token required")
