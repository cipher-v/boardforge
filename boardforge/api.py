from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

from boardforge.agent import JobAgent
from boardforge.auth import auth_required, authorize_admin, authorize_any_profile, authorize_profile, session_profile
from boardforge.connectors.detect import detect_source
from boardforge.database import create_store
from boardforge.discovery import queue_and_dispatch_discovery
from boardforge.models import (
    CandidateProfile,
    CandidateProfileUpdate,
    FeedbackRequest,
    Preferences,
    PreferenceUpdate,
    SearchRequest,
)
from boardforge.observability import metrics_response, request_observability
from boardforge.resume import extract_candidate_profile, extract_resume_text
from boardforge.sync import sync_url
from boardforge.intelligent_agent import ToolCallingAgent


app = FastAPI(
    title="BoardForge",
    version="0.6.0",
    description="Job intelligence infrastructure: ingest, track, personalize, rank and evaluate.",
)
store = create_store()
core_agent = JobAgent(store)
tool_agent = ToolCallingAgent(core_agent)
app.middleware("http")(request_observability)


@app.middleware("http")
async def private_response_cache_control(request: Request, call_next):
    response = await call_next(request)
    private_prefixes = (
        "/candidate-profile/", "/recommendations/", "/discovery/", "/preferences/",
    )
    if request.url.path == "/session" or request.url.path.startswith(private_prefixes):
        response.headers["Cache-Control"] = "no-store, private"
        vary = {value.strip() for value in response.headers.get("Vary", "").split(",") if value.strip()}
        vary.add("Authorization")
        response.headers["Vary"] = ", ".join(sorted(vary))
    return response


def _reject_nul_param(*values: str) -> None:
    if any("\x00" in value for value in values):
        raise HTTPException(status_code=400, detail="input must not contain NUL bytes")


class SyncRequest(BaseModel):
    url: HttpUrl


_UI_PATH = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(_UI_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.6.0",
        "deployment_mode": "cloud-run" if os.getenv("K_SERVICE") else "local-demo",
    }


@app.get("/session")
def session(request: Request):
    return {"profile_id": session_profile(request), "auth_required": auth_required()}


@app.get("/ready")
def ready():
    if not store.ping():
        raise HTTPException(status_code=503, detail="database unavailable")
    embeddings = core_agent.embeddings
    embedding_state = {
        "enabled": embeddings.enabled,
        "model": embeddings.model if embeddings.enabled else None,
    }
    if embeddings.enabled:
        embedding_state.update(store.embedding_status(embeddings.model))
    return {
        "status": "ready",
        "store": type(store).__name__,
        "embeddings": embedding_state,
    }


@app.get("/metrics", include_in_schema=False)
def metrics(request: Request):
    authorize_admin(request)
    return metrics_response()


async def _schedule_discovery(
    profile_id: str,
    profile: CandidateProfile,
    query: str = "",
) -> dict:
    preferences = store.load_preferences(profile_id)
    return await queue_and_dispatch_discovery(
        store, profile_id, profile, preferences, query=query,
    )


async def _read_limited_body(request: Request, max_bytes: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        raise HTTPException(status_code=413, detail="resume must be 5 MB or smaller")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise HTTPException(status_code=413, detail="resume must be 5 MB or smaller")
    return bytes(body)


@app.post("/sync")
async def sync(req: SyncRequest, request: Request):
    authorize_admin(request)
    try:
        url = str(req.url)
        summary = await sync_url(url, store)
        source = detect_source(url)
        store.record_discovered_source(
            source.kind, source.key, source.board_url, "manual-source",
            synced=True, status="synced",
        )
        return summary.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/resume")
async def upload_resume(
    request: Request,
    profile_id: str = "default",
    filename: str = "resume.pdf",
    recommend: bool = True,
):
    _reject_nul_param(profile_id, filename)
    authorize_profile(request, profile_id)
    data = await _read_limited_body(request, 5 * 1024 * 1024)
    if not data:
        raise HTTPException(status_code=400, detail="resume file is empty")
    try:
        text = extract_resume_text(filename, data)
        profile = extract_candidate_profile(text, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not profile.skills and not profile.role_signals and not profile.education:
        raise HTTPException(status_code=400, detail="could not extract a useful profile from this resume")
    store.save_candidate_profile(profile_id, profile)
    if not recommend:
        return {"profile": profile.model_dump(), "matches": []}
    recommendations = await core_agent.recommend_jobs(profile_id, 15)
    return {
        "profile": profile.model_dump(),
        "discovery": await _schedule_discovery(profile_id, profile),
        **recommendations.data,
    }


@app.get("/candidate-profile/{profile_id}")
def candidate_profile(profile_id: str, request: Request):
    _reject_nul_param(profile_id)
    authorize_profile(request, profile_id)
    profile = store.load_candidate_profile(profile_id)
    return {"profile": profile.model_dump() if profile else None}


@app.put("/candidate-profile/{profile_id}")
def update_candidate_profile(profile_id: str, req: CandidateProfileUpdate, request: Request):
    _reject_nul_param(profile_id)
    authorize_profile(request, profile_id)
    profile = store.load_candidate_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="candidate profile not found")
    updates = req.model_dump(exclude_none=True)
    for field in ("skills", "target_roles", "domains"):
        if field in updates:
            updates[field] = list(dict.fromkeys(
                value.strip()[:80]
                for value in updates[field]
                if value.strip()
            ))[:30]
    profile = profile.model_copy(update=updates)
    store.save_candidate_profile(profile_id, profile)
    return {"profile": profile.model_dump()}


@app.delete("/candidate-profile/{profile_id}")
def delete_candidate_profile(profile_id: str, request: Request):
    _reject_nul_param(profile_id)
    authorize_profile(request, profile_id)
    store.delete_profile(profile_id)
    return {"deleted": True, "profile_id": profile_id}


@app.get("/recommendations/{profile_id}")
async def recommendations(
    profile_id: str,
    request: Request,
    top_k: int = Query(default=15, ge=1, le=50),
):
    _reject_nul_param(profile_id)
    authorize_profile(request, profile_id)
    result = await core_agent.recommend_jobs(profile_id, top_k)
    return result.data


@app.post("/discover/{profile_id}")
async def discover(
    profile_id: str,
    req: SearchRequest,
    request: Request,
):
    # req.profile_id is ignored here; ownership follows the path parameter.
    _reject_nul_param(profile_id)
    authorize_profile(request, profile_id)
    profile = store.load_candidate_profile(profile_id) or CandidateProfile()
    result = await core_agent.search_jobs(profile_id, req.query, req.top_k)
    queued = await _schedule_discovery(profile_id, profile, req.query)
    return {"discovery": queued, **result.data}


@app.get("/discovery/{request_id}")
def discovery_status(request_id: int, request: Request):
    authorize_any_profile(request)
    record = store.discovery_request(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="discovery request not found")
    if auth_required() and record.get("profile_id") != session_profile(request):
        raise HTTPException(status_code=404, detail="discovery request not found")
    return record


@app.get("/sources")
def sources(request: Request):
    authorize_admin(request)
    return {"sources": store.discovered_sources()}


@app.get("/preferences/{profile_id}")
def get_preferences(profile_id: str, request: Request):
    _reject_nul_param(profile_id)
    authorize_profile(request, profile_id)
    return {
        "profile_id": profile_id,
        "preferences": store.load_preferences(profile_id).model_dump(),
    }


@app.delete("/preferences/{profile_id}")
def reset_preferences(profile_id: str, request: Request):
    _reject_nul_param(profile_id)
    authorize_profile(request, profile_id)
    store.reset_preferences(profile_id)
    return {
        "reset": True,
        "profile_id": profile_id,
        "preferences": Preferences().model_dump(),
    }


@app.post("/preferences")
async def preferences(req: PreferenceUpdate, request: Request):
    authorize_profile(request, req.profile_id)
    result = await core_agent.update_preferences(req.profile_id, req.text)
    return {"message":result.message,"preferences":result.data}


@app.post("/search")
async def search(req: SearchRequest, request: Request):
    authorize_profile(request, req.profile_id)
    result = await core_agent.search_jobs(req.profile_id, req.query, req.top_k)
    return result.data


@app.post("/feedback")
async def feedback(req: FeedbackRequest, request: Request):
    authorize_profile(request, req.profile_id)
    result = await core_agent.record_feedback(req)
    return {"message":result.message,**result.data}


@app.post("/agent")
async def agent_turn(req: SearchRequest, request: Request):
    authorize_profile(request, req.profile_id)
    return await tool_agent.run(req.profile_id, req.query, top_k=req.top_k)
