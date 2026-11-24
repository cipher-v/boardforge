from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

from boardforge.agent import JobAgent
from boardforge.models import FeedbackRequest, PreferenceUpdate, SearchRequest
from boardforge.resume import extract_candidate_profile, extract_resume_text
from boardforge.store import Store
from boardforge.sync import sync_url
from boardforge.tool_agent import ToolCallingAgent


app = FastAPI(
    title="BoardForge",
    version="0.5.0",
    description="Job intelligence infrastructure: ingest, track, personalize, rank and evaluate.",
)
store = Store()
core_agent = JobAgent(store)
tool_agent = ToolCallingAgent(core_agent)


class SyncRequest(BaseModel):
    url: HttpUrl


_UI_PATH = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(_UI_PATH)


@app.get("/health")
def health():
    return {"status":"ok","version":"0.5.0"}


@app.post("/sync")
async def sync(req: SyncRequest):
    try:
        return (await sync_url(str(req.url), store)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/resume")
async def upload_resume(
    request: Request,
    profile_id: str = "default",
    filename: str = "resume.pdf",
):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="resume file is empty")
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="resume must be 5 MB or smaller")
    try:
        text = extract_resume_text(filename, data)
        profile = extract_candidate_profile(text, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not profile.skills and not profile.role_signals and not profile.education:
        raise HTTPException(status_code=400, detail="could not extract a useful profile from this resume")
    store.save_candidate_profile(profile_id, profile)
    recommendations = await core_agent.search_jobs(profile_id, "", 10)
    return {
        "profile": profile.model_dump(),
        **recommendations.data,
    }


@app.get("/candidate-profile/{profile_id}")
def candidate_profile(profile_id: str):
    profile = store.load_candidate_profile(profile_id)
    return {"profile": profile.model_dump() if profile else None}


@app.post("/preferences")
async def preferences(req: PreferenceUpdate):
    result = await core_agent.update_preferences(req.profile_id, req.text)
    return {"message":result.message,"preferences":result.data}


@app.post("/search")
async def search(req: SearchRequest):
    result = await core_agent.search_jobs(req.profile_id, req.query, req.top_k)
    return result.data


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    result = await core_agent.record_feedback(req)
    return {"message":result.message,**result.data}


@app.post("/agent")
async def agent_turn(req: SearchRequest):
    return await tool_agent.run(req.profile_id, req.query)
