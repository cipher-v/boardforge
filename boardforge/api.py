from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from boardforge.agent import JobAgent
from boardforge.models import FeedbackRequest, PreferenceUpdate, SearchRequest
from boardforge.store import Store
from boardforge.sync import sync_url
from boardforge.tool_agent import ToolCallingAgent


app = FastAPI(
    title="BoardForge",
    version="0.4.0",
    description="Job intelligence infrastructure: ingest, track, personalize, rank and evaluate.",
)
store = Store()
core_agent = JobAgent(store)
tool_agent = ToolCallingAgent(core_agent)


class SyncRequest(BaseModel):
    url: HttpUrl


@app.get("/health")
def health():
    return {"status":"ok","version":"0.4.0"}


@app.post("/sync")
async def sync(req: SyncRequest):
    try:
        return (await sync_url(str(req.url), store)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
