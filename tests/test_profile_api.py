from fastapi.testclient import TestClient

import boardforge.api as api
from boardforge.agent import JobAgent
from boardforge.store import Store
from boardforge.tool_agent import ToolCallingAgent


RESUME = b"""
Tanish Example
B.Tech Computer Science, Example University
Built a RAG assistant using Python, FastAPI, PostgreSQL, embeddings and Docker.
Software Engineer Intern working on backend APIs and semantic search.
"""


def test_resume_profile_can_be_reviewed_and_deleted(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "api.db"))
    agent = JobAgent(store)
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "core_agent", agent)
    monkeypatch.setattr(api, "tool_agent", ToolCallingAgent(agent))
    monkeypatch.delenv("BOARDFORGE_BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BOARDFORGE_ENABLE_WEB_DISCOVERY", raising=False)
    client = TestClient(api.app)

    uploaded = client.post(
        "/resume?profile_id=u&filename=resume.txt",
        content=RESUME,
        headers={"Content-Type": "text/plain"},
    )
    assert uploaded.status_code == 200
    body = uploaded.json()
    assert body["discovery"]["status"] == "disabled"
    assert body["profile"]["evidence"]
    assert body["profile"]["reviewed"] is False

    reviewed = client.put(
        "/candidate-profile/u",
        json={"target_roles": ["AI/ML"], "skills": ["Python", "RAG"], "reviewed": True},
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["profile"]["target_roles"] == ["AI/ML"]
    assert reviewed.json()["profile"]["reviewed"] is True

    deleted = client.delete("/candidate-profile/u")
    assert deleted.status_code == 200
    assert client.get("/candidate-profile/u").json() == {"profile": None}


def test_resume_upload_rejects_oversized_body(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "large.db"))
    agent = JobAgent(store)
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "core_agent", agent)
    client = TestClient(api.app)
    response = client.post(
        "/resume?profile_id=u&filename=resume.txt",
        content=b"x" * (5 * 1024 * 1024 + 1),
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 413
