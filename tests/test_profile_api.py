import json

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
    monkeypatch.setenv("BOARDFORGE_SEARCH_PROVIDER", "none")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BOARDFORGE_BRAVE_API_KEY", raising=False)
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


def test_two_user_profile_confirmation_persistence_deletion_and_isolation(tmp_path, monkeypatch):
    database = tmp_path / "isolated-profiles.db"
    store = Store(str(database))
    agent = JobAgent(store)
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "core_agent", agent)
    monkeypatch.setenv("BOARDFORGE_PROFILE_TOKENS", json.dumps({
        "token-a": "profile-a", "token-b": "profile-b",
    }))
    monkeypatch.setenv("BOARDFORGE_SEARCH_PROVIDER", "none")
    monkeypatch.setenv("BOARDFORGE_ENABLE_EMBEDDINGS", "0")
    monkeypatch.setenv("BOARDFORGE_ENABLE_RELEVANCE_VERIFIER", "0")
    monkeypatch.setenv("BOARDFORGE_ENABLE_INTELLIGENT_AGENT", "0")
    for key in ("TAVILY_API_KEY", "BOARDFORGE_BRAVE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    client = TestClient(api.app)
    headers_a = {"Authorization": "Bearer token-a"}
    headers_b = {"Authorization": "Bearer token-b"}
    resume_a = b"Education\nM.Tech Computer Science, Example IIT\nProjects\nBuilt a Python FastAPI RAG service."
    resume_b = b"Education\nB.Tech Computer Science, Example University\nProjects\nBuilt a PyTorch OpenCV vision pipeline."

    uploaded_a = client.post(
        "/resume?profile_id=profile-a&filename=a.txt&recommend=false",
        content=resume_a, headers={**headers_a, "Content-Type": "text/plain"},
    )
    uploaded_b = client.post(
        "/resume?profile_id=profile-b&filename=b.txt&recommend=false",
        content=resume_b, headers={**headers_b, "Content-Type": "text/plain"},
    )
    assert uploaded_a.status_code == uploaded_b.status_code == 200
    assert uploaded_a.json()["profile"]["resume_filename"] == "a.txt"
    assert "OpenCV" in uploaded_b.json()["profile"]["skills"]

    confirmed = client.put(
        "/candidate-profile/profile-a", headers=headers_a,
        json={"target_roles": ["Backend"], "skills": ["Python", "FastAPI"], "reviewed": True},
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["profile"]["reviewed"] is True
    assert client.get("/candidate-profile/profile-b", headers=headers_a).status_code == 403

    client.post("/preferences", headers=headers_a, json={
        "profile_id": "profile-a", "text": "India only",
    })
    client.post("/feedback", headers=headers_a, json={
        "profile_id": "profile-a", "job_id": "test:board:a", "verdict": "save", "note": "",
    })

    # Reopen the same isolated database to prove restoration is durable rather
    # than an in-memory artifact of the first Store instance.
    restored_store = Store(str(database))
    monkeypatch.setattr(api, "store", restored_store)
    monkeypatch.setattr(api, "core_agent", JobAgent(restored_store))
    restored = client.get("/candidate-profile/profile-a", headers=headers_a)
    assert restored.status_code == 200
    assert restored.json()["profile"]["target_roles"] == ["Backend"]
    assert restored.headers["cache-control"] == "no-store, private"
    assert "Authorization" in restored.headers["vary"]

    deleted = client.delete("/candidate-profile/profile-a", headers=headers_a)
    assert deleted.status_code == 200
    assert client.get("/candidate-profile/profile-a", headers=headers_a).json() == {"profile": None}
    assert restored_store.feedback_history("profile-a") == []
    assert restored_store.load_preferences("profile-a").countries == []
    remaining_b = client.get("/candidate-profile/profile-b", headers=headers_b)
    assert remaining_b.status_code == 200
    assert remaining_b.json()["profile"]["resume_filename"] == "b.txt"
