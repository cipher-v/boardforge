import json

from fastapi.testclient import TestClient

from boardforge.api import app


def test_profile_tokens_enforce_ownership(monkeypatch):
    monkeypatch.setenv("BOARDFORGE_PROFILE_TOKENS", json.dumps({"token-a": "profile-a"}))
    client = TestClient(app)

    missing = client.post("/search", json={"profile_id": "profile-a", "query": "backend"})
    wrong_profile = client.post(
        "/search",
        json={"profile_id": "profile-b", "query": "backend"},
        headers={"Authorization": "Bearer token-a"},
    )
    allowed = client.post(
        "/search",
        json={"profile_id": "profile-a", "query": "backend"},
        headers={"Authorization": "Bearer token-a"},
    )

    assert missing.status_code == 401
    assert wrong_profile.status_code == 403
    assert allowed.status_code == 200


def test_admin_token_protects_source_mutation(monkeypatch):
    monkeypatch.setenv("BOARDFORGE_ADMIN_TOKEN", "admin-secret")
    client = TestClient(app)
    blocked = client.post("/sync", json={"url": "https://jobs.lever.co/example"})
    assert blocked.status_code == 401


def test_readiness_and_prometheus_metrics_are_exposed():
    client = TestClient(app)
    assert client.get("/ready").status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "boardforge_http_requests_total" in metrics.text
