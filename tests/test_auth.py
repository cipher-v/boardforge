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


def test_invite_session_resolves_identity(monkeypatch):
    monkeypatch.setenv("BOARDFORGE_PROFILE_TOKENS", json.dumps({
        "token-a": "profile-a", "token-b": "profile-b",
    }))
    client = TestClient(app)
    assert client.get("/session").status_code == 401
    assert client.get("/session", headers={"Authorization": "Bearer invalid"}).status_code == 401
    for token, profile in (("token-a", "profile-a"), ("token-b", "profile-b")):
        response = client.get("/session", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json() == {"profile_id": profile, "auth_required": True}


def test_cross_profile_operations_rejected(monkeypatch):
    monkeypatch.setenv("BOARDFORGE_PROFILE_TOKENS", json.dumps({
        "token-a": "profile-a", "token-b": "profile-b",
    }))
    client = TestClient(app)
    for token, other in (("token-a", "profile-b"), ("token-b", "profile-a")):
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get(f"/candidate-profile/{other}", headers=headers).status_code == 403
        assert client.put(f"/candidate-profile/{other}", headers=headers,
                          json={"skills": ["Python"]}).status_code == 403
        assert client.delete(f"/candidate-profile/{other}", headers=headers).status_code == 403
        assert client.post(f"/resume?profile_id={other}&recommend=false",
                           headers=headers, content=b"test").status_code == 403
        assert client.get(f"/recommendations/{other}", headers=headers).status_code == 403
        for path in ("/search", "/agent", f"/discover/{other}"):
            assert client.post(path, headers=headers,
                               json={"profile_id": other, "query": "backend"}).status_code == 403
        assert client.post("/preferences", headers=headers,
                           json={"profile_id": other, "text": "India only"}).status_code == 403
        assert client.post("/feedback", headers=headers,
                           json={"profile_id": other, "job_id": "test:board:1",
                                 "verdict": "save", "note": ""}).status_code == 403


def test_discovery_status_checks_owner(monkeypatch):
    from boardforge.api import store

    monkeypatch.setenv("BOARDFORGE_PROFILE_TOKENS", json.dumps({
        "token-a": "profile-a", "token-b": "profile-b",
    }))
    record = {"request_id": 42, "profile_id": "profile-a", "status": "completed"}
    monkeypatch.setattr(store, "discovery_request", lambda request_id: record if request_id == 42 else None)
    client = TestClient(app)
    assert client.get("/discovery/42").status_code == 401
    assert client.get("/discovery/42", headers={"Authorization": "Bearer token-b"}).status_code == 404
    response = client.get("/discovery/42", headers={"Authorization": "Bearer token-a"})
    assert response.status_code == 200
    assert response.json() == record


def test_cloud_mode_fails_closed_without_secrets(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "boardforge-test")
    monkeypatch.delenv("BOARDFORGE_PROFILE_TOKENS", raising=False)
    monkeypatch.delenv("BOARDFORGE_ADMIN_TOKEN", raising=False)
    client = TestClient(app)
    assert client.get("/session").status_code == 401
    assert client.get("/candidate-profile/local-demo").status_code == 401
    assert client.get("/metrics").status_code == 403
    assert client.get("/sources").status_code == 403


def test_candidate_cannot_access_admin_endpoints(monkeypatch):
    monkeypatch.setenv("BOARDFORGE_PROFILE_TOKENS", json.dumps({"token-a": "profile-a"}))
    monkeypatch.setenv("BOARDFORGE_ADMIN_TOKEN", "admin-secret")
    client = TestClient(app)
    headers = {"Authorization": "Bearer token-a"}
    assert client.get("/sources", headers=headers).status_code == 401
    assert client.get("/metrics", headers=headers).status_code == 401
    assert client.post("/sync", headers=headers,
                       json={"url": "https://jobs.lever.co/example"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer admin-secret"}).status_code == 200
