from fastapi.testclient import TestClient

from boardforge.api import app


def test_root_serves_boardforge_workspace():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "BoardForge" in response.text
    assert 'id="workspace"' in response.text
    assert 'id="searchBtn"' in response.text
    assert 'id="resumeInput"' in response.text
    assert "Upload your resume" in response.text
