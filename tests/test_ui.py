from fastapi.testclient import TestClient

from boardforge.api import app


def test_root_serves_resume_first_boardforge_ui():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "BoardForge" in response.text
    assert "Find jobs that fit you." in response.text
    assert 'id="resumeInput"' in response.text
    assert "Upload your resume" in response.text
    assert 'id="searchBtn"' in response.text
    assert 'id="results"' in response.text
    assert "Preferences & data sources" in response.text
