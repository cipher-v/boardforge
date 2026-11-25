from fastapi.testclient import TestClient

import boardforge.api as api
from boardforge.intelligent_agent import ToolCallingAgent


app = api.app


def test_root_serves_resume_first_boardforge_ui():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "BoardForge" in response.text
    assert "Find work that fits you." in response.text
    assert 'id="resumeInput"' in response.text
    assert "Upload your resume" in response.text
    assert 'id="searchBtn"' in response.text
    assert 'id="results"' in response.text
    assert "Preferences & data sources" in response.text
    assert "Delete my data" in response.text
    assert "targetRoles" in response.text
    assert "scorePercent" not in response.text
    assert "relative rank" in response.text


def test_api_agent_endpoint_is_wired_to_bounded_intelligent_agent():
    assert isinstance(api.tool_agent, ToolCallingAgent)
