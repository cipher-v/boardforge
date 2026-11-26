from types import SimpleNamespace
import asyncio

import pytest

from boardforge.agent_tools import AgentToolExecutor
from boardforge.intelligent_agent import ToolCallingAgent
from boardforge.models import Preferences

QUERY = "Find entry-level machine learning engineering jobs in India requiring up to one year of experience."

@pytest.mark.parametrize("number,expected", [("three", 3), ("5", 5), ("ten", 10)])
def test_explicit_condition_parser(number, expected):
    text = f"{QUERY} If fewer than {number} suitable opportunities exist, discover additional job sources while preserving my location and experience constraints."
    assert ToolCallingAgent._explicit_discovery_condition(text) == (QUERY.rstrip("."), expected)

@pytest.mark.parametrize("suffix", ["", " If fewer than three suitable jobs exist, do not discover additional job sources.", " If fewer than three suitable jobs exist, discover additional job sources and change my preferences."])
def test_condition_parser_does_not_authorize_other_actions(suffix):
    assert ToolCallingAgent._explicit_discovery_condition(QUERY + suffix) is None

@pytest.mark.parametrize("count,threshold,queued", [(0, 3, True), (2, 3, True), (3, 3, False), (3, 5, True), (5, 5, False)])
def test_conditional_execution(monkeypatch, count, threshold, queued):
    calls = []
    class Store:
        def load_candidate_profile(self, profile_id): return None
        def load_preferences(self, profile_id): return Preferences()
        def feedback_history(self, profile_id, limit): return []
    class Core:
        store = Store()
        async def search_jobs(self, profile_id, query, top_k, **kwargs):
            calls.append(("search", query))
            return SimpleNamespace(data={"matches": [
                {"job": {"title": "ML Engineer", "company": "Example", "external_id": str(i)}, "status": "recommended"}
                for i in range(count)
            ], "retrieval": {}})
    class Provider:
        enabled = True
        provider_name = "fake"
        model = "fake"
        async def generate(self, request):
            raise AssertionError("No LLM reasoning allowed for this condition")
    async def queue(store, profile_id, profile, preferences, *, query, client):
        calls.append(("queue", query))
        assert "India" in preferences.countries
        assert preferences.max_required_experience_years == 1
        return {"status": "queued", "request_id": 42}
    monkeypatch.setattr("boardforge.agent_tools.queue_and_dispatch_discovery", queue)
    core = Core()
    agent = ToolCallingAgent(core, provider=Provider(), executor=AgentToolExecutor(core), enabled=True)
    result = asyncio.run(agent.run("test", f"{QUERY} If fewer than {threshold} suitable opportunities exist, discover additional job sources."))
    assert result["agent"]["provider_calls"] == 0
    assert result["agent"]["orchestration_mode"] == "deterministic_conditional"
    assert result["agent"]["discovery_requested"] is queued
    assert [name for name, _ in calls] == (["search", "queue"] if queued else ["search"])
    assert len(result["matches"]) == count
