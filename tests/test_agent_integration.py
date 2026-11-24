import asyncio
from pathlib import Path

from boardforge.agent import JobAgent
from boardforge.models import Job, Preferences
from boardforge.preferences import deterministic_extract
from boardforge.store import Store


class DeterministicExtractor:
    async def extract(self, text: str, current: Preferences | None = None) -> Preferences:
        return deterministic_extract(text, current)


class DisabledEmbeddings:
    enabled = False


def make_agent(tmp_path: Path) -> tuple[JobAgent, Store]:
    store = Store(str(tmp_path / "agent.db"))
    return JobAgent(store, extractor=DeterministicExtractor(), embeddings=DisabledEmbeddings()), store


def test_actual_agent_runs_preference_only_plan(tmp_path: Path):
    agent, store = make_agent(tmp_path)
    store.save_preferences("u", Preferences(roles=["QA", "Backend"]))
    result = asyncio.run(agent.handle("u", "Do not show QA roles anymore"))
    assert [step["tool"] for step in result.data["trace"]] == ["update_preferences"]
    assert store.load_preferences("u").roles == ["Backend"]
    assert "matches" not in result.data


def test_actual_agent_executes_feedback_with_arguments(tmp_path: Path):
    agent, store = make_agent(tmp_path)
    result = asyncio.run(agent.handle("u", "Reject benchmark:v2:ai-llm-blr"))
    assert result.data["trace"] == [{
        "tool": "record_feedback",
        "arguments": {
            "job_id": "benchmark:v2:ai-llm-blr",
            "verdict": "reject",
            "note": "",
        },
    }]
    assert store.rejected_job_ids("u") == {"benchmark:v2:ai-llm-blr"}


def test_actual_agent_updates_preferences_before_search(tmp_path: Path):
    agent, store = make_agent(tmp_path)
    store.upsert_source_jobs("benchmark", "v2", [Job(
        source_kind="benchmark",
        source_key="v2",
        external_id="ai-1",
        title="AI Engineer",
        company="Demo",
        country="India",
        location="Bengaluru, India",
    )])
    result = asyncio.run(agent.handle("u", "Only India roles. Show me AI jobs"))
    assert [step["tool"] for step in result.data["trace"]] == [
        "update_preferences",
        "search_jobs",
    ]
    assert result.data["matches"][0]["job"]["external_id"] == "ai-1"


def test_search_uses_transient_role_without_persisting_it(tmp_path: Path):
    agent, store = make_agent(tmp_path)
    store.save_preferences("u", Preferences(roles=["AI/ML"]))
    store.upsert_source_jobs("benchmark", "v2", [
        Job(source_kind="benchmark", source_key="v2", external_id="ai", title="AI Engineer", company="Demo"),
        Job(source_kind="benchmark", source_key="v2", external_id="backend", title="Backend Engineer", company="Demo"),
    ])
    result = asyncio.run(agent.handle("u", "Find backend roles"))
    assert [match["job"]["external_id"] for match in result.data["matches"]] == ["backend"]
    assert store.load_preferences("u").roles == ["AI/ML"]
