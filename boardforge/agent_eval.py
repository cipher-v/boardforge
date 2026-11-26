from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import httpx

from boardforge.planning import tool_names
from boardforge.tool_agent import TOOLS


DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "agent_tools.json"


def deterministic_tool_plan(text: str) -> list[str]:
    return tool_names(text)


def _sequence_metrics(predicted: list[str], expected: list[str]) -> dict[str, float | bool]:
    predicted_set, expected_set = set(predicted), set(expected)
    true_positive = len(predicted_set & expected_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(expected_set) if expected_set else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "exact": predicted == expected, "set_precision": precision,
        "set_recall": recall, "set_f1": f1,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "exact_sequence_accuracy": round(sum(bool(row["exact"]) for row in rows) / len(rows), 4),
        "tool_set_precision": round(sum(float(row["set_precision"]) for row in rows) / len(rows), 4),
        "tool_set_recall": round(sum(float(row["set_recall"]) for row in rows) / len(rows), 4),
        "tool_set_f1": round(sum(float(row["set_f1"]) for row in rows) / len(rows), 4),
    }


def run_deterministic_tool_eval(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for case in payload["cases"]:
        predicted = deterministic_tool_plan(case["text"])
        rows.append({
            "name": case["name"], "text": case["text"],
            "expected": case["expected_tools"], "predicted": predicted,
            **_sequence_metrics(predicted, case["expected_tools"]),
        })
    return {
        "mode": "deterministic", "dataset_version": payload.get("version", "unknown"),
        "cases": len(rows), "macro": _aggregate(rows),
        "failures": [row for row in rows if not row["exact"]],
    }


async def _llm_plan(text: str) -> list[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM tool evaluation")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "messages": [
            {"role": "system", "content": (
                "Select the BoardForge tools needed. Emit required calls in order."
            )},
            {"role": "user", "content": text},
        ],
        "tools": TOOLS, "tool_choice": "required", "parallel_tool_calls": False,
    }
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=payload,
        )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return [call["function"]["name"] for call in message.get("tool_calls") or []]


async def run_llm_tool_eval(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "skipped", "reason": "OPENAI_API_KEY is not configured."}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for case in payload["cases"]:
        predicted = await _llm_plan(case["text"])
        rows.append({
            "name": case["name"], "expected": case["expected_tools"],
            "predicted": predicted,
            **_sequence_metrics(predicted, case["expected_tools"]),
        })
    return {
        "status": "ok", "mode": "llm",
        "dataset_version": payload.get("version", "unknown"), "cases": len(rows),
        "macro": _aggregate(rows), "failures": [row for row in rows if not row["exact"]],
    }


@dataclass(frozen=True)
class AgentScenario:
    name: str
    user_text: str
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    allowed_repeated_tools: tuple[str, ...] = ()
    acceptable_stops: tuple[str, ...] = (
        "sufficient_results", "discovery_queued", "provider_finished", "provider_failure",
        "token_limit", "tool_call_limit", "iteration_limit", "total_timeout",
    )
    expected_state: dict[str, Any] = field(default_factory=dict)


def load_agent_scenarios(path: str | Path) -> list[AgentScenario]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        AgentScenario(
            name=item["name"], user_text=item.get("user_text", item["name"]),
            required_tools=tuple(item.get("required_tools", [])),
            forbidden_tools=tuple(item.get("forbidden_tools", [])),
        )
        for item in document["scenarios"]
    ]


def score_agent_scenario(
    scenario: AgentScenario,
    result: dict[str, Any],
    *,
    state_checks: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Score a deterministic trace without judging model prose."""
    trace = result.get("trace", [])
    tools = [step.get("tool") for step in trace if step.get("ok")]
    attempted = [step.get("tool") for step in trace]
    required_ok = all(tool in tools for tool in scenario.required_tools)
    forbidden_ok = not any(tool in attempted for tool in scenario.forbidden_tools)
    counts = {tool: attempted.count(tool) for tool in set(attempted)}
    unnecessary = sum(
        max(0, count - 1)
        for tool, count in counts.items()
        if tool not in scenario.allowed_repeated_tools
    ) + sum(not step.get("ok") for step in trace)
    agent = result.get("agent", {})
    stopping_ok = agent.get("stopped_reason") in scenario.acceptable_stops
    checks = state_checks or {}
    state_correct = all(checks.values()) if checks else True
    constraints_preserved = not any(
        step.get("error") in {
            "preference_update_not_authorized", "feedback_not_explicitly_authorized",
            "source_not_in_trusted_registry",
        } and step.get("ok")
        for step in trace
    )
    task_completion = required_ok and forbidden_ok and stopping_ok and state_correct
    return {
        "name": scenario.name,
        "task_completion": task_completion,
        "constraint_preservation": constraints_preserved,
        "appropriate_tool_selection": required_ok and forbidden_ok,
        "unnecessary_calls": unnecessary,
        "stopping_behavior": stopping_ok,
        "state_correctness": state_correct,
        "provider_calls": agent.get("provider_calls", 0),
        "tool_calls": agent.get("attempted_tool_calls", len(trace)),
        "token_usage": agent.get("token_usage", {}),
        "estimated_cost_usd": agent.get("estimated_cost_usd", 0.0),
        "stopped_reason": agent.get("stopped_reason"),
    }


def summarize_agent_scenarios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    rate = lambda key: round(sum(bool(row[key]) for row in rows) / count, 4) if count else 0.0
    return {
        "scenarios": count,
        "task_completion_rate": rate("task_completion"),
        "constraint_preservation_rate": rate("constraint_preservation"),
        "appropriate_tool_selection_rate": rate("appropriate_tool_selection"),
        "stopping_behavior_rate": rate("stopping_behavior"),
        "state_correctness_rate": rate("state_correctness"),
        "unnecessary_calls": sum(row["unnecessary_calls"] for row in rows),
        "provider_calls": sum(row["provider_calls"] for row in rows),
        "tool_calls": sum(row["tool_calls"] for row in rows),
        "estimated_cost_usd": round(sum(row["estimated_cost_usd"] for row in rows), 8),
    }
