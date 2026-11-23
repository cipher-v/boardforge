from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from boardforge.tool_agent import TOOLS


DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "agent_tools.json"

SEARCH_CUES = re.compile(
    r"\b(find|search|show|openings?|alternatives?|anything|what matches|what are|any good|similar)\b",
    re.I,
)
PREFERENCE_CUES = re.compile(
    r"\b(do not|don't|dont|not|no|avoid|exclude|remove|stop showing|only|from now on|minimum|at least|changed my mind|"
    r"salary matters|okay with|lower priority|focus on|want internships|want remote|worldwide|onsite|hybrid)\b",
    re.I,
)
NEGATED_SHOW_CUES = re.compile(r"\b(do not|don't|dont|stop)\s+show(?:ing)?\b", re.I)
FEEDBACK_CUES = re.compile(r"\b(reject|ignore|save|applied|interview|mark)\b", re.I)
JOB_ID_CUE = re.compile(r"\b[\w-]+:[\w-]+:[\w-]+\b")


def deterministic_tool_plan(text: str) -> list[str]:
    tools: list[str] = []
    if FEEDBACK_CUES.search(text) and JOB_ID_CUE.search(text):
        tools.append("record_feedback")
    if PREFERENCE_CUES.search(text):
        tools.append("update_preferences")
    search_requested = bool(SEARCH_CUES.search(text))
    if NEGATED_SHOW_CUES.search(text):
        search_requested = bool(re.search(r"\b(find|search|show me|openings?|alternatives?|similar)\b", text, re.I))
    if search_requested:
        tools.append("search_jobs")
    if not tools:
        tools.append("search_jobs")
    return tools


def _sequence_metrics(predicted: list[str], expected: list[str]) -> dict[str, float | bool]:
    pset, eset = set(predicted), set(expected)
    tp = len(pset & eset)
    precision = tp / len(pset) if pset else 0.0
    recall = tp / len(eset) if eset else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "exact": predicted == expected,
        "set_precision": precision,
        "set_recall": recall,
        "set_f1": f1,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "exact_sequence_accuracy": round(sum(bool(r["exact"]) for r in rows) / len(rows), 4),
        "tool_set_precision": round(sum(float(r["set_precision"]) for r in rows) / len(rows), 4),
        "tool_set_recall": round(sum(float(r["set_recall"]) for r in rows) / len(rows), 4),
        "tool_set_f1": round(sum(float(r["set_f1"]) for r in rows) / len(rows), 4),
    }


def run_deterministic_tool_eval(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for case in payload["cases"]:
        predicted = deterministic_tool_plan(case["text"])
        metrics = _sequence_metrics(predicted, case["expected_tools"])
        rows.append({
            "name": case["name"],
            "text": case["text"],
            "expected": case["expected_tools"],
            "predicted": predicted,
            **metrics,
        })
    return {
        "mode": "deterministic",
        "dataset_version": payload.get("version", "unknown"),
        "cases": len(rows),
        "macro": _aggregate(rows),
        "failures": [row for row in rows if not row["exact"]],
    }


async def _llm_plan(text: str) -> list[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM tool evaluation")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Select the BoardForge tools needed for this request. "
                    "Do not answer the user. Emit all required tool calls in execution order."
                ),
            },
            {"role": "user", "content": text},
        ],
        "tools": TOOLS,
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return [call["function"]["name"] for call in (message.get("tool_calls") or [])]


async def run_llm_tool_eval(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "skipped", "reason": "OPENAI_API_KEY is not configured."}

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for case in payload["cases"]:
        predicted = await _llm_plan(case["text"])
        metrics = _sequence_metrics(predicted, case["expected_tools"])
        rows.append({
            "name": case["name"],
            "expected": case["expected_tools"],
            "predicted": predicted,
            **metrics,
        })
    return {
        "status": "ok",
        "mode": "llm",
        "dataset_version": payload.get("version", "unknown"),
        "cases": len(rows),
        "macro": _aggregate(rows),
        "failures": [row for row in rows if not row["exact"]],
    }
