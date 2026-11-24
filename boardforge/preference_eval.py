from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from boardforge.models import Preferences
from boardforge.preferences import deterministic_extract


DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "preference_cases.json"


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list) and isinstance(actual, list):
        return set(actual) == set(expected)
    return actual == expected


def run_preference_eval(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    total_fields = correct_fields = 0
    for case in payload["cases"]:
        current = Preferences.model_validate(case.get("current", {}))
        actual = deterministic_extract(case["text"], current).model_dump()
        field_results = {
            field: _same(actual.get(field), expected)
            for field, expected in case["expected"].items()
        }
        total_fields += len(field_results)
        correct_fields += sum(field_results.values())
        rows.append({
            "name": case["name"],
            "exact": all(field_results.values()),
            "field_results": field_results,
            "expected": case["expected"],
            "actual": {field: actual.get(field) for field in case["expected"]},
        })
    exact = sum(row["exact"] for row in rows) / len(rows) if rows else 0.0
    field_accuracy = correct_fields / total_fields if total_fields else 0.0
    return {
        "dataset_version": payload.get("version", "unknown"),
        "cases": len(rows),
        "exact_case_accuracy": round(exact, 4),
        "field_accuracy": round(field_accuracy, 4),
        "failures": [row for row in rows if not row["exact"]],
    }
