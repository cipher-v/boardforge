from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from boardforge.resume import extract_candidate_profile


DEFAULT_PATH = files("boardforge").joinpath("data/resume_cases.json")


def _set_metrics(predicted: set[str], expected: set[str]) -> dict[str, float]:
    overlap = len(predicted & expected)
    precision = overlap / len(predicted) if predicted else float(not expected)
    recall = overlap / len(expected) if expected else float(not predicted)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def run_resume_eval(path: str | Path | None = None) -> dict:
    dataset = json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    results = []
    for case in dataset["cases"]:
        profile = extract_candidate_profile(case["text"])
        skill = _set_metrics(set(profile.skills), set(case["expected_skills"]))
        role = _set_metrics(set(profile.role_signals), set(case["expected_roles"]))
        results.append({
            "name": case["name"],
            "skill": skill,
            "role": role,
            "predicted_skills": profile.skills,
            "predicted_roles": profile.role_signals,
        })
    count = len(results)
    macro = {
        f"{kind}_{metric}": sum(row[kind][metric] for row in results) / count
        for kind in ("skill", "role")
        for metric in ("precision", "recall", "f1")
    }
    return {
        "dataset_version": dataset["dataset_version"],
        "cases": count,
        "macro": macro,
        "results": results,
    }
