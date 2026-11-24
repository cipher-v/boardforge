from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from boardforge.evals import ndcg_at_k, precision_among_returned, precision_at_k, recall_at_k
from boardforge.matching import rank_jobs
from boardforge.models import Job, Preferences


DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "benchmark_v2.json"


def run_benchmark(path: str | Path = DEFAULT_PATH, top_k: int = 5) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs = [Job.model_validate(item) for item in payload["jobs"]]
    results: list[dict[str, Any]] = []

    for case in payload["cases"]:
        preferences = Preferences.model_validate(case["preferences"])
        matches = rank_jobs(
            jobs,
            preferences,
            query=case.get("query", ""),
            top_k=top_k,
            rejected_ids=set(case.get("rejected_ids", [])),
        )
        predicted = [match.job.external_id for match in matches]
        relevant = set(case["relevant_ids"])
        forbidden = set(case["forbidden_ids"])
        forbidden_hits = [job_id for job_id in predicted if job_id in forbidden]
        row: dict[str, Any] = {
            "name": case["name"],
            "predicted": predicted,
            "forbidden_hits": forbidden_hits,
            "forbidden_rate": len(forbidden_hits) / len(predicted) if predicted else 0.0,
        }
        if relevant:
            row.update({
                "precision_at_k": precision_at_k(predicted, relevant, top_k),
                "precision_among_returned": precision_among_returned(predicted, relevant, top_k),
                "recall_at_k": recall_at_k(predicted, relevant, top_k),
                "ndcg_at_k": ndcg_at_k(predicted, relevant, top_k),
            })
        results.append(row)

    scored = [row for row in results if "precision_at_k" in row]
    macro = {
        metric: sum(row[metric] for row in scored) / len(scored) if scored else 0.0
        for metric in ["precision_at_k", "precision_among_returned", "recall_at_k", "ndcg_at_k"]
    }
    macro["forbidden_rate"] = (
        sum(row["forbidden_rate"] for row in results) / len(results) if results else 0.0
    )
    return {"cases": len(results), "top_k": top_k, "macro": macro, "results": results}
