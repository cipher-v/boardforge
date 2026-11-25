from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from boardforge.evals import ndcg_at_k, precision_among_returned, precision_at_k, recall_at_k
from boardforge.database import create_store
from boardforge.matching import hard_violations, lexical_score, rank_jobs, role_evidence
from boardforge.models import Job, Match, Preferences
from boardforge.semantic import EmbeddingClient, hybrid_rank_jobs


DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "benchmark_v2.json"


def _lexical_only(
    jobs: list[Job], preferences: Preferences, query: str, top_k: int, rejected_ids: set[str]
) -> list[Match]:
    # Baseline intentionally has no personalization memory.
    candidates = list(jobs)
    ranked = sorted(
        candidates,
        key=lambda job: (-lexical_score(job, preferences, query), job.title.casefold()),
    )
    return [
        Match(job=job, score=round(lexical_score(job, preferences, query), 4),
              reasons=[], risks=[], hard_violations=[])
        for job in ranked[:top_k]
    ]


def _constrained_lexical(
    jobs: list[Job], preferences: Preferences, query: str, top_k: int, rejected_ids: set[str]
) -> list[Match]:
    candidates: list[Job] = []
    for job in jobs:
        # Constrained lexical adds correctness gates but still has no feedback memory.
        if hard_violations(job, preferences):
            continue
        if preferences.roles and not any(role_evidence(job, role) for role in preferences.roles):
            continue
        candidates.append(job)
    ranked = sorted(
        candidates,
        key=lambda job: (-lexical_score(job, preferences, query), job.title.casefold()),
    )
    return [
        Match(job=job, score=round(lexical_score(job, preferences, query), 4),
              reasons=[], risks=[], hard_violations=[])
        for job in ranked[:top_k]
    ]


def _full_no_feedback(
    jobs: list[Job], preferences: Preferences, query: str, top_k: int, rejected_ids: set[str]
) -> list[Match]:
    return rank_jobs(jobs, preferences, query=query, top_k=top_k, rejected_ids=set())


def _full_feedback(
    jobs: list[Job], preferences: Preferences, query: str, top_k: int, rejected_ids: set[str]
) -> list[Match]:
    return rank_jobs(jobs, preferences, query=query, top_k=top_k, rejected_ids=rejected_ids)


def _case_metrics(matches: list[Match], case: dict[str, Any], top_k: int) -> dict[str, Any]:
    predicted = [match.job.external_id for match in matches]
    relevant = set(case.get("relevant_ids", []))
    forbidden = set(case.get("forbidden_ids", []))
    forbidden_hits = [job_id for job_id in predicted if job_id in forbidden]
    return {
        "predicted": predicted,
        "precision_at_k": precision_at_k(predicted, relevant, top_k) if relevant else 0.0,
        "precision_among_returned": precision_among_returned(predicted, relevant, top_k) if relevant else 0.0,
        "recall_at_k": recall_at_k(predicted, relevant, top_k) if relevant else 0.0,
        "ndcg_at_k": ndcg_at_k(predicted, relevant, top_k) if relevant else 0.0,
        "forbidden_rate": len(forbidden_hits) / len(predicted) if predicted else 0.0,
        "forbidden_hits": forbidden_hits,
        "empty": not predicted,
        "top1_relevant": bool(predicted and predicted[0] in relevant),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metrics = ["precision_at_k", "precision_among_returned", "recall_at_k", "ndcg_at_k", "forbidden_rate"]
    out = {name: sum(float(row[name]) for row in rows) / len(rows) for name in metrics}
    out["top1_accuracy"] = sum(bool(row["top1_relevant"]) for row in rows) / len(rows)
    out["empty_rate"] = sum(bool(row["empty"]) for row in rows) / len(rows)
    return {key: round(value, 4) for key, value in out.items()}


def _family_aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    return {family: _aggregate(items) for family, items in sorted(grouped.items())}


def run_comparison(path: str | Path = DEFAULT_PATH, top_k: int = 5) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs = [Job.model_validate(item) for item in payload["jobs"]]

    variants: dict[str, Callable[..., list[Match]]] = {
        "lexical_only": _lexical_only,
        "constrained_lexical": _constrained_lexical,
        "full_evidence": _full_no_feedback,
        "full_evidence_feedback": _full_feedback,
    }

    report: dict[str, Any] = {
        "dataset_version": payload.get("version", "unknown"),
        "cases": len(payload["cases"]),
        "jobs": len(jobs),
        "top_k": top_k,
        "variants": {},
    }

    for name, runner in variants.items():
        rows = []
        for case in payload["cases"]:
            preferences = Preferences.model_validate(case["preferences"])
            rejected = set(case.get("rejected_ids", []))
            matches = runner(jobs, preferences, case.get("query", ""), top_k, rejected)
            row = {
                "name": case["name"],
                "family": case.get("family", "unknown"),
                **_case_metrics(matches, case, top_k),
            }
            rows.append(row)
        report["variants"][name] = {
            "macro": _aggregate(rows),
            "by_family": _family_aggregate(rows),
            "failures": [
                {
                    "name": row["name"],
                    "predicted": row["predicted"],
                    "forbidden_hits": row["forbidden_hits"],
                }
                for row in rows
                if row["forbidden_hits"] or not row["top1_relevant"]
            ][:25],
        }

    return report


async def run_embedding_comparison(
    path: str | Path = DEFAULT_PATH,
    top_k: int = 5,
) -> dict[str, Any]:
    client = EmbeddingClient(budget_store=create_store())
    if not client.enabled:
        return {
            "status": "skipped",
            "reason": "Configure BOARDFORGE_EMBEDDING_PROVIDER credentials and set BOARDFORGE_ENABLE_EMBEDDINGS=1.",
        }

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs = [Job.model_validate(item) for item in payload["jobs"]]
    rows = []
    for case in payload["cases"]:
        preferences = Preferences.model_validate(case["preferences"])
        matches = await hybrid_rank_jobs(
            jobs,
            preferences,
            query=case.get("query", ""),
            top_k=top_k,
            rejected_ids=set(case.get("rejected_ids", [])),
            client=client,
        )
        rows.append({
            "name": case["name"],
            "family": case.get("family", "unknown"),
            **_case_metrics(matches, case, top_k),
        })

    return {
        "status": "ok",
        "dataset_version": payload.get("version", "unknown"),
        "cases": len(rows),
        "top_k": top_k,
        "macro": _aggregate(rows),
        "by_family": _family_aggregate(rows),
    }


def comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BoardForge ranking benchmark",
        "",
        f"Dataset: v{report['dataset_version']} — {report['cases']} cases / {report['jobs']} jobs",
        "",
        f"| Variant | P@{report['top_k']} | P/returned | Recall@{report['top_k']} | NDCG@{report['top_k']} | Forbidden | Top-1 | Empty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in report["variants"].items():
        m = data["macro"]
        lines.append(
            f"| {name} | {m['precision_at_k']:.3f} | {m['precision_among_returned']:.3f} | {m['recall_at_k']:.3f} | "
            f"{m['ndcg_at_k']:.3f} | {m['forbidden_rate']:.3f} | "
            f"{m['top1_accuracy']:.3f} | {m['empty_rate']:.3f} |"
        )
    lines += [
        "",
        "> Synthetic adversarial benchmark. Useful for regression and ablation evidence; not a substitute for real-user relevance labels.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(run_comparison(), indent=2))
