from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boardforge.evals import ndcg_at_k, precision_among_returned, precision_at_k, recall_at_k
from boardforge.matching import hard_violations, job_text
from boardforge.models import Job, Preferences
from boardforge.preferences import effective_search_preferences
from boardforge.semantic import (
    RETRIEVAL_DOCUMENT,
    EmbeddingClient,
    hybrid_rank_jobs,
)


DEFAULT_REAL_EVAL_PATH = (
    Path(__file__).resolve().parent / "data" / "real_india_eval_v1_1.json"
)


def build_corpus_snapshot(
    jobs: list[Job],
    *,
    country: str,
    include_test: bool = False,
) -> dict:
    """Freeze a real active-job subset for independently labeled relevance evals."""
    wanted_country = country.casefold()
    country_jobs = [job for job in jobs if job.country.casefold() == wanted_country]
    excluded_test_jobs = sum(job.source_kind == "test" for job in country_jobs)

    selected = [
        job
        for job in country_jobs
        if include_test or job.source_kind != "test"
    ]
    selected.sort(
        key=lambda job: (
            job.company.casefold(),
            job.title.casefold(),
            job.stable_id,
        )
    )

    return {
        "version": "1.0",
        "kind": "real_corpus_snapshot",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "country": country,
        "excluded_test_jobs": 0 if include_test else excluded_test_jobs,
        "jobs": [job.model_dump(mode="json") for job in selected],
    }


class _SnapshotEmbeddingCache:
    """In-memory document-vector cache for a frozen eval corpus."""

    def __init__(self) -> None:
        self._vectors: dict[tuple[str, str, str], list[float]] = {}

    def load_embedding(
        self,
        stable_id: str,
        model: str,
        content_hash: str,
    ) -> list[float] | None:
        return self._vectors.get((stable_id, model, content_hash))

    def save_embedding(
        self,
        stable_id: str,
        model: str,
        content_hash: str,
        vector: list[float],
    ) -> None:
        self._vectors[(stable_id, model, content_hash)] = vector


def _graded_ndcg_at_k(
    predicted: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    import math

    gains = [
        (2 ** relevance_grades.get(stable_id, 0)) - 1
        for stable_id in predicted[:k]
    ]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted(
        ((2 ** grade) - 1 for grade in relevance_grades.values()),
        reverse=True,
    )[:k]
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metrics = [
        "precision_at_k",
        "precision_among_returned",
        "direct_precision_among_returned",
        "recall_at_k",
        "ndcg_at_k",
        "graded_ndcg_at_k",
        "forbidden_rate",
        "hard_constraint_violation_rate",
        "recommended_irrelevant_rate",
        "recommended_adjacent_rate",
        "needs_verification_rate",
    ]
    out = {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in metrics
    }
    out["top1_accuracy"] = (
        sum(bool(row["top1_relevant"]) for row in rows) / len(rows)
    )
    out["top1_direct_accuracy"] = (
        sum(bool(row["top1_direct"]) for row in rows) / len(rows)
    )
    out["empty_rate"] = sum(bool(row["empty"]) for row in rows) / len(rows)
    return {key: round(value, 4) for key, value in out.items()}


def _family_aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    return {
        family: _aggregate(items)
        for family, items in sorted(grouped.items())
    }


def _validate_dataset(
    jobs: list[Job],
    dataset: dict[str, Any],
) -> None:
    stable_ids = {job.stable_id for job in jobs}
    expected_jobs = dataset.get("corpus", {}).get("expected_jobs")
    if expected_jobs is not None and len(jobs) != int(expected_jobs):
        raise ValueError(
            f"real eval corpus mismatch: expected {expected_jobs} jobs, got {len(jobs)}"
        )

    missing: list[str] = []
    invalid_grades: list[str] = []
    overlaps: list[str] = []
    for case in dataset.get("cases", []):
        grades = case.get("relevance_grades", {})
        for stable_id, grade in grades.items():
            if grade not in {1, 2}:
                invalid_grades.append(f"{case['name']}:{stable_id}={grade}")
        positive_ids = set(grades) or set(case.get("relevant_ids", []))
        forbidden_ids = set(case.get("forbidden_ids", []))
        for stable_id in [*positive_ids, *forbidden_ids]:
            if stable_id not in stable_ids:
                missing.append(f"{case['name']}:{stable_id}")
        for stable_id in positive_ids & forbidden_ids:
            overlaps.append(f"{case['name']}:{stable_id}")
    if invalid_grades:
        sample = ", ".join(invalid_grades[:5])
        raise ValueError(
            f"real eval relevance grades must be 1 or 2: {sample}"
        )
    if overlaps:
        sample = ", ".join(overlaps[:5])
        raise ValueError(
            f"real eval job cannot be both relevant and forbidden: {sample}"
        )
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"real eval labels reference jobs missing from the frozen corpus: {sample}"
        )


async def run_real_eval(
    corpus_path: str | Path,
    dataset_path: str | Path = DEFAULT_REAL_EVAL_PATH,
    top_k: int = 5,
    *,
    client: EmbeddingClient | None = None,
) -> dict[str, Any]:
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    jobs = [Job.model_validate(item) for item in corpus["jobs"]]
    _validate_dataset(jobs, dataset)

    embeddings = client or EmbeddingClient()
    if not embeddings.enabled:
        return {
            "status": "skipped",
            "reason": (
                "real-eval requires enabled embeddings; set "
                "BOARDFORGE_ENABLE_EMBEDDINGS=1 and provider credentials"
            ),
            "corpus_jobs": len(jobs),
            "cases": len(dataset.get("cases", [])),
        }

    documents = [job_text(job)[:6000] for job in jobs]
    vectors = await embeddings.embed(
        documents,
        task_type=RETRIEVAL_DOCUMENT,
    )
    if len(vectors) != len(jobs):
        raise RuntimeError(
            "embedding provider returned an unexpected document vector count"
        )

    import hashlib

    cache = _SnapshotEmbeddingCache()
    for job, document, vector in zip(jobs, documents, vectors):
        cache.save_embedding(
            job.stable_id,
            embeddings.model,
            hashlib.sha256(document.encode("utf-8")).hexdigest(),
            vector,
        )

    rows: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        query = case["query"]
        saved = Preferences.model_validate(case.get("saved_preferences", {}))
        preferences = effective_search_preferences(saved, query)
        matches = await hybrid_rank_jobs(
            jobs,
            preferences,
            query=query,
            top_k=top_k,
            rejected_ids=set(case.get("rejected_ids", [])),
            client=embeddings,
            store=cache,
        )

        predicted = [match.job.stable_id for match in matches]
        relevance_grades = {
            stable_id: int(grade)
            for stable_id, grade in case.get("relevance_grades", {}).items()
        }
        if not relevance_grades:
            relevance_grades = {
                stable_id: 2 for stable_id in case.get("relevant_ids", [])
            }
        relevant = set(relevance_grades)
        direct = {
            stable_id for stable_id, grade in relevance_grades.items()
            if grade == 2
        }
        adjacent = {
            stable_id for stable_id, grade in relevance_grades.items()
            if grade == 1
        }
        forbidden = set(case.get("forbidden_ids", []))
        forbidden_hits = [stable_id for stable_id in predicted if stable_id in forbidden]
        hard_hits = [
            match.job.stable_id
            for match in matches
            if hard_violations(match.job, preferences)
        ]
        recommended = [match for match in matches if match.status == "recommended"]
        recommended_irrelevant = [
            match.job.stable_id
            for match in recommended
            if match.job.stable_id not in relevant
        ]
        recommended_adjacent = [
            match.job.stable_id
            for match in recommended
            if match.job.stable_id in adjacent
        ]

        rows.append({
            "name": case["name"],
            "family": case["family"],
            "query": query,
            "predicted": predicted,
            "statuses": [match.status for match in matches],
            "precision_at_k": precision_at_k(predicted, relevant, top_k),
            "precision_among_returned": precision_among_returned(
                predicted, relevant, top_k
            ),
            "direct_precision_among_returned": precision_among_returned(
                predicted, direct, top_k
            ),
            "recall_at_k": recall_at_k(predicted, relevant, top_k),
            "ndcg_at_k": ndcg_at_k(predicted, relevant, top_k),
            "graded_ndcg_at_k": _graded_ndcg_at_k(
                predicted, relevance_grades, top_k
            ),
            "forbidden_rate": (
                len(forbidden_hits) / len(predicted) if predicted else 0.0
            ),
            "forbidden_hits": forbidden_hits,
            "hard_constraint_violation_rate": (
                len(hard_hits) / len(matches) if matches else 0.0
            ),
            "hard_constraint_hits": hard_hits,
            "recommended_irrelevant_rate": (
                len(recommended_irrelevant) / len(recommended)
                if recommended else 0.0
            ),
            "recommended_irrelevant": recommended_irrelevant,
            "recommended_adjacent_rate": (
                len(recommended_adjacent) / len(recommended)
                if recommended else 0.0
            ),
            "recommended_adjacent": recommended_adjacent,
            "needs_verification_rate": (
                sum(match.status == "needs_verification" for match in matches)
                / len(matches)
                if matches else 0.0
            ),
            "top1_relevant": bool(predicted and predicted[0] in relevant),
            "top1_direct": bool(predicted and predicted[0] in direct),
            "empty": not predicted,
        })

    failures = [
        {
            "name": row["name"],
            "family": row["family"],
            "predicted": row["predicted"],
            "forbidden_hits": row["forbidden_hits"],
            "recommended_irrelevant": row["recommended_irrelevant"],
        }
        for row in rows
        if (
            not row["top1_relevant"]
            or row["forbidden_hits"]
            or row["recommended_irrelevant"]
        )
    ]

    return {
        "status": "ok",
        "dataset_version": dataset.get("version", "unknown"),
        "corpus_snapshot": {
            "captured_at": corpus.get("captured_at"),
            "country": corpus.get("country"),
            "jobs": len(jobs),
        },
        "cases": len(rows),
        "top_k": top_k,
        "embedding_model": embeddings.model,
        "document_embeddings_created": len(vectors),
        "macro": _aggregate(rows),
        "by_family": _family_aggregate(rows),
        "failure_count": len(failures),
        "failures": failures,
        "results": rows,
    }
