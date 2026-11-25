from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from boardforge.evals import ndcg_at_k, precision_at_k, precision_among_returned, recall_at_k
from boardforge.holdout import canonical_sha256
from boardforge.matching import hard_violations, job_text
from boardforge.models import Job, Preferences
from boardforge.observability import EMBEDDING_REQUESTS
from boardforge.postgres_store import PostgresStore
from boardforge.preferences import effective_search_preferences
from boardforge.semantic import (
    POSTGRES_EMBEDDING_DIMENSIONS,
    RETRIEVAL_DOCUMENT,
    RETRIEVAL_QUERY,
    EmbeddingClient,
    VertexGeminiEmbeddingProvider,
    hybrid_rank_jobs,
    retrieve_and_rank_jobs,
    semantic_query_text,
)


def assert_isolated_database_url(database_url: str) -> str:
    database = urlparse(database_url).path.lstrip("/").casefold()
    if not database or not any(token in database for token in ("holdout", "eval", "test")):
        raise ValueError(
            "holdout evaluation refuses a non-isolated database; use a database "
            "whose name contains holdout, eval, or test"
        )
    return database


def _load_frozen(corpus_path: str | Path, labels_path: str | Path) -> tuple[dict, dict, list[Job]]:
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    if canonical_sha256(corpus["records"]) != corpus["records_sha256"]:
        raise ValueError("holdout corpus checksum mismatch")
    if canonical_sha256(labels["cases"]) != labels["labels_sha256"]:
        raise ValueError("holdout labels checksum mismatch")
    if labels["corpus_records_sha256"] != corpus["records_sha256"]:
        raise ValueError("labels were not frozen against this corpus")
    jobs = [Job.model_validate(record["job"]) for record in corpus["records"]]
    stable_ids = {job.stable_id for job in jobs}
    for case in labels["cases"]:
        if set(case["labels"]) != stable_ids:
            raise ValueError(f"label coverage mismatch for {case['name']}")
    return corpus, labels, jobs


class CachedEmbeddingProvider:
    """Persistent task-aware cache with explicit paid-request accounting."""

    def __init__(self, provider, cache_path: str | Path):
        self.provider = provider
        self.enabled = provider.enabled
        self.model = provider.model
        self.dimensions = provider.dimensions
        self.cache_path = Path(cache_path)
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._cache = raw.get("vectors", {})
        self.provider_calls = 0
        self.provider_texts = 0
        self.cache_hits = 0
        self.embed_invocations = 0

    def _key(self, text: str, task_type: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.model}|{task_type}|{digest}"

    def has(self, text: str, task_type: str) -> bool:
        return self._key(text, task_type) in self._cache

    async def embed(self, texts: list[str], *, task_type: str = RETRIEVAL_DOCUMENT):
        self.embed_invocations += 1
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, tuple[str, list[int]]] = {}
        for index, text in enumerate(texts):
            key = self._key(text, task_type)
            if key in self._cache:
                self.cache_hits += 1
                vectors[index] = self._cache[key]
            else:
                missing.setdefault(key, (text, []))[1].append(index)
        if missing:
            unique_texts = [value[0] for value in missing.values()]
            self.provider_calls += 1
            self.provider_texts += len(unique_texts)
            embedded = await self.provider.embed(unique_texts, task_type=task_type)
            if len(embedded) != len(unique_texts):
                raise RuntimeError("embedding provider returned an unexpected vector count")
            for (key, (_, indexes)), vector in zip(missing.items(), embedded):
                if len(vector) != self.dimensions:
                    raise RuntimeError("cached embedding dimension mismatch")
                self._cache[key] = vector
                for index in indexes:
                    vectors[index] = vector
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps({"model": self.model, "vectors": self._cache}),
                encoding="utf-8",
            )
        return [vector for vector in vectors if vector is not None]


class DisabledEmbeddingProvider:
    enabled = False
    model = "disabled"
    dimensions = POSTGRES_EMBEDDING_DIMENSIONS

    async def embed(self, texts: list[str], *, task_type: str = RETRIEVAL_DOCUMENT):
        raise AssertionError("disabled lexical provider must not be called")


def plan_vertex_usage(
    corpus_path: str | Path,
    labels_path: str | Path,
    cache_path: str | Path,
    *,
    model: str = "vertex:gemini-embedding-001:768",
) -> dict[str, Any]:
    _, labels, jobs = _load_frozen(corpus_path, labels_path)
    vectors: dict[str, Any] = {}
    cache = Path(cache_path)
    if cache.exists():
        vectors = json.loads(cache.read_text(encoding="utf-8")).get("vectors", {})
    documents = {job_text(job)[:6000] for job in jobs}
    queries = set()
    for case in labels["cases"]:
        prefs = effective_search_preferences(
            Preferences.model_validate(case.get("saved_preferences", {})), case["query"]
        )
        query = semantic_query_text(prefs, case["query"])
        if query:
            queries.add(query)
    def missing(texts: set[str], task: str) -> int:
        return sum(
            f"{model}|{task}|{hashlib.sha256(text.encode('utf-8')).hexdigest()}" not in vectors
            for text in texts
        )
    missing_documents = missing(documents, RETRIEVAL_DOCUMENT)
    missing_queries = missing(queries, RETRIEVAL_QUERY)
    return {
        "status": "plan_only_no_api_calls_executed",
        "model": model,
        "unique_documents": len(documents),
        "unique_queries": len(queries),
        "cached_vectors": len(vectors),
        "planned_document_embedding_requests": missing_documents,
        "planned_query_embedding_requests": missing_queries,
        "planned_total_vertex_prediction_requests": missing_documents + missing_queries,
        "note": "Vertex Gemini sends one prediction request per uncached text; all three evaluation variants share this cache.",
    }


def prepare_isolated_database(store: PostgresStore, jobs: list[Job]) -> dict[str, Any]:
    existing = {job.stable_id for job in store.active_jobs()}
    expected = {job.stable_id for job in jobs}
    unexpected = sorted(existing - expected)
    if unexpected:
        raise ValueError(
            "isolated evaluation database contains unrelated active jobs; refusing to overwrite it"
        )
    groups: dict[tuple[str, str], list[Job]] = {}
    for job in jobs:
        groups.setdefault((job.source_kind, job.source_key), []).append(job)
    summaries = [
        store.upsert_source_jobs(kind, key, source_jobs).model_dump()
        for (kind, key), source_jobs in sorted(groups.items())
    ]
    return {"jobs": len(jobs), "sources": len(groups), "sync_summaries": summaries}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _graded_ndcg(predicted: list[str], direct: set[str], adjacent: set[str], k: int) -> float:
    grades = {stable_id: 2 for stable_id in direct}
    grades.update({stable_id: 1 for stable_id in adjacent})
    gains = [(2 ** grades.get(stable_id, 0)) - 1 for stable_id in predicted[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted(((2 ** grade) - 1 for grade in grades.values()), reverse=True)[:k]
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _embedding_attempt_snapshot() -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in EMBEDDING_REQUESTS.collect():
        for sample in metric.samples:
            if not sample.name.endswith("_total"):
                continue
            labels = sample.labels
            if labels.get("provider") != "vertex":
                continue
            key = f"{labels.get('task_type', 'unknown')}:{labels.get('status', 'unknown')}"
            values[key] = float(sample.value)
    return values


def _counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {
        key: int(round(after.get(key, 0.0) - before.get(key, 0.0)))
        for key in sorted(keys)
        if after.get(key, 0.0) - before.get(key, 0.0)
    }


def _row_metrics(case: dict, matches, candidates: list[Job], prefs: Preferences, top_k: int) -> dict:
    direct = {sid for sid, row in case["labels"].items() if row["label"] == "direct"}
    adjacent = {sid for sid, row in case["labels"].items() if row["label"] == "adjacent"}
    relevant = direct | adjacent
    forbidden = {sid for sid, row in case["labels"].items() if row["label"] == "forbidden"}
    predicted = [match.job.stable_id for match in matches]
    candidate_ids = {job.stable_id for job in candidates}
    recommended = [match for match in matches if match.status == "recommended"]
    recommended_ids = [match.job.stable_id for match in recommended]
    recommended_irrelevant = [stable_id for stable_id in recommended_ids if stable_id not in relevant]
    forbidden_hits = [stable_id for stable_id in predicted if stable_id in forbidden]
    hard_hits = [match.job.stable_id for match in matches if hard_violations(match.job, prefs)]
    unresolved = {
        stable_id for stable_id, row in case["labels"].items()
        if row.get("eligibility") == "unresolved"
    }
    return {
        "has_relevant_jobs": bool(relevant),
        "relevant_count": len(relevant),
        "candidate_count": len(candidate_ids),
        "candidate_ids": sorted(candidate_ids),
        "candidate_recall": len(candidate_ids & relevant) / len(relevant) if relevant else None,
        "predicted": predicted,
        "statuses": [match.status for match in matches],
        "precision_at_10": precision_at_k(predicted, relevant, top_k),
        "recall_at_10": recall_at_k(predicted, relevant, top_k) if relevant else None,
        "ndcg_at_10": ndcg_at_k(predicted, relevant, top_k) if relevant else None,
        "graded_ndcg_at_10": _graded_ndcg(predicted, direct, adjacent, top_k) if relevant else None,
        "recommended": recommended_ids,
        "recommended_precision": precision_among_returned(recommended_ids, relevant, len(recommended_ids)),
        "recommended_recall": recall_at_k(recommended_ids, relevant, len(recommended_ids)) if relevant else None,
        "recommendation_coverage": len(recommended_ids) / len(matches) if matches else 0.0,
        "abstention_rate": sum(m.status == "needs_verification" for m in matches) / len(matches) if matches else 1.0,
        "recommended_irrelevant_rate": len(recommended_irrelevant) / len(recommended_ids) if recommended_ids else 0.0,
        "recommended_irrelevant": recommended_irrelevant,
        "forbidden_result_rate": len(forbidden_hits) / len(predicted) if predicted else 0.0,
        "forbidden_hits": forbidden_hits,
        "hard_constraint_violation_rate": len(hard_hits) / len(matches) if matches else 0.0,
        "hard_constraint_hits": hard_hits,
        "false_recommendation_count": len(recommended_irrelevant),
        "unresolved_retrieved": [stable_id for stable_id in predicted if stable_id in unresolved],
        "unresolved_recommended": [stable_id for stable_id in recommended_ids if stable_id in unresolved],
    }


_AGGREGATE_METRICS = (
    "candidate_recall", "precision_at_10", "recall_at_10", "ndcg_at_10",
    "graded_ndcg_at_10", "recommended_precision", "recommended_recall",
    "recommendation_coverage", "abstention_rate", "recommended_irrelevant_rate",
    "forbidden_result_rate", "hard_constraint_violation_rate",
)


def _aggregate_rows(rows: list[dict[str, Any]], latency_ms: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"queries": len(rows)}
    macro_metrics = (
        "candidate_recall", "precision_at_10", "recall_at_10", "ndcg_at_10",
        "graded_ndcg_at_10",
    )
    for metric in macro_metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        out[metric] = round(statistics.fmean(values), 4) if values else None
    returned = sum(len(row["predicted"]) for row in rows)
    recommended = sum(len(row["recommended"]) for row in rows)
    false_recommendations = sum(row["false_recommendation_count"] for row in rows)
    relevant_recommendations = recommended - false_recommendations
    relevant_total = sum(row["relevant_count"] for row in rows)
    abstained = sum(
        status == "needs_verification"
        for row in rows for status in row["statuses"]
    )
    forbidden_hits = sum(len(row["forbidden_hits"]) for row in rows)
    hard_hits = sum(len(row["hard_constraint_hits"]) for row in rows)
    out.update({
        "recommended_precision": (
            round(relevant_recommendations / recommended, 4) if recommended else 0.0
        ),
        "recommended_recall": (
            round(relevant_recommendations / relevant_total, 4) if relevant_total else None
        ),
        "recommendation_coverage": round(recommended / returned, 4) if returned else 0.0,
        "abstention_rate": round(abstained / returned, 4) if returned else 1.0,
        "recommended_irrelevant_rate": (
            round(false_recommendations / recommended, 4) if recommended else 0.0
        ),
        "forbidden_result_rate": round(forbidden_hits / returned, 4) if returned else 0.0,
        "hard_constraint_violation_rate": round(hard_hits / returned, 4) if returned else 0.0,
    })
    out.update({
        "returned_results": returned,
        "recommended_results": recommended,
        "false_recommendations": false_recommendations,
        "queries_with_false_recommendations": sum(bool(row["false_recommendation_count"]) for row in rows),
        "queries_with_false_recommendations_rate": round(
            sum(bool(row["false_recommendation_count"]) for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "latency_ms_p50": round(statistics.median(latency_ms), 2) if latency_ms else 0.0,
        "latency_ms_p95": round(_percentile(latency_ms, 0.95), 2) if latency_ms else 0.0,
    })
    return out


def refresh_report_aggregates(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute summaries from saved per-query rows without rerunning retrieval."""
    for variant in ("lexical", "hybrid", "exhaustive"):
        data = report["variants"][variant]
        rows = data["per_query"]
        relevant = [row for row in rows if row["has_relevant_jobs"]]
        no_relevant = [row for row in rows if not row["has_relevant_jobs"]]
        data["relevant_queries"] = _aggregate_rows(
            relevant, [row["latency_ms"] for row in relevant]
        )
        data["no_relevant_queries"] = _aggregate_rows(
            no_relevant, [row["latency_ms"] for row in no_relevant]
        )
        data["all_queries_operational"] = _aggregate_rows(
            rows, [row["latency_ms"] for row in rows]
        )
    report["aggregation_semantics"] = {
        "ranking_metrics": "macro average across queries with defined ground truth",
        "recommendation_and_constraint_rates": "micro average over returned/recommended results",
        "empty_ground_truth": "excluded from candidate recall, recall@10, NDCG, graded NDCG, and recommended recall",
    }
    return report


async def run_production_holdout_eval(
    database_url: str,
    corpus_path: str | Path,
    labels_path: str | Path,
    cache_path: str | Path,
    *,
    top_k: int = 10,
    candidate_limit: int = 50,
) -> dict[str, Any]:
    assert_isolated_database_url(database_url)
    corpus, labels, jobs = _load_frozen(corpus_path, labels_path)
    provider = CachedEmbeddingProvider(VertexGeminiEmbeddingProvider(), cache_path)
    client = EmbeddingClient(provider=provider)
    if not client.enabled:
        raise RuntimeError("Vertex provider is not configured; run the usage plan first")
    lexical_client = EmbeddingClient(provider=DisabledEmbeddingProvider())
    store = PostgresStore(database_url, min_size=1, max_size=2)
    provider.provider.budget_store = store
    try:
        prepared = prepare_isolated_database(store, jobs)
        http_attempts_before = _embedding_attempt_snapshot()
        initial_cache_vectors = len(provider._cache)
        documents = [job_text(job)[:6000] for job in jobs]
        requests_before_documents = provider.provider_texts
        document_started = time.perf_counter()
        document_vectors = await client.embed(documents, task_type=RETRIEVAL_DOCUMENT)
        document_embedding_latency_ms = (time.perf_counter() - document_started) * 1000
        document_index_requests = provider.provider_texts - requests_before_documents
        postgres_index_started = time.perf_counter()
        for job, document, vector in zip(jobs, documents, document_vectors):
            store.save_embedding(
                job.stable_id, client.model,
                hashlib.sha256(document.encode("utf-8")).hexdigest(), vector,
            )
        postgres_index_latency_ms = (time.perf_counter() - postgres_index_started) * 1000

        case_context: dict[str, tuple[Preferences, str]] = {}
        unique_query_texts: list[str] = []
        for case in labels["cases"]:
            prefs = effective_search_preferences(
                Preferences.model_validate(case.get("saved_preferences", {})), case["query"]
            )
            query_text = semantic_query_text(prefs, case["query"])
            case_context[case["name"]] = (prefs, query_text)
            if query_text and query_text not in unique_query_texts:
                unique_query_texts.append(query_text)

        requests_before_queries = provider.provider_texts
        query_started = time.perf_counter()
        query_vectors_list = await client.embed(
            unique_query_texts, task_type=RETRIEVAL_QUERY
        )
        query_embedding_latency_ms = (time.perf_counter() - query_started) * 1000
        query_embedding_requests = provider.provider_texts - requests_before_queries
        query_vectors = dict(zip(unique_query_texts, query_vectors_list))
        http_attempts_after_cold = _embedding_attempt_snapshot()

        variants = {
            name: {"rows": [], "latency_ms": [], "embedding_invocations": 0, "cache_hits": 0}
            for name in ("lexical", "hybrid", "exhaustive")
        }
        explain_by_query: list[dict[str, Any]] = []
        for case in labels["cases"]:
            prefs, retrieval_query = case_context[case["name"]]

            started = time.perf_counter()
            lexical_matches, _ = await retrieve_and_rank_jobs(
                store, prefs, query=case["query"], top_k=top_k,
                rejected_ids=set(), client=lexical_client, candidate_limit=candidate_limit,
            )
            lexical_latency = (time.perf_counter() - started) * 1000
            lexical_candidates = store.candidate_jobs(
                retrieval_query, limit=candidate_limit, preferences=prefs
            )
            lexical_row = _row_metrics(case, lexical_matches, lexical_candidates, prefs, top_k)
            lexical_row.update({
                "name": case["name"], "family": case["family"],
                "query": case["query"], "latency_ms": round(lexical_latency, 3),
            })
            variants["lexical"]["latency_ms"].append(lexical_latency)
            variants["lexical"]["rows"].append(lexical_row)

            invocations_before = provider.embed_invocations
            hits_before = provider.cache_hits
            started = time.perf_counter()
            hybrid_matches, _ = await retrieve_and_rank_jobs(
                store, prefs, query=case["query"], top_k=top_k,
                rejected_ids=set(), client=client, candidate_limit=candidate_limit,
            )
            hybrid_latency = (time.perf_counter() - started) * 1000
            variants["hybrid"]["embedding_invocations"] += provider.embed_invocations - invocations_before
            variants["hybrid"]["cache_hits"] += provider.cache_hits - hits_before
            query_vector = query_vectors.get(retrieval_query)
            hybrid_candidates = store.candidate_jobs(
                retrieval_query, limit=candidate_limit, query_vector=query_vector,
                embedding_model=client.model, preferences=prefs,
            )
            hybrid_row = _row_metrics(case, hybrid_matches, hybrid_candidates, prefs, top_k)
            hybrid_row.update({
                "name": case["name"], "family": case["family"],
                "query": case["query"], "latency_ms": round(hybrid_latency, 3),
            })
            variants["hybrid"]["latency_ms"].append(hybrid_latency)
            variants["hybrid"]["rows"].append(hybrid_row)

            invocations_before = provider.embed_invocations
            hits_before = provider.cache_hits
            started = time.perf_counter()
            exhaustive_matches = await hybrid_rank_jobs(
                jobs, prefs, query=case["query"], top_k=top_k, rejected_ids=set(),
                client=client, store=store, query_vector=query_vector,
                query_embedding_attempted=bool(retrieval_query),
            )
            exhaustive_latency = (time.perf_counter() - started) * 1000
            variants["exhaustive"]["embedding_invocations"] += provider.embed_invocations - invocations_before
            variants["exhaustive"]["cache_hits"] += provider.cache_hits - hits_before
            exhaustive_row = _row_metrics(case, exhaustive_matches, jobs, prefs, top_k)
            exhaustive_row.update({
                "name": case["name"], "family": case["family"],
                "query": case["query"], "latency_ms": round(exhaustive_latency, 3),
            })
            variants["exhaustive"]["latency_ms"].append(exhaustive_latency)
            variants["exhaustive"]["rows"].append(exhaustive_row)

            explain_by_query.append({
                "name": case["name"], "query": case["query"],
                "lexical": store.explain_candidate_jobs(
                    retrieval_query, limit=candidate_limit, preferences=prefs
                ),
                "hybrid": store.explain_candidate_jobs(
                    retrieval_query, limit=candidate_limit, query_vector=query_vector,
                    embedding_model=client.model, preferences=prefs,
                ),
            })

        http_attempts_after_run = _embedding_attempt_snapshot()
        actual_http_attempts = _counter_delta(http_attempts_before, http_attempts_after_run)
        cold_http_attempts = _counter_delta(http_attempts_before, http_attempts_after_cold)
        total_http_attempt_count = sum(actual_http_attempts.values())
        report_variants: dict[str, Any] = {}
        for name, data in variants.items():
            relevant_rows = [row for row in data["rows"] if row["has_relevant_jobs"]]
            no_relevant_rows = [row for row in data["rows"] if not row["has_relevant_jobs"]]
            report_variants[name] = {
                "relevant_queries": _aggregate_rows(
                    relevant_rows, [row["latency_ms"] for row in relevant_rows]
                ),
                "no_relevant_queries": _aggregate_rows(
                    no_relevant_rows, [row["latency_ms"] for row in no_relevant_rows]
                ),
                "all_queries_operational": _aggregate_rows(data["rows"], data["latency_ms"]),
                "embedding_invocations_during_warm_search": data["embedding_invocations"],
                "embedding_cache_hits_during_warm_search": data["cache_hits"],
                "per_query": data["rows"],
            }

        plan_summary: dict[str, Any] = {}
        for variant in ("lexical", "hybrid"):
            plans = [item[variant] for item in explain_by_query]
            execution_times = [
                float(plan["execution_time_ms"])
                for plan in plans if plan.get("execution_time_ms") is not None
            ]
            index_counts: Counter[str] = Counter(
                index for plan in plans for index in plan["actual_indexes"]
            )
            plan_summary[variant] = {
                "queries_explained": len(plans),
                "actual_index_query_counts": dict(sorted(index_counts.items())),
                "queries_using_gin": sum(plan["gin_used"] for plan in plans),
                "queries_using_hnsw": sum(plan["hnsw_used"] for plan in plans),
                "execution_time_ms_p50": round(statistics.median(execution_times), 3),
                "execution_time_ms_p95": round(_percentile(execution_times, 0.95), 3),
            }

        unresolved_source = [
            item for item in labels.get("unresolved_cases", [])
        ]
        unresolved_report = []
        for item in unresolved_source:
            row = {**item, "variant_outcomes": {}}
            for name, data in variants.items():
                match = next(case_row for case_row in data["rows"] if case_row["name"] == item["case"])
                row["variant_outcomes"][name] = {
                    "retrieved": item["stable_id"] in match["predicted"],
                    "recommended": item["stable_id"] in match["recommended"],
                }
            unresolved_report.append(row)

        no_relevant_names = [
            case["name"] for case in labels["cases"]
            if not any(row["label"] in {"direct", "adjacent"} for row in case["labels"].values())
        ]
        return {
            "status": "ok",
            "corpus_sha256": corpus["records_sha256"],
            "labels_sha256": labels["labels_sha256"],
            "label_verification_status": labels["verification_status"],
            "jobs": len(jobs), "queries": len(labels["cases"]),
            "top_k": top_k, "candidate_limit": candidate_limit,
            "database": assert_isolated_database_url(database_url),
            "prepared": prepared,
            "variants": report_variants,
            "embedding_usage": {
                "initial_cache_vectors": initial_cache_vectors,
                "final_cache_vectors": len(provider._cache),
                "provider_batch_calls": provider.provider_calls,
                "provider_prediction_requests": provider.provider_texts,
                "document_index_prediction_requests": document_index_requests,
                "query_prediction_requests": query_embedding_requests,
                "actual_http_attempts_by_task_and_status": actual_http_attempts,
                "cold_http_attempts_by_task_and_status": cold_http_attempts,
                "actual_http_attempt_count": total_http_attempt_count,
                "additional_http_attempts_from_retries": max(
                    0, total_http_attempt_count - provider.provider_texts
                ),
                "variant_prediction_requests": {
                    "lexical": 0,
                    "hybrid": 0,
                    "exhaustive_incremental": 0,
                },
                "cache_hits": provider.cache_hits,
            },
            "latency_phases": {
                "cold_document_embedding_ms": round(document_embedding_latency_ms, 2),
                "postgres_document_index_write_ms": round(postgres_index_latency_ms, 2),
                "cold_query_embedding_ms": round(query_embedding_latency_ms, 2),
                "cold_total_embedding_ms": round(
                    document_embedding_latency_ms + query_embedding_latency_ms, 2
                ),
                "warm_search_latency_note": "Variant p50/p95 excludes embedding creation, candidate-metric queries, and EXPLAIN ANALYZE.",
            },
            "ground_truth_groups": {
                "queries_with_relevant_jobs": len(labels["cases"]) - len(no_relevant_names),
                "queries_with_no_relevant_jobs": len(no_relevant_names),
                "no_relevant_query_names": no_relevant_names,
                "conventional_recall_ndcg_note": "Recall and NDCG are null for empty-ground-truth rows and aggregated only over queries with relevant jobs.",
            },
            "unresolved_eligibility": unresolved_report,
            "explain_analyze": {
                "summary": plan_summary,
                "by_query": explain_by_query,
            },
        }
    finally:
        store.close()


def holdout_comparison_markdown(report: dict[str, Any]) -> str:
    variants = report["variants"]
    names = ("lexical", "hybrid", "exhaustive")

    def value(name: str, group: str, metric: str) -> str:
        item = variants[name][group].get(metric)
        return "N/A" if item is None else str(item)

    relevant_metrics = (
        ("Candidate recall", "candidate_recall"),
        ("Precision@10", "precision_at_10"),
        ("Recall@10", "recall_at_10"),
        ("NDCG@10", "ndcg_at_10"),
        ("Graded NDCG@10", "graded_ndcg_at_10"),
        ("Recommended precision", "recommended_precision"),
        ("Recommended recall", "recommended_recall"),
        ("Recommendation coverage", "recommendation_coverage"),
        ("Abstention rate", "abstention_rate"),
        ("Recommended irrelevant rate", "recommended_irrelevant_rate"),
        ("Forbidden-result rate", "forbidden_result_rate"),
        ("Hard-constraint violation rate", "hard_constraint_violation_rate"),
        ("Warm latency p50 (ms)", "latency_ms_p50"),
        ("Warm latency p95 (ms)", "latency_ms_p95"),
    )
    lines = [
        "# BoardForge controlled holdout evaluation",
        "",
        "> Labels: **AI-audited, not human-verified**.",
        "",
        f"- Corpus jobs: {report['jobs']}; queries: {report['queries']}"
        f" ({report['ground_truth_groups']['queries_with_relevant_jobs']} with relevant jobs,"
        f" {report['ground_truth_groups']['queries_with_no_relevant_jobs']} with none)",
        f"- Corpus checksum: `{report['corpus_sha256']}`",
        f"- Label checksum: `{report['labels_sha256']}`",
        "",
        "## Queries with relevant jobs",
        "",
        "| Metric | Lexical | Hybrid PostgreSQL | Exhaustive reference |",
        "|---|---:|---:|---:|",
    ]
    for label, metric in relevant_metrics:
        lines.append(
            f"| {label} | {value('lexical', 'relevant_queries', metric)} | "
            f"{value('hybrid', 'relevant_queries', metric)} | "
            f"{value('exhaustive', 'relevant_queries', metric)} |"
        )

    lines.extend([
        "",
        "## Queries with no relevant jobs",
        "",
        "Recall and NDCG are N/A and are excluded from conventional aggregates.",
        "",
        "| Metric | Lexical | Hybrid PostgreSQL | Exhaustive reference |",
        "|---|---:|---:|---:|",
    ])
    no_relevant_metrics = (
        ("False recommendations", "false_recommendations"),
        ("Queries with false recommendations", "queries_with_false_recommendations"),
        ("Recommendation coverage", "recommendation_coverage"),
        ("Abstention rate", "abstention_rate"),
        ("Recommended irrelevant rate", "recommended_irrelevant_rate"),
        ("Forbidden-result rate", "forbidden_result_rate"),
        ("Hard-constraint violation rate", "hard_constraint_violation_rate"),
        ("Warm latency p50 (ms)", "latency_ms_p50"),
        ("Warm latency p95 (ms)", "latency_ms_p95"),
    )
    for label, metric in no_relevant_metrics:
        lines.append(
            f"| {label} | {value('lexical', 'no_relevant_queries', metric)} | "
            f"{value('hybrid', 'no_relevant_queries', metric)} | "
            f"{value('exhaustive', 'no_relevant_queries', metric)} |"
        )

    usage = report["embedding_usage"]
    phases = report["latency_phases"]
    lines.extend([
        "",
        "## Embeddings and latency phases",
        "",
        f"- Vertex prediction texts: {usage['provider_prediction_requests']} "
        f"({usage['document_index_prediction_requests']} documents + "
        f"{usage['query_prediction_requests']} queries).",
        f"- Actual HTTP attempts: {usage['actual_http_attempt_count']}; "
        f"additional retry attempts: {usage['additional_http_attempts_from_retries']}.",
        f"- Embedding-cache hits: {usage['cache_hits']}.",
        f"- Cold document embedding: {phases['cold_document_embedding_ms']} ms; "
        f"cold query embedding: {phases['cold_query_embedding_ms']} ms; "
        f"PostgreSQL embedding writes: {phases['postgres_document_index_write_ms']} ms.",
        "- Warm retrieval p50/p95 values exclude embedding creation and EXPLAIN overhead.",
        "",
        "## Actual PostgreSQL plans",
        "",
    ])
    for name in ("lexical", "hybrid"):
        plan = report["explain_analyze"]["summary"][name]
        lines.append(
            f"- **{name}**: {plan['queries_explained']} queries explained; "
            f"GIN used by {plan['queries_using_gin']}; HNSW used by "
            f"{plan['queries_using_hnsw']}; indexes observed: "
            f"`{json.dumps(plan['actual_index_query_counts'], sort_keys=True)}`."
        )

    lines.extend(["", "## False recommendations on no-relevant-result queries", ""])
    no_relevant = set(report["ground_truth_groups"]["no_relevant_query_names"])
    for name in names:
        rows = [row for row in variants[name]["per_query"] if row["name"] in no_relevant]
        bad = [row for row in rows if row["recommended"]]
        if not bad:
            lines.append(f"- **{name}**: none.")
        else:
            for row in bad:
                lines.append(
                    f"- **{name} / {row['name']}**: "
                    + ", ".join(f"`{stable_id}`" for stable_id in row["recommended"])
                )

    lines.extend(["", "## Unresolved eligibility", ""])
    if not report["unresolved_eligibility"]:
        lines.append("None.")
    for item in report["unresolved_eligibility"]:
        lines.append(
            f"- `{item['case']}` / `{item['stable_id']}` ({item['label']}): "
            f"{item['reason']} Outcomes: `{json.dumps(item['variant_outcomes'], sort_keys=True)}`"
        )
    lines.extend([
        "",
        "Complete per-query metrics and raw `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` "
        "plans are stored in the machine-readable JSON report.",
    ])
    return "\n".join(lines) + "\n"
