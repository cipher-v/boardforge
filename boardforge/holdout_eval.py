from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from boardforge.evals import ndcg_at_k, precision_among_returned, recall_at_k
from boardforge.holdout import canonical_sha256
from boardforge.matching import hard_violations, job_text
from boardforge.models import Job, Preferences
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

    def _key(self, text: str, task_type: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.model}|{task_type}|{digest}"

    def has(self, text: str, task_type: str) -> bool:
        return self._key(text, task_type) in self._cache

    async def embed(self, texts: list[str], *, task_type: str = RETRIEVAL_DOCUMENT):
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


def _row_metrics(case: dict, matches, candidates: list[Job], prefs: Preferences, top_k: int) -> dict:
    direct = {sid for sid, row in case["labels"].items() if row["label"] == "direct"}
    adjacent = {sid for sid, row in case["labels"].items() if row["label"] == "adjacent"}
    relevant = direct | adjacent
    forbidden = {sid for sid, row in case["labels"].items() if row["label"] == "forbidden"}
    predicted = [match.job.stable_id for match in matches]
    candidate_ids = {job.stable_id for job in candidates}
    recommended = [match for match in matches if match.status == "recommended"]
    recommended_ids = [match.job.stable_id for match in recommended]
    return {
        "candidate_recall": len(candidate_ids & relevant) / len(relevant) if relevant else 1.0,
        "precision_at_k": precision_among_returned(predicted, relevant, top_k),
        "recall_at_k": recall_at_k(predicted, relevant, top_k),
        "ndcg_at_k": ndcg_at_k(predicted, relevant, top_k),
        "graded_ndcg_at_k": _graded_ndcg(predicted, direct, adjacent, top_k),
        "recommended_precision": precision_among_returned(recommended_ids, relevant, len(recommended_ids)),
        "recommended_recall": recall_at_k(recommended_ids, relevant, len(recommended_ids)),
        "recommendation_coverage": len(recommended_ids) / len(matches) if matches else 0.0,
        "abstention_rate": sum(m.status == "needs_verification" for m in matches) / len(matches) if matches else 1.0,
        "forbidden_result_rate": len(set(predicted) & forbidden) / len(predicted) if predicted else 0.0,
        "hard_constraint_violation_rate": sum(bool(hard_violations(m.job, prefs)) for m in matches) / len(matches) if matches else 0.0,
    }


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
    try:
        prepared = prepare_isolated_database(store, jobs)
        documents = [job_text(job)[:6000] for job in jobs]
        requests_before_documents = provider.provider_texts
        document_vectors = await client.embed(documents, task_type=RETRIEVAL_DOCUMENT)
        document_index_requests = provider.provider_texts - requests_before_documents
        for job, document, vector in zip(jobs, documents, document_vectors):
            store.save_embedding(
                job.stable_id, client.model,
                hashlib.sha256(document.encode("utf-8")).hexdigest(), vector,
            )

        variants = {name: {"rows": [], "latency_ms": []} for name in ("lexical", "hybrid", "exhaustive")}
        explain: dict[str, Any] = {}
        for index, case in enumerate(labels["cases"]):
            prefs = effective_search_preferences(
                Preferences.model_validate(case.get("saved_preferences", {})), case["query"]
            )
            retrieval_query = semantic_query_text(prefs, case["query"])

            started = time.perf_counter()
            lexical_matches, _ = await retrieve_and_rank_jobs(
                store, prefs, query=case["query"], top_k=top_k,
                rejected_ids=set(), client=lexical_client, candidate_limit=candidate_limit,
            )
            lexical_candidates = store.candidate_jobs(
                retrieval_query, limit=candidate_limit, preferences=prefs
            )
            variants["lexical"]["latency_ms"].append((time.perf_counter() - started) * 1000)
            variants["lexical"]["rows"].append(_row_metrics(case, lexical_matches, lexical_candidates, prefs, top_k))

            started = time.perf_counter()
            hybrid_matches, _ = await retrieve_and_rank_jobs(
                store, prefs, query=case["query"], top_k=top_k,
                rejected_ids=set(), client=client, candidate_limit=candidate_limit,
            )
            query_text = semantic_query_text(prefs, case["query"])
            query_vector = (await client.embed([query_text], task_type=RETRIEVAL_QUERY))[0] if query_text else None
            hybrid_candidates = store.candidate_jobs(
                retrieval_query, limit=candidate_limit, query_vector=query_vector,
                embedding_model=client.model, preferences=prefs,
            )
            variants["hybrid"]["latency_ms"].append((time.perf_counter() - started) * 1000)
            variants["hybrid"]["rows"].append(_row_metrics(case, hybrid_matches, hybrid_candidates, prefs, top_k))

            started = time.perf_counter()
            exhaustive_matches = await hybrid_rank_jobs(
                jobs, prefs, query=case["query"], top_k=top_k, rejected_ids=set(),
                client=client, store=store, query_vector=query_vector,
                query_embedding_attempted=bool(query_text),
            )
            variants["exhaustive"]["latency_ms"].append((time.perf_counter() - started) * 1000)
            variants["exhaustive"]["rows"].append(_row_metrics(case, exhaustive_matches, jobs, prefs, top_k))

            if index == 0:
                explain["lexical"] = store.explain_candidate_jobs(
                    retrieval_query, limit=candidate_limit, preferences=prefs
                )
                explain["hybrid"] = store.explain_candidate_jobs(
                    retrieval_query, limit=candidate_limit, query_vector=query_vector,
                    embedding_model=client.model, preferences=prefs,
                )

        metric_names = list(variants["lexical"]["rows"][0])
        report_variants = {}
        for name, data in variants.items():
            report_variants[name] = {
                **{metric: round(statistics.fmean(row[metric] for row in data["rows"]), 4) for metric in metric_names},
                "latency_ms_p50": round(statistics.median(data["latency_ms"]), 2),
                "latency_ms_p95": round(_percentile(data["latency_ms"], 0.95), 2),
            }
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
                "provider_batch_calls": provider.provider_calls,
                "provider_prediction_requests": provider.provider_texts,
                "document_index_prediction_requests": document_index_requests,
                "variant_prediction_requests": {
                    "lexical": 0,
                    "hybrid": provider.provider_texts - document_index_requests,
                    "exhaustive_incremental": 0,
                },
                "cache_hits": provider.cache_hits,
            },
            "explain_analyze": explain,
        }
    finally:
        store.close()
