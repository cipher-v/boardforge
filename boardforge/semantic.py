from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
import uuid
from typing import Protocol

import httpx

from boardforge.database import StoreProtocol
from boardforge.confidence import GENERIC_QUERY_TERMS, assess_query_relevance, occupational_query_text
from boardforge.matching import _has_query_overlap, job_text, rank_jobs, tokens
from boardforge.models import Job, Match, Preferences
from boardforge.observability import observe_embedding_request
from boardforge.preferences import extract_experience_floor, is_entry_level_target


RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
POSTGRES_EMBEDDING_DIMENSIONS = 768
QUERY_EMBEDDING_CACHE_VERSION = 1


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def semantic_query_text(preferences: Preferences, query: str = "") -> str:
    """Keep semantic retrieval focused on role/skill intent, not hard-filter tokens."""
    cleaned = occupational_query_text(query)
    for term in [*preferences.countries, *preferences.locations]:
        if term:
            cleaned = re.sub(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )
    cleaned = re.sub(
        r"\b(?:remote|remote-only|onsite|on-site|onsite-only|hybrid)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:lpa|lakhs?|lakh)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    parts = [*preferences.roles, *preferences.prefer_terms, " ".join(cleaned.split())]
    return " ".join(part for part in parts if part).strip()


def lexical_query_text(preferences: Preferences, query: str = "") -> str:
    """Build a PostgreSQL FTS query from explicit search intent without AND-ing soft resume terms."""
    cleaned = occupational_query_text(query)
    for term in [*preferences.countries, *preferences.locations]:
        if term:
            cleaned = re.sub(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )
    cleaned = re.sub(
        r"\b(?:remote|remote-only|onsite|on-site|onsite-only|hybrid)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:lpa|lakhs?|lakh)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    explicit_tokens = [
        tok for tok in tokens(cleaned)
        if tok not in GENERIC_QUERY_TERMS
    ]
    role_tokens = [
        tok
        for role in preferences.roles
        for tok in tokens(role.replace("/", " "))
        if tok not in GENERIC_QUERY_TERMS
    ]
    if query.strip():
        # Keep explicit search intent isolated from soft resume prefer_terms so
        # websearch_to_tsquery never turns a 15-skill resume into an impossible AND.
        combined = list(dict.fromkeys([*explicit_tokens, *role_tokens]))
        return " or ".join(combined[:12])
    soft_tokens = [
        tok
        for term in [*preferences.roles, *preferences.prefer_terms[:8]]
        for tok in tokens(term.replace("/", " "))
        if tok not in GENERIC_QUERY_TERMS
    ]
    return " or ".join(list(dict.fromkeys(soft_tokens))[:12])


def query_embedding_cache_identity(query: str, client: "EmbeddingClient") -> tuple[str, str, str]:
    normalized = " ".join(query.casefold().split())
    configuration = {
        "version": QUERY_EMBEDDING_CACHE_VERSION,
        "model": client.model,
        "dimensions": client.dimensions,
        "task_type": RETRIEVAL_QUERY,
    }
    config_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_key = hashlib.sha256(
        json.dumps(
            {"normalized_query": normalized, "config_hash": config_hash},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return cache_key, normalized, config_hash


class EmbeddingProvider(Protocol):
    enabled: bool
    model: str
    dimensions: int

    async def embed(
        self,
        texts: list[str],
        *,
        task_type: str = RETRIEVAL_DOCUMENT,
    ) -> list[list[float]]:
        ...


class OpenAIEmbeddingProvider:
    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY", "")
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.provider_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.dimensions = int(os.getenv("BOARDFORGE_EMBEDDING_DIMENSIONS", "768"))
        self.model = f"openai:{self.provider_model}:{self.dimensions}"
        self.enabled = bool(self.key)

    async def embed(
        self,
        texts: list[str],
        *,
        task_type: str = RETRIEVAL_DOCUMENT,
    ) -> list[list[float]]:
        del task_type
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI-compatible embeddings")
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.base}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.provider_model,
                    "input": texts,
                    "dimensions": self.dimensions,
                },
            )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [row["embedding"] for row in rows]
        self._validate(vectors)
        return vectors

    def _validate(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}"
                )


class VertexGeminiEmbeddingProvider:
    def __init__(self, *, budget_store=None):
        self.budget_store = budget_store
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.provider_model = os.getenv(
            "BOARDFORGE_EMBEDDING_MODEL", "gemini-embedding-001"
        )
        self.dimensions = int(os.getenv("BOARDFORGE_EMBEDDING_DIMENSIONS", "768"))
        if self.dimensions != POSTGRES_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "BoardForge PostgreSQL embedding index is fixed at 768 dimensions; "
                "set BOARDFORGE_EMBEDDING_DIMENSIONS=768"
            )
        self.model = f"vertex:{self.provider_model}:{self.dimensions}"
        self.enabled = bool(self.project)
        self._credentials = None
        self._auth_request = None
        self._http_client: httpx.AsyncClient | None = None
        self._concurrency = max(
            1, min(8, int(os.getenv("BOARDFORGE_EMBEDDING_CONCURRENCY", "4")))
        )
        self._max_retries = max(
            0, min(5, int(os.getenv("BOARDFORGE_EMBEDDING_MAX_RETRIES", "3")))
        )
        self._semaphore = asyncio.Semaphore(self._concurrency)

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=45)
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _ensure_credentials(self):
        if self._credentials is not None:
            return
        import google.auth
        from google.auth.transport.requests import Request

        credentials, detected_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
            quota_project_id=self.project or None,
        )
        if not self.project:
            self.project = detected_project or ""
        if not self.project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is required for Vertex embeddings"
            )
        self._credentials = credentials
        self._auth_request = Request()

    async def _access_token(self) -> str:
        self._ensure_credentials()
        if not self._credentials.valid or not self._credentials.token:
            await asyncio.to_thread(self._credentials.refresh, self._auth_request)
        if not self._credentials.token:
            raise RuntimeError("failed to obtain Google Cloud access token")
        return self._credentials.token

    @property
    def endpoint(self) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project}/locations/{self.location}/publishers/google/models/"
            f"{self.provider_model}:predict"
        )

    async def _embed_one(
        self,
        client: httpx.AsyncClient,
        token: str,
        text: str,
        task_type: str,
        semaphore: asyncio.Semaphore,
    ) -> list[float]:
        payload = {
            "instances": [{"content": text, "task_type": task_type}],
            "parameters": {
                "autoTruncate": True,
                "outputDimensionality": self.dimensions,
            },
        }
        response = None
        for attempt in range(self._max_retries + 1):
            reservation_id = None
            estimated_tokens = max(1, math.ceil(len(text.encode("utf-8")) / 4))
            estimated_cost = estimated_tokens * float(os.getenv(
                "BOARDFORGE_VERTEX_EMBEDDING_USD_PER_MILLION", "0.15"
            )) / 1_000_000
            if self.budget_store is not None:
                reservation_id = str(uuid.uuid4())
                admission = self.budget_store.reserve_llm_usage(
                    "vertex-embedding", self.model, reservation_id, estimated_tokens,
                    tpm_limit=int(os.getenv("BOARDFORGE_VERTEX_EMBEDDING_TPM_LIMIT", "1000000")),
                    tpd_limit=int(os.getenv("BOARDFORGE_VERTEX_EMBEDDING_TPD_LIMIT", "10000000")),
                    reserved_cost_usd=estimated_cost,
                    cumulative_cost_limit_usd=float(os.getenv(
                        "BOARDFORGE_CUMULATIVE_SPEND_LIMIT_USD", "2.0"
                    )),
                )
                if not admission["allowed"]:
                    raise RuntimeError(
                        f"Vertex embedding budget denied: {admission['reason']}"
                    )
            started = time.perf_counter()
            try:
                async with semaphore:
                    response = await client.post(
                        self.endpoint,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            except asyncio.CancelledError:
                if reservation_id:
                    self.budget_store.finalize_llm_usage(
                        reservation_id, status="failed", estimated_cost_usd=estimated_cost
                    )
                observe_embedding_request(
                    "vertex", task_type, "cancelled", time.perf_counter() - started
                )
                raise
            except (httpx.TimeoutException, httpx.TransportError):
                if reservation_id:
                    self.budget_store.finalize_llm_usage(
                        reservation_id, status="failed", estimated_cost_usd=estimated_cost
                    )
                observe_embedding_request(
                    "vertex", task_type, "transport_error", time.perf_counter() - started
                )
                if attempt >= self._max_retries:
                    raise
            else:
                if reservation_id:
                    self.budget_store.finalize_llm_usage(
                        reservation_id, status="completed",
                        input_tokens=estimated_tokens, total_tokens=estimated_tokens,
                        estimated_cost_usd=estimated_cost,
                    )
                observe_embedding_request(
                    "vertex", task_type, str(response.status_code), time.perf_counter() - started
                )
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
                if attempt >= self._max_retries:
                    break

            delay = min(8.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.25)
            await asyncio.sleep(delay)

        if response is None:
            raise RuntimeError("Vertex embedding request failed before receiving a response")
        if response.status_code in {401, 403}:
            raise RuntimeError(
                f"Vertex embedding authentication/permission failure ({response.status_code})"
            )
        if response.status_code == 429:
            raise RuntimeError("Vertex embedding rate limit reached (429)")
        response.raise_for_status()
        predictions = response.json().get("predictions") or []
        if len(predictions) != 1:
            raise RuntimeError("Vertex embedding response did not contain one prediction")
        vector = predictions[0].get("embeddings", {}).get("values")
        if not isinstance(vector, list):
            raise RuntimeError("Vertex embedding response is missing embeddings.values")
        values = [float(value) for value in vector]
        if len(values) != self.dimensions:
            raise RuntimeError(
                f"embedding dimension mismatch: expected {self.dimensions}, got {len(values)}"
            )
        return values

    async def embed(
        self,
        texts: list[str],
        *,
        task_type: str = RETRIEVAL_DOCUMENT,
    ) -> list[list[float]]:
        if not self.enabled:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is required for Vertex embeddings"
            )
        if task_type not in {RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT}:
            raise ValueError(f"unsupported BoardForge retrieval task type: {task_type}")
        if not texts:
            return []

        token = await self._access_token()
        client = self._get_http_client()
        return await asyncio.gather(
            *(
                self._embed_one(client, token, text, task_type, self._semaphore)
                for text in texts
            )
        )


def create_embedding_provider() -> EmbeddingProvider:
    provider = os.getenv("BOARDFORGE_EMBEDDING_PROVIDER", "openai").strip().casefold()
    if provider == "vertex":
        return VertexGeminiEmbeddingProvider()
    if provider in {"openai", "openai-compatible"}:
        return OpenAIEmbeddingProvider()
    raise ValueError(
        "unsupported BOARDFORGE_EMBEDDING_PROVIDER; expected vertex or openai"
    )


class EmbeddingClient:
    """Feature-gated facade over the configured embedding provider."""

    def __init__(self, provider: EmbeddingProvider | None = None, *, budget_store=None):
        self._provider_injected = provider is not None
        self.provider = provider or create_embedding_provider()
        if budget_store is not None and hasattr(self.provider, "budget_store"):
            self.provider.budget_store = budget_store

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    @property
    def enabled(self) -> bool:
        # An explicitly injected provider is already an explicit opt-in. This
        # keeps tests/evaluations deterministic without weakening the runtime
        # feature gate used by the default environment-created provider.
        feature_enabled = self._provider_injected or (
            os.getenv("BOARDFORGE_ENABLE_EMBEDDINGS", "0") == "1"
        )
        return feature_enabled and self.provider.enabled

    async def embed(
        self,
        texts: list[str],
        *,
        task_type: str = RETRIEVAL_DOCUMENT,
    ) -> list[list[float]]:
        return await self.provider.embed(texts, task_type=task_type)


async def hybrid_rank_jobs(
    jobs: list[Job],
    prefs: Preferences,
    query: str,
    top_k: int,
    rejected_ids: set[str],
    client: EmbeddingClient,
    store: StoreProtocol | None = None,
    query_vector: list[float] | None = None,
    query_embedding_attempted: bool = False,
) -> list[Match]:
    if top_k <= 0:
        return []
    baseline = rank_jobs(
        jobs,
        prefs,
        query=query,
        top_k=len(jobs),
        rejected_ids=rejected_ids,
        allow_semantic_candidates=client.enabled,
    )
    if not baseline or not client.enabled:
        return baseline[:top_k]

    def _deterministic_fallback(candidates: list[Match]) -> list[Match]:
        if not query.strip():
            return candidates[:top_k]
        filtered = [
            m for m in candidates
            if _has_query_overlap(
                m.job, prefs, query, m.evidence, assess_query_relevance(m.job, prefs, query)
            )
        ]
        return filtered[:top_k]

    if query_vector is not None:
        qv = query_vector
    elif query_embedding_attempted:
        # retrieve_and_rank_jobs already tried this exact query. A provider
        # failure must degrade once to deterministic ranking, not trigger a
        # duplicate paid request from the hybrid rescoring layer.
        return _deterministic_fallback(baseline)
    else:
        qtext = semantic_query_text(prefs, query)
        try:
            qv = (await client.embed([qtext], task_type=RETRIEVAL_QUERY))[0]
        except Exception:
            return _deterministic_fallback(baseline)

    documents = [job_text(match.job)[:6000] for match in baseline]
    vectors: list[list[float] | None] = [None] * len(documents)
    missing_indexes: list[int] = []
    content_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in documents
    ]

    for index, (match, content_hash) in enumerate(zip(baseline, content_hashes)):
        cached = (
            store.load_embedding(match.job.stable_id, client.model, content_hash)
            if store
            else None
        )
        if cached is None:
            missing_indexes.append(index)
        else:
            vectors[index] = cached

    # Production search must never fan out to the external embedding provider
    # for missing documents. Document embeddings are populated explicitly by
    # embed-index / background maintenance. In-memory evals (store=None) may
    # embed documents on demand because they do not persist or serve requests.
    if missing_indexes and store is None:
        try:
            embedded = await client.embed(
                [documents[index] for index in missing_indexes],
                task_type=RETRIEVAL_DOCUMENT,
            )
        except Exception:
            return _deterministic_fallback(baseline)
        if len(embedded) != len(missing_indexes):
            raise RuntimeError(
                "embedding provider returned an unexpected vector count"
            )
        for index, vector in zip(missing_indexes, embedded):
            vectors[index] = vector

    entry_level_target = is_entry_level_target(prefs, query)
    from boardforge.confidence import SENIORITY_TITLE_RE

    rescored = []
    for match, vector in zip(baseline, vectors):
        if vector is None:
            # Keep the deterministic lexical/evidence result while the offline
            # embedding index catches up, excluding zero-overlap non-semantic rows.
            if query.strip() and not _has_query_overlap(
                match.job,
                prefs,
                query,
                match.evidence,
                assess_query_relevance(match.job, prefs, query),
            ):
                continue
            rescored.append(match)
            continue
        semantic = cosine(qv, vector)
        match.score = round((0.55 * match.score) + (0.45 * semantic), 4)

        if semantic > 0.45:
            match.reasons.append(f"semantic similarity score {semantic:.3f}")

        rescored.append(match)

    rescored.sort(
        key=lambda match: (
            bool(match.hard_violations),
            match.status == "needs_verification",
            (entry_level_target and bool(SENIORITY_TITLE_RE.search(match.job.title))),
            (prefs.max_required_experience_years is not None
             and match.job.min_experience_years is None),
            -match.score,
            match.job.title.lower(),
        )
    )
    return rescored[:top_k]


async def retrieve_and_rank_jobs(
    store: StoreProtocol,
    prefs: Preferences,
    *,
    query: str,
    top_k: int,
    rejected_ids: set[str],
    client: EmbeddingClient,
    candidate_limit: int,
    deadline_monotonic: float | None = None,
) -> tuple[list[Match], dict]:
    total_started = time.perf_counter()
    retrieval_query = semantic_query_text(prefs, query)
    lexical_query = lexical_query_text(prefs, query)
    min_floor = extract_experience_floor(query) if query.strip() else None
    query_vector: list[float] | None = None
    semantic_retrieval = False
    embedding_error: str | None = None
    query_embedding_cache_hit = False
    query_embedding_source = "disabled"
    embedding_started = time.perf_counter()
    if client.enabled and retrieval_query:
        cache_key, normalized_query, config_hash = query_embedding_cache_identity(
            retrieval_query, client
        )
        query_vector = (
            store.load_query_embedding(cache_key)
            if hasattr(store, "load_query_embedding") else None
        )
        if query_vector is not None:
            semantic_retrieval = True
            query_embedding_cache_hit = True
            query_embedding_source = "persistent_cache"
        else:
            min_remaining = max(0.1, float(os.getenv(
                "BOARDFORGE_EMBEDDING_MIN_REMAINING_SECONDS", "3"
            )))
            remaining = (
                deadline_monotonic - time.perf_counter()
                if deadline_monotonic is not None else None
            )
            if remaining is not None and remaining < min_remaining:
                embedding_error = "insufficient_remaining_time_for_query_embedding"
                query_embedding_source = "deadline_skip"
            else:
                try:
                    query_vector = (
                        await client.embed([retrieval_query], task_type=RETRIEVAL_QUERY)
                    )[0]
                    if hasattr(store, "save_query_embedding"):
                        store.save_query_embedding(
                            cache_key, normalized_query, client.model, config_hash, query_vector
                        )
                    semantic_retrieval = True
                    query_embedding_source = "provider"
                except Exception as exc:
                    query_vector = None
                    embedding_error = f"{type(exc).__name__}: {exc}"
                    query_embedding_source = "provider_failure"
    query_embedding_ms = (time.perf_counter() - embedding_started) * 1000

    database_started = time.perf_counter()
    jobs = store.candidate_jobs(
        lexical_query,
        limit=candidate_limit,
        query_vector=query_vector,
        embedding_model=client.model if query_vector is not None else "",
        preferences=prefs,
    )
    database_ms = (time.perf_counter() - database_started) * 1000
    ranking_started = time.perf_counter()
    matches = await hybrid_rank_jobs(
        jobs,
        prefs,
        query=query,
        top_k=top_k,
        rejected_ids=rejected_ids,
        client=client,
        store=store,
        query_vector=query_vector,
        query_embedding_attempted=client.enabled and bool(retrieval_query),
    )
    ranking_ms = (time.perf_counter() - ranking_started) * 1000
    return matches, {
        "store": type(store).__name__,
        "candidate_count": len(jobs),
        "candidate_limit": candidate_limit,
        "semantic_retrieval": semantic_retrieval,
        "query_embedding_cache_hit": query_embedding_cache_hit,
        "query_embedding_source": query_embedding_source,
        "embedding_model": client.model if client.enabled else None,
        "embedding_error": embedding_error,
        "lexical_query": lexical_query,
        "unsupported_constraints": (
            [f"minimum_experience_years>={min_floor:g}"]
            if min_floor is not None else []
        ),
        "latency_ms": {
            "query_embedding": round(query_embedding_ms, 2),
            "database_retrieval": round(database_ms, 2),
            "ranking": round(ranking_ms, 2),
            "retrieval_total": round((time.perf_counter() - total_started) * 1000, 2),
        },
        "hard_filters_pushed_down": isinstance(prefs, Preferences) and bool(
            prefs.countries
            or prefs.locations
            or prefs.max_required_experience_years is not None
            or not prefs.remote_ok
            or not prefs.onsite_ok
            or prefs.paid_only
            or prefs.min_salary_lpa is not None
        ),
    }


async def warm_job_embeddings(
    store: StoreProtocol,
    client: EmbeddingClient,
    *,
    batch_size: int = 64,
    max_jobs: int = 10_000,
    countries: list[str] | None = None,
) -> dict:
    if not client.enabled:
        raise RuntimeError(
            "embeddings are disabled or the configured provider is unavailable; "
            "set BOARDFORGE_ENABLE_EMBEDDINGS=1 and provider credentials"
        )
    processed = 0
    while processed < max_jobs:
        jobs = store.embedding_candidates(
            client.model,
            limit=min(batch_size, max_jobs - processed),
            countries=countries,
        )
        if not jobs:
            break
        documents = [job_text(job)[:6000] for job in jobs]
        vectors = await client.embed(
            documents,
            task_type=RETRIEVAL_DOCUMENT,
        )
        if len(vectors) != len(jobs):
            raise RuntimeError(
                "embedding provider returned an unexpected vector count"
            )
        for job, document, vector in zip(jobs, documents, vectors):
            content_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()
            store.save_embedding(
                job.stable_id,
                client.model,
                content_hash,
                vector,
            )
        processed += len(jobs)
    return {
        "processed": processed,
        "model": client.model,
        "dimensions": client.dimensions,
        "countries": countries or [],
    }
