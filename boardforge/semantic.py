from __future__ import annotations

import asyncio
import hashlib
import math
import os
from typing import Protocol

import httpx

from boardforge.database import StoreProtocol
from boardforge.matching import job_text, query_text, rank_jobs
from boardforge.models import Job, Match, Preferences


RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
POSTGRES_EMBEDDING_DIMENSIONS = 768


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


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
        self.dimensions = int(os.getenv("BOARDFORGE_EMBEDDING_DIMENSIONS", "1536"))
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
                json={"model": self.provider_model, "input": texts},
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
    def __init__(self):
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
        self._concurrency = max(
            1, min(8, int(os.getenv("BOARDFORGE_EMBEDDING_CONCURRENCY", "4")))
        )

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
        async with semaphore:
            response = await client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "instances": [
                        {
                            "content": text,
                            "task_type": task_type,
                        }
                    ],
                    "parameters": {
                        "autoTruncate": True,
                        "outputDimensionality": self.dimensions,
                    },
                },
            )
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
        semaphore = asyncio.Semaphore(self._concurrency)
        async with httpx.AsyncClient(timeout=45) as client:
            return await asyncio.gather(
                *(
                    self._embed_one(client, token, text, task_type, semaphore)
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

    def __init__(self, provider: EmbeddingProvider | None = None):
        self.provider = provider or create_embedding_provider()

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    @property
    def enabled(self) -> bool:
        return (
            os.getenv("BOARDFORGE_ENABLE_EMBEDDINGS", "0") == "1"
            and self.provider.enabled
        )

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
) -> list[Match]:
    baseline = rank_jobs(
        jobs, prefs, query=query, top_k=len(jobs), rejected_ids=rejected_ids
    )
    if not baseline or not client.enabled:
        return baseline[:top_k]

    qtext = query_text(prefs, query)
    try:
        qv = query_vector or (
            await client.embed([qtext], task_type=RETRIEVAL_QUERY)
        )[0]
    except Exception:
        return baseline[:top_k]

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

    if missing_indexes:
        try:
            embedded = await client.embed(
                [documents[index] for index in missing_indexes],
                task_type=RETRIEVAL_DOCUMENT,
            )
        except Exception:
            return baseline[:top_k]
        if len(embedded) != len(missing_indexes):
            raise RuntimeError(
                "embedding provider returned an unexpected vector count"
            )
        for index, vector in zip(missing_indexes, embedded):
            vectors[index] = vector
            if store:
                store.save_embedding(
                    baseline[index].job.stable_id,
                    client.model,
                    content_hashes[index],
                    vector,
                )

    rescored = []
    for match, vector in zip(baseline, vectors):
        if vector is None:  # pragma: no cover
            continue
        semantic = cosine(qv, vector)
        match.score = round((0.55 * match.score) + (0.45 * semantic), 4)
        if semantic > 0.45:
            match.reasons.append("semantic similarity supports the match")
        rescored.append(match)

    rescored.sort(
        key=lambda match: (
            bool(match.hard_violations),
            match.status == "needs_verification",
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
) -> tuple[list[Match], dict]:
    retrieval_query = query_text(prefs, query).strip()
    query_vector: list[float] | None = None
    semantic_retrieval = False
    if client.enabled and retrieval_query:
        try:
            query_vector = (
                await client.embed([retrieval_query], task_type=RETRIEVAL_QUERY)
            )[0]
            semantic_retrieval = True
        except Exception:
            query_vector = None

    jobs = store.candidate_jobs(
        retrieval_query,
        limit=candidate_limit,
        query_vector=query_vector,
        embedding_model=client.model if query_vector is not None else "",
    )
    matches = await hybrid_rank_jobs(
        jobs,
        prefs,
        query=query,
        top_k=top_k,
        rejected_ids=rejected_ids,
        client=client,
        store=store,
        query_vector=query_vector,
    )
    return matches, {
        "store": type(store).__name__,
        "candidate_count": len(jobs),
        "candidate_limit": candidate_limit,
        "semantic_retrieval": semantic_retrieval,
        "embedding_model": client.model if client.enabled else None,
    }


async def warm_job_embeddings(
    store: StoreProtocol,
    client: EmbeddingClient,
    *,
    batch_size: int = 64,
    max_jobs: int = 10_000,
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
    }
