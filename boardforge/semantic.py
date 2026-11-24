from __future__ import annotations

import math
import os
import hashlib
import httpx

from boardforge.matching import job_text, rank_jobs
from boardforge.models import Job, Match, Preferences
from boardforge.store import Store


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na*nb)


class EmbeddingClient:
    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY")
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    @property
    def enabled(self) -> bool:
        return bool(self.key and os.getenv("BOARDFORGE_ENABLE_EMBEDDINGS", "0") == "1")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings")
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(
                f"{self.base}/embeddings",
                headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json"},
                json={"model":self.model,"input":texts},
            )
        r.raise_for_status()
        rows = sorted(r.json()["data"], key=lambda x:x["index"])
        return [row["embedding"] for row in rows]


async def hybrid_rank_jobs(
    jobs: list[Job],
    prefs: Preferences,
    query: str,
    top_k: int,
    rejected_ids: set[str],
    client: EmbeddingClient,
    store: Store | None = None,
) -> list[Match]:
    baseline = rank_jobs(jobs, prefs, query=query, top_k=len(jobs), rejected_ids=rejected_ids)
    if not baseline or not client.enabled:
        return baseline[:top_k]

    qtext = " ".join(prefs.roles + prefs.locations + prefs.prefer_terms + [query])
    qv = (await client.embed([qtext]))[0]
    documents = [job_text(match.job)[:6000] for match in baseline]
    vectors: list[list[float] | None] = [None] * len(documents)
    missing_indexes: list[int] = []
    content_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in documents]

    for index, (match, content_hash) in enumerate(zip(baseline, content_hashes)):
        cached = store.load_embedding(match.job.stable_id, client.model, content_hash) if store else None
        if cached is None:
            missing_indexes.append(index)
        else:
            vectors[index] = cached

    if missing_indexes:
        embedded = await client.embed([documents[index] for index in missing_indexes])
        if len(embedded) != len(missing_indexes):
            raise RuntimeError("embedding provider returned an unexpected vector count")
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
    for match, vec in zip(baseline, vectors):
        if vec is None:  # pragma: no cover - guarded by cache/fetch paths above
            continue
        semantic = cosine(qv, vec)
        match.score = round((0.55 * match.score) + (0.45 * semantic), 4)
        if semantic > 0.45:
            match.reasons.append("semantic similarity supports the match")
        rescored.append(match)

    rescored.sort(
        key=lambda m: (
            bool(m.hard_violations),
            m.status == "needs_verification",
            -m.score,
            m.job.title.lower(),
        )
    )
    return rescored[:top_k]
