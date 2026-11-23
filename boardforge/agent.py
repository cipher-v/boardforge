from __future__ import annotations

from dataclasses import dataclass
from boardforge.models import FeedbackRequest
from boardforge.preferences import LLMPreferenceExtractor
from boardforge.semantic import EmbeddingClient, hybrid_rank_jobs
from boardforge.store import Store


@dataclass
class AgentResult:
    message: str
    data: dict


class JobAgent:
    """Explicit tools around a persistent, testable job-search core."""

    def __init__(
        self,
        store: Store,
        extractor: LLMPreferenceExtractor | None = None,
        embeddings: EmbeddingClient | None = None,
    ):
        self.store = store
        self.extractor = extractor or LLMPreferenceExtractor()
        self.embeddings = embeddings or EmbeddingClient()

    async def update_preferences(self, profile_id: str, text: str) -> AgentResult:
        current = self.store.load_preferences(profile_id)
        updated = await self.extractor.extract(text, current)
        self.store.save_preferences(profile_id, updated)
        return AgentResult("Preferences updated.", updated.model_dump())

    async def search_jobs(self, profile_id: str, query: str = "", top_k: int = 10) -> AgentResult:
        prefs = self.store.load_preferences(profile_id)
        matches = await hybrid_rank_jobs(
            self.store.active_jobs(),
            prefs,
            query=query,
            top_k=top_k,
            rejected_ids=self.store.rejected_job_ids(profile_id),
            client=self.embeddings,
        )
        return AgentResult(
            f"Found {len(matches)} ranked jobs.",
            {"matches":[m.model_dump(mode="json") for m in matches]},
        )

    async def record_feedback(self, req: FeedbackRequest) -> AgentResult:
        self.store.record_feedback(req)
        return AgentResult("Feedback recorded.", req.model_dump())

    async def handle(self, profile_id: str, text: str, top_k: int = 10) -> AgentResult:
        low = text.lower()
        if any(x in low for x in ["don't show","do not show","prefer","i want","avoid","only","remote"]):
            await self.update_preferences(profile_id, text)
        return await self.search_jobs(profile_id, query=text, top_k=top_k)
