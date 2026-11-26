from __future__ import annotations

from dataclasses import dataclass
import time
from boardforge.models import FeedbackRequest
from boardforge.database import StoreProtocol
from boardforge.planning import plan_tools
from boardforge.preferences import LLMPreferenceExtractor, effective_search_preferences, preserve_search_constraints
from boardforge.resume import apply_candidate_context
from boardforge.relevance import RelevanceVerifier
from boardforge.semantic import EmbeddingClient, retrieve_and_rank_jobs


@dataclass
class AgentResult:
    message: str
    data: dict


class JobAgent:
    """Explicit tools around a persistent, testable job-search core."""

    def __init__(
        self,
        store: StoreProtocol,
        extractor: LLMPreferenceExtractor | None = None,
        embeddings: EmbeddingClient | None = None,
        relevance_verifier: RelevanceVerifier | None = None,
    ):
        self.store = store
        self.extractor = extractor or LLMPreferenceExtractor()
        self.embeddings = embeddings or EmbeddingClient(budget_store=store)
        if hasattr(self.embeddings, "provider") and hasattr(
            self.embeddings.provider, "budget_store"
        ):
            self.embeddings.provider.budget_store = store
        self.relevance_verifier = relevance_verifier or RelevanceVerifier.from_env(store)

    async def update_preferences(self, profile_id: str, text: str) -> AgentResult:
        current = self.store.load_preferences(profile_id)
        updated = await self.extractor.extract(text, current)
        self.store.save_preferences(profile_id, updated)
        return AgentResult("Preferences updated.", updated.model_dump())

    @staticmethod
    def _record_sanitized_telemetry(store: StoreProtocol, event_kind: str, payload: dict) -> None:
        recorder = getattr(store, "record_telemetry_event", None)
        if not callable(recorder):
            return
        try:
            recorder(event_kind, payload)
        except Exception:
            pass

    async def search_jobs(
        self, profile_id: str, query: str = "", top_k: int = 10,
        *, deadline_monotonic: float | None = None, original_query: str | None = None,
        record_telemetry: bool = True,
    ) -> AgentResult:
        if "\x00" in query or "\x00" in profile_id:
            raise ValueError("input must not contain NUL bytes")
        search_started = time.perf_counter()
        top_k = max(1, min(int(top_k), 50))
        saved = self.store.load_preferences(profile_id)
        prefs = effective_search_preferences(saved, query) if query else saved
        if original_query is not None:
            prefs = preserve_search_constraints(saved, original_query, prefs)
        candidate_profile = self.store.load_candidate_profile(profile_id)
        if candidate_profile:
            prefs = apply_candidate_context(prefs, candidate_profile)
        candidate_limit = max(100, min(1000, top_k * 25))
        rejected_ids = self.store.rejected_job_ids(profile_id)
        retrieval_k = (
            max(top_k, self.relevance_verifier.max_candidates)
            if self.relevance_verifier.enabled else top_k
        )
        matches, retrieval = await retrieve_and_rank_jobs(
            self.store,
            prefs,
            query=query,
            top_k=retrieval_k,
            rejected_ids=rejected_ids,
            client=self.embeddings,
            candidate_limit=candidate_limit,
            deadline_monotonic=deadline_monotonic,
        )
        matches, verification = await self.relevance_verifier.verify_matches(
            matches,
            intent=query,
            preferences=prefs,
            rejected_ids=rejected_ids,
            deadline_monotonic=deadline_monotonic,
        )
        retrieval["relevance_verification"] = verification
        retrieval["pre_verification_pool_size"] = retrieval_k
        ver_ms = float((verification.get("latency_ms") or {}).get("total", 0.0))
        retrieval.setdefault("latency_ms", {})["relevance_verification"] = round(ver_ms, 2)
        retrieval["latency_ms"]["total_search"] = round((time.perf_counter() - search_started) * 1000, 2)
        emb_cost = float(retrieval.get("query_embedding_cost_usd", 0.0) or 0.0)
        ver_cost = float(verification.get("estimated_cost_usd", 0.0) or 0.0)
        retrieval["cost_usd"] = {
            "query_embedding": round(emb_cost, 8),
            "relevance_verification": round(ver_cost, 8),
            "total": round(emb_cost + ver_cost, 8),
        }
        matches = matches[:top_k]
        if record_telemetry:
            self._record_sanitized_telemetry(
                self.store,
                "search",
                {
                    "endpoint": "search",
                    "orchestration_mode": "direct_search",
                    "stopped_reason": "completed",
                    "tools_executed": [
                        {
                            "tool": "search_jobs",
                            "ok": True,
                            "duration_ms": retrieval["latency_ms"]["total_search"],
                            "error": None,
                        }
                    ],
                    "match_count": len(matches),
                    "recommended_count": sum(m.status == "recommended" for m in matches),
                    "alternative_count": sum(m.status == "alternative" for m in matches),
                    "needs_verification_count": sum(m.status == "needs_verification" for m in matches),
                    "cache_hits": {
                        "query_embedding": bool(retrieval.get("query_embedding_cache_hit")),
                        "relevance_verification": int(verification.get("cache_hits", 0) or 0),
                    },
                    "latency_ms": retrieval["latency_ms"],
                    "cost_usd": retrieval["cost_usd"],
                },
            )
        return AgentResult(
            f"Found {len(matches)} ranked jobs.",
            {
                "matches":[m.model_dump(mode="json") for m in matches],
                "profile_used": candidate_profile is not None,
                "retrieval": retrieval,
            },
        )

    async def recommend_jobs(self, profile_id: str, top_k: int = 15) -> AgentResult:
        """Resume-led recommendations that stay within relevant role families."""
        rec_started = time.perf_counter()
        top_k = max(1, min(int(top_k), 50))
        saved = self.store.load_preferences(profile_id)
        candidate_profile = self.store.load_candidate_profile(profile_id)
        if candidate_profile is None:
            return await self.search_jobs(profile_id, "", top_k)

        prefs = saved.model_copy(deep=True)
        # Only roles explicitly confirmed in the profile become hard title
        # constraints. Resume-inferred roles remain soft evidence.
        if not prefs.roles and candidate_profile.target_roles:
            prefs.roles = candidate_profile.target_roles[:4]
        prefs = apply_candidate_context(prefs, candidate_profile)

        candidate_limit = max(100, min(1000, top_k * 25))
        rejected_ids = self.store.rejected_job_ids(profile_id)
        retrieval_k = (
            max(top_k, self.relevance_verifier.max_candidates)
            if self.relevance_verifier.enabled else top_k
        )
        matches, retrieval = await retrieve_and_rank_jobs(
            self.store,
            prefs,
            query="",
            top_k=retrieval_k,
            rejected_ids=rejected_ids,
            client=self.embeddings,
            candidate_limit=candidate_limit,
        )
        matches, verification = await self.relevance_verifier.verify_matches(
            matches,
            intent="",
            preferences=prefs,
            rejected_ids=rejected_ids,
        )
        retrieval["relevance_verification"] = verification
        retrieval["pre_verification_pool_size"] = retrieval_k
        ver_ms = float((verification.get("latency_ms") or {}).get("total", 0.0))
        retrieval.setdefault("latency_ms", {})["relevance_verification"] = round(ver_ms, 2)
        retrieval["latency_ms"]["total_search"] = round((time.perf_counter() - rec_started) * 1000, 2)
        emb_cost = float(retrieval.get("query_embedding_cost_usd", 0.0) or 0.0)
        ver_cost = float(verification.get("estimated_cost_usd", 0.0) or 0.0)
        retrieval["cost_usd"] = {
            "query_embedding": round(emb_cost, 8),
            "relevance_verification": round(ver_cost, 8),
            "total": round(emb_cost + ver_cost, 8),
        }
        matches = matches[:top_k]
        return AgentResult(
            f"Found {len(matches)} resume-based recommendations.",
            {
                "matches": [m.model_dump(mode="json") for m in matches],
                "profile_used": True,
                "mode": "resume_recommendations",
                "retrieval": retrieval,
            },
        )

    async def record_feedback(self, req: FeedbackRequest) -> AgentResult:
        self.store.record_feedback(req)
        return AgentResult("Feedback recorded.", req.model_dump())

    async def handle(self, profile_id: str, text: str, top_k: int = 10) -> AgentResult:
        top_k = max(1, min(int(top_k), 50))
        data: dict = {"trace": []}
        executed: list[str] = []
        for planned in plan_tools(text, top_k=top_k):
            data["trace"].append({"tool": planned.name, "arguments": planned.arguments})
            executed.append(planned.name)
            if planned.name == "update_preferences":
                result = await self.update_preferences(profile_id, planned.arguments["text"])
                data["preferences"] = result.data
            elif planned.name == "record_feedback":
                result = await self.record_feedback(FeedbackRequest(
                    profile_id=profile_id,
                    job_id=planned.arguments["job_id"],
                    verdict=planned.arguments["verdict"],
                    note=planned.arguments.get("note", ""),
                ))
                data["feedback"] = result.data
            elif planned.name == "search_jobs":
                result = await self.search_jobs(
                    profile_id,
                    planned.arguments.get("query", text),
                    planned.arguments.get("top_k", top_k),
                    record_telemetry=False,
                )
                data.update(result.data)
            else:  # pragma: no cover - planner emits only registered tools
                raise ValueError(f"unknown planned tool: {planned.name}")
        return AgentResult(f"Executed: {', '.join(executed)}.", data)
