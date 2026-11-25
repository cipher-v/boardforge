from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from boardforge.agent import JobAgent
from boardforge.discovery.service import queue_discovery
from boardforge.models import CandidateProfile, FeedbackRequest
from boardforge.sync import sync_url


class SearchIndexedJobsArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=20)
    mandatory_constraints: list[str] = Field(default_factory=list, max_length=10)
    soft_preferences: list[str] = Field(default_factory=list, max_length=10)
    conditional_alternatives: list[str] = Field(default_factory=list, max_length=10)


class DiscoverJobSourcesArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=240)


class SyncTrustedSourceArgs(BaseModel):
    source_kind: str = Field(min_length=1, max_length=40)
    source_key: str = Field(min_length=1, max_length=160)


class ReadUserContextArgs(BaseModel):
    include_feedback: bool = True


class UpdatePreferencesArgs(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    authorization_quote: str = Field(min_length=1, max_length=240)


class RetrieveFeedbackArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=50)


class RecordFeedbackArgs(BaseModel):
    job_id: str = Field(min_length=3, max_length=300)
    verdict: Literal["save", "reject", "applied", "interview", "ignore"]
    note: str = Field(default="", max_length=500)


TOOL_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "search_indexed_jobs": SearchIndexedJobsArgs,
    "discover_job_sources": DiscoverJobSourcesArgs,
    "synchronize_trusted_source": SyncTrustedSourceArgs,
    "read_user_context": ReadUserContextArgs,
    "update_preferences": UpdatePreferencesArgs,
    "retrieve_search_feedback": RetrieveFeedbackArgs,
    "record_job_feedback": RecordFeedbackArgs,
}


TOOL_DESCRIPTIONS = {
    "search_indexed_jobs": (
        "Search BoardForge's existing indexed jobs first. Separate mandatory constraints, "
        "soft preferences, and conditional alternatives."
    ),
    "discover_job_sources": (
        "Queue targeted official-ATS discovery only after an indexed search demonstrates "
        "that too few suitable recommendations exist."
    ),
    "synchronize_trusted_source": (
        "Synchronize one already-known trusted ATS source. Never accepts arbitrary URLs."
    ),
    "read_user_context": "Read the candidate profile and durable preferences.",
    "update_preferences": (
        "Persist an explicit user-authorized preference change. The authorization quote "
        "must be copied exactly from the current user message."
    ),
    "retrieve_search_feedback": "Read prior save/reject/application feedback.",
    "record_job_feedback": "Record explicit feedback for a stable job ID.",
}


def native_tool_declarations(*, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    declarations = []
    for name, model in TOOL_ARGUMENT_MODELS.items():
        if name in excluded:
            continue
        schema = model.model_json_schema()
        schema.pop("title", None)
        declarations.append({
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parameters": schema,
        })
    return [{"function_declarations": declarations}]


@dataclass
class AgentExecutionState:
    profile_id: str
    user_text: str
    tool_calls: int = 0
    searches: int = 0
    recommended_count: int = 0
    last_match_count: int = 0
    discovery_requested: bool = False
    sufficient_results: bool = False
    stopped_reason: str = ""
    seen_calls: set[str] = field(default_factory=set)
    trace: list[dict[str, Any]] = field(default_factory=list)
    last_search: dict[str, Any] | None = None
    best_search: dict[str, Any] | None = None


Syncer = Callable[[str, Any], Awaitable[Any]]


class AgentToolExecutor:
    """Typed, deterministic authorization boundary around BoardForge capabilities."""

    def __init__(
        self,
        core: JobAgent,
        *,
        min_sufficient_results: int = 3,
        syncer: Syncer = sync_url,
        discovery_client: Any | None = None,
        allow_discovery: bool = True,
        allow_sync: bool = True,
    ) -> None:
        self.core = core
        self.store = core.store
        self.min_sufficient_results = max(1, min(10, min_sufficient_results))
        self.syncer = syncer
        self.discovery_client = discovery_client
        self.allow_discovery = allow_discovery
        self.allow_sync = allow_sync

    @staticmethod
    def _call_key(name: str, arguments: BaseModel) -> str:
        return f"{name}:{json.dumps(arguments.model_dump(mode='json'), sort_keys=True)}"

    @staticmethod
    def _explicit_preference_authorization(user_text: str, quote: str) -> bool:
        if quote not in user_text:
            return False
        return bool(re.search(
            r"\b(?:only|must|do not|don't|dont|avoid|exclude|remove|from now on|"
            r"changed my mind|prefer|want|minimum|remote|onsite|hybrid|worldwide)\b",
            quote, re.I,
        ))

    @staticmethod
    def _should_replace_best_search(
        current: dict[str, Any] | None, candidate: dict[str, Any],
    ) -> bool:
        if current is None:
            return True
        current_recommended = int(current.get("recommended_count", 0))
        candidate_recommended = int(candidate.get("recommended_count", 0))
        if candidate_recommended != current_recommended:
            return candidate_recommended > current_recommended
        outcome_rank = {
            "verification_provider_failure": 0,
            "verification_incomplete": 1,
            "no_verified_recommendations": 2,
            "verified_recommendations": 3,
        }
        current_rank = outcome_rank.get(current.get("result_state"), 0)
        candidate_rank = outcome_rank.get(candidate.get("result_state"), 0)
        if candidate_rank != current_rank:
            if candidate_rank > current_rank:
                return True
            # Partial candidates are still more useful than an empty earlier
            # result, even when the later verification outcome is incomplete.
            return not current.get("matches") and bool(candidate.get("matches"))
        if candidate.get("matches"):
            return True
        return not current.get("matches")

    @staticmethod
    def _bounded_match(match: dict[str, Any]) -> dict[str, Any]:
        job = match.get("job", {})
        return {
            "stable_id": ":".join(str(job.get(key, "")) for key in (
                "source_kind", "source_key", "external_id"
            )),
            "title": str(job.get("title", ""))[:180],
            "company": str(job.get("company", ""))[:120],
            "location": str(job.get("location", ""))[:160],
            "country": str(job.get("country", ""))[:80],
            "remote_policy": str(job.get("remote_policy", "unknown"))[:20],
            "min_experience_years": job.get("min_experience_years"),
            "apply_url": job.get("apply_url"),
            "salary_text": job.get("salary_text"),
            "last_seen_at": job.get("last_seen_at"),
            "source_kind": job.get("source_kind"),
            "status": match.get("status"),
            "score": match.get("score"),
            "reasons": [str(value)[:300] for value in match.get("reasons", [])[:4]],
            "risks": [str(value)[:300] for value in match.get("risks", [])[:4]],
            "evidence": [
                {
                    "kind": item.get("kind"),
                    "detail": str(item.get("detail", ""))[:500],
                }
                for item in match.get("evidence", [])[:4]
            ],
            "untrusted_source_data": True,
        }

    async def execute(
        self, state: AgentExecutionState, name: str, raw_arguments: dict[str, Any],
        *, deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        model = TOOL_ARGUMENT_MODELS.get(name)
        if model is None:
            return {"ok": False, "error": "unknown_tool", "tool": name}
        try:
            arguments = model.model_validate(raw_arguments)
        except ValidationError as exc:
            return {
                "ok": False, "error": "invalid_arguments",
                "details": exc.errors(include_url=False)[:3],
            }
        key = self._call_key(name, arguments)
        if key in state.seen_calls:
            return {"ok": False, "error": "duplicate_tool_call"}
        state.seen_calls.add(key)

        if name == "read_user_context":
            profile = self.store.load_candidate_profile(state.profile_id)
            preferences = self.store.load_preferences(state.profile_id)
            result = {
                "profile": profile.model_dump(mode="json") if profile else None,
                "preferences": preferences.model_dump(mode="json"),
            }
            if arguments.include_feedback:
                result["feedback"] = self.store.feedback_history(state.profile_id, limit=20)
        elif name == "retrieve_search_feedback":
            result = {"feedback": self.store.feedback_history(
                state.profile_id, limit=arguments.limit
            )}
        elif name == "update_preferences":
            if not self._explicit_preference_authorization(
                state.user_text, arguments.authorization_quote
            ):
                return {"ok": False, "error": "preference_update_not_authorized"}
            updated = await self.core.update_preferences(state.profile_id, arguments.text)
            result = {"preferences": updated.data}
        elif name == "record_job_feedback":
            if arguments.job_id not in state.user_text or arguments.verdict not in state.user_text.casefold():
                return {"ok": False, "error": "feedback_not_explicitly_authorized"}
            recorded = await self.core.record_feedback(FeedbackRequest(
                profile_id=state.profile_id, **arguments.model_dump()
            ))
            result = {"feedback": recorded.data}
        elif name == "search_indexed_jobs":
            searched = await self.core.search_jobs(
                state.profile_id, arguments.query, arguments.top_k,
                deadline_monotonic=deadline_monotonic,
            )
            matches = searched.data.get("matches", [])
            bounded = [self._bounded_match(match) for match in matches[:10]]
            state.searches += 1
            state.last_match_count = len(bounded)
            state.recommended_count = sum(
                item["status"] == "recommended" for item in bounded
            )
            state.sufficient_results = (
                state.recommended_count >= self.min_sufficient_results
            )
            result = {
                "matches": bounded,
                "match_count": len(bounded),
                "recommended_count": state.recommended_count,
                "sufficient_results": state.sufficient_results,
                "intent": {
                    "mandatory_constraints": arguments.mandatory_constraints,
                    "soft_preferences": arguments.soft_preferences,
                    "conditional_alternatives": arguments.conditional_alternatives,
                },
                "retrieval": self._bounded_retrieval_summary(
                    searched.data.get("retrieval", {})
                ),
            }
            verification = searched.data.get(
                "retrieval", {}
            ).get("verification", {})

            if state.recommended_count:
                result["result_state"] = "verified_recommendations"

            elif verification.get("provider_failures"):
                result["result_state"] = "verification_provider_failure"

            elif any((
                verification.get("candidate_validation_failures"),
                verification.get("budget_skipped_candidates"),
                verification.get("deadline_skipped_candidates"),
                verification.get("quota_denials"),
            )):
                result["result_state"] = "verification_incomplete"

            else:
                result["result_state"] = "no_verified_recommendations"
            state.last_search = result
            if self._should_replace_best_search(state.best_search, result):
                state.best_search = result
        elif name == "discover_job_sources":
            if not self.allow_discovery:
                return {"ok": False, "error": "discovery_disabled_for_evaluation"}
            if state.searches == 0:
                return {"ok": False, "error": "search_index_required_before_discovery"}
            if state.sufficient_results:
                return {"ok": False, "error": "discovery_unnecessary_sufficient_results"}
            profile = self.store.load_candidate_profile(state.profile_id) or CandidateProfile()
            preferences = self.store.load_preferences(state.profile_id)
            queued = queue_discovery(
                self.store, state.profile_id, profile, preferences,
                query=arguments.query, client=self.discovery_client,
            )
            state.discovery_requested = queued.get("status") in {"queued", "running"}
            result = {"discovery": queued}
        else:  # synchronize_trusted_source
            if not self.allow_sync:
                return {"ok": False, "error": "source_sync_disabled_for_evaluation"}
            trusted = next((source for source in self.store.discovered_sources()
                if source["source_kind"] == arguments.source_kind
                and source["source_key"] == arguments.source_key), None)
            if trusted is None:
                return {"ok": False, "error": "source_not_in_trusted_registry"}
            summary = await self.syncer(trusted["board_url"], self.store)
            result = {
                "source": {"kind": arguments.source_kind, "key": arguments.source_key},
                "summary": summary.model_dump(mode="json")
                if hasattr(summary, "model_dump") else summary,
            }

        state.tool_calls += 1
        return {
            "ok": True,
            **result,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @staticmethod
    def _bounded_retrieval_summary(retrieval: dict[str, Any]) -> dict[str, Any]:
        verification = retrieval.get("relevance_verification") or {}
        return {
            "mode": retrieval.get("mode"),
            "candidate_count": retrieval.get("candidate_count"),
            "query_embedding_used": retrieval.get("query_embedding_used"),
            "latency_ms": retrieval.get("latency_ms", {}),
            "verification": {
                key: verification.get(key)
                for key in (
                    "enabled", "shortlisted", "verified", "upgraded",
                    "alternatives", "downgraded", "uncertain", "cache_hits",
                    "provider_calls", "estimated_cost_usd",
                    "latency_ms", "token_usage", "provider_failures",
                    "deadline_skipped_candidates",
                )
                if key in verification
            },
        }
