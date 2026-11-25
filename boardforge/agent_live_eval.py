from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from boardforge.agent import JobAgent
from boardforge.agent_tools import AgentToolExecutor
from boardforge.intelligent_agent import ToolCallingAgent, VertexGeminiLLMProvider
from boardforge.models import CandidateProfile, Preferences
from boardforge.postgres_store import PostgresStore
from boardforge.relevance import RelevanceVerifier, VertexGeminiRelevanceProvider
from boardforge.semantic import EmbeddingClient


SCENARIO_TEXT = (
    "Use my saved preferences to find backend engineering roles that fit me. "
    "If there are not enough strong backend matches, software engineering roles "
    "in cloud security are acceptable alternatives. Keep my saved preferences "
    "unchanged. Search only the existing index; do not discover or sync sources."
)


def _usage_rows(store: PostgresStore, started_at: datetime) -> list[dict[str, Any]]:
    with store.connect() as connection:
        return connection.execute(
            """SELECT reservation_id,provider,model,status,input_tokens,output_tokens,
                      thinking_tokens,total_tokens,reserved_tokens,
                      reserved_cost_usd,estimated_cost_usd,created_at,completed_at
               FROM llm_usage_reservations WHERE created_at >= %s
               ORDER BY created_at,reservation_id""",
            (started_at,),
        ).fetchall()


def _cache_counts(store: PostgresStore, embedding_model: str) -> dict[str, int]:
    with store.connect() as connection:
        jobs = connection.execute(
            "SELECT COUNT(*) AS count FROM jobs WHERE status='active'"
        ).fetchone()["count"]
        embeddings = connection.execute(
            "SELECT COUNT(*) AS count FROM job_embeddings WHERE model=%s",
            (embedding_model,),
        ).fetchone()["count"]
        relevance = connection.execute(
            "SELECT COUNT(*) AS count FROM relevance_verifications"
        ).fetchone()["count"]
    return {
        "active_jobs": int(jobs), "cached_document_embeddings": int(embeddings),
        "cached_relevance_judgments": int(relevance),
    }


def _atomic_write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _refinement_assessment(search_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for index, event in enumerate(search_events):
        result = event.get("result", {})
        arguments = event.get("arguments", {})
        if previous is None:
            justified = True
            reason = "initial indexed search"
        else:
            prior_result = previous.get("result", {})
            changed = arguments != previous.get("arguments", {})
            insufficient = not bool(prior_result.get("sufficient_results"))
            justified = changed and insufficient
            reason = (
                "materially changed search after insufficient results"
                if justified else "repeated after sufficient results or without material change"
            )
        rows.append({
            "search_number": index + 1, "arguments": arguments,
            "recommended_count": result.get("recommended_count"),
            "sufficient_results": result.get("sufficient_results"),
            "justified": justified, "reason": reason,
        })
        previous = event
    return rows


async def run_live_agent_evaluation(
    *, database_url: str, output_path: Path, profile_id: str,
    max_incremental_cost_usd: float,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError(
            f"refusing to repeat paid scenario because artifact exists: {output_path}"
        )
    if not 0 < max_incremental_cost_usd <= 0.04:
        raise ValueError("incremental cost limit must be greater than zero and at most $0.04")

    os.environ.update({
        "BOARDFORGE_ENABLE_EMBEDDINGS": "1",
        "BOARDFORGE_EMBEDDING_PROVIDER": "vertex",
        "BOARDFORGE_ENABLE_RELEVANCE_VERIFIER": "1",
        "BOARDFORGE_RELEVANCE_PROVIDER": "vertex",
        "BOARDFORGE_ENABLE_INTELLIGENT_AGENT": "1",
        "BOARDFORGE_AGENT_PROVIDER": "vertex",
        "BOARDFORGE_RELEVANCE_MAX_RETRIES": "0",
        "BOARDFORGE_EMBEDDING_MAX_RETRIES": "0",
        "BOARDFORGE_RELEVANCE_MAX_CANDIDATES": "4",
        "BOARDFORGE_RELEVANCE_BATCH_SIZE": "4",
        "BOARDFORGE_RELEVANCE_MAX_CALLS_PER_SEARCH": "1",
        "BOARDFORGE_RELEVANCE_MAX_DESCRIPTION_CHARS": "4000",
        "BOARDFORGE_AGENT_MAX_ITERATIONS": "3",
        "BOARDFORGE_AGENT_MAX_TOOL_CALLS": "5",
        "BOARDFORGE_AGENT_MIN_SUFFICIENT_RESULTS": "1",
    })

    store = PostgresStore(database_url)
    end_to_end_started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    # These exact diagnostics must succeed before the first paid request.
    diagnostic_usage = _usage_rows(store, started_at)
    baseline = store.usage_cost_summary()
    absolute_ceiling = baseline["estimated_cost_usd"] + max_incremental_cost_usd
    os.environ["BOARDFORGE_LLM_CUMULATIVE_COST_LIMIT_USD"] = str(absolute_ceiling)
    os.environ["BOARDFORGE_CUMULATIVE_SPEND_LIMIT_USD"] = str(absolute_ceiling)

    preferences = Preferences(
        roles=["Backend"], countries=["USA"],
        max_required_experience_years=5,
        prefer_terms=["distributed systems", "APIs"],
        notes=[
            "If backend results are insufficient, cloud-security software "
            "engineering roles are acceptable alternatives."
        ],
    )
    profile = CandidateProfile(
        skills=["Python", "FastAPI", "PostgreSQL", "distributed systems"],
        role_signals=["Backend", "Software Engineering"],
        target_roles=["Backend"], domains=["Backend / APIs"],
        experience=["Four years building backend APIs and distributed services"],
        reviewed=True, resume_filename="isolated-agent-evaluation.txt",
    )
    existing_profile = store.load_candidate_profile(profile_id)
    reused_profile = existing_profile is not None
    if existing_profile is None:
        store.save_preferences(profile_id, preferences)
        store.save_candidate_profile(profile_id, profile)
    elif (
        existing_profile != profile
        or store.load_preferences(profile_id) != preferences
    ):
        raise RuntimeError("isolated evaluation profile differs from the frozen scenario state")
    before_state = {
        "preferences": store.load_preferences(profile_id).model_dump(mode="json"),
        "profile": store.load_candidate_profile(profile_id).model_dump(mode="json"),
        "feedback": store.feedback_history(profile_id),
    }

    embeddings = EmbeddingClient(budget_store=store)
    relevance = RelevanceVerifier(
        store, VertexGeminiRelevanceProvider(), enabled=True,
        batch_size=4, max_candidates=4, max_description_chars=4000,
    )
    relevance.max_calls_per_search = 1
    relevance.cumulative_cost_limit_usd = absolute_ceiling
    core = JobAgent(
        store, embeddings=embeddings, relevance_verifier=relevance,
    )
    executor = AgentToolExecutor(
        core, min_sufficient_results=1,
        allow_discovery=False, allow_sync=False,
    )
    provider = VertexGeminiLLMProvider()
    provider.max_retries = 0
    agent = ToolCallingAgent(
        core, provider=provider, executor=executor, enabled=True,
    )
    agent.cumulative_cost_limit_usd = absolute_ceiling

    report: dict[str, Any] = {
        "status": "incomplete", "scenario_version": "live-agent-smoke-v2",
        "labels": "not_a_ranking_holdout_evaluation",
        "profile_id": profile_id, "scenario": SCENARIO_TEXT,
        "authorized_incremental_cost_usd": max_incremental_cost_usd,
        "absolute_ledger_ceiling_usd": absolute_ceiling,
        "budget_before": baseline,
        "provider": {
            "name": provider.provider_name, "model": provider.model,
            "location": provider.location, "max_retries": provider.max_retries,
        },
        "capabilities": {"discovery": False, "source_sync": False},
        "profile_reused": reused_profile,
        "diagnostic_sql_preflight": {
            "status": "passed", "usage_rows_since_start": len(diagnostic_usage),
            "queries": ["usage ledger export", "cache counts", "usage cost summary"],
        },
        "incremental_trace": [],
    }
    try:
        cache_before = _cache_counts(store, embeddings.model)
        report["cache_before"] = cache_before
        _atomic_write(output_path, report)

        async def checkpoint(event: dict[str, Any]) -> None:
            report["incremental_trace"].append({
                **event, "budget_checkpoint": store.usage_cost_summary(),
                "elapsed_ms": round((time.perf_counter() - end_to_end_started) * 1000, 2),
            })
            _atomic_write(output_path, report)

        result = await agent.run(
            profile_id, SCENARIO_TEXT, event_sink=checkpoint,
        )
        # Persist the complete execution before any diagnostic post-processing.
        # A reporting query must never be able to erase the paid-call trace.
        report["execution"] = result
        _atomic_write(output_path, report)
        after_state = {
            "preferences": store.load_preferences(profile_id).model_dump(mode="json"),
            "profile": store.load_candidate_profile(profile_id).model_dump(mode="json"),
            "feedback": store.feedback_history(profile_id),
        }
        budget_after = store.usage_cost_summary()
        incremental_cost = (
            budget_after["estimated_cost_usd"] - baseline["estimated_cost_usd"]
        )
        usage_rows = _usage_rows(store, started_at)
        tools = [item["tool"] for item in result.get("trace", []) if item.get("ok")]
        attempted_tools = [item["tool"] for item in result.get("trace", [])]
        duplicates = len(attempted_tools) - len(set(
            json.dumps({"tool": item["tool"], "arguments": item["arguments"]}, sort_keys=True)
            for item in result.get("trace", [])
        ))
        forbidden = {"discover_job_sources", "synchronize_trusted_source"}
        tool_events = [
            event for event in report["incremental_trace"]
            if event.get("type") == "tool_executed"
        ]
        search_events = [
            event for event in tool_events
            if event.get("tool") == "search_indexed_jobs"
        ]
        refinements = _refinement_assessment(search_events)
        recommendations = [
            item for item in result.get("matches", [])
            if item.get("status") == "recommended"
        ]
        message_grounded = bool(recommendations) and all(
            item["title"] in result["message"] and item["company"] in result["message"]
            for item in recommendations[:3]
        ) and not result["message"].count("http")
        evidence_grounded = bool(recommendations) and all(
            item.get("evidence")
            and all(evidence.get("detail") for evidence in item["evidence"])
            for item in recommendations
        )
        relevance_supported = bool(recommendations) and all(
            item.get("status") == "recommended"
            and any(
                evidence.get("kind") == "llm_relevance"
                for evidence in item.get("evidence", [])
            )
            for item in recommendations
        )
        constraints_satisfied = bool(recommendations) and all(
            item.get("country") == "USA"
            and item.get("min_experience_years") is not None
            and float(item["min_experience_years"]) <= 5
            for item in recommendations
        )
        constraints_preserved = before_state == after_state
        context_events = [
            event for event in report["incremental_trace"]
            if event.get("type") == "context_preloaded"
        ]
        read_then_search = bool(context_events) and "search_indexed_jobs" in tools
        no_forbidden_calls = not any(tool in forbidden for tool in attempted_tools)
        allowed_tools = {"search_indexed_jobs"}
        unnecessary_calls = sum(
            not event.get("result", {}).get("ok", False)
            or event.get("tool") not in allowed_tools
            for event in tool_events
        ) + sum(not row["justified"] for row in refinements[1:])
        trace_complete = (
            len(result.get("events", [])) == len(report["incremental_trace"])
            and bool(report["incremental_trace"])
        )
        complete = all((
            read_then_search, no_forbidden_calls, constraints_preserved,
            bool(recommendations), result["agent"]["sufficient_results"],
            result["agent"]["stopped_reason"] == "sufficient_results",
            message_grounded, evidence_grounded, relevance_supported,
            constraints_satisfied, duplicates == 0, unnecessary_calls == 0,
            trace_complete,
            incremental_cost <= max_incremental_cost_usd + 1e-12,
        ))
        report.update({
            "status": "complete" if complete else "incomplete",
            "assessment": {
                "read_preferences_before_search": read_then_search,
                "constraints_preserved": constraints_preserved,
                "recommendation_constraints_satisfied": constraints_satisfied,
                "tool_selection_correct": (
                    read_then_search and no_forbidden_calls
                    and all(tool in allowed_tools for tool in attempted_tools)
                ),
                "unnecessary_calls": unnecessary_calls,
                "duplicate_calls": duplicates,
                "search_refinements": refinements,
                "stopping_behavior_correct": (
                    result["agent"]["stopped_reason"] == "sufficient_results"
                ),
                "grounded_final_recommendations": message_grounded,
                "evidence_grounding_valid": evidence_grounded,
                "final_recommendation_relevance_supported": relevance_supported,
                "recommendation_count": len(recommendations),
                "state_mutations_during_agent_turn": []
                if constraints_preserved else ["isolated_user_state_changed"],
                "native_function_calling_compatible": bool(
                    result["agent"].get("provider_trace")
                    and any(row.get("function_calls") for row in result["agent"]["provider_trace"])
                ),
                "incremental_trace_complete": trace_complete,
            },
            "state_before": before_state, "state_after": after_state,
            "cache_before": cache_before,
            "cache_after": _cache_counts(store, embeddings.model),
            "usage_ledger_rows": usage_rows,
            "budget_after": budget_after,
            "incremental_cost_usd": round(incremental_cost, 8),
            "incremental_tokens": budget_after["total_tokens"] - baseline["total_tokens"],
            "latency_ms": {
                "end_to_end": round((time.perf_counter() - end_to_end_started) * 1000, 2),
                "agent": result["agent"].get("latency_ms", {}),
                "model_calls": [
                    row.get("latency_ms")
                    for row in result["agent"].get("provider_trace", [])
                ],
                "tool_calls": [row.get("duration_ms") for row in result.get("trace", [])],
                "search_stages": [
                    event.get("result", {}).get("retrieval", {}).get("latency_ms", {})
                    for event in search_events
                ],
                "relevance_stages": [
                    event.get("result", {}).get("retrieval", {})
                    .get("verification", {}).get("latency_ms", {})
                    for event in search_events
                ],
            },
        })
    except Exception as exc:
        budget_after = store.usage_cost_summary()
        report.update({
            "status": "incomplete", "error": f"{type(exc).__name__}: {exc}",
            "budget_after": budget_after,
            "incremental_cost_usd": round(
                budget_after["estimated_cost_usd"] - baseline["estimated_cost_usd"], 8
            ),
            "usage_ledger_rows": _usage_rows(store, started_at),
        })
    finally:
        store.close()
        _atomic_write(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-id", default="agent-live-eval-20261125")
    parser.add_argument("--max-incremental-cost-usd", type=float, default=0.04)
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing live evaluation without --execute")
    report = asyncio.run(run_live_agent_evaluation(
        database_url=args.database_url, output_path=args.output,
        profile_id=args.profile_id,
        max_incremental_cost_usd=args.max_incremental_cost_usd,
    ))
    print(json.dumps({
        "status": report["status"], "artifact": str(args.output),
        "incremental_cost_usd": report.get("incremental_cost_usd"),
        "error": report.get("error"),
    }, indent=2))


if __name__ == "__main__":
    main()
