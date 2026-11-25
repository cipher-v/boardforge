from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from boardforge.holdout_eval import _aggregate_rows, _load_frozen, _row_metrics
from boardforge.models import Match, Preferences
from boardforge.preferences import effective_search_preferences
from boardforge.relevance import (
    PROMPT_VERSION,
    RelevanceVerifier,
    _eligibility_restrictions,
    verification_cache_key,
)


def plan_relevance_verifier_eval(
    corpus_path: str | Path,
    labels_path: str | Path,
    deterministic_report_path: str | Path,
    *,
    verifier: RelevanceVerifier | None = None,
    batch_size: int = 8,
    max_candidates: int = 10,
    max_description_chars: int = 6000,
    assumed_call_latency_ms: int = 1500,
) -> dict[str, Any]:
    _, labels, jobs = _load_frozen(corpus_path, labels_path)
    report = json.loads(Path(deterministic_report_path).read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in report["variants"]["hybrid"]["per_query"]}
    jobs_by_id = {job.stable_id: job for job in jobs}
    model = verifier.provider.model if verifier else "configured-model"
    calls = candidates = cache_hits = input_tokens = output_tokens = 0
    calls_by_query: dict[str, int] = {}
    candidates_by_query: dict[str, int] = {}

    for case in labels["cases"]:
        row = rows[case["name"]]
        prefs = effective_search_preferences(
            Preferences.model_validate(case.get("saved_preferences", {})), case["query"]
        )
        shortlisted = []
        for index, (stable_id, status) in enumerate(zip(row["predicted"], row["statuses"])):
            if status not in {"recommended", "needs_verification"}:
                continue
            candidate = Match(
                job=jobs_by_id[stable_id], score=float(len(row["predicted"]) - index),
                status=status, reasons=[], risks=[], hard_violations=[],
            )
            if not _eligibility_restrictions(candidate, prefs, set()):
                shortlisted.append(stable_id)
            if len(shortlisted) >= max_candidates:
                break
        missing = []
        for stable_id in shortlisted:
            job = jobs_by_id[stable_id]
            if verifier is not None:
                key = verification_cache_key(job, case["query"], prefs, model)
                if verifier.store.load_relevance_verification(key) is not None:
                    cache_hits += 1
                    continue
            missing.append(Match(
                job=job, score=0.0, status="needs_verification",
                reasons=[], risks=[], hard_violations=[],
            ))
        if verifier is not None:
            batches = verifier._token_aware_batches(missing, case["query"], prefs)
            query_calls = len(batches)
            for _, payload, reserved, _ in batches:
                output_cap = verifier._output_tokens_for_payload(payload)
                input_tokens += max(0, reserved - output_cap)
                output_tokens += output_cap
        else:
            query_calls = math.ceil(len(missing) / batch_size) if missing else 0
            for match in missing:
                input_tokens += math.ceil(
                    (len(match.job.title) + len(match.job.description[:max_description_chars])) / 4
                )
                output_tokens += 120
            input_tokens += query_calls * 300
        calls += query_calls
        candidates += len(missing)
        calls_by_query[case["name"]] = query_calls
        candidates_by_query[case["name"]] = len(missing)

    query_call_counts = list(calls_by_query.values())
    sorted_calls = sorted(query_call_counts)
    p95_index = max(0, math.ceil(0.95 * len(sorted_calls)) - 1)
    input_price = float(os.getenv("BOARDFORGE_RELEVANCE_INPUT_USD_PER_MILLION", "0.75"))
    output_price = float(os.getenv("BOARDFORGE_RELEVANCE_OUTPUT_USD_PER_MILLION", "3.75"))
    thinking_budget = int(os.getenv("BOARDFORGE_RELEVANCE_THINKING_BUDGET", "128"))
    max_thinking_tokens = calls * max(0, thinking_budget)
    estimated_cost = (
        input_tokens * input_price
        + (output_tokens + max_thinking_tokens) * output_price
    ) / 1_000_000
    return {
        "status": "plan_only_no_llm_calls_executed",
        "labels_verification_status": labels["verification_status"],
        "candidate_source": "saved hybrid per-query candidate sets",
        "candidate_sets_identical": True,
        "queries": len(labels["cases"]),
        "shortlisted_candidate_judgments": candidates,
        "cache_hits": cache_hits,
        "proposed_model_calls": calls,
        "batching_plan": "current_token_aware_implementation" if verifier else "legacy_estimate",
        "batch_size": batch_size,
        "max_candidates_per_query": max_candidates,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": input_tokens + output_tokens,
        "estimated_max_thinking_tokens": max_thinking_tokens,
        "estimated_cost_usd_including_thinking_budget": round(estimated_cost, 6),
        "pricing_usd_per_million_tokens": {
            "input": input_price,
            "output_and_thinking": output_price,
        },
        "estimated_sequential_eval_latency_ms": calls * assumed_call_latency_ms,
        "estimated_added_production_latency_ms_p50": round(
            statistics.median(query_call_counts) * assumed_call_latency_ms
        ),
        "estimated_added_production_latency_ms_p95": (
            sorted_calls[p95_index] * assumed_call_latency_ms
        ),
        "latency_assumption": (
            f"{assumed_call_latency_ms} ms per provider call; actual latency depends on model, "
            "queueing, prompt length, and cache hit rate"
        ),
        "calls_by_query": calls_by_query,
        "candidates_by_query": candidates_by_query,
    }


async def run_relevance_verifier_eval(
    corpus_path: str | Path,
    labels_path: str | Path,
    deterministic_report_path: str | Path,
    verifier: RelevanceVerifier,
    *,
    checkpoint_path: str | Path | None = None,
    max_estimated_cost_usd: float = 0.60,
) -> dict[str, Any]:
    _, labels, jobs = _load_frozen(corpus_path, labels_path)
    deterministic = json.loads(
        Path(deterministic_report_path).read_text(encoding="utf-8")
    )
    source_rows = {
        row["name"]: row
        for row in deterministic["variants"]["hybrid"]["per_query"]
    }
    jobs_by_id = {job.stable_id: job for job in jobs}
    checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
    checkpoint = {}
    if checkpoint_file and checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if checkpoint.get("prompt_version") != PROMPT_VERSION:
            raise RuntimeError("evaluation checkpoint prompt version does not match")
    verified_rows = list(checkpoint.get("verified_rows", []))
    verification_meta = list(checkpoint.get("verification_meta", []))
    completed = {row["name"] for row in verified_rows}

    def save_checkpoint(status: str) -> None:
        if not checkpoint_file:
            return
        payload = {
            "status": status,
            "prompt_version": PROMPT_VERSION,
            "labels_verification_status": labels["verification_status"],
            "verified_rows": verified_rows,
            "verification_meta": verification_meta,
            "completed_queries": sorted(completed),
            "updated_at_unix": time.time(),
        }
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_file.with_suffix(checkpoint_file.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(checkpoint_file)

    paused_reason = None
    for case in labels["cases"]:
        if case["name"] in completed:
            continue
        source = source_rows[case["name"]]
        matches = [
            Match(
                job=jobs_by_id[stable_id], score=float(len(source["predicted"]) - index),
                status=source["statuses"][index], reasons=[], risks=[], hard_violations=[],
            )
            for index, stable_id in enumerate(source["predicted"])
        ]
        prefs = effective_search_preferences(
            Preferences.model_validate(case.get("saved_preferences", {})), case["query"]
        )
        preverification_ids = [match.job.stable_id for match in matches]
        source_status = {match.job.stable_id: match.status for match in matches}
        verified, meta = await verifier.verify_matches(
            matches, intent=case["query"], preferences=prefs, rejected_ids=set()
        )
        spent = sum(
            item.get("estimated_cost_usd", 0.0) for item in verification_meta
        ) + meta.get("estimated_cost_usd", 0.0)
        if meta.get("quota_denials") or meta.get("budget_skipped_candidates"):
            paused_reason = (
                meta.get("quota_denials", [{}])[0].get("reason")
                if meta.get("quota_denials") else "per_search_budget"
            )
            save_checkpoint("paused_budget_or_quota")
            break
        if spent > max_estimated_cost_usd:
            raise RuntimeError("provider exceeded the evaluation cost ceiling")
        if set(preverification_ids) != set(source["predicted"]):
            raise RuntimeError("pre-verification candidate set changed")
        candidates = [jobs_by_id[stable_id] for stable_id in source["candidate_ids"]]
        row = _row_metrics(case, verified, candidates, prefs, top_k=10)
        relevant_ids = {
            stable_id for stable_id, label in case["labels"].items()
            if label["label"] in {"direct", "adjacent"}
        }
        final_recommended = {
            match.job.stable_id for match in verified if match.status == "recommended"
        }
        recoverable = {
            stable_id for stable_id in preverification_ids
            if source_status[stable_id] == "needs_verification" and stable_id in relevant_ids
        }
        recovered = recoverable & final_recommended
        row.update({
            "name": case["name"], "family": case["family"], "query": case["query"],
            "latency_ms": 0.0,
            "pre_verification_candidate_ids": preverification_ids,
            "pre_verification_candidate_set_preserved": True,
            "recoverable_relevant_count": len(recoverable),
            "recovered_relevant_recommendations": sorted(recovered),
            "recovered_relevant_count": len(recovered),
            "recovery_rate": len(recovered) / len(recoverable) if recoverable else None,
            "alternatives": [
                match.job.stable_id for match in verified if match.status == "alternative"
            ],
        })
        verified_rows.append(row)
        verification_meta.append({"name": case["name"], **meta})
        completed.add(case["name"])
        save_checkpoint("running")

    relevant = [row for row in verified_rows if row["has_relevant_jobs"]]
    no_relevant = [row for row in verified_rows if not row["has_relevant_jobs"]]
    recoverable_total = sum(row["recoverable_relevant_count"] for row in verified_rows)
    recovered_total = sum(row["recovered_relevant_count"] for row in verified_rows)
    verified_relevant_aggregate = _aggregate_rows(relevant, [])
    deterministic_relevant = deterministic["variants"]["hybrid"]["relevant_queries"]
    usage = {
        "provider_calls": sum(item.get("provider_calls", 0) for item in verification_meta),
        "provider_http_attempts": sum(
            item.get("provider_http_attempts", 0) for item in verification_meta
        ),
        "input_tokens": sum(
            item.get("token_usage", {}).get("input_tokens", 0)
            for item in verification_meta
        ),
        "output_tokens": sum(
            item.get("token_usage", {}).get("output_tokens", 0)
            for item in verification_meta
        ),
        "thinking_tokens": sum(
            item.get("token_usage", {}).get("thinking_tokens", 0)
            for item in verification_meta
        ),
        "estimated_cost_usd": round(sum(
            item.get("estimated_cost_usd", 0.0) for item in verification_meta
        ), 8),
    }
    usage["additional_retry_attempts"] = max(
        0, usage["provider_http_attempts"] - usage["provider_calls"]
    )
    result = {
        "status": "paused_budget_or_quota" if paused_reason else "ok",
        "paused_reason": paused_reason,
        "completed_queries": len(completed),
        "remaining_queries": len(labels["cases"]) - len(completed),
        "labels_verification_status": labels["verification_status"],
        "candidate_sets_identical": True,
        "deterministic_hybrid": {
            "relevant_queries": deterministic["variants"]["hybrid"]["relevant_queries"],
            "no_relevant_queries": deterministic["variants"]["hybrid"]["no_relevant_queries"],
        },
        "llm_verified": {
            "relevant_queries": verified_relevant_aggregate,
            "no_relevant_queries": _aggregate_rows(no_relevant, []),
            "per_query": verified_rows,
        },
        "impact": {
            "recommended_precision_delta": round(
                verified_relevant_aggregate["recommended_precision"]
                - deterministic_relevant["recommended_precision"], 4
            ),
            "recommended_recall_delta": round(
                verified_relevant_aggregate["recommended_recall"]
                - deterministic_relevant["recommended_recall"], 4
            ),
            "recoverable_relevant_candidates": recoverable_total,
            "recovered_relevant_recommendations": recovered_total,
            "recovery_rate": (
                round(recovered_total / recoverable_total, 4)
                if recoverable_total else None
            ),
        },
        "provider_usage": usage,
        "verification": verification_meta,
    }
    save_checkpoint(result["status"])
    return result
