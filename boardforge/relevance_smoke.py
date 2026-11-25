from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from boardforge.holdout_eval import _load_frozen
from boardforge.models import Match, Preferences
from boardforge.preferences import effective_search_preferences
from boardforge.relevance import RelevanceVerifier, create_relevance_provider
from boardforge.store import Store


async def run_once(
    *,
    corpus_path: str = "boardforge/data/holdout_corpus_v1.json",
    labels_path: str = "boardforge/data/holdout_labels_v1_1_ai_audited.json",
    deterministic_report_path: str = "results/holdout-eval-v1_1-ai-audited.json",
    cache_path: str | None = None,
    output_path: str | None = None,
    case_name: str = "backend_general",
    store=None,
) -> dict:
    """Run one guarded live batch; existing output/cache prevents repetition."""
    provider = create_relevance_provider()
    provider_name = os.getenv("BOARDFORGE_RELEVANCE_PROVIDER", "vertex").casefold()
    slug = "groq" if provider_name == "groq" else "vertex"
    run_id = os.getenv("BOARDFORGE_SMOKE_RUN_ID", "").strip()
    artifact_slug = f"{slug}-{run_id}" if run_id else slug
    cache = Path(cache_path or f"results/relevance-smoke-{artifact_slug}.db")
    output = Path(output_path or f"results/relevance-smoke-{artifact_slug}.json")
    if output.exists() or (store is None and cache.exists()):
        raise RuntimeError("smoke artifacts already exist; refusing a repeated paid request")

    _, labels, jobs = _load_frozen(corpus_path, labels_path)
    jobs_by_id = {job.stable_id: job for job in jobs}
    report = json.loads(Path(deterministic_report_path).read_text(encoding="utf-8"))
    source = next(
        row for row in report["variants"]["hybrid"]["per_query"]
        if row["name"] == case_name
    )
    case = next(case for case in labels["cases"] if case["name"] == case_name)
    selected_ids = source["predicted"][:8]
    matches = [
        Match(
            job=jobs_by_id[stable_id],
            score=float(len(selected_ids) - index),
            status=source["statuses"][index],
            reasons=[], risks=[], hard_violations=[],
        )
        for index, stable_id in enumerate(selected_ids)
    ]
    prefs = effective_search_preferences(
        Preferences.model_validate(case.get("saved_preferences", {})), case["query"]
    )

    if hasattr(provider, "backend"):
        provider.backend.max_retries = 0
    if hasattr(provider, "max_retries"):
        provider.max_retries = 0
    verifier = RelevanceVerifier(
        store or Store(str(cache)), provider, enabled=True,
        batch_size=4 if slug == "groq" else 8, max_candidates=8,
    )
    verifier.max_calls_per_search = 8
    verified, metadata = await verifier.verify_matches(
        matches, intent=case["query"], preferences=prefs, rejected_ids=set()
    )
    result = {
        "status": "ok" if not metadata["provider_failures"] else "provider_failure",
        "candidate_batch_authorized": 8,
        "provider_calls": metadata["provider_calls"],
        "provider_http_attempts": metadata["provider_http_attempts"],
        "additional_retry_attempts": metadata["additional_retry_attempts"],
        "provider": slug,
        "model": provider.model,
        "location": provider.location,
        "case": case_name,
        "intent": case["query"],
        "candidate_count": len(selected_ids),
        "structured_output_compliant": (
            not metadata["provider_failures"]
            and len(metadata["judgments"]) == len(selected_ids)
        ),
        "judgments": metadata["judgments"],
        "final_results": [
            {
                "candidate_id": match.job.stable_id,
                "title": match.job.title,
                "status": match.status,
            }
            for match in verified
        ],
        "excluded_ids": metadata["excluded_ids"],
        "token_usage": metadata["token_usage"],
        "provider_latencies_ms": metadata["provider_latencies_ms"],
        "provider_response_metadata": metadata.get("provider_response_metadata", []),
        "estimated_cost_usd": metadata["estimated_cost_usd"],
        "provider_failures": metadata["provider_failures"],
        "latency_ms": metadata.get("latency_ms", {}),
        "batch_plan": metadata["batch_plan"],
        "quota_denials": metadata["quota_denials"],
        "rate_limit_waits_seconds": metadata["rate_limit_waits_seconds"],
        "labels_verification_status": labels["verification_status"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "boardforge-ai-2026")
    print(json.dumps(asyncio.run(run_once()), indent=2, ensure_ascii=False))
