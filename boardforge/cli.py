from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from boardforge.agent import JobAgent
from boardforge.database import create_store
from boardforge.agent_eval import run_deterministic_tool_eval, run_llm_tool_eval
from boardforge.benchmark import run_benchmark
from boardforge.comparison import comparison_markdown, run_comparison, run_embedding_comparison
from boardforge.preference_eval import run_preference_eval
from boardforge.resume_eval import run_resume_eval
from boardforge.real_eval import build_corpus_snapshot, run_real_eval
from boardforge.semantic import EmbeddingClient, warm_job_embeddings
from boardforge.workers import sync_due_sources
from boardforge.discovery.har import discover_from_har
from boardforge.discovery import run_pending_discoveries
from boardforge.discovery.manifest import ConnectorManifest
from boardforge.connectors.manifest import fetch_manifest
from boardforge.store import Store
from boardforge.sync import sync_url
from boardforge.holdout import collect_holdout
from boardforge.holdout_labels import label_holdout
from boardforge.holdout_eval import (
    holdout_comparison_markdown,
    plan_vertex_usage,
    run_production_holdout_eval,
)
from boardforge.relevance import RelevanceVerifier, create_relevance_provider
from boardforge.relevance_eval import (
    plan_relevance_verifier_eval,
    run_relevance_verifier_eval,
)


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(prog="boardforge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync")
    p_sync.add_argument("url")

    p_pref = sub.add_parser("prefs")
    p_pref.add_argument("text")
    p_pref.add_argument("--profile", default="default")

    p_search = sub.add_parser("search")
    p_search.add_argument("query", nargs="?", default="")
    p_search.add_argument("--profile", default="default")
    p_search.add_argument("--top-k", type=int, default=10)

    p_learn = sub.add_parser("learn-har")
    p_learn.add_argument("har_file")
    p_learn.add_argument("--out", default="connector-manifests.json")

    p_manifest = sub.add_parser("sync-manifest")
    p_manifest.add_argument("manifest_file")
    p_manifest.add_argument("--index", type=int, default=0)

    p_benchmark = sub.add_parser("benchmark")
    p_benchmark.add_argument("--dataset", default=None)
    p_benchmark.add_argument("--top-k", type=int, default=5)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--dataset", default=None)
    p_compare.add_argument("--top-k", type=int, default=5)
    p_compare.add_argument("--markdown-out", default=None)

    p_embed = sub.add_parser("compare-embeddings")
    p_embed.add_argument("--dataset", default=None)
    p_embed.add_argument("--top-k", type=int, default=5)

    p_agent_eval = sub.add_parser("agent-eval")
    p_agent_eval.add_argument("--dataset", default=None)

    p_agent_llm = sub.add_parser("agent-eval-llm")
    p_agent_llm.add_argument("--dataset", default=None)

    p_preference_eval = sub.add_parser("preference-eval")
    p_preference_eval.add_argument("--dataset", default=None)

    p_resume_eval = sub.add_parser("resume-eval")
    p_resume_eval.add_argument("--dataset", default=None)

    p_discovery_worker = sub.add_parser("discovery-worker")
    p_discovery_worker.add_argument("--limit", type=int, default=10)

    p_embed_index = sub.add_parser("embed-index")
    p_embed_index.add_argument("--batch-size", type=int, default=64)
    p_embed_index.add_argument("--max-jobs", type=int, default=10000)
    p_embed_index.add_argument(
        "--country",
        action="append",
        default=[],
        help="Only backfill active jobs in this country; repeat for multiple countries.",
    )
    p_embed_index.add_argument(
        "--allow-sqlite",
        action="store_true",
        help="Allow embedding backfill against the SQLite fallback store.",
    )

    p_sync_sources = sub.add_parser("sync-sources")
    p_sync_sources.add_argument("--interval-seconds", type=int, default=3600)
    p_sync_sources.add_argument("--limit", type=int, default=25)
    p_sync_sources.add_argument("--concurrency", type=int, default=4)

    p_reenrich = sub.add_parser("reenrich-index")
    p_reenrich.add_argument(
        "--apply",
        action="store_true",
        help="Persist recomputed metadata in place; defaults to read-only dry-run.",
    )

    p_snapshot = sub.add_parser("snapshot-eval-corpus")
    p_snapshot.add_argument("--country", default="India")
    p_snapshot.add_argument("--out", default="results/real-corpus.json")
    p_snapshot.add_argument(
        "--include-test",
        action="store_true",
        help="Include source_kind=test rows. Excluded by default.",
    )

    p_real_eval = sub.add_parser("real-eval")
    p_real_eval.add_argument(
        "--corpus",
        default="results/india-corpus.json",
        help="Frozen corpus created by snapshot-eval-corpus.",
    )
    p_real_eval.add_argument("--dataset", default=None)
    p_real_eval.add_argument("--top-k", type=int, default=5)
    p_real_eval.add_argument(
        "--details",
        action="store_true",
        help="Print all per-query rows instead of a compact summary.",
    )

    p_holdout_collect = sub.add_parser("collect-holdout")
    p_holdout_collect.add_argument("--sources", default="boardforge/data/holdout_sources_v1.json")
    p_holdout_collect.add_argument("--out", default="boardforge/data/holdout_corpus_v1.json")
    p_holdout_collect.add_argument("--target", type=int, default=100)
    p_holdout_collect.add_argument("--force", action="store_true")

    p_holdout_label = sub.add_parser("label-holdout")
    p_holdout_label.add_argument("--corpus", default="boardforge/data/holdout_corpus_v1.json")
    p_holdout_label.add_argument("--intents", default="boardforge/data/holdout_intents_v1.json")
    p_holdout_label.add_argument("--out", default="boardforge/data/holdout_labels_v1_provisional.json")
    p_holdout_label.add_argument("--review", default="docs/holdout-v1-label-review.md")
    p_holdout_label.add_argument("--force", action="store_true")

    p_holdout_eval = sub.add_parser("holdout-eval")
    p_holdout_eval.add_argument("--corpus", default="boardforge/data/holdout_corpus_v1.json")
    p_holdout_eval.add_argument("--labels", default="boardforge/data/holdout_labels_v1_1_ai_audited.json")
    p_holdout_eval.add_argument("--cache", default="results/holdout-embedding-cache.json")
    p_holdout_eval.add_argument("--out", default="results/holdout-eval.json")
    p_holdout_eval.add_argument("--markdown-out", default="results/holdout-eval.md")
    p_holdout_eval.add_argument("--database-url", default=None)
    p_holdout_eval.add_argument("--top-k", type=int, default=10)
    p_holdout_eval.add_argument("--candidate-limit", type=int, default=50)
    p_holdout_eval.add_argument(
        "--execute-vertex", action="store_true",
        help="Execute the paid Vertex/PostgreSQL evaluation; without this flag only usage is planned.",
    )

    p_relevance_eval = sub.add_parser("relevance-eval")
    p_relevance_eval.add_argument("--corpus", default="boardforge/data/holdout_corpus_v1.json")
    p_relevance_eval.add_argument("--labels", default="boardforge/data/holdout_labels_v1_1_ai_audited.json")
    p_relevance_eval.add_argument("--deterministic-report", default="results/holdout-eval-v1_1-ai-audited.json")
    p_relevance_eval.add_argument("--out", default="results/relevance-verifier-eval.json")
    p_relevance_eval.add_argument("--database-url", default=None)
    p_relevance_eval.add_argument(
        "--checkpoint", default="results/relevance-verifier-eval.checkpoint.json"
    )
    p_relevance_eval.add_argument("--max-cost-usd", type=float, default=0.60)
    p_relevance_eval.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    if args.cmd == "collect-holdout":
        _print_json(asyncio.run(collect_holdout(
            args.sources, args.out, target_count=args.target, force=args.force
        )))
        return

    if args.cmd == "label-holdout":
        _print_json(label_holdout(
            args.corpus, args.intents, args.out, args.review, force=args.force
        ))
        return

    if args.cmd == "holdout-eval":
        plan = plan_vertex_usage(args.corpus, args.labels, args.cache)
        if not args.execute_vertex:
            _print_json(plan)
            return
        database_url = args.database_url or os.getenv("BOARDFORGE_HOLDOUT_DATABASE_URL", "")
        if not database_url:
            raise RuntimeError(
                "--execute-vertex requires --database-url or BOARDFORGE_HOLDOUT_DATABASE_URL"
            )
        report = asyncio.run(run_production_holdout_eval(
            database_url, args.corpus, args.labels, args.cache,
            top_k=args.top_k, candidate_limit=args.candidate_limit,
        ))
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(holdout_comparison_markdown(report), encoding="utf-8")
        _print_json({
            **plan,
            "execution_status": report["status"],
            "embedding_usage": report["embedding_usage"],
            "output": str(output_path),
            "markdown_output": str(markdown_path),
        })
        return

    if args.cmd == "relevance-eval":
        plan = plan_relevance_verifier_eval(
            args.corpus, args.labels, args.deterministic_report
        )
        if not args.execute:
            _print_json(plan)
            return
        database_url = args.database_url or os.getenv(
            "BOARDFORGE_HOLDOUT_DATABASE_URL", ""
        )
        if not database_url:
            raise RuntimeError(
                "--execute requires an isolated --database-url or "
                "BOARDFORGE_HOLDOUT_DATABASE_URL"
            )
        from boardforge.holdout_eval import assert_isolated_database_url
        from boardforge.postgres_store import PostgresStore

        assert_isolated_database_url(database_url)
        store = PostgresStore(database_url, min_size=1, max_size=2)
        try:
            verifier = RelevanceVerifier(
                store, create_relevance_provider(), enabled=True
            )
            verifier.cumulative_cost_limit_usd = args.max_cost_usd
            if hasattr(verifier.provider, "backend"):
                verifier.provider.backend.max_retries = max(0, min(
                    1, int(os.getenv("BOARDFORGE_EVAL_PROVIDER_MAX_RETRIES", "0"))
                ))
            report = asyncio.run(run_relevance_verifier_eval(
                args.corpus, args.labels, args.deterministic_report, verifier,
                checkpoint_path=args.checkpoint,
                max_estimated_cost_usd=args.max_cost_usd,
            ))
        finally:
            store.close()
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _print_json({**plan, "execution_status": report["status"], "output": str(output_path)})
        return

    if args.cmd == "benchmark":
        out = run_benchmark(args.dataset, args.top_k) if args.dataset else run_benchmark(top_k=args.top_k)
        _print_json(out)
        return

    if args.cmd == "compare":
        out = run_comparison(args.dataset, args.top_k) if args.dataset else run_comparison(top_k=args.top_k)
        if args.markdown_out:
            output_path = Path(args.markdown_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(comparison_markdown(out), encoding="utf-8")
        _print_json(out)
        return

    if args.cmd == "compare-embeddings":
        out = asyncio.run(
            run_embedding_comparison(args.dataset, args.top_k)
            if args.dataset
            else run_embedding_comparison(top_k=args.top_k)
        )
        _print_json(out)
        return

    if args.cmd == "agent-eval":
        out = run_deterministic_tool_eval(args.dataset) if args.dataset else run_deterministic_tool_eval()
        _print_json(out)
        return

    if args.cmd == "agent-eval-llm":
        out = asyncio.run(run_llm_tool_eval(args.dataset) if args.dataset else run_llm_tool_eval())
        _print_json(out)
        return

    if args.cmd == "preference-eval":
        out = run_preference_eval(args.dataset) if args.dataset else run_preference_eval()
        _print_json(out)
        return

    if args.cmd == "resume-eval":
        out = run_resume_eval(args.dataset) if args.dataset else run_resume_eval()
        _print_json(out)
        return

    if args.cmd == "discovery-worker":
        out = asyncio.run(run_pending_discoveries(create_store(), limit=args.limit))
        _print_json(out)
        return

    if args.cmd == "embed-index":
        store = create_store()
        if type(store).__name__ == "Store" and not args.allow_sqlite:
            raise RuntimeError(
                "embed-index refuses the SQLite fallback by default. "
                "Set DATABASE_URL for PostgreSQL, or pass --allow-sqlite intentionally."
            )
        out = asyncio.run(warm_job_embeddings(
            store, EmbeddingClient(budget_store=store),
            batch_size=args.batch_size,
            max_jobs=args.max_jobs,
            countries=args.country or None,
        ))
        out["store"] = type(store).__name__
        _print_json(out)
        return

    if args.cmd == "sync-sources":
        out = asyncio.run(sync_due_sources(
            create_store(), interval_seconds=args.interval_seconds,
            limit=args.limit, concurrency=args.concurrency,
        ))
        _print_json(out)
        return

    if args.cmd == "reenrich-index":
        from boardforge.enrich import reenrich_store_jobs

        store = create_store()
        try:
            _print_json(reenrich_store_jobs(store, apply=args.apply))
        finally:
            if hasattr(store, "close"):
                store.close()
        return

    if args.cmd == "snapshot-eval-corpus":
        store = create_store()
        payload = build_corpus_snapshot(
            store.active_jobs(),
            country=args.country,
            include_test=args.include_test,
        )
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _print_json({
            "output": str(output_path),
            "country": payload["country"],
            "jobs": len(payload["jobs"]),
            "excluded_test_jobs": payload["excluded_test_jobs"],
        })
        return

    if args.cmd == "real-eval":
        report = asyncio.run(
            run_real_eval(args.corpus, args.dataset, args.top_k)
            if args.dataset
            else run_real_eval(args.corpus, top_k=args.top_k)
        )
        if args.details or report.get("status") != "ok":
            _print_json(report)
        else:
            _print_json({
                "status": report["status"],
                "dataset_version": report["dataset_version"],
                "corpus_snapshot": report["corpus_snapshot"],
                "cases": report["cases"],
                "top_k": report["top_k"],
                "embedding_model": report["embedding_model"],
                "macro": report["macro"],
                "by_family": report["by_family"],
                "failure_count": report["failure_count"],
                "failures": report["failures"][:8],
            })
        return

    store = create_store()
    agent = JobAgent(store)

    if args.cmd == "sync":
        out = asyncio.run(sync_url(args.url, store)).model_dump()
    elif args.cmd == "prefs":
        result = asyncio.run(agent.update_preferences(args.profile, args.text))
        out = result.data
    elif args.cmd == "search":
        result = asyncio.run(agent.search_jobs(args.profile, args.query, args.top_k))
        out = result.data
    elif args.cmd == "learn-har":
        har = json.loads(Path(args.har_file).read_text(encoding="utf-8"))
        manifests = discover_from_har(har)
        Path(args.out).write_text(
            json.dumps([m.model_dump() for m in manifests], indent=2),
            encoding="utf-8",
        )
        out = {
            "discovered": len(manifests),
            "output": args.out,
            "manifests": [m.model_dump() for m in manifests],
        }
    elif args.cmd == "sync-manifest":
        raw = json.loads(Path(args.manifest_file).read_text(encoding="utf-8"))
        item = raw[args.index] if isinstance(raw, list) else raw
        manifest = ConnectorManifest.model_validate(item)
        jobs = asyncio.run(fetch_manifest(manifest))
        summary = store.upsert_source_jobs("learned", manifest.name, jobs)
        out = summary.model_dump()
    _print_json(out)


if __name__ == "__main__":
    main()
