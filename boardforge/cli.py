from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from boardforge.agent import JobAgent
from boardforge.database import create_store
from boardforge.agent_eval import run_deterministic_tool_eval, run_llm_tool_eval
from boardforge.benchmark import run_benchmark
from boardforge.comparison import comparison_markdown, run_comparison, run_embedding_comparison
from boardforge.preference_eval import run_preference_eval
from boardforge.resume_eval import run_resume_eval
from boardforge.semantic import EmbeddingClient, warm_job_embeddings
from boardforge.workers import sync_due_sources
from boardforge.discovery.har import discover_from_har
from boardforge.discovery import run_pending_discoveries
from boardforge.discovery.manifest import ConnectorManifest
from boardforge.connectors.manifest import fetch_manifest
from boardforge.store import Store
from boardforge.sync import sync_url


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
        "--allow-sqlite",
        action="store_true",
        help="Allow embedding backfill against the SQLite fallback store.",
    )

    p_sync_sources = sub.add_parser("sync-sources")
    p_sync_sources.add_argument("--interval-seconds", type=int, default=3600)
    p_sync_sources.add_argument("--limit", type=int, default=25)
    p_sync_sources.add_argument("--concurrency", type=int, default=4)

    args = parser.parse_args()

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
            store, EmbeddingClient(),
            batch_size=args.batch_size, max_jobs=args.max_jobs,
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
