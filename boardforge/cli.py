from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from boardforge.agent import JobAgent
from boardforge.agent_eval import run_deterministic_tool_eval, run_llm_tool_eval
from boardforge.benchmark import run_benchmark
from boardforge.comparison import comparison_markdown, run_comparison, run_embedding_comparison
from boardforge.discovery.har import discover_from_har
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

    store = Store()
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
