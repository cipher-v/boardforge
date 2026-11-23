from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from boardforge.agent import JobAgent
from boardforge.benchmark import run_benchmark
from boardforge.discovery.har import discover_from_har
from boardforge.discovery.manifest import ConnectorManifest
from boardforge.connectors.manifest import fetch_manifest
from boardforge.store import Store
from boardforge.sync import sync_url


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

    args = parser.parse_args()
    if args.cmd == "benchmark":
        out = run_benchmark(args.dataset, args.top_k) if args.dataset else run_benchmark(top_k=args.top_k)
        print(json.dumps(out, indent=2, default=str))
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
        out = {"discovered":len(manifests),"output":args.out,"manifests":[m.model_dump() for m in manifests]}
    elif args.cmd == "sync-manifest":
        raw = json.loads(Path(args.manifest_file).read_text(encoding="utf-8"))
        item = raw[args.index] if isinstance(raw, list) else raw
        manifest = ConnectorManifest.model_validate(item)
        jobs = asyncio.run(fetch_manifest(manifest))
        summary = store.upsert_source_jobs("learned", manifest.name, jobs)
        out = summary.model_dump()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
