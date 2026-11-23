from __future__ import annotations

import argparse
import asyncio
import json

from boardforge.agent import JobAgent
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

    args = parser.parse_args()
    store = Store()
    agent = JobAgent(store)

    if args.cmd == "sync":
        out = asyncio.run(sync_url(args.url, store)).model_dump()
    elif args.cmd == "prefs":
        result = asyncio.run(agent.update_preferences(args.profile, args.text))
        out = result.data
    else:
        result = asyncio.run(agent.search_jobs(args.profile, args.query, args.top_k))
        out = result.data

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
