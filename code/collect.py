"""EngageIQ data collector — run the streaming/ingestion pipeline (Capability 1).

Usage:
    python code/collect.py --domains all --limit 40
    python code/collect.py --domains machine_learning,devops_k8s --limit 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `engageiq` importable

from engageiq.domains import DOMAINS, DOMAIN_LABELS          # noqa: E402
from engageiq.sources.bluesky import BlueskyAdapter          # noqa: E402
from engageiq.sources.devto import DevToAdapter              # noqa: E402
from engageiq.sources.github import GitHubAdapter            # noqa: E402
from engageiq.sources.hackernews import HackerNewsAdapter    # noqa: E402
from engageiq.sources.reddit import RedditAdapter            # noqa: E402
from engageiq.storage import Store                           # noqa: E402

ADAPTERS = [HackerNewsAdapter(), DevToAdapter(), GitHubAdapter(),
            RedditAdapter(), BlueskyAdapter()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="all", help="'all' or comma-separated domain keys")
    ap.add_argument("--limit", type=int, default=40, help="items per source per domain")
    ap.add_argument("--sources", default="all", help="'all' or comma-separated source names")
    args = ap.parse_args()

    domains = DOMAINS if args.domains == "all" else args.domains.split(",")
    selected = (ADAPTERS if args.sources == "all"
                else [a for a in ADAPTERS if a.name in args.sources.split(",")])
    store = Store()
    print(f"DB start count: {store.count()} | domains: {len(domains)} | limit/source/domain: {args.limit}\n")

    for adapter in selected:
        if not adapter.available():
            print(f"[{adapter.name:11}] skipped — no credentials")
            continue
        ins = skip = 0
        for d in domains:
            i, s = store.add_many(adapter.fetch(d, args.limit))
            ins += i
            skip += s
        print(f"[{adapter.name:11}] +{ins} new, {skip} dup")

    print(f"\nTotal in DB: {store.count()}")
    print("By source :", store.counts_by("source"))
    print("By domain :", {DOMAIN_LABELS.get(k, k): v for k, v in store.counts_by("domain").items()})

    print("\nSample rows:")
    for src, dom, score, title in store.conn.execute(
        "SELECT source, domain, score, substr(title,1,68) FROM opportunities "
        "WHERE title != '' ORDER BY RANDOM() LIMIT 7"
    ):
        print(f"  [{src:10}] {DOMAIN_LABELS.get(dom, dom):20} ★{str(score):>4}  {title}")


if __name__ == "__main__":
    main()
