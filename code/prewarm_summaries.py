"""Pre-generate LLM card summaries for the demo personas' top opportunities, so the
live feed shows real summaries instantly (the API only reads the cache). Run once after
ranking changes:  PYTHONPATH=code .venv/bin/python -m prewarm_summaries
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from engageiq import personas, rank, summarize


def main() -> None:
    rec = rank.Recommender()
    summ = summarize.Summarizer()
    seen: set[str] = set()
    items: list[tuple] = []
    # The plan-driven Hub widens each persona's feed to ALL sources (its activity
    # buckets pull Dev.to/Bluesky even when the base persona did not), so prewarm
    # across all sources, otherwise Dev.to/Bluesky cards show no summary.
    all_sources = set(personas.ALL_SOURCES)
    for pid, p in personas.PERSONAS.items():
        out = rec.recommend(p, k=80, sources=all_sources)
        for r in out["ranked"]:
            oid = r["id"]
            if oid in seen:
                continue
            seen.add(oid)
            title = r.get("title") or r.get("disp") or ""
            body = r.get("body_raw") or r.get("text") or ""
            items.append((oid, title, body, r["source"]))
    print(f"prewarming summaries for {len(items)} unique items across {len(personas.PERSONAS)} personas "
          f"(already cached: {len(summ.cache)})...")
    n = summ.prewarm(items)
    print(f"done: {n} new summaries written; cache size now {len(summ.cache)}")


if __name__ == "__main__":
    main()
