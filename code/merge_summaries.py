"""Merge the sub-agent summary result files (data/summ_results/chunk_*.json, each a
{oid: summary} map) into data/summaries.json, the cache the API reads. Run after the
summarize-cards Workflow finishes:  PYTHONPATH=code .venv/bin/python -m merge_summaries
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from engageiq.embed import DB_PATH
from engageiq.summarize import CACHE_PATH

RESULTS = Path("data/summ_results")


def clean(s: str) -> str:
    """Match summarize.py's house style: no markdown, no dashes, single-spaced, capped."""
    s = re.sub(r"[*#`]+", "", s or "")
    s = s.replace(" — ", ", ").replace("—", ", ").replace("–", "-")
    return " ".join(s.split())[:650]


def main() -> None:
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    before = len(cache)
    added = 0
    for f in sorted(RESULTS.glob("chunk_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        for oid, summ in d.items():
            if isinstance(summ, str) and summ.strip() and oid not in cache:
                cache[oid] = clean(summ)
                added += 1
    CACHE_PATH.write_text(json.dumps(cache))

    # per-source coverage report
    ids = set(cache)
    conn = sqlite3.connect(str(DB_PATH))
    bysrc = Counter()
    for oid, src in conn.execute("SELECT opportunity_id, source FROM opportunities"):
        if oid in ids:
            bysrc[src] += 1
    conn.close()
    print(f"merged: +{added} summaries ({before} -> {len(cache)})")
    print("cached summaries by source: " + ", ".join(f"{s}={bysrc[s]}" for s in
          ["github", "devto", "bluesky", "reddit", "hackernews"]))


if __name__ == "__main__":
    main()
