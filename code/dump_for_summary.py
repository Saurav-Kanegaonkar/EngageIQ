"""Dump un-summarized surfaced opportunities (across the demo personas, ALL sources,
matching the plan-driven Hub's widened feed) into chunk files, so a Workflow of Claude
sub-agents can write structured summaries in parallel (no NVIDIA rate limit).
Run: PYTHONPATH=code python -m dump_for_summary
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import shutil
from pathlib import Path

from engageiq import personas, rank, summarize


def main() -> None:
    rec = rank.Recommender()
    summ = summarize.Summarizer()
    all_sources = set(personas.ALL_SOURCES)
    seen: set[str] = set()
    items: list[dict] = []
    for pid, p in personas.PERSONAS.items():
        # widened feed = what the plan-driven Hub actually shows (all 5 sources)
        for r in rec.recommend(p, k=80, sources=all_sources)["ranked"]:
            oid = r["id"]
            if oid in seen or oid in summ.cache:        # skip what NVIDIA already cached
                continue
            seen.add(oid)
            items.append({"oid": oid,
                          "title": (r.get("title") or r.get("disp") or "").strip(),
                          "source": r["source"],
                          "body": (r.get("body_raw") or r.get("text") or "")[:1500]})
    cdir, rdir = Path("data/summ_chunks"), Path("data/summ_results")
    for d in (cdir, rdir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    n_chunks = max(1, min(16, math.ceil(len(items) / 10))) if items else 0
    chunks = [items[i::n_chunks] for i in range(n_chunks)] if n_chunks else []
    for i, c in enumerate(chunks):
        (cdir / f"chunk_{i}.json").write_text(json.dumps(c, indent=2))
    # machine-readable summary line for the orchestrator
    print(json.dumps({"items": len(items), "chunks": n_chunks, "cached": len(summ.cache)}))


if __name__ == "__main__":
    main()
