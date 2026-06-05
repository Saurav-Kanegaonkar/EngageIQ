"""Dump EVERY surfaced-but-unsummarized opportunity into chunk files, so a Workflow
of Claude sub-agents can write card summaries in parallel (no NVIDIA rate limit).

Unlike dump_for_summary.py (4 demo personas only), this sweeps a REPRESENTATIVE set:
the 4 demo personas + one synthetic profile per domain (all 15), across all 5 sources,
top-80 each. The union approximates what ANY custom profile (gallery) could surface,
so once filled, custom-profile feeds stop falling back to title-only/snippet cards.

The data is static (we are not re-scraping), so this is a one-time fill.
Run: PYTHONPATH=code python -m dump_card_gap
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import itertools
import json
import math
import shutil
import sqlite3
from pathlib import Path

from engageiq import personas, rank, summarize
from engageiq.embed import DB_PATH

CHUNK_SIZE = 15   # items per sub-agent (smaller = more agents = more parallel + careful)


def _profile(pid, doms, sources):
    label = ", ".join(d.replace("_", " ") for d in doms)
    return {"id": pid, "name": label, "goal": "work on " + label,
            "interests": [d.replace("_", " ") for d in doms], "domains": list(doms),
            "platforms": list(sources), "time_budget_hours": 5, "skills": label}


def main() -> None:
    rec = rank.Recommender()
    summ = summarize.Summarizer()
    all_sources = set(personas.ALL_SOURCES)

    conn = sqlite3.connect(str(DB_PATH))
    domains = [d for (d,) in conn.execute(
        "SELECT DISTINCT domain FROM opportunities WHERE domain IS NOT NULL")]
    conn.close()

    # representative profile set: 4 demo personas + every single domain + every
    # PAIR of domains (custom profiles are usually multi-domain, and a pair surfaces
    # items neither single-domain profile ranks into its top 80). This approximates
    # what any realistic custom profile could surface.
    profiles = [dict(p) for p in personas.PERSONAS.values()]
    for d in domains:
        profiles.append(_profile("d_" + d, [d], all_sources))
    for a, b in itertools.combinations(sorted(domains), 2):
        profiles.append(_profile(f"p_{a}_{b}", [a, b], all_sources))

    seen: set[str] = set()
    items: list[dict] = []
    for p in profiles:
        for r in rec.recommend(p, k=80, sources=all_sources)["ranked"]:
            oid = r["id"]
            if oid in seen or oid in summ.cache:   # skip dups + already-summarized
                continue
            seen.add(oid)
            items.append({
                "oid": oid,
                "title": (r.get("title") or r.get("disp") or "").strip(),
                "source": r["source"],
                "body": (r.get("body_raw") or r.get("text") or "")[:1500],
            })

    cdir, rdir = Path("data/summ_chunks"), Path("data/summ_results")
    for d in (cdir, rdir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(len(items) / CHUNK_SIZE) if items else 0
    chunks = [items[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE] for i in range(n_chunks)]
    for i, c in enumerate(chunks):
        (cdir / f"chunk_{i}.json").write_text(json.dumps(c, indent=2))

    print(json.dumps({"items": len(items), "chunks": n_chunks, "cached": len(summ.cache)}))


if __name__ == "__main__":
    main()
